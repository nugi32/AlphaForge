"""
Stage 3: univariate analysis and go/no-go (spec section 8). TRAIN rows only; no rule search, no selection of atoms.

For every kept feature x bin x label column (net R = gross - cost) the stage reports n, mean net R, cluster-robust SE (calendar weeks),
and the LIFT over the baseline (mean net R of all train rows for the same label). Consistency over time is the share of usable periods
(years, or quarters if the train segment is short) in which the cell beats the baseline of that period.

Go/no-go (decisions D-016, D-017): hits are judged on the GROSS lift (information about price direction), not on net R, because cost in
R varies with hour and volatility regime and would create "hits" on pure noise. A "hit" is a cell with gross lift > 0, cluster
t(gross lift) >= t_min, t(gross mean) >= min_t_gross_mean (the cell itself must be positive, otherwise a cell that is merely "less bad"
than an edge-contaminated baseline, e.g. the mirror side of a long edge, would count), n >= min_n_cell and >= min_periods usable periods. A hit is "consistent" if the share of periods in which the
cell's gross mean beats the baseline of that period is >= min_frac_positive_periods. Net R, cost and the net lift are reported for every cell.
  no hits                     -> NO_GO              (stop and report)
  hits, none consistent       -> GO_REGIME_CAUTION  (strong only in some periods)
  consistent, best lift small -> GO_WEAK            (continue with low expectations)
  otherwise                   -> GO
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from edge_research.splits import Splits

VERDICTS = ("NO_GO", "GO_REGIME_CAUTION", "GO_WEAK", "GO")


def resolve_t_min(e: Dict[str, Any], n_tested: int) -> float:
    """``t_min: auto`` = one-sided Bonferroni over the tested cells at ``alpha`` (decision D-017); otherwise the configured number."""
    if str(e["t_min"]).lower() == "auto":
        return float(norm.isf(float(e["alpha"]) / max(n_tested, 1)))
    return float(e["t_min"])


def cluster_codes(index: pd.DatetimeIndex, how: str = "week") -> np.ndarray:
    if how not in ("week", "month"):
        raise ValueError(f"unknown cluster mode: {how!r}")
    per = index.to_period("W" if how == "week" else "M")
    return pd.factorize(per.asi8)[0].astype(np.int64)


def period_codes(index: pd.DatetimeIndex, min_periods: int, min_n_period: int) -> Tuple[np.ndarray, str]:
    """Years if at least ``min_periods`` years have enough rows, else quarters."""
    years = index.year.to_numpy()
    counts = pd.Series(years).value_counts()
    if int((counts >= 10 * min_n_period).sum()) >= min_periods:
        return pd.factorize(years)[0].astype(np.int64), "year"
    return pd.factorize(years * 4 + (index.quarter.to_numpy() - 1))[0].astype(np.int64), "quarter"


def cluster_mean_se(y: np.ndarray, g: np.ndarray, n_groups: int) -> Tuple[float, float, int]:
    """Mean, cluster-robust SE of the mean (CR1 style small-sample factor) and n, ignoring non-finite y."""
    v = np.isfinite(y)
    n = int(v.sum())
    if n < 2:
        return float("nan"), float("nan"), n
    yy, gg = y[v], g[v]
    mean = yy.mean()
    c = np.bincount(gg, weights=yy - mean, minlength=n_groups)
    g_eff = int((np.bincount(gg, minlength=n_groups) > 0).sum())
    if g_eff < 2:
        return float(mean), float("nan"), n
    return float(mean), float(np.sqrt(g_eff / (g_eff - 1) * (c**2).sum()) / n), n


def make_bins(x: np.ndarray, kind: str, n_bins: int, max_levels: int) -> Tuple[np.ndarray, List[Tuple[float, float]], bool]:
    """Bin id per row (-1 for non-finite), (lo, hi) per bin, and whether the feature was treated as categorical."""
    ok = np.isfinite(x)
    b = np.full(len(x), -1, dtype=np.int64)
    if not ok.any():
        return b, [], False
    levels = np.unique(x[ok])
    if kind == "time" or len(levels) <= max_levels:
        b[ok] = np.searchsorted(levels, x[ok])
        return b, [(float(v), float(v)) for v in levels], True
    r = pd.Series(x[ok]).rank(method="first").to_numpy()               # equal-frequency bins, deterministic tie order
    bi = np.minimum(((r - 1) * n_bins / ok.sum()).astype(np.int64), n_bins - 1)
    b[ok] = bi
    edges = [(float(x[ok][bi == k].min()), float(x[ok][bi == k].max())) for k in range(n_bins) if (bi == k).any()]
    return b, edges, False


def _cell_stats(b: np.ndarray, k: int, y: np.ndarray, g: np.ndarray, n_g: int, p: np.ndarray, n_p: int,
                base_period_mean: np.ndarray, min_n_period: int) -> Dict[str, np.ndarray]:
    v = (b >= 0) & np.isfinite(y)
    bv, yv, gv, pv = b[v], y[v], g[v], p[v]
    n = np.bincount(bv, minlength=k).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(bv, weights=yv, minlength=k) / n
        r = yv - mean[bv]
        key = bv * n_g + gv
        c = np.bincount(key, weights=r, minlength=k * n_g).reshape(k, n_g)
        g_eff = (np.bincount(key, minlength=k * n_g).reshape(k, n_g) > 0).sum(1).astype(np.float64)
        se = np.sqrt(g_eff / (g_eff - 1) * (c**2).sum(1)) / n
        kp = bv * n_p + pv
        n_pk = np.bincount(kp, minlength=k * n_p).reshape(k, n_p).astype(np.float64)
        mp = np.bincount(kp, weights=yv, minlength=k * n_p).reshape(k, n_p) / n_pk
        used = n_pk >= min_n_period
        pos = used & (mp > base_period_mean[None, :])
        used_n = used.sum(1)
        frac = np.where(used_n > 0, pos.sum(1) / np.maximum(used_n, 1), np.nan)
    return {"n": n, "mean": mean, "se": se, "periods_used": used_n.astype(np.int64), "frac_pos": frac}


@dataclass
class EdaResult:
    baseline: pd.DataFrame
    baseline_by_period: pd.DataFrame
    univariate: pd.DataFrame
    time_profile: pd.DataFrame
    hits: pd.DataFrame
    verdict: Dict[str, Any]
    summary_md: str


def run_univariate(features: pd.DataFrame, labels: pd.DataFrame, kinds: Dict[str, str], kept: Sequence[str], cfg: Dict[str, Any],
                   splits: Splits, first_valid_row: int = 0) -> EdaResult:
    e = cfg["eda"]
    sl = slice(max(splits.train.start, int(first_valid_row)), splits.train.stop)          # TRAIN ONLY
    X, L = features.iloc[sl], labels.iloc[sl]
    idx = X.index
    g = cluster_codes(idx, e["cluster"])
    n_g = int(g.max()) + 1
    p, pname = period_codes(idx, int(e["min_periods"]), int(e["min_n_period"]))
    n_p = int(p.max()) + 1

    names = [c[: -len("_gross")] for c in L.columns if c.endswith("_gross")]
    gross = {nm: L[f"{nm}_gross"].to_numpy(np.float64) for nm in names}
    cost = {nm: L[f"{nm}_cost"].to_numpy(np.float64) for nm in names}
    net = {nm: gross[nm] - cost[nm] for nm in names}

    # ---- baseline (random entry) -----------------------------------------------------------------------------------
    brow, bper = [], []
    base_mean, base_pm, base_mean_g, base_pm_g = {}, {}, {}, {}
    for nm in names:
        y = net[nm]
        v = np.isfinite(y)
        base_mean[nm] = float(y[v].mean()) if v.any() else float("nan")
        cnt = np.bincount(p[v], minlength=n_p).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            base_pm[nm] = np.bincount(p[v], weights=y[v], minlength=n_p) / cnt
        vg = np.isfinite(gross[nm])
        base_mean_g[nm] = float(gross[nm][vg].mean()) if vg.any() else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            base_pm_g[nm] = np.bincount(p[vg], weights=gross[nm][vg], minlength=n_p) / np.bincount(p[vg], minlength=n_p)
        mg, seg, _ = cluster_mean_se(gross[nm], g, n_g)
        mc = float(np.nanmean(cost[nm])) if v.any() else float("nan")
        mn, sen, nn = cluster_mean_se(y, g, n_g)
        t_net = mn / sen if sen and np.isfinite(sen) and sen > 0 else float("nan")
        t_gross = mg / seg if seg and np.isfinite(seg) and seg > 0 else float("nan")
        flag = ""
        if np.isfinite(t_gross) and abs(t_gross) >= 3:
            flag = "gross drift is significant: check data drift or cost/label bugs"
        elif np.isfinite(t_net) and mn > 0 and t_net >= 2:
            flag = "baseline net R is positive: suspicious"
        brow.append({"label": nm, "n": nn, "mean_gross_R": mg, "mean_cost_R": mc, "mean_net_R": mn, "se_net": sen, "t_net": t_net, "t_gross": t_gross, "flag": flag})
        for j in range(n_p):
            bper.append({"label": nm, "period": j, "n": int(cnt[j]), "mean_net_R": base_pm[nm][j]})
    baseline = pd.DataFrame(brow)
    baseline_by_period = pd.DataFrame(bper)

    t_used = float("nan")
    # ---- per feature x bin x label -----------------------------------------------------------------------------------
    parts = []
    for feat in kept:
        b, edges, categorical = make_bins(X[feat].to_numpy(np.float64), kinds.get(feat, "numeric"), int(e["n_bins"]), int(e["max_categorical_levels"]))
        k = len(edges)
        if k == 0:
            continue
        lo = np.array([a for a, _ in edges])
        hi = np.array([c for _, c in edges])
        for nm in names:
            s = _cell_stats(b, k, net[nm], g, n_g, p, n_p, base_pm[nm], int(e["min_n_period"]))
            sg = _cell_stats(b, k, gross[nm], g, n_g, p, n_p, base_pm_g[nm], int(e["min_n_period"]))
            parts.append(pd.DataFrame({
                "feature": feat, "categorical": categorical, "bin": np.arange(k), "bin_lo": lo, "bin_hi": hi, "label": nm,
                "n": s["n"].astype(np.int64), "mean_R": s["mean"], "se": s["se"], "lift_R": s["mean"] - base_mean[nm],
                "mean_gross_R": sg["mean"], "se_gross": sg["se"], "lift_gross_R": sg["mean"] - base_mean_g[nm],
                "periods_used": sg["periods_used"], "frac_pos_periods": sg["frac_pos"]}))
    uni = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not uni.empty:
        uni["direction"] = uni["label"].str.split("_").str[0]
        uni["family"] = uni["label"].str.split("_", n=1).str[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            uni["t_lift"] = uni["lift_R"] / uni["se"]
            uni["t_lift_gross"] = uni["lift_gross_R"] / uni["se_gross"]
            uni["t_gross"] = uni["mean_gross_R"] / uni["se_gross"]
        uni["ci_lo"] = uni["mean_R"] - 1.96 * uni["se"]
        uni["ci_hi"] = uni["mean_R"] + 1.96 * uni["se"]
        n_tested_cells = int((uni["n"] >= e["min_n_cell"]).sum())
        t_used = resolve_t_min(e, n_tested_cells)
        uni["hit"] = (uni["lift_gross_R"] > 0) & (uni["t_lift_gross"] >= t_used) & (uni["t_gross"] >= e["min_t_gross_mean"]) & (uni["n"] >= e["min_n_cell"]) & (uni["periods_used"] >= e["min_periods"])
        uni["consistent"] = uni["hit"] & (uni["frac_pos_periods"] >= e["min_frac_positive_periods"])

    # ---- time profile (hour, day of week) for the time-stop labels ------------------------------------------------------
    tp = []
    tnames = [nm for nm in names if nm.split("_", 1)[1].startswith("T")]
    for by, key in (("hour", idx.hour.to_numpy()), ("dayofweek", idx.dayofweek.to_numpy())):
        for nm in tnames:
            d = pd.DataFrame({"k": key, "gross": gross[nm], "cost": cost[nm], "net": net[nm]}).dropna()
            a = d.groupby("k").agg(n=("net", "size"), mean_gross_R=("gross", "mean"), mean_cost_R=("cost", "mean"), mean_net_R=("net", "mean")).reset_index()
            a.insert(0, "label", nm)
            a.insert(0, "by", by)
            tp.append(a.rename(columns={"k": "value"}))
    time_profile = pd.concat(tp, ignore_index=True) if tp else pd.DataFrame()

    # ---- verdict -----------------------------------------------------------------------------------------------------
    tested = uni[uni["n"] >= e["min_n_cell"]] if not uni.empty else uni
    hits = uni[uni["hit"]].sort_values("t_lift_gross", ascending=False) if not uni.empty else uni
    cons = hits[hits["consistent"]] if not hits.empty else hits
    if hits.empty:
        verdict_name = "NO_GO"
    elif cons.empty:
        verdict_name = "GO_REGIME_CAUTION"
    elif cons["lift_gross_R"].max() < e["weak_lift_R"]:
        verdict_name = "GO_WEAK"
    else:
        verdict_name = "GO"
    n_tested = int(len(tested))
    verdict = {
        "verdict": verdict_name, "period_unit": pname, "n_periods": n_p, "n_clusters": n_g, "train_rows": int(len(X)),
        "n_cells_tested": n_tested, "n_hits": int(len(hits)), "n_consistent_hits": int(len(cons)),
        "expected_false_hits_if_independent": float(n_tested * norm.sf(t_used)) if n_tested else 0.0,
        "features_with_hits": sorted(hits["feature"].unique().tolist()) if not hits.empty else [],
        "features_with_consistent_hits": sorted(cons["feature"].unique().tolist()) if not cons.empty else [],
        "best_consistent_gross_lift_R": float(cons["lift_gross_R"].max()) if not cons.empty else None,
        "baseline_flags": baseline.loc[baseline["flag"] != "", "label"].tolist(),
        "t_min_used": t_used, "stop_recommended": verdict_name == "NO_GO", "train_slice": [sl.start, sl.stop],
    }
    return EdaResult(baseline, baseline_by_period, uni, time_profile, hits, verdict, _summary(baseline, hits, cons, verdict, e))


def _summary(baseline: pd.DataFrame, hits: pd.DataFrame, cons: pd.DataFrame, v: Dict[str, Any], e: Dict[str, Any]) -> str:
    L = [f"# Stage 3 univariate summary", "", f"**Verdict: {v['verdict']}**", ""]
    L += [f"- Train rows: {v['train_rows']}; weekly clusters: {v['n_clusters']}; consistency unit: {v['period_unit']} ({v['n_periods']} periods)",
          f"- Cells tested (n >= {e['min_n_cell']}): {v['n_cells_tested']}; hits (gross lift > 0, t >= {v['t_min_used']:.2f}): {v['n_hits']}; consistent hits: {v['n_consistent_hits']}",
          f"- Expected false hits if cells were independent: {v['expected_false_hits_if_independent']:.2f} (t_min is a Bonferroni bound; cells are correlated, so it is conservative)", ""]
    L += ["## Baseline (random entry, train)", "", "| label | n | gross R | cost R | net R | t net |", "|---|---|---|---|---|---|"]
    show = baseline.sort_values("label").head(12)
    for _, r in show.iterrows():
        L.append(f"| {r['label']} | {int(r['n'])} | {r['mean_gross_R']:.4f} | {r['mean_cost_R']:.4f} | {r['mean_net_R']:.4f} | {r['t_net']:.2f} |")
    if len(baseline) > 12:
        L.append(f"| ... {len(baseline) - 12} more labels in baseline.csv | | | | | |")
    L.append("")
    if v["baseline_flags"]:
        L += ["**Baseline flags (investigate before trusting anything below):** " + ", ".join(v["baseline_flags"]), ""]
    else:
        L += ["Baseline check passed: no significant gross drift and no positive random-entry net R.", ""]
    if hits.empty:
        L += ["## Hits", "", "No cell passed the hit rule. Per the spec this means: stop and report; consider another timeframe, instrument or "
              "feature family. Caveat (NOTES Q-003): an edge that exists only as a CONJUNCTION of conditions can be invisible to a "
              "one-feature view at this sample size, so NO_GO is a recommendation to review, not proof that no edge exists.", ""]
    else:
        L += ["## Features with hits (best cell per feature)", "", "| feature | hit cells | consistent | best label | bin | gross lift R | net mean R | t | positive periods |", "|---|---|---|---|---|---|---|---|---|"]
        best = hits.sort_values("t_lift_gross", ascending=False).groupby("feature", sort=False).head(1)
        for _, r in best.head(25).iterrows():
            nh = int((hits["feature"] == r["feature"]).sum())
            nc = int(((hits["feature"] == r["feature"]) & hits["consistent"]).sum())
            L.append(f"| {r['feature']} | {nh} | {nc} | {r['label']} | {int(r['bin'])} ({r['bin_lo']:.4g}..{r['bin_hi']:.4g}) | {r['lift_gross_R']:.3f} | {r['mean_R']:.3f} | {r['t_lift_gross']:.1f} | {r['frac_pos_periods']:.2f} |")
        L.append("")
        if v["verdict"] == "GO_REGIME_CAUTION":
            L += ["Hits exist but none is consistent across periods: treat as regime dependent (spec chapter 11).", ""]
        elif v["verdict"] == "GO_WEAK":
            L += [f"Consistent patterns exist but the best gross lift is below {e['weak_lift_R']} R: continue with low expectations.", ""]
    L += ["## Notes", "", "- Hits use the GROSS lift over the baseline of the same label; the cell may still have negative net R after costs (see net mean R). CI columns in univariate.csv refer to net R.",
          "- This stage selects nothing: thresholds and atoms are built in Stage 4. Use this output only to see where information may sit.", ""]
    return "\n".join(L)


def write_eda(res: EdaResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.univariate.round(6).to_csv(out / "univariate.csv", index=False)
    res.baseline.round(6).to_csv(out / "baseline.csv", index=False)
    res.baseline_by_period.round(6).to_csv(out / "baseline_by_period.csv", index=False)
    res.time_profile.round(6).to_csv(out / "time_profile.csv", index=False)
    res.hits.round(6).to_csv(out / "hits.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(res.verdict, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text(res.summary_md, encoding="utf-8")
