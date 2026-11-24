"""Stage 9 (M7): null test over the WHOLE search (spec 13).

Question: is the best score found on the real data better than the best score the identical search finds on data with no feature-label relation?

* The search statistic is computed by ONE function (``search_statistic``) used for the real data and for every replicate: coarse screen, exact statistics
  with lift and minimum episodes, F2. Only the alignment of the label table changes.
* Replicates rotate the label rows circularly INSIDE the train segment (never touching validation or holdout) by an offset in [embargo, n_train - embargo].
  The rows of the label table move as whole rows, so the cross-section of the 24 exit families and the cost columns stay intact.
* Variant ``random``: any offset. Variant ``calendar``: offsets that are multiples of one week, which keep hour/weekday alignment (13.4): a rule that
  beats the random null but not the calendar null is ``calendar_driven`` (a session effect, not the indicator).
* S = largest cluster t (over all 24 exit families with enough episodes) among the rules that pass F1 and F2; 0 when no rule passes.
  S_cont = largest exact cluster t over all screen survivors and families with enough episodes, whatever the other filters say: a continuous statistic
  that is defined for every replicate.
* p_global = (1 + #{S_null >= S_real}) / (R + 1). False-discovery estimate = mean(rules passing F1-F2 in the null) / (rules passing F1-F2 on the real data).
* Each replicate is stored as its own JSON file with the fingerprint of everything it depends on, so a stopped run resumes where it stopped and a changed
  config never reuses stale replicates.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_research.atoms.atoms import AtomSet
from edge_research.discovery.candidates import build_candidates
from edge_research.discovery.deep import deep_scope_note, expand_survivors, normalise_rules
from edge_research.discovery.filters import run_filters
from edge_research.discovery.pairs import coarse_screen
from edge_research.discovery.trees import induce_rules
from edge_research.splits import max_hold_bars


def deep_rules_for_search(aset: AtomSet, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice, surv: pd.DataFrame,
                          feature_table: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The deep part of the search (stage 5b) as the null test needs it: identical inputs, identical code, only the labels differ.

    ``deep.apriori.include_in_null`` / ``deep.trees.include_in_null`` decide what runs INSIDE the search statistic. The flags are read the
    same way for the real data and for every replicate (spec 13.3: only the label alignment may change), and the scope is reported.
    """
    d = cfg.get("deep", {})
    parts: List[pd.DataFrame] = []
    used: List[str] = []
    if bool(d.get("apriori", {}).get("enabled")) and bool(d.get("apriori", {}).get("include_in_null")):
        ap, _ = expand_survivors(aset, surv, label_table, cfg, rows)
        if len(ap):
            parts.append(ap.assign(source="apriori"))
        used.append("apriori")
    note = deep_scope_note(cfg)
    if bool(d.get("trees", {}).get("enabled")) and bool(d.get("trees", {}).get("include_in_null")):
        if feature_table is None:
            # The real path always passes the feature table; a caller that cannot supply it would silently shrink
            # the null procedure below the real search (spec 13.3 forbids that). Report the shrink instead of hiding it.
            note += " ATTENTION: the tree engine is REQUESTED inside the null but was SKIPPED here because no feature table was given."
        else:
            tr, _ = induce_rules(aset, feature_table, label_table, cfg, rows)
            if len(tr):
                parts.append(tr)
            used.append("trees")
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["atoms", "direction", "score", "source"])
    return frame, {"used": used, "note": note}


def search_statistic(aset: AtomSet, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice, feature_table: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """The search procedure, once. Identical for real data and for every null replicate."""
    surv, _ = coarse_screen(aset, label_table, cfg, rows)
    deep, dscope = deep_rules_for_search(aset, label_table, cfg, rows, surv, feature_table)
    rules = normalise_rules(surv, aset, "pairs")
    if len(deep):
        rules = pd.concat([rules, normalise_rules(deep, aset, "deep")], ignore_index=True)
    cand, fam, cinfo = build_candidates(rules, aset, label_table, cfg, rows)
    out: Dict[str, Any] = {"n_surv": int(len(surv)), "n_deep": int(len(deep)), "n_cand": int(len(cand)), "S_cont": float(cinfo["max_t_exact"]),
                           "S": 0.0, "n_pass_f2": 0, "n_pass_all": 0, "deep_used": "+".join(dscope["used"]) if dscope["used"] else "none",
                           "deep_note": dscope["note"]}
    if cand.empty:
        return out
    log, passed, _ = run_filters(cand, fam, aset, label_table, cfg, rows)
    min_ep = int(cfg["candidates"]["min_episodes_train"])
    ok = set(log.loc[log["pass_f2"], "cand_id"])
    out["n_pass_f2"], out["n_pass_all"] = len(ok), int(len(passed))
    if ok:
        f = fam[fam["cand_id"].isin(ok) & (fam["n_episodes"] >= min_ep) & fam["t"].notna()]
        out["S"] = float(f["t"].max()) if len(f) else 0.0
    return out


def rotate_labels(label_table: pd.DataFrame, rows: slice, offset: int) -> pd.DataFrame:
    """Copy of the label table whose rows ``rows`` are rotated circularly by ``offset`` (row i of the segment gets the label row i - offset). Other rows are untouched."""
    out = label_table.copy()
    seg = label_table.iloc[rows].to_numpy()
    out.iloc[rows] = np.roll(seg, int(offset), axis=0)
    return out


def estimate_week_bars(index: pd.DatetimeIndex) -> int:
    """Bars per calendar week = the MOST COMMON number of bars in a calendar week (partial first / last weeks and holiday weeks do not matter).
    A calendar-variant offset must be a multiple of this number to keep hour-of-day AND weekday alignment, so an off-by-one estimate would drift."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2 or (idx[-1] - idx[0]).total_seconds() <= 0:
        return 1
    week = (idx.normalize() - pd.to_timedelta(idx.dayofweek, unit="D")).asi8
    counts = np.unique(week, return_counts=True)[1]
    vals, freq = np.unique(counts, return_counts=True)
    best = vals[freq == freq.max()]
    if freq.max() >= 2:
        return int(best.max())
    span_days = (idx[-1] - idx[0]).total_seconds() / 86400.0
    return max(1, int(round(len(idx) / (span_days / 7.0))))


def rotation_offsets(n_train: int, embargo: int, n: int, rng: np.random.Generator, step: Optional[int] = None) -> np.ndarray:
    """``n`` offsets in [embargo, n_train - embargo]; with ``step`` only multiples of it. PREFIX-STABLE: the first k offsets do not depend on ``n``, so a run with
    R = 50 can later be extended to R = 100 without recomputing. Repeats appear only when there are fewer distinct offsets than ``n`` (whole permutations are appended)."""
    lo, hi = int(embargo), int(n_train - embargo)
    if hi < lo:
        raise ValueError(f"train segment too short for rotation with embargo {embargo}: {n_train} rows")
    pool = np.arange(lo, hi + 1) if not step else np.arange(((lo + step - 1) // step) * step, hi + 1, step)
    if pool.size == 0:
        raise ValueError(f"no offset that is a multiple of {step} between {lo} and {hi}")
    parts: List[np.ndarray] = []
    while sum(len(x) for x in parts) < n:
        parts.append(rng.permutation(pool))
    return np.concatenate(parts)[:n]


def replicate_fingerprint(base_fp: str, variant: str, rep: int, offset: int) -> str:
    return hashlib.sha256(f"{base_fp}|{variant}|{rep}|{offset}".encode()).hexdigest()[:16]


# Version of the null search PROCEDURE itself. It is folded into the base fingerprint so that a change to the
# procedure (search_statistic / deep_rules_for_search, e.g. the deep engines joining the replicates) invalidates
# every stored replicate and real.json even when data, config and seed are unchanged. Bump it on any such change.
SEARCH_VERSION = "2"


def null_base_fingerprint(upstream_fp: str, seed: Any) -> str:
    """Base fingerprint shared by the real statistic and all replicates: the upstream filters stage, the seed and SEARCH_VERSION."""
    return hashlib.sha256(f"{upstream_fp}|{seed}|{SEARCH_VERSION}".encode()).hexdigest()[:16]


def run_null(aset: AtomSet, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice, out_dir: str | Path, base_fp: str,
             max_replicates_this_call: Optional[int] = None, progress: Optional[Callable[[str], None]] = None,
             feature_table: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Real statistic + all replicates of all variants. Resumable: finished replicate files with a matching fingerprint are reused.

    ``feature_table`` is only needed when the deep stage runs inside the null (``deep.*.include_in_null``); without it the tree engine is
    skipped inside the statistic and the scope note says so.
    """
    n_cfg = cfg["nulltest"]
    out = Path(out_dir)
    rep_dir = out / "replicates"
    rep_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    budget = n_cfg.get("max_seconds")
    real_path = rep_dir / "real.json"
    real = None
    if real_path.exists():
        j = json.loads(real_path.read_text(encoding="utf-8"))
        if j.get("fingerprint") == base_fp:
            real = j
    if real is None:
        real = {**search_statistic(aset, label_table, cfg, rows, feature_table), "fingerprint": base_fp}
        real_path.write_text(json.dumps(real), encoding="utf-8")
    n_tr = rows.stop - rows.start
    embargo = max_hold_bars(cfg)
    week = estimate_week_bars(label_table.index[rows]) if str(n_cfg["week_bars"]) == "auto" else int(n_cfg["week_bars"])
    R = int(n_cfg["replicates"])
    done = computed = 0
    complete = True
    for vi, variant in enumerate(n_cfg["variants"]):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 * (vi + 1))
        offs = rotation_offsets(n_tr, embargo, R, rng, step=week if variant == "calendar" else None)
        for rep, off in enumerate(offs):
            f = rep_dir / f"{variant}_{rep:04d}.json"
            fp = replicate_fingerprint(base_fp, variant, rep, int(off))
            if f.exists():
                j = json.loads(f.read_text(encoding="utf-8"))
                if j.get("fingerprint") == fp:
                    done += 1
                    continue
            if (max_replicates_this_call is not None and computed >= max_replicates_this_call) or (budget is not None and time.time() - t_start > float(budget)):
                complete = False
                continue
            t0 = time.time()
            rec = search_statistic(aset, rotate_labels(label_table, rows, int(off)), cfg, rows, feature_table)
            rec.update(variant=variant, rep=rep, offset=int(off), fingerprint=fp, seconds=round(time.time() - t0, 2))
            f.write_text(json.dumps(rec), encoding="utf-8")
            computed += 1
            done += 1
            if progress:
                progress(f"{variant} {rep + 1}/{R} S={rec['S']:.2f} S_cont={rec['S_cont']:.2f}")
    return {"real": real, "week_bars": week, "embargo": embargo, "n_done": done, "n_computed": computed, "complete": complete, "n_expected": R * len(n_cfg["variants"])}


def load_distribution(out_dir: str | Path, base_fp: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for f in sorted((Path(out_dir) / "replicates").glob("*_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        if j.get("fingerprint") == replicate_fingerprint(base_fp, j["variant"], j["rep"], j["offset"]):
            rows.append({k: j[k] for k in ("variant", "rep", "offset", "S", "S_cont", "n_surv", "n_cand", "n_pass_f2", "n_pass_all", "seconds")})
    return pd.DataFrame(rows)


def p_global(s_null: np.ndarray, s_real: float) -> float:
    s_null = np.asarray(s_null, dtype=float)
    return float((1 + int((s_null >= s_real).sum())) / (len(s_null) + 1))


def null_summary(real: Dict[str, Any], dist: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Thresholds, p_global, false-discovery estimate and the G1 verdict, per variant."""
    q = float(cfg["nulltest"]["quantile"])
    out: Dict[str, Any] = {"real": {k: real[k] for k in ("S", "S_cont", "n_surv", "n_cand", "n_pass_f2", "n_pass_all")}, "variants": {}, "min_replicates": int(cfg["nulltest"]["min_replicates"])}
    out["real"]["deep_used"] = real.get("deep_used", "unknown")
    out["real"]["deep_note"] = real.get("deep_note", "the null scope of this stored real.json is unknown (written before the scope was recorded).")
    for v, d in dist.groupby("variant"):
        R = len(d)
        item: Dict[str, Any] = {"n_replicates": R, "below_minimum": bool(R < int(cfg["nulltest"]["min_replicates"]))}
        for stat in ("S", "S_cont"):
            thr = float(np.quantile(d[stat].to_numpy(float), q, method="higher"))
            item[f"{stat}_threshold"] = thr
            item[f"{stat}_null_max"] = float(d[stat].max())
            item[f"p_global_{stat}"] = p_global(d[stat].to_numpy(), float(real[stat]))
        item["null_pass_f2_mean"] = float(d["n_pass_f2"].mean())
        item["null_with_any_pass"] = float((d["n_pass_f2"] > 0).mean())
        rp = real["n_pass_f2"]
        item["false_discovery_estimate"] = float(item["null_pass_f2_mean"] / rp) if rp > 0 else float("nan")
        out["variants"][v] = item
    rv = out["variants"].get("random")
    if rv is not None:
        fd = rv["false_discovery_estimate"]
        out["g1_pass"] = bool(real["n_pass_f2"] > 0 and rv["p_global_S"] <= (1 - q) + 1e-9 and np.isfinite(fd) and fd < 0.5 and not rv["below_minimum"])
        out["g1_reason"] = ("passes: p_global_S={:.3f} <= {:.2f}, false discovery {:.3f} < 0.5".format(rv["p_global_S"], 1 - q, fd) if out["g1_pass"] else
                            "fails: " + "; ".join(r for r, bad in (("no rule passes F1-F2 on the real data", real["n_pass_f2"] == 0), (f"p_global_S={rv['p_global_S']:.3f} > {1 - q:.2f}", rv["p_global_S"] > (1 - q) + 1e-9),
                                                                   (f"false discovery {fd:.3f} >= 0.5 or undefined", not (np.isfinite(fd) and fd < 0.5)),
                                                                   (f"only {rv['n_replicates']} replicates (< {out['min_replicates']})", rv["below_minimum"])) if bad))
    else:
        out["g1_pass"], out["g1_reason"] = False, "no random-variant replicates"
    return out


def calendar_driven(t_rule: float, thr_random: float, thr_calendar: float) -> Dict[str, bool]:
    """A rule that beats the random null but not the calendar null is calendar driven (spec 13.4)."""
    beats_random = bool(t_rule > thr_random)
    beats_calendar = bool(t_rule > thr_calendar)
    return {"beats_random": beats_random, "beats_calendar": beats_calendar, "calendar_driven": beats_random and not beats_calendar}


def rule_flags(chosen: pd.DataFrame, summary: Dict[str, Any]) -> pd.DataFrame:
    """Frozen representatives vs the null thresholds. Like for like: S is a MAXIMUM over exit families, so the rule's statistic is its best-family t (``best_t``), not the t of the
    later-chosen exit (which can only be lower). Without calendar replicates (partial run, config without the variant) the calendar comparison is UNKNOWN: NaN,
    never a silent ``calendar_driven = beats_random`` (spec 13.4 needs BOTH variants to decide)."""
    reps = chosen[chosen["is_representative"]].copy()
    rv, cv = summary["variants"].get("random"), summary["variants"].get("calendar")
    if reps.empty or rv is None:
        return reps.assign(beats_random=False, beats_calendar=False, calendar_driven=False)
    def one(t: float) -> Dict[str, Any]:
        beats_random = bool(t > rv["S_threshold"])
        if cv is None:
            return {"beats_random": beats_random, "beats_calendar": np.nan, "calendar_driven": np.nan}
        return calendar_driven(t, rv["S_threshold"], cv["S_threshold"])
    flags = [one(float(t)) for t in reps["best_t"]]
    return pd.concat([reps.reset_index(drop=True), pd.DataFrame(flags)], axis=1)


def null_summary_md(summary: Dict[str, Any], flags: pd.DataFrame, run: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    r = summary["real"]
    L = ["# Stage 9 - null test over the whole search (labels rotated inside the train segment)", "",
         f"- Real data: S = {r['S']:.2f} (max cluster t among rules passing F1-F2), S_cont = {r['S_cont']:.2f}, rules passing F1-F2: {r['n_pass_f2']}, all filters: {r['n_pass_all']}.",
         f"- Replicates per variant: {cfg['nulltest']['replicates']} (minimum {summary['min_replicates']}); {run['n_done']} of {run['n_expected']} done; complete: {run['complete']}; week = {run['week_bars']} bars; embargo = {run['embargo']} bars.",
         f"- Deep scope (engines priced by the null statistic): {r.get('deep_used', 'unknown')} - {r.get('deep_note', '')}", ""]
    for v, it in summary["variants"].items():
        L += [f"## Variant `{v}` ({it['n_replicates']} replicates)" + ("  - BELOW THE MINIMUM NUMBER OF REPLICATES" if it["below_minimum"] else ""),
              f"- S: threshold (q={cfg['nulltest']['quantile']}) = {it['S_threshold']:.2f}, null max = {it['S_null_max']:.2f}, p_global = {it['p_global_S']:.4f}",
              f"- S_cont: threshold = {it['S_cont_threshold']:.2f}, null max = {it['S_cont_null_max']:.2f}, p_global = {it['p_global_S_cont']:.4f}",
              f"- Replicates with any rule passing F1-F2: {it['null_with_any_pass']:.0%}; mean rules passing per replicate: {it['null_pass_f2_mean']:.2f}; false-discovery estimate = {it['false_discovery_estimate']:.3f}", ""]
    L += [f"**G1 (null test): {'PASS' if summary['g1_pass'] else 'FAIL'}** - {summary['g1_reason']}", ""]
    L += ["A resolution note: with R replicates the smallest possible p_global is 1/(R+1). The search's own thresholds make most null replicates empty (S = 0), so S_cont is the more informative continuous comparison."]
    if len(flags):
        unknown = " ?" if flags["calendar_driven"].isna().any() else ""   # calendar variant missing (partial run): the flag is unknown, not False
        L += ["", f"## Frozen representatives against the null thresholds{unknown}", "",
              "| rule A | rule B | exit | best-family t | beats random | beats calendar | calendar_driven |", "|---|---|---|---|---|---|---|"]
        for x in flags.head(20).itertuples():
            bc = "?" if pd.isna(x.beats_calendar) else str(x.beats_calendar)
            cd = "?" if pd.isna(x.calendar_driven) else str(x.calendar_driven)
            L.append(f"| {x.expr_a} | {x.expr_b} | {x.chosen_family} | {x.best_t:.2f} | {x.beats_random} | {bc} | {cd} |")
    return "\n".join(L) + "\n"


def write_null(summary: Dict[str, Any], dist: pd.DataFrame, flags: pd.DataFrame, run: Dict[str, Any], cfg: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    dist.to_csv(out / "null_distribution.csv", index=False)
    flags.to_csv(out / "rule_flags.csv", index=False)
    (out / "null_info.json").write_text(json.dumps({"summary": summary, "run": {k: v for k, v in run.items() if k != "real"}}, indent=2, default=float), encoding="utf-8")
    (out / "summary.md").write_text(null_summary_md(summary, flags, run, cfg), encoding="utf-8")
