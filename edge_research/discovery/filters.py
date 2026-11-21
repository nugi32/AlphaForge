"""Stage 7 (M5b): candidate filters F2 (time stability), F7 (cost stress), F4 (threshold plateau), F6 (monotonicity, soft) and dedup again (spec 11).

All filters look at the candidate's ``best_family`` (the passing family with the highest cluster t; a SELECTED choice, which is why F2/F4/F7 are
checked on it rather than trusted from Stage 6). Train rows only. Failing candidates stay visible in ``filter_log`` with every metric.

* F2: folds, periods (years; quarters if the train spans fewer than ``min_periods`` years), concentration of the profit (period share, top-5
  episodes), leave-one-period-out. All on the trigger rows with a finite label, episodes as in Stage 6.
* F7: ``mean_gross - mult * cost`` > 0 on the same rows (the cost of a family is linear in spread and slippage, so this is exact).
* F4: shift each atom by +-1..steps neighbours of the SAME feature and operator (the quantile ladder), the other atom fixed; the median
  neighbour mean must be >= ``plateau_min_ratio`` x the original and >= ``plateau_min_pos_share`` of the neighbours positive. A rule with no
  numeric atom has no ladder: it passes with ``plateau_na`` and score 1.0.
* F6 (soft, not a gate): mean |Spearman| between the threshold ladder and the result, no sign reversal; ``pass_f6_soft`` = consistent AND |rho| >= ``mono_min_abs_rho``.
* F8 (spec 11): backward elimination on the VALIDATION rows (``_f8_backward_eliminate``) — an atom is dropped when the simpler rule does not fall
  beyond noise; the simplified rule's stage 7 metrics and family rows are recomputed, so no stale column reaches the gates.
* F9 again (spec 11): leader clustering of the passing candidates; a candidate joins the best-ranked representative with trigger-row Jaccard >=
  ``dedup_jaccard`` OR a per-bar P&L correlation >= ``dedup_corr`` (its own best-family net R on train). The rank is SIMPLEST first (fewest atoms),
  then the best plateau score, so the representative is the simplest rule of the cluster. The plateau score is the SMOOTHED result, the mean of the
  original and its finite neighbours (R units); a ratio to the original would reward loose thresholds whose neighbours happen to be better (NOTES D-031).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from edge_research.atoms.atoms import Atom, AtomSet, compatible
from edge_research.discovery.candidates import FamilyLabels, _MaskCache, build_family_labels, evaluate_rule, parent_order
from edge_research.discovery.deep import rule_mask
from edge_research.discovery.episodes import episode_ids
from edge_research.discovery.pairs import DIRECTIONS, family_hold, fold_bounds
from edge_research.eda.univariate import cluster_mean_se
from edge_research.labels.builder import net_R
from edge_research.splits import make_splits


def period_keys(index: pd.DatetimeIndex, min_periods: int) -> Tuple[np.ndarray, str]:
    """Calendar year per row, or year*4+quarter when the rows span fewer than ``min_periods`` distinct years."""
    y = index.year.to_numpy()
    if len(np.unique(y)) >= min_periods:
        return y.astype(np.int64), "year"
    return (y * 4 + (index.month.to_numpy() - 1) // 3).astype(np.int64), "quarter"


def time_stability(idx: np.ndarray, yy: np.ndarray, ep: np.ndarray, period: np.ndarray, bounds: np.ndarray, c: Dict[str, Any]) -> Dict[str, Any]:
    """F2 on observations ``yy`` at row positions ``idx`` (episode ids ``ep``, period key per observation ``period``)."""
    n = len(yy)
    total = float(yy.sum())
    out: Dict[str, Any] = {"n_obs": n, "total_profit": total}
    nf = len(bounds) - 1
    fold = np.searchsorted(np.asarray(bounds), idx, side="right") - 1
    f_sum = np.bincount(fold, weights=yy, minlength=nf)[:nf]
    f_cnt = np.bincount(fold, minlength=nf)[:nf]
    out["pos_folds"] = int(((f_cnt > 0) & (f_sum > 0)).sum())
    keys, inv = np.unique(period, return_inverse=True)
    p_sum = np.bincount(inv, weights=yy, minlength=len(keys))
    p_cnt = np.bincount(inv, minlength=len(keys))
    p_ep = np.array([len(np.unique(ep[inv == k])) for k in range(len(keys))])
    usable = p_ep >= int(c["min_episodes_period"])
    out["n_periods"] = int(usable.sum())
    if usable.any():
        out["pos_period_share"] = float((p_sum[usable] > 0).mean())
        out["loyo_share"] = float(np.mean([(total - p_sum[k]) / (n - p_cnt[k]) > 0 if n > p_cnt[k] else False for k in np.flatnonzero(usable)]))
    else:
        out["pos_period_share"] = out["loyo_share"] = float("nan")
    if total > 0:
        shares = np.concatenate([np.maximum(f_sum, 0), np.maximum(p_sum[usable], 0)]) / total
        out["max_period_share"] = float(shares.max()) if shares.size else float("nan")
        e_sum = np.bincount(ep, weights=yy)
        out["top5_share"] = float(np.sort(e_sum)[::-1][:5].sum() / total)
    else:
        out["max_period_share"] = out["top5_share"] = float("inf")
    with np.errstate(invalid="ignore"):
        out["pass_folds"] = bool(out["pos_folds"] >= int(c["min_pos_folds"]))
        period_ok = out["n_periods"] >= 2 and out["pos_period_share"] >= float(c["min_pos_period_share"]) and out["loyo_share"] >= float(c["min_loyo_share"])
        out["pass_periods"] = bool(period_ok)
        out["pass_concentration"] = bool(total > 0 and out["max_period_share"] <= float(c["max_period_profit_share"]) and out["top5_share"] <= float(c["max_top5_episode_share"]))
    out["pass_f2"] = bool(out["pass_folds"] and out["pass_periods"] and out["pass_concentration"])
    return out


class _Ladders:
    """Quantile atoms of one (feature, operator) sorted by threshold: the neighbourhood used by F4 and F6."""

    def __init__(self, aset: AtomSet):
        d: Dict[Tuple[str, str], List[Atom]] = {}
        for a in aset.atoms:
            if a.kind == "quantile":
                d.setdefault((a.feature, a.op), []).append(a)
        self.d = {k: sorted(v, key=lambda a: a.threshold) for k, v in d.items()}

    def of(self, a: Atom) -> Tuple[List[Atom], int]:
        if a.kind != "quantile":
            return [], -1
        lst = self.d[(a.feature, a.op)]
        return lst, [x.id for x in lst].index(a.id)


def _ladder_means(lst: List[Atom], others: Sequence[Atom], other_mask: np.ndarray, dense, y: np.ndarray, min_obs: int) -> List[Tuple[Atom, float]]:
    """Mean of ``y`` for each ladder atom combined with the fixed other atoms (AND); NaN when incompatible or too few observations.
    ``others`` may also be a single :class:`Atom` (the documented two-atom case)."""
    fin = np.isfinite(y)
    others = [others] if isinstance(others, Atom) else list(others)
    oid = {o.id for o in others}
    res = []
    for x in lst:
        if x.id in oid or any(not compatible(x, o) for o in others):
            res.append((x, float("nan")))
            continue
        m = dense(x.id) & other_mask & fin
        res.append((x, float(y[m].mean()) if int(m.sum()) >= min_obs else float("nan")))
    return res


def plateau_and_monotonic_multi(atoms: Sequence[Atom], orig_mean: float, ladders: _Ladders, dense, y: np.ndarray, c: Dict[str, Any]) -> Dict[str, Any]:
    """F4 / F6 soft score for a rule of 2-4 atoms: every atom is shifted +-steps along its own quantile ladder while the others stay fixed."""
    nbr: List[float] = []
    rhos: List[float] = []
    steps = int(c["plateau_steps"])
    for x in atoms:
        others = [o for o in atoms if o.id != x.id]
        other_mask: Optional[np.ndarray] = None
        for o in others:
            m = dense(o.id)
            other_mask = m if other_mask is None else (other_mask & m)
        lst, pos = ladders.of(x)
        if pos < 0:
            continue
        means = _ladder_means(lst, others, other_mask, dense, y, int(c["plateau_min_obs"]))
        for k in range(max(0, pos - steps), min(len(lst), pos + steps + 1)):
            if k == pos or any(not compatible(lst[k], o) for o in others):
                continue
            nbr.append(means[k][1])
        pts = [(lst[k].threshold, means[k][1]) for k in range(len(lst)) if np.isfinite(means[k][1])]
        if len(pts) >= int(c["mono_min_points"]):
            th, mv = zip(*pts)
            if np.ptp(mv) > 0:
                rhos.append(float(spearmanr(th, mv)[0]))                        # SIGNED: the sign is the direction of the relation
    signed = [r_ for r_ in rhos if np.isfinite(r_)]
    out: Dict[str, Any] = {"n_neighbours": len(nbr),
                           "mono_score": float(np.mean([abs(r_) for r_ in signed])) if signed else float("nan"),
                           "mono_signs": ",".join("+" if r_ >= 0 else "-" for r_ in signed),
                           "mono_consistent": bool(signed) and (all(r_ >= 0 for r_ in signed) or all(r_ < 0 for r_ in signed))}
    out["pass_f6_soft"] = bool(out.get("mono_consistent") and np.isfinite(out["mono_score"]) and out["mono_score"] >= float(c.get("mono_min_abs_rho", 0.6)))
    if not nbr:
        out.update(plateau_na=True, plateau_ratio=1.0, plateau_pos_share=1.0, plateau_smooth=float(orig_mean), pass_f4=True,
                   mono_score=float("nan"), mono_consistent=False, pass_f6_soft=False)
        return out
    arr = np.array(nbr)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = float(np.nanmedian(arr) / orig_mean) if np.isfinite(arr).any() and orig_mean > 0 else 0.0
        pos = float(np.mean(np.where(np.isfinite(arr), arr > 0, False)))
    out.update(plateau_na=False, plateau_ratio=ratio, plateau_pos_share=pos, plateau_smooth=float(np.mean(np.concatenate([[orig_mean], arr[np.isfinite(arr)]]))),
               pass_f4=bool(ratio >= float(c["plateau_min_ratio"]) and pos >= float(c["plateau_min_pos_share"])))
    return out


def plateau_and_monotonic(a: Atom, b: Atom, orig_mean: float, ladders: _Ladders, dense, y: np.ndarray, c: Dict[str, Any]) -> Dict[str, Any]:
    """Depth-2 wrapper of :func:`plateau_and_monotonic_multi` (kept as the documented two-atom entry point)."""
    return plateau_and_monotonic_multi([a, b], orig_mean, ladders, dense, y, c)


def _ensure_f8_cols(passed: pd.DataFrame) -> None:
    """The three F8 flag columns ALWAYS exist on the passed frame (False / 0 / NaN when F8 simplified nothing), so downstream
    stages (freeze META_COLS, reporting) see one stable schema whether or not any rule was simplified."""
    for col, val in (("f8_simplified", False), ("f8_dropped", 0), ("f8_val_mean", np.nan)):
        if col not in passed.columns:
            passed[col] = val


def _f8_backward_eliminate(passed: pd.DataFrame, fam: pd.DataFrame, aset: AtomSet, dense, ladders: "_Ladders", labels: Dict[str, FamilyLabels],
                           bounds: np.ndarray, period: np.ndarray, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice,
                           c: Dict[str, Any], info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """F8 (spec 11): backward elimination on the VALIDATION rows — drop an atom when the simplified rule does not fall beyond noise.

    "Noise" = ``f8_noise_se`` (default 1) cluster SEs of the ORIGINAL rule's validation mean (episode clusters, gap = the family hold).
    A drop is accepted only when the simplified rule still qualifies on train (>= ``min_families_pass`` families with episodes + t + lift,
    the chosen family still passing) AND still passes F2 / F7 / F4 (its metrics are recomputed, so no stale column reaches the gates);
    otherwise the original rule is kept. Rules of two atoms are never touched (dropping one would leave the rule representation).

    Returns (passed with rewritten atoms for the simplified rules, fam with their recomputed family rows, flags frame by cand_id).
    """
    flags = pd.DataFrame(columns=["cand_id", "f8_simplified", "f8_dropped", "f8_val_mean"])
    counts = {"f8_simplified": 0, "f8_dropped_atoms": 0, "f8_reverted": 0}
    if not bool(c.get("f8_enabled", True)) or passed.empty:
        info.update(counts)
        _ensure_f8_cols(passed)
        return passed, fam, flags
    val = make_splits(len(label_table), cfg).val
    idx_t = label_table.index[rows]
    n_years = (idx_t[-1] - idx_t[0]).total_seconds() / (365.25 * 86400) if len(idx_t) else 0.0
    noise_se = float(c.get("f8_noise_se", 1.0))
    need = int(cfg["candidates"]["min_families_pass"])
    cc = cfg["candidates"]
    changed: Dict[int, Dict[str, Any]] = {}
    for pos in range(len(passed)):
        row = passed.iloc[pos]
        ids = [int(x) for x in row["atoms"]]
        if len(ids) < 3:
            continue
        direction, fam_name = str(row["direction"]), str(row["best_family"])
        y_val = net_R(label_table, direction, fam_name).to_numpy(np.float64)[val]
        fin = np.isfinite(y_val)
        hold = family_hold(fam_name, cfg)

        def val_stats(mask_ids: Sequence[int]) -> Tuple[float, float]:
            m = np.zeros(y_val.size, dtype=bool)
            for i in mask_ids:
                m |= aset.dense(int(i), val)
            idx = np.flatnonzero(m & fin)
            if idx.size < 2:
                return float("nan"), float("nan")
            ep = episode_ids(idx, hold)
            mean, se, _ = cluster_mean_se(y_val[idx], ep, int(ep[-1]) + 1)
            return float(mean), float(se)

        def train_ev(trial: Sequence[int]) -> Dict[str, Any]:
            trig = rule_mask(dense, trial)
            return evaluate_rule(trig, [rule_mask(dense, p) for p in parent_order(trial)], labels[direction], bounds, cc, n_years=n_years)

        mean_ref, se_ref = val_stats(ids)
        if not (np.isfinite(mean_ref) and np.isfinite(se_ref)):
            continue
        cur, dropped = list(ids), 0
        while len(cur) > 2:
            progressed = False
            for i in list(cur):
                trial = [x for x in cur if x != i]
                ev = train_ev(trial)
                famrow = next((x for x in ev["families"] if x["family"] == fam_name), None)
                if ev["n_pass"] < need or famrow is None or not famrow["passed"]:
                    continue
                ms, _ss = val_stats(trial)
                if np.isfinite(ms) and ms >= mean_ref - noise_se * se_ref:
                    cur, dropped, mean_ref = sorted(trial), dropped + 1, ms     # canonical order: the sorted atom tuple (deep.rule_key)
                    progressed = True
                    break
            if not progressed:
                break
        if not dropped:
            continue
        ev_keep = train_ev(cur)                                          # the accepted set may differ from the last trial: recompute once
        passing = [x for x in ev_keep["families"] if x["passed"]]
        famrow = next((x for x in ev_keep["families"] if x["family"] == fam_name), None)
        if ev_keep["n_pass"] < need or not passing or famrow is None or not famrow["passed"]:
            counts["f8_reverted"] += 1
            continue
        best = max(passing, key=lambda x: x["t"])
        fl = labels[direction]
        k = fl.names.index(best["family"])
        y_tr, g = fl.y[k], int(fl.hold[k])
        trig_tr = rule_mask(dense, cur)
        idx_tr = np.flatnonzero(trig_tr & np.isfinite(y_tr))
        f2 = time_stability(idx_tr, y_tr[idx_tr], episode_ids(idx_tr, g), period[idx_tr], bounds, c)
        stress = float(best["mean_gross"] - float(c["cost_stress_mult"]) * best["cost"])
        f4 = plateau_and_monotonic_multi([aset.atoms[i] for i in cur], float(best["mean"]), ladders, dense, y_tr, c)
        if not (f2["pass_f2"] and stress > 0 and f4["pass_f4"]):
            counts["f8_reverted"] += 1
            continue
        changed[int(row["cand_id"])] = {"atoms": cur, "ev": ev_keep, "best": best, "fam_name": fam_name, "dropped": dropped,
                                        "val_mean": mean_ref, "f2": f2, "stress": stress, "f4": f4, "direction": direction}
        counts["f8_simplified"] += 1
        counts["f8_dropped_atoms"] += dropped
    if not changed:
        info.update(counts)
        _ensure_f8_cols(passed)
        return passed, fam, flags
    # rewrite the passed rows (atoms, expressions and recomputed statistics) so every downstream stage sees the simplified rule
    _ensure_f8_cols(passed)                                          # the f8 flag columns must exist with their proper dtypes BEFORE the
    for cid, ch in changed.items():                                  # .at writes below (creating them with np.nan would force float64, and
        pos = passed.index[passed["cand_id"] == cid]                 # pandas >= 2 rejects writing True into a float64 column)
        if not len(pos):
            continue
        r0 = pos[0]
        ev, best = ch["ev"], ch["best"]
        cur = ch["atoms"]
        passing = [x for x in ev["families"] if x["passed"]]
        upd = {"atoms": cur, "n_atoms": len(cur), "atom_a": cur[0], "atom_b": cur[1], "rule_key": ",".join(str(i) for i in sorted(cur)),
               "expr_a": aset.atoms[cur[0]].expr(), "expr_b": aset.atoms[cur[1]].expr(),
               "rule_expr": " AND ".join(aset.atoms[i].expr() for i in cur),
               "n_families_pass": len(passing), "families_pass": ",".join(x["family"] for x in passing),
               "rank_score": float(np.median([x["t"] if np.isfinite(x["t"]) else -np.inf for x in ev["families"]])),
               "best_family": best["family"], "best_t": best["t"], "best_mean": best["mean"], "best_mean_gross": best["mean_gross"],
               "best_cost": best["cost"], "best_n_episodes": best["n_episodes"], "best_n_trigger": best["n_trigger"],
               "best_lift_min": float(best["lift_min"]), "best_pos_folds": int(sum(1 for m, n in zip(best["fold_mean"], best["fold_n"]) if n > 0 and m > 0)),
               "plateau_smooth": ch["f4"]["plateau_smooth"], "plateau_score": ch["f4"]["plateau_smooth"],
               "f8_simplified": True, "f8_dropped": int(ch["dropped"]), "f8_val_mean": float(ch["val_mean"])}
        for k_, v_ in upd.items():
            if k_ not in passed.columns:                     # F8 runs before the flag merge: create the column it introduces (.at cannot enlarge)
                passed[k_] = np.nan
            passed.at[r0, k_] = v_
        for k_, v_ in ch["f2"].items():
            passed.at[r0, "pass_f2" if k_ == "pass_f2" else f"f2_{k_}"] = v_
        passed.at[r0, "cost_stress_mean"] = ch["stress"]
        passed.at[r0, "pass_f7"] = bool(ch["stress"] > 0)
        for k_, v_ in ch["f4"].items():
            passed.at[r0, k_] = v_
    flags = pd.DataFrame([{"cand_id": cid, "f8_simplified": True, "f8_dropped": int(ch["dropped"]), "f8_val_mean": float(ch["val_mean"])}
                          for cid, ch in changed.items()])
    # replace the family rows of the simplified candidates with the recomputed ones (the exit surface must be computed on the new atoms)
    kept = fam[~fam["cand_id"].isin(list(changed))]
    rows_new: List[Dict[str, Any]] = []
    for cid, ch in changed.items():
        for x in ch["ev"]["families"]:
            row: Dict[str, Any] = {"cand_id": cid, "direction": ch["direction"]}
            row.update({k_: v_ for k_, v_ in x.items() if k_ not in ("fold_mean", "fold_n")})
            for f, (m_, n_) in enumerate(zip(x["fold_mean"], x["fold_n"])):
                row[f"fold{f}_mean"], row[f"fold{f}_n"] = m_, n_
            rows_new.append(row)
    fam = pd.concat([kept, pd.DataFrame(rows_new)], ignore_index=True) if rows_new else fam
    info.update(counts)
    return passed, fam, flags


def run_filters(cand: pd.DataFrame, fam: pd.DataFrame, aset: AtomSet, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice
                ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Returns (log, passed, info): ``log`` = every candidate with all filter metrics; ``passed`` = the candidates passing F2, F7 and F4, re-clustered."""
    t0 = time.time()
    c = cfg["filters"]
    n_folds = int(cfg["discovery"]["n_folds"])
    info: Dict[str, Any] = {"funnel": {"candidates": int(len(cand)), "f2": 0, "f7": 0, "f4": 0}}
    if cand.empty:
        info.update(n_passed=0, n_clusters=0, seconds=0.0, period_kind=None)
        return cand.copy(), cand.copy(), info
    labels = {d: build_family_labels(label_table, cfg, d, rows) for d in DIRECTIONS}
    bounds = fold_bounds(labels["long"].y.shape[1], n_folds)
    period, pkind = period_keys(label_table.index[rows], int(c["min_periods"]))
    dense = _MaskCache(aset, rows, cfg["candidates"]["cache_atoms"])
    ladders = _Ladders(aset)
    famidx = fam.set_index(["cand_id", "family"])
    recs: List[Dict[str, Any]] = []
    for r in cand.itertuples(index=False):
        fl: FamilyLabels = labels[r.direction]
        k = fl.names.index(r.best_family)
        y, g = fl.y[k], int(fl.hold[k])
        ids = [int(x) for x in getattr(r, "atoms", [getattr(r, "atom_a"), getattr(r, "atom_b")])]
        a_atoms = [aset.atoms[i] for i in ids]
        trig = rule_mask(dense, ids)
        idx = np.flatnonzero(trig & np.isfinite(y))
        yy = y[idx]
        f2 = time_stability(idx, yy, episode_ids(idx, g), period[idx], bounds, c)
        frow = famidx.loc[(int(r.cand_id), r.best_family)]
        stress = float(frow["mean_gross"] - float(c["cost_stress_mult"]) * frow["cost"])
        f4 = plateau_and_monotonic_multi(a_atoms, float(r.best_mean), ladders, dense, y, c)
        rec = {"cand_id": int(r.cand_id), "cost_stress_mean": stress, "pass_f7": bool(stress > 0)}
        rec.update({f"f2_{k_}": v for k_, v in f2.items() if k_ not in ("pass_f2",)})
        rec["pass_f2"] = f2["pass_f2"]
        rec.update(f4)
        recs.append(rec)
    log = cand.merge(pd.DataFrame(recs), on="cand_id")
    ok2 = log["pass_f2"]
    ok7 = ok2 & log["pass_f7"]
    ok4 = ok7 & log["pass_f4"]
    info["funnel"].update(f2=int(ok2.sum()), f7=int(ok7.sum()), f4=int(ok4.sum()))
    log["passed_filters"] = ok4
    passed = log[ok4].copy()
    passed["plateau_score"] = passed["plateau_smooth"]
    passed, fam, f8_flags = _f8_backward_eliminate(passed, fam, aset, dense, ladders, labels, bounds, period, label_table, cfg, rows, c, info)
    if len(f8_flags):
        log = log.merge(f8_flags, on="cand_id", how="left")
    for col, val in (("f8_simplified", False), ("f8_dropped", 0)):      # both frames always carry the F8 flags with clean dtypes
        for frame in (log, passed):
            if col in frame.columns:
                frame[col] = frame[col].fillna(val).astype(bool if col == "f8_simplified" else int)
            else:
                frame[col] = val
    if "f8_val_mean" not in passed.columns:
        passed["f8_val_mean"] = np.nan
    info["funnel"].update(f8=int(passed["f8_simplified"].sum()))
    # F9 representative rule (spec 11): SIMPLEST first, then the best plateau score (the order the leader clustering follows)
    passed = passed.sort_values(["n_atoms", "plateau_score", "rank_score", "atom_a", "atom_b", "direction"], ascending=[True, False, False, True, True, True],
                                kind="stable").reset_index(drop=True)
    passed["fcluster"] = -1
    passed["f_is_representative"] = False
    nxt = 0
    n_by_corr = 0
    jac_thr = float(cfg["candidates"]["dedup_jaccard"])
    corr_thr = float(cfg["candidates"].get("dedup_corr", 0.7))
    for d in DIRECTIONS:
        sel = np.flatnonzero((passed["direction"] == d).to_numpy())
        if sel.size == 0:
            continue
        ids_list = [list(x) for x in passed["atoms"].to_numpy()[sel]]
        # F9 second criterion: the per-bar P&L series (net R of the candidate's best family on train rows); corr >= dedup_corr merges too
        pnl: List[Tuple[np.ndarray, np.ndarray]] = []
        for pos in sel:
            row = passed.iloc[pos]
            fl = labels[d]
            y_i = fl.y[fl.names.index(row["best_family"])]
            ids_i = [int(x) for x in row["atoms"]] if "atoms" in passed.columns else [int(row["atom_a"]), int(row["atom_b"])]
            m = rule_mask(dense, ids_i) & np.isfinite(y_i)
            idx_i = np.flatnonzero(m)
            pnl.append((idx_i, y_i[idx_i]))
        reps: List[int] = []
        rep_masks: List[np.ndarray] = []
        rep_pnl: List[Tuple[np.ndarray, np.ndarray]] = []
        cid_local = np.full(sel.size, -1, dtype=np.int64)
        rep_local = np.zeros(sel.size, dtype=bool)
        for i in range(sel.size):
            m_i = rule_mask(dense, ids_list[i])
            assigned = -1
            for k, rk in enumerate(reps):
                inter = int((m_i & rep_masks[k]).sum())
                union = int((m_i | rep_masks[k]).sum())
                jac = inter / union if union else 0.0
                c_ = _pnl_corr(rep_pnl[k], pnl[i])
                if jac >= jac_thr:
                    assigned = k
                    break
                if c_ is not None and c_ >= corr_thr:
                    assigned = k
                    n_by_corr += 1
                    break
            if assigned < 0:
                reps.append(i)
                rep_masks.append(m_i)
                rep_pnl.append(pnl[i])
                rep_local[i] = True
                assigned = len(reps) - 1
            cid_local[i] = assigned
        passed.loc[sel, "fcluster"] = cid_local + nxt
        passed.loc[sel, "f_is_representative"] = rep_local
        nxt += len(reps)
    if len(passed):
        passed["f_cluster_size"] = passed.groupby("fcluster")["fcluster"].transform("size")
    info.update(n_passed=int(len(passed)), n_clusters=int(passed["fcluster"].nunique()) if len(passed) else 0, merged_by_corr=int(n_by_corr),
                period_kind=pkind, seconds=round(time.time() - t0, 2))
    return log, passed, info


def _pnl_corr(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray], min_overlap: int = 30) -> Optional[float]:
    """Pearson correlation of two sparse per-bar P&L series over the union of their non-zero bars (zeros fill the gaps); None when undefined."""
    if len(a[0]) < min_overlap or len(b[0]) < min_overlap:
        return None
    union = np.union1d(a[0], b[0])
    lookup_a = dict(zip(a[0].tolist(), a[1].tolist()))
    lookup_b = dict(zip(b[0].tolist(), b[1].tolist()))
    va = np.array([lookup_a.get(int(i), 0.0) for i in union])
    vb = np.array([lookup_b.get(int(i), 0.0) for i in union])
    sa, sb = va.std(), vb.std()
    if sa == 0 or sb == 0:
        return None
    return float(((va - va.mean()) * (vb - vb.mean())).mean() / (sa * sb))
