"""Stage 6 (M5): exact statistics on the survivors of the coarse screen, lift over parents, minimum episodes, dedup.

For every survivor (pair, direction) of Stage 5 and for EVERY label family of that direction (not only the three screening labels):
  * trigger = A & B on TRAIN rows; observations = trigger rows with a finite label; episodes merge triggers closer than ``g`` bars with
    ``g`` = hold of the family (spec 10.4); ``discovery.episodes.rule_stats`` gives episodes, effective n, mean, cluster t, per-fold
    means and the gross mean on the same rows;
  * lift over BOTH parents (spec 10.5): mean(P&C) - mean(P&~C) with a cluster-robust t (``episodes.lift_stats``), clusters = time blocks of
    ``lift_block_mult`` x hold bars (parent episodes degenerate to ONE cluster for a dense parent, NOTES D-027);
  * a family PASSES when episodes >= ``min_episodes_train``, t >= ``t_exact`` and both lift t >= ``lift_t_min``;
  * a candidate qualifies when >= ``min_families_pass`` families of its direction pass.
Qualified candidates are ranked (median t over ALL label families of their direction, so breadth counts) and clustered by the Jaccard overlap of their trigger rows
(leader clustering, same direction only, best-ranked member is the representative; spec F9). Only train rows are read.

Not here (later steps): time-stability beyond the folds (F2), exit neighbourhood (F3), threshold plateau (F4), monotonicity (F6),
cost stress (F7), simplification (F8), exit selection (M6), null test over the whole search (M7). The statistics are NOT evidence yet:
the search is wide, so every number here is selected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from edge_research.eda.univariate import cluster_mean_se

from edge_research.atoms.atoms import AtomSet, _POPCOUNT
from edge_research.discovery.deep import normalise_rules, rule_key, rule_mask
from edge_research.discovery.episodes import bootstrap_mean_ci, lift_stats, rule_stats
from edge_research.discovery.trees import week_blocks
from edge_research.discovery.pairs import DIRECTIONS, family_hold, fold_bounds
from edge_research.labels.builder import family_list, net_R


@dataclass
class FamilyLabels:
    names: List[str]            # label families of one direction, e.g. "T3", "B2_1_12"
    hold: np.ndarray            # (F,) bars = episode gap g
    y: np.ndarray               # (F, n_rows) target (net by default), NaN = no valid outcome
    gross: np.ndarray           # (F, n_rows) gross result on the same rows
    week: Optional[np.ndarray] = None   # (n_rows,) calendar-week id per row for the optional ``cluster_extra`` SE (spec 10.4)


def build_family_labels(label_table: pd.DataFrame, cfg: Dict[str, Any], direction: str, rows: slice) -> FamilyLabels:
    """ALL label families of one direction on ``rows`` (the whole label set, not only the screening labels)."""
    fams = [name for name, _ in family_list(cfg)]
    target = cfg["discovery"]["target"]
    ys, gs = [], []
    for fam in fams:
        gross = label_table[f"{direction}_{fam}_gross"].to_numpy(np.float64)[rows]
        net = net_R(label_table, direction, fam).to_numpy(np.float64)[rows]
        ys.append(np.array(net if target == "net" else gross, copy=True))
        gs.append(np.array(gross, copy=True))
    hold = np.array([family_hold(f, cfg) for f in fams], dtype=np.int64)
    week = week_blocks(label_table.index[rows]) if len(label_table) else None
    return FamilyLabels(fams, hold, np.vstack(ys), np.vstack(gs), week)


class _MaskCache:
    """Dense boolean masks of atoms on the train rows, with a hard cap on the number kept (oldest dropped first)."""

    def __init__(self, aset: AtomSet, rows: slice, cap: int):
        self.aset, self.rows, self.cap = aset, rows, max(2, int(cap))
        self._d: Dict[int, np.ndarray] = {}

    def __call__(self, atom_id: int) -> np.ndarray:
        m = self._d.get(atom_id)
        if m is None:
            if len(self._d) >= self.cap:
                self._d.pop(next(iter(self._d)))
            m = self._d[atom_id] = self.aset.dense(int(atom_id), self.rows)
        return m


FAMILY_COLS = ["n_trigger", "n_episodes", "mean", "mean_gross", "cost", "se", "se_week", "t", "pct_positive", "profit_factor", "wl_ratio", "net_per_year",
               "lift_a", "lift_a_t", "lift_b", "lift_b_t", "lift_min", "lift_min_t", "n_parents"]


def evaluate_rule(trig: np.ndarray, parent_masks: Sequence[np.ndarray], fl: FamilyLabels, bounds: np.ndarray, c: Dict[str, Any],
                  n_years: Optional[float] = None) -> Dict[str, Any]:
    """Exact statistics of one rule (``trig`` = AND of its atoms) for every family of one direction.

    ``parent_masks`` holds the leave-one-out parents of the rule (for a pair: [A, B], so parent A = the rule without atom B). The lift is the
    child's mean minus the parent's remainder mean, with the Stage 6 estimator; a family passes lift when the MINIMUM t over all parents is at
    least ``lift_t_min`` (for a pair this is exactly the old ``a AND b`` rule). Lift is computed only for families that already pass episodes and t.

    Extras (spec 10.4): per-rule profit factor / W-L ratio / net per year; with ``c['cluster_extra'] == 'week'`` the SE also clusters on
    calendar weeks and the CONSERVATIVE (larger) SE is used for ``t``.
    """
    recs: List[Dict[str, Any]] = []
    extra_week = str(c.get("cluster_extra") or "") == "week"
    for k, fam in enumerate(fl.names):
        g = int(fl.hold[k])
        s = rule_stats(trig, fl.y[k], g, bounds=bounds, gross=fl.gross[k])
        r = {"family": fam, "g": g, "n_trigger": s["n_trigger"], "n_episodes": s["n_episodes"], "mean": s["mean"], "mean_gross": s["mean_gross"],
             "cost": s["mean_gross"] - s["mean"], "se": s["se"], "se_week": float("nan"), "t": s["t"], "pct_positive": s["pct_positive"],
             "profit_factor": float("nan"), "wl_ratio": float("nan"), "net_per_year": float("nan"),
             "fold_mean": s["fold_mean"], "fold_n": s["fold_n"], "n_parents": int(len(parent_masks)),
             "lift_a": np.nan, "lift_a_t": np.nan, "lift_b": np.nan, "lift_b_t": np.nan, "lift_min": np.nan, "lift_min_t": np.nan}
        m = trig & np.isfinite(fl.y[k])
        yy = fl.y[k][m]
        if yy.size:
            pos, neg = yy[yy > 0], yy[yy < 0]
            if neg.size and neg.sum() < 0:
                r["profit_factor"] = float(pos.sum() / -neg.sum()) if pos.size else 0.0
                r["wl_ratio"] = float(pos.mean() / -neg.mean()) if pos.size else 0.0
            elif pos.size:
                r["profit_factor"] = float("inf")                                     # no losing trade: an infinite PF passes (spec 14.5 G8)
            if n_years is not None and n_years > 0:
                r["net_per_year"] = float(yy.sum() / n_years)
        if extra_week and fl.week is not None and yy.size >= 4:                       # optional calendar-week cross-clustering (spec 10.4)
            wk = fl.week[m]
            n_w = int(wk.max()) + 1 if wk.size else 0
            if n_w >= 2:
                _, se_w, _ = cluster_mean_se(yy, wk, n_w)
                r["se_week"] = se_w
                if np.isfinite(se_w) and se_w > 0 and (not np.isfinite(r["se"]) or se_w > r["se"]):
                    r["se"] = se_w
                    r["t"] = r["mean"] / se_w if r["mean"] == r["mean"] else float("nan")
        r["pass_episodes"] = bool(s["n_episodes"] >= int(c["min_episodes_train"]))
        with np.errstate(invalid="ignore"):
            r["pass_t"] = bool(r["pass_episodes"] and s["t"] >= float(c["t_exact"]))
        recs.append(r)
    n_et = sum(r["pass_t"] for r in recs)
    if n_et >= int(c["min_families_pass"]):                          # lift only where it can still change the outcome
        for k, r in enumerate(recs):
            if not r["pass_t"]:
                continue
            blk = max(1, int(c["lift_block_mult"]) * r["g"])            # time blocks, not parent episodes: dense parents would be ONE episode (D-027)
            lifts = [lift_stats(pm, trig, fl.y[k], r["g"], block=blk) for pm in parent_masks]
            ts = [float(x["t"]) for x in lifts]
            ls = [float(x["lift"]) for x in lifts]
            finite = all(np.isfinite(ts))
            r.update(lift_a=lifts[0]["lift"], lift_a_t=ts[0],
                     lift_b=(ls[1] if len(ls) > 1 else np.nan), lift_b_t=(ts[1] if len(ts) > 1 else np.nan),
                     lift_min=(float(np.min(ls)) if finite else np.nan), lift_min_t=(float(np.min(ts)) if finite else np.nan))
    for r in recs:
        with np.errstate(invalid="ignore"):
            r["pass_lift"] = bool(r["pass_t"] and r["lift_min_t"] >= float(c["lift_t_min"]))
        r["passed"] = bool(r["pass_lift"])
    return {"families": recs, "n_pass_episodes": sum(r["pass_episodes"] for r in recs), "n_pass_t": n_et,
            "n_pass": sum(r["passed"] for r in recs)}


def evaluate_pair(a: np.ndarray, b: np.ndarray, fl: FamilyLabels, bounds: np.ndarray, c: Dict[str, Any]) -> Dict[str, Any]:
    """Exact statistics of the depth-2 rule A & B (kept for callers and tests): ``evaluate_rule(a & b, [a, b], ...)``."""
    return evaluate_rule(a & b, [a, b], fl, bounds, c)


def _popcount_rows(packed: np.ndarray) -> np.ndarray:
    return _POPCOUNT[packed].sum(axis=-1)


def leader_clusters(get_mask: Callable[[int], np.ndarray], n: int, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Leader clustering by Jaccard overlap of boolean masks. Candidates are visited in rank order (0 = best).

    Candidate i joins the best-ranked existing representative r with jaccard(mask_i, mask_r) >= ``thr``; otherwise it becomes a new
    representative. Returns (cluster id per candidate = rank of its representative's cluster, is_representative). Masks are bit-packed;
    only the representatives are kept in memory. Empty masks are their own clusters.
    """
    cid = np.full(n, -1, dtype=np.int64)
    is_rep = np.zeros(n, dtype=bool)
    reps: Optional[np.ndarray] = None                       # (cap, nbytes) packed masks of representatives
    rep_size = np.zeros(0, dtype=np.int64)
    rep_cluster: List[int] = []
    for i in range(n):
        m = np.asarray(get_mask(i), dtype=bool)
        packed = np.packbits(m)
        size = int(_POPCOUNT[packed].sum())
        hit = -1
        if reps is not None and size > 0 and len(rep_cluster):
            R = len(rep_cluster)
            ratio_ok = np.minimum(rep_size, size) >= thr * np.maximum(rep_size, size)      # Jaccard <= min/max, so this prunes safely
            cand = np.flatnonzero(ratio_ok)
            if cand.size:
                inter = _popcount_rows(reps[cand] & packed)
                union = rep_size[cand] + size - inter
                good = np.flatnonzero(inter >= thr * union)
                if good.size:
                    hit = int(cand[good[0]])                  # cand is ascending, reps are in rank order -> best-ranked representative
        if hit >= 0:
            cid[i] = rep_cluster[hit]
            continue
        if reps is None:
            reps = np.empty((16, packed.size), dtype=np.uint8)
        elif len(rep_cluster) >= len(reps):
            reps = np.concatenate([reps, np.empty_like(reps)])
        reps[len(rep_cluster)] = packed
        rep_size = np.append(rep_size, size)
        cid[i] = len(rep_cluster)                                  # cluster ids are 0..R-1 in rank order of the representatives
        rep_cluster.append(int(cid[i]))
        is_rep[i] = True
    return cid, is_rep


def parent_order(ids: Sequence[int]) -> List[List[int]]:
    """Leave-one-out parents in the lift column order: first the rule without its LAST atom (``lift_a``), then without its FIRST (``lift_b``).
    For a pair that is exactly [A, B] (lift_a = over parent A, lift_b = over parent B), so stored depth-2 numbers keep their meaning."""
    k = len(ids)
    order = [k - 1, 0] + [i for i in range(1, k - 1)]
    return [[int(x) for j, x in enumerate(ids) if j != i] for i in order]


def build_candidates(surv: pd.DataFrame, aset: AtomSet, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Exact statistics, filters and dedup on the rule candidates of stages 5 and 5b.

    ``surv`` holds one row per candidate rule: the depth-2 survivors of ``coarse_screen`` (``atom_a`` / ``atom_b``), Apriori survivors or Engine B
    leaves (both with an ``atoms`` list). Every shape is normalised to the atom-list form first. Returns (candidates, family_stats, info);
    ``info['with_upstream']`` counts the candidates by source.
    """
    t0 = time.time()
    c = cfg["candidates"]
    n_folds = int(cfg["discovery"]["n_folds"])
    labels = {d: build_family_labels(label_table, cfg, d, rows) for d in DIRECTIONS}
    n_rows = labels["long"].y.shape[1]
    bounds = fold_bounds(n_rows, n_folds)
    idx_t = label_table.index[rows]
    n_years = (idx_t[-1] - idx_t[0]).total_seconds() / (365.25 * 86400) if n_rows else 0.0
    dense = _MaskCache(aset, rows, c["cache_atoms"])
    rules = normalise_rules(surv, aset, "pairs")
    funnel = {"survivors": int(len(rules)), "episodes": 0, "t": 0, "lift": 0}
    max_t = 0.0                                                      # largest exact t over survivors and families with enough episodes (null test, M7)
    rows_c: List[Dict[str, Any]] = []
    for r in rules.itertuples(index=False):
        ids = [int(x) for x in r.atoms]
        trig = rule_mask(dense, ids)
        parents = [rule_mask(dense, p) for p in parent_order(ids)]
        ev = evaluate_rule(trig, parents, labels[r.direction], bounds, c, n_years=n_years)
        for x in ev["families"]:
            if x["pass_episodes"] and np.isfinite(x["t"]) and x["t"] > max_t:
                max_t = float(x["t"])
        need = int(c["min_families_pass"])
        funnel["episodes"] += ev["n_pass_episodes"] >= need
        funnel["t"] += ev["n_pass_t"] >= need
        funnel["lift"] += ev["n_pass"] >= need
        if ev["n_pass"] < need:
            continue
        passing = [x for x in ev["families"] if x["passed"]]
        best = max(passing, key=lambda x: x["t"])
        rows_c.append({"atoms": ids, "n_atoms": len(ids), "rule_key": rule_key(ids), "source": str(r.source),
                       "atom_a": ids[0], "atom_b": ids[1], "expr_a": aset.atoms[ids[0]].expr(), "expr_b": aset.atoms[ids[1]].expr(),
                       "rule_expr": " AND ".join(aset.atoms[i].expr() for i in ids), "direction": r.direction,
                       "screen_score": float(r.score),
                       "tree_hits": int(getattr(r, "hits")) if getattr(r, "hits", None) is not None and getattr(r, "hits", None) == getattr(r, "hits", None) else None,
                       "tree_family": str(getattr(r, "family")) if getattr(r, "family", None) is not None else None,
                       "n_families_pass": len(passing), "families_pass": ",".join(x["family"] for x in passing),
                       "rank_score": float(np.median([x["t"] if np.isfinite(x["t"]) else -np.inf for x in ev["families"]])), "best_family": best["family"], "best_t": best["t"],
                       "best_mean": best["mean"], "best_mean_gross": best["mean_gross"], "best_cost": best["cost"],
                       "best_n_episodes": best["n_episodes"], "best_n_trigger": best["n_trigger"],
                       "best_lift_min": float(best["lift_min"]),
                       "best_pos_folds": int(sum(1 for m, n in zip(best["fold_mean"], best["fold_n"]) if n > 0 and m > 0)),
                       "_families": ev["families"]})
        # spec 10.4: a block-bootstrap CI of the mean for the candidate's best family (blocks = the lift time blocks, >= the episode gap)
        fl_b = labels[r.direction]
        y_b = fl_b.y[fl_b.names.index(best["family"])]
        m_b = trig & np.isfinite(y_b)
        idx_b = np.flatnonzero(m_b)
        blk_w = max(1, int(c["lift_block_mult"]) * int(best["g"]))
        lo, hi = bootstrap_mean_ci(y_b[idx_b], idx_b // blk_w, int(c.get("bootstrap_n", 200)), seed=int(cfg["seed"]))
        rows_c[-1]["best_ci_lo"], rows_c[-1]["best_ci_hi"] = lo, hi
    cand = pd.DataFrame(rows_c)
    info: Dict[str, Any] = {"funnel": funnel, "max_t_exact": max_t, "n_train_rows": int(n_rows), "n_folds": n_folds,
                            "n_families": {d: len(labels[d].names) for d in DIRECTIONS},
                            "with_upstream": {k: int(v) for k, v in rules["source"].value_counts().items()} if len(rules) else {}}
    if cand.empty:
        info.update(n_candidates=0, n_clusters=0, seconds=round(time.time() - t0, 2), n_both_directions=0, by_depth={}, by_source={})
        return cand.drop(columns=["_families"], errors="ignore"), pd.DataFrame(), info
    cand = cand.sort_values(["rank_score", "n_atoms", "atom_a", "atom_b", "direction"], ascending=[False, True, True, True, True], kind="stable").reset_index(drop=True)
    cand["cluster"] = -1
    cand["is_representative"] = False
    next_cluster = 0
    for d in DIRECTIONS:                                             # a long and a short rule are never merged
        sel = np.flatnonzero((cand["direction"] == d).to_numpy())
        if sel.size == 0:
            continue
        ids_list = [list(x) for x in cand["atoms"].to_numpy()[sel]]
        cid, rep = leader_clusters(lambda i: rule_mask(dense, ids_list[i]), sel.size, float(c["dedup_jaccard"]))
        cand.loc[sel, "cluster"] = cid + next_cluster
        cand.loc[sel, "is_representative"] = rep
        next_cluster += int(cid.max()) + 1
    cand["cluster_size"] = cand.groupby("cluster")["cluster"].transform("size")
    rule_dirs = cand.groupby("rule_key")["direction"].transform("nunique")
    cand["both_directions"] = rule_dirs > 1                          # suspected volatility marker (spec 10.6)
    cand.insert(0, "cand_id", np.arange(len(cand)))
    fam_rows: List[Dict[str, Any]] = []
    for pos in range(len(cand)):
        for x in cand.at[pos, "_families"]:
            row = {"cand_id": int(cand.at[pos, "cand_id"]), "direction": cand.at[pos, "direction"]}
            row.update({k: v for k, v in x.items() if k not in ("fold_mean", "fold_n")})
            for f, (m, n) in enumerate(zip(x["fold_mean"], x["fold_n"])):
                row[f"fold{f}_mean"], row[f"fold{f}_n"] = m, n
            fam_rows.append(row)
    cand = cand.drop(columns=["_families"])
    fam = pd.DataFrame(fam_rows)
    info.update(n_candidates=int(len(cand)), n_clusters=int(cand["cluster"].nunique()), n_both_directions=int(cand[cand["both_directions"]]["cand_id"].count()),
                by_direction={d: int((cand["direction"] == d).sum()) for d in DIRECTIONS},
                clusters_by_direction={d: int(cand.loc[cand["direction"] == d, "cluster"].nunique()) for d in DIRECTIONS},
                by_depth={int(k): int(v) for k, v in cand["n_atoms"].value_counts().sort_index().items()},
                by_source={str(k): int(v) for k, v in cand["source"].value_counts().items()},
                seconds=round(time.time() - t0, 2))
    return cand, fam, info
