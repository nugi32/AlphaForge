"""Engine B (spec 10.3): shallow regression trees as an INDEPENDENT rule-induction engine, feeding the same stage 6 as Engine A.

Design (spec 10.3, project conventions D-019 / D-021 preserved):
  * Target = ONE label column (direction x exit family), winsorized at +-``deep.trees.winsor_R`` so single outliers cannot dominate.
  * Splits are ONLY the atoms' own quantile thresholds of that feature, so every leaf is exactly an AND of atoms and no threshold has to
    be mapped back afterwards; the rule the leaf encodes is the rule stage 6 evaluates.
  * Feature bagging per tree (``feature_frac``), and a bootstrap over CALENDAR-WEEK blocks (``include_in_tree_bootstrap``), so serial
    dependence and the session structure stay in the resamples.
  * A leaf becomes a rule when it holds at least ``min_leaf_rows`` bootstrap rows and its bootstrap mean is positive. The leaf path is
    canonicalised (same feature and operator kept at the strictest bound, an empty interval dropped, incompatible pairs dropped).
  * All trees share one RNG seeded from ``deep.trees.seed`` (or the global seed), so the engine is a deterministic function of the config
    and the data. That is required for the null test, where only the label alignment may change (spec 13.3).
  * Rules are counted across trees (``hits``); those with ``hits`` >= ``min_hits`` are kept, scored by their TRAIN mean on the trigger rows
    of the family they were induced on, and capped at ``max_rules``. Exact statistics, filters, exits and validation are stage 6+ business.

Trees are greedy and can miss pure interactions; that is why this is the comparison engine, not the only one (spec 10.3).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from edge_research.atoms.atoms import AtomSet
from edge_research.discovery.deep import rule_compatible, rule_mask
from edge_research.discovery.pairs import DIRECTIONS
from edge_research.labels.builder import net_R

TREE_COLS = ["atoms", "direction", "score", "hits", "family", "n_atoms", "source"]


def week_blocks(index: pd.DatetimeIndex) -> np.ndarray:
    """Dense calendar-week id per row (rows in time order). Non-datetime indexes fall back to fixed 100-row blocks."""
    if not isinstance(index, pd.DatetimeIndex):
        return (np.arange(len(index)) // 100).astype(np.int64)
    key = (index.normalize() - pd.to_timedelta(index.dayofweek, unit="D")).asi8
    _, inv = np.unique(key, return_inverse=True)
    return inv.astype(np.int64)


def split_table(aset: AtomSet) -> Dict[str, List[Tuple[float, int, int]]]:
    """feature -> [(threshold, ``<=`` atom id, ``>`` atom id)] sorted by threshold: the atom quantile ladder usable as tree splits."""
    raw: Dict[str, Dict[float, List[Optional[int]]]] = {}
    for a in aset.atoms:
        if a.kind != "quantile":
            continue
        slot = raw.setdefault(a.feature, {}).setdefault(float(a.threshold), [None, None])
        slot[0 if a.op == "<=" else 1] = int(a.id)
    out: Dict[str, List[Tuple[float, int, int]]] = {}
    for f, d in raw.items():
        rows = [(thr, x[0], x[1]) for thr, x in sorted(d.items()) if x[0] is not None and x[1] is not None]
        if rows:
            out[f] = rows
    return out


def atom_lookup(aset: AtomSet) -> Dict[Tuple[str, str, float], int]:
    return {(a.feature, a.op, float(a.threshold)): int(a.id) for a in aset.atoms}


def canonical_atoms(constraints: Sequence[Tuple[str, str, float]], aset: AtomSet, lookup: Dict[Tuple[str, str, float], int]) -> Optional[Tuple[int, ...]]:
    """Atom ids of a leaf path. Same feature and operator is tightened to the strictest bound; an interval with ``lo >= hi`` is empty and the
    rule is dropped; atoms that may not appear together are dropped. Returns None when the path does not describe a legal rule of >= 2 atoms."""
    per: Dict[str, Dict[str, List[float]]] = {}
    for f, op, thr in constraints:
        per.setdefault(f, {"<=": [], ">": []})[op].append(float(thr))
    ids: List[int] = []
    for f, parts in per.items():
        le = min(parts["<="]) if parts["<="] else None
        gt = max(parts[">"]) if parts[">"] else None
        if le is not None and gt is not None and not (gt < le):
            return None
        for op, thr in (("<=", le), (">", gt)):
            if thr is None:
                continue
            i = lookup.get((f, op, float(thr)))
            if i is None:
                return None
            ids.append(int(i))
    ids = sorted(set(ids))
    if len(ids) < 2:
        return None
    return tuple(ids) if rule_compatible([aset.atoms[i] for i in ids]) else None


def _grow(yv: np.ndarray, sq: np.ndarray, M: np.ndarray, idx: np.ndarray, depth: int, max_depth: int, min_leaf: int,
          path: Tuple[Tuple[int, int], ...], leaves: List[Tuple[Tuple[Tuple[int, int], ...], np.ndarray]]) -> None:
    """Grow one tree on rows ``idx`` (positions in the restricted array). ``M`` is (n_splits, n_rows) uint8/float of split membership."""
    n = int(idx.size)
    if depth >= int(max_depth) or n < 2 * int(min_leaf):
        leaves.append((path, idx))
        return
    sub = M[:, idx]
    cnt = sub.sum(1).astype(np.float64)
    n_r = n - cnt
    ok = (cnt >= int(min_leaf)) & (n_r >= int(min_leaf))
    if not ok.any():
        leaves.append((path, idx))
        return
    s = sub @ yv[idx]
    ss = sub @ sq[idx]
    s_node, ss_node = float(yv[idx].sum()), float(sq[idx].sum())
    sse_node = ss_node - s_node * s_node / n
    with np.errstate(invalid="ignore", divide="ignore"):
        sse_l = ss - (s * s) / np.maximum(cnt, 1.0)
        sse_r = (ss_node - ss) - ((s_node - s) ** 2) / np.maximum(n_r, 1.0)
    gain = np.where(ok, sse_node - sse_l - sse_r, -np.inf)
    k = int(np.argmax(gain))
    if not np.isfinite(gain[k]) or gain[k] <= 1e-12:
        leaves.append((path, idx))
        return
    left = np.asarray(M[k, idx]).astype(np.float64) > 0.5
    _grow(yv, sq, M, idx[left], depth + 1, max_depth, min_leaf, path + ((k, 0),), leaves)
    _grow(yv, sq, M, idx[~left], depth + 1, max_depth, min_leaf, path + ((k, 1),), leaves)


def induce_rules(aset: AtomSet, feature_table: pd.DataFrame, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice
                 ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Engine B rules on the TRAIN rows. Returns (frame with ``atoms``/``direction``/``score``/``hits``/``family``, info)."""
    t = cfg["deep"]["trees"]
    t0 = time.time()
    info: Dict[str, Any] = {"enabled": bool(t["enabled"]), "n_trees": int(t["n_trees"]), "n_features": 0, "n_splits": 0, "n_leaf_rules": 0,
                            "n_rules": 0, "seconds": 0.0, "by_direction": {k: 0 for k in DIRECTIONS}}
    if not t["enabled"] or len(label_table) == 0:
        return pd.DataFrame(columns=TREE_COLS), info
    seed = int(t["seed"]) if t.get("seed") is not None else int(cfg["seed"])
    rng = np.random.default_rng(seed)
    depths = [int(t["max_depth"])] if isinstance(t["max_depth"], (int, np.integer)) else [int(x) for x in t["max_depth"]]   # spec 10.3: depth 3-4
    index = label_table.index[rows]
    n_rows = int(len(index))
    blocks = week_blocks(index)
    n_blocks = int(blocks.max()) + 1 if n_rows else 0
    ladder = split_table(aset)
    feats = [f for f in sorted(ladder) if f in feature_table.columns and len(ladder[f])]
    if len(feats) < 2 or n_rows == 0:
        return pd.DataFrame(columns=TREE_COLS), info
    values = {f: feature_table[f].to_numpy(np.float64)[rows] for f in feats}
    splits: List[Tuple[str, float, int, int]] = [(f, thr, le, gt) for f in feats for (thr, le, gt) in ladder[f]]
    feat_of_split = np.array([feats.index(s[0]) for s in splits], dtype=np.int64)
    M = np.stack([values[s[0]] <= s[1] for s in splits]).astype(np.float32)
    lookup = atom_lookup(aset)
    n_leaf = int(t["min_leaf_rows"])
    frac = float(t["feature_frac"])
    winsor = float(t["winsor_R"])
    n_use = max(2, min(len(feats), int(round(frac * len(feats)))))
    info.update(n_features=len(feats), n_splits=len(splits), depths=depths)
    found: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
    y_cache: Dict[Tuple[str, str], np.ndarray] = {}
    for direction in DIRECTIONS:
        for fam in t["families"]:
            key_y = (direction, str(fam))
            if key_y not in y_cache:
                y_cache[key_y] = np.array(net_R(label_table, direction, str(fam)).to_numpy(np.float64)[rows], copy=True)
            y = y_cache[key_y]
            finite = np.isfinite(y)
            n_valid = int(finite.sum())
            if n_valid < 4 * n_leaf:
                continue
            yv = np.where(finite, np.clip(y, -winsor, winsor), 0.0)
            sq = yv * yv
            pos = np.flatnonzero(finite)
            blk = blocks[pos]
            order = np.argsort(blk, kind="stable")
            pos_sorted = pos[order]
            counts = np.bincount(blk, minlength=max(n_blocks, 1))
            edges = np.concatenate([[0], np.cumsum(counts)])
            groups = [pos_sorted[edges[b]:edges[b + 1]] for b in range(len(counts)) if counts[b] > 0]
            if not groups:
                continue
            for depth in depths:
                for _ in range(int(t["n_trees"])):
                    chosen = rng.choice(len(feats), size=n_use, replace=False)
                    allowed = np.flatnonzero(np.isin(feat_of_split, chosen))
                    if allowed.size < 2:
                        continue
                    draw = rng.integers(0, len(groups), size=len(groups))
                    rows_boot = np.concatenate([groups[b] for b in draw]) if len(groups) else np.zeros(0, dtype=np.int64)
                    if rows_boot.size < 2 * n_leaf:
                        continue
                    leaves: List[Tuple[Tuple[Tuple[int, int], ...], np.ndarray]] = []
                    _grow(yv, sq, M[allowed], rows_boot, 0, depth, n_leaf, (), leaves)
                    # harvest THIS tree now: hits are counted ACROSS trees (min_hits), so the leaves of every tree
                    # must be seen — harvesting only after the loop would keep the last tree alone and its leaf
                    # rules can never reach min_hits >= 2.
                    for path, idx_leaf in leaves:
                        if idx_leaf.size < n_leaf or float(yv[idx_leaf].mean()) <= 0:
                            continue
                        cons = []
                        for k, branch in path:
                            f, thr, _le, _gt = splits[int(allowed[k])]
                            cons.append((f, "<=" if branch == 0 else ">", float(thr)))
                        ids = canonical_atoms(cons, aset, lookup)
                        if ids is None:
                            continue
                        rec = found.setdefault((direction, ids), {"hits": 0, "fam_means": {}})
                        rec["hits"] += 1
                        rec["fam_means"][str(fam)] = float(yv[idx_leaf].mean())
                        info["n_leaf_rules"] += 1
    rows_recs: List[Dict[str, Any]] = []
    min_hits = int(t["min_hits"])
    dense = lambda i: aset.dense(int(i), rows)                                     # noqa: E731 (local mask accessor)
    for (direction, ids), rec in found.items():
        if int(rec["hits"]) < min_hits:
            continue
        mask = rule_mask(dense, ids)
        best_fam, best_mean = None, -np.inf
        for fam, _ in rec["fam_means"].items():
            y = y_cache[(direction, str(fam))]
            m = mask & np.isfinite(y)
            if not m.any():
                continue
            mu = float(y[m].mean())
            if mu > best_mean:
                best_fam, best_mean = fam, mu
        if best_fam is None or best_mean <= 0:                                     # a leaf whose TRAIN result is not positive is not a candidate
            continue
        rows_recs.append({"atoms": list(ids), "direction": direction, "score": float(best_mean), "hits": int(rec["hits"]),
                          "family": str(best_fam), "n_atoms": len(ids), "source": "tree"})
    frame = pd.DataFrame(rows_recs, columns=TREE_COLS)
    if len(frame):
        cap = int(t["max_rules"])
        frame = frame.sort_values(["score", "hits"], ascending=[False, False], kind="stable").head(cap).reset_index(drop=True)
        info["by_direction"] = {k: int((frame["direction"] == k).sum()) for k in DIRECTIONS}
    info["n_rules"] = int(len(frame))
    info["seconds"] = round(time.time() - t0, 2)
    return frame, info
