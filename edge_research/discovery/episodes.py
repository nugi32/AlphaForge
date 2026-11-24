"""Episodes and effective n (spec 10.4).

Triggers closer than ``g`` bars are merged into one episode; the number of episodes is the coarse effective n and the
episodes are the clusters of the cluster-robust SE. Only triggers whose label is finite take part (a trigger without an
outcome is not an observation), so merging never bridges over rows that carry no information.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from edge_research.eda.univariate import cluster_mean_se


def episode_ids(rows: np.ndarray, g: int) -> np.ndarray:
    """Episode id (0..E-1) for each trigger row index. ``rows`` must be strictly increasing.

    A new episode starts when the distance to the previous trigger is greater than ``g`` bars
    (distance <= g merges, so g = 1 merges only adjacent bars).
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    if rows.size > 1 and not (np.diff(rows) > 0).all():
        raise ValueError("trigger rows must be strictly increasing")
    return np.concatenate([[0], np.cumsum(np.diff(rows) > int(g))]).astype(np.int64)


def rule_stats(trigger: np.ndarray, y: np.ndarray, g: int, rows: Optional[slice] = None, bounds: Optional[np.ndarray] = None,
               gross: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Mean, episode-clustered SE and t of label ``y`` on the rows where ``trigger`` is True.

    ``trigger`` and ``y`` are aligned row arrays; ``rows`` restricts both (e.g. the train segment). Non-finite labels are dropped
    before episodes are formed. Positions in the returned counts refer to the restricted array.

    Optional extras, computed on the SAME observations (trigger rows with a finite ``y``):
      ``bounds``  fold edges (F + 1 row positions in the restricted array) -> ``fold_mean`` / ``fold_n`` (F values; NaN / 0 for an empty fold);
      ``gross``   a second label aligned with ``y`` -> ``mean_gross`` (so ``mean_gross - mean`` is the cost on exactly these rows).
    """
    t = np.asarray(trigger, dtype=bool)
    yy = np.asarray(y, dtype=np.float64)
    gg = None if gross is None else np.asarray(gross, dtype=np.float64)
    if rows is not None:
        t, yy = t[rows], yy[rows]
        gg = None if gg is None else gg[rows]
    idx = np.flatnonzero(t & np.isfinite(yy))
    out: Dict[str, Any] = {"n_trigger": int(idx.size), "n_episodes": 0, "mean": float("nan"), "se": float("nan"), "t": float("nan"),
                           "pct_positive": float("nan")}
    if bounds is not None:
        out["fold_mean"] = [float("nan")] * (len(bounds) - 1)
        out["fold_n"] = [0] * (len(bounds) - 1)
    if gg is not None:
        out["mean_gross"] = float("nan")
    if idx.size == 0:
        return out
    ep = episode_ids(idx, g)
    n_ep = int(ep[-1]) + 1
    mean, se, _ = cluster_mean_se(yy[idx], ep, n_ep)
    out.update(n_episodes=n_ep, mean=mean, se=se, pct_positive=float((yy[idx] > 0).mean()))
    out["t"] = mean / se if np.isfinite(se) and se > 0 else float("nan")
    if bounds is not None:
        fold = np.searchsorted(np.asarray(bounds), idx, side="right") - 1
        n_f = len(bounds) - 1
        cnt = np.bincount(fold, minlength=n_f)[:n_f]
        sm = np.bincount(fold, weights=yy[idx], minlength=n_f)[:n_f]
        out["fold_n"] = [int(c) for c in cnt]
        out["fold_mean"] = [float(s / c) if c > 0 else float("nan") for s, c in zip(sm, cnt)]
    if gg is not None:
        out["mean_gross"] = float(gg[idx].mean())
    return out


def bootstrap_mean_ci(y_obs: np.ndarray, block: np.ndarray, n_boot: int = 200, seed: int = 0, lo_q: float = 0.025, hi_q: float = 0.975
                      ) -> Tuple[float, float]:
    """Block-bootstrap CI of the mean (spec 10.4, "bootstrap blok" for survivor decisions).

    ``block`` is a cluster id per observation (episodes or fixed time blocks of >= the episode gap). Whole blocks are resampled with
    replacement, so serial dependence inside a block is preserved. Deterministic for a given (seed, data) — a null-test requirement.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    block = np.asarray(block, dtype=np.int64)
    if y_obs.size == 0 or y_obs.size != block.size:
        return float("nan"), float("nan")
    n_blocks = int(block.max()) + 1
    if n_blocks < 2:
        return float("nan"), float("nan")
    sums = np.bincount(block, weights=y_obs, minlength=n_blocks)
    counts = np.bincount(block, minlength=n_blocks).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, n_blocks, size=(int(n_boot), n_blocks))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def lift_stats(parent: np.ndarray, child: np.ndarray, y: np.ndarray, g: int, rows: Optional[slice] = None, block: Optional[int] = None) -> Dict[str, Any]:
    """Lift of ``child`` over its ``parent`` (spec 10.5): mean(P & C) - mean(P & ~C), with a cluster-robust SE.

    Observations are the parent rows with a finite label. Clusters (a cluster can hold rows of both groups):
      * ``block`` None: episodes of the PARENT (gap ``g``). Only sensible for a sparse parent: a dense parent (support above roughly 1/g)
        chains the whole segment into ONE episode, and then the SE is undefined (NaN);
      * ``block`` = B: fixed time blocks of B rows (``position // B``). Rows closer than ``g`` share a price path, so B >= g captures that
        dependence, and the number of clusters stays large for any parent. This is what Stage 6 uses (NOTES D-027).
    With group means m1 (C) and m0 (~C) and n1, n0 rows, the SE is the sandwich estimate of the dummy regression:
    ``sqrt(G / (G - 1) * sum_g (r1_g / n1 - r0_g / n0)^2)`` where r1_g, r0_g are the sums of residuals of cluster g in each group.
    A group with no rows, or a single cluster, gives NaN (which fails any threshold).
    """
    p = np.asarray(parent, dtype=bool)
    c = np.asarray(child, dtype=bool)
    yy = np.asarray(y, dtype=np.float64)
    if rows is not None:
        p, c, yy = p[rows], c[rows], yy[rows]
    idx = np.flatnonzero(p & np.isfinite(yy))
    out = {"n_child": 0, "n_rest": 0, "lift": float("nan"), "se": float("nan"), "t": float("nan")}
    if idx.size == 0:
        return out
    cc = c[idx]
    n1, n0 = int(cc.sum()), int((~cc).sum())
    out.update(n_child=n1, n_rest=n0)
    if n1 == 0 or n0 == 0:
        return out
    if block is None:
        ep = episode_ids(idx, g)
    else:
        if int(block) < 1:
            raise ValueError("block must be >= 1")
        ep = np.unique(idx // int(block), return_inverse=True)[1].astype(np.int64)      # dense cluster ids 0..G-1
    n_ep = int(ep.max()) + 1
    if n_ep < 2:
        return out
    v = yy[idx]
    m1, m0 = v[cc].mean(), v[~cc].mean()
    r1 = np.bincount(ep[cc], weights=(v - m1)[cc], minlength=n_ep)
    r0 = np.bincount(ep[~cc], weights=(v - m0)[~cc], minlength=n_ep)
    infl = r1 / n1 - r0 / n0
    se = float(np.sqrt(n_ep / (n_ep - 1) * (infl**2).sum()))
    out.update(lift=float(m1 - m0), se=se, t=float((m1 - m0) / se) if se > 0 else float("nan"))
    return out
