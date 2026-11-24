"""Engine A, depth 2 (spec 10.2): statistics of ALL atom pairs through matrix products, then a coarse screen.

For atom matrix A (rows x atoms, 0/1) and a label y with a validity mask v (labels are NaN where the window leaves the data
or crosses a gap; NOTES D-023):
    pair row count  N = A^T diag(v) A          pair label sum  S = A^T diag(y0) A      (y0 = y with NaN -> 0)
computed per fold (time block of the train segment), per row chunk, and per atom tile (block of columns). Combination masks are
never stored. The cells A&B, A&~B, ~A&B come from the pair cell and the two marginals (~ is the logical complement, so NaN
feature rows count as "not A").

Only the train segment is ever read. Survivors of the coarse screen are returned; everything else is dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from edge_research.atoms.atoms import Atom, AtomSet, compatible
from edge_research.labels.builder import family_list, net_R

DIRECTIONS = ("long", "short")


def family_hold(family: str, cfg: Dict[str, Any]) -> int:
    """Maximum hold length (bars) of a label family: n for time stops, horizon for barrier families (the L of the t deflation)."""
    for name, p in family_list(cfg):
        if name == family:
            return int(p["n"] if p["kind"] == "T" else p["h"])
    raise KeyError(f"unknown label family {family!r}")


@dataclass
class ScreenLabels:
    names: List[str]            # e.g. "long_T3"
    directions: List[str]
    families: List[str]
    hold: np.ndarray            # (L,) bars
    y: np.ndarray               # (L, n_rows) float64, NaN = no valid outcome
    sd: np.ndarray              # (L,) global sd of y over valid rows


def build_screen_labels(label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice) -> ScreenLabels:
    d = cfg["discovery"]
    names, dirs, fams, holds, ys = [], [], [], [], []
    for direction in DIRECTIONS:
        for fam in d["screen_families"]:
            s = net_R(label_table, direction, fam) if d["target"] == "net" else label_table[f"{direction}_{fam}_gross"]
            names.append(f"{direction}_{fam}"); dirs.append(direction); fams.append(fam)
            holds.append(family_hold(fam, cfg)); ys.append(np.array(s.to_numpy(np.float64)[rows], copy=True))
    y = np.vstack(ys)
    sd = np.array([np.nanstd(r, ddof=1) for r in y])
    return ScreenLabels(names, dirs, fams, np.array(holds, dtype=np.float64), y, sd)


def legal_matrix(atoms: Sequence[Atom]) -> np.ndarray:
    """K x K bool: may atoms i and j be combined (same rule as ``atoms.compatible``, vectorized). Diagonal is False."""
    fam = pd.factorize(np.array([a.family for a in atoms]))[0]
    legal = fam[:, None] != fam[None, :]
    by_feature: Dict[str, List[int]] = {}
    for k, a in enumerate(atoms):
        by_feature.setdefault(a.feature, []).append(k)
    for ks in by_feature.values():                       # same feature: only intervals (handled by the scalar rule)
        for x in ks:
            for z in ks:
                if x != z and compatible(atoms[x], atoms[z]):
                    legal[x, z] = True
    return legal


def fold_bounds(n_rows: int, n_folds: int) -> np.ndarray:
    return np.linspace(0, n_rows, n_folds + 1).round().astype(np.int64)


def _dense(aset: AtomSet, ids: Sequence[int], rows: slice) -> np.ndarray:
    return np.stack([aset.dense(int(i), rows) for i in ids], axis=1).astype(np.uint8)


def _valid_groups(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Labels with identical validity masks share one count product. Returns (valid float (G, n), group index per label (L,))."""
    v = np.isfinite(y)
    keys: Dict[bytes, int] = {}
    gid = np.empty(len(v), dtype=np.int64)
    for l, row in enumerate(v):
        gid[l] = keys.setdefault(row.tobytes(), len(keys))
    uniq = np.zeros((len(keys), v.shape[1]), dtype=np.float64)
    for l, row in enumerate(v):
        uniq[gid[l]] = row
    return uniq, gid


def tile_stats(di: np.ndarray, dj: Optional[np.ndarray], y: np.ndarray, bounds: np.ndarray, chunk_rows: int = 4096,
               dtype: str = "float32") -> Tuple[np.ndarray, np.ndarray]:
    """Pair statistics of one tile. di: (n, bi) uint8 atom columns, dj: (n, bj) or None (= di). y: (L, n) labels with NaN.

    Returns N (L, F, bi, bj) valid-row counts and S (L, F, bi, bj) label sums per fold, as float64.
    """
    dt = np.dtype(dtype)
    L, n = y.shape
    F = len(bounds) - 1
    vg, gid = _valid_groups(y)
    y0 = np.where(np.isfinite(y), y, 0.0)
    bi = di.shape[1]
    bj = bi if dj is None else dj.shape[1]
    Ng = np.zeros((len(vg), F, bi, bj))
    S = np.zeros((L, F, bi, bj))
    for f in range(F):
        for c0 in range(int(bounds[f]), int(bounds[f + 1]), chunk_rows):
            c1 = min(int(bounds[f + 1]), c0 + chunk_rows)
            ai = di[c0:c1].astype(dt)
            aj = ai if dj is None else dj[c0:c1].astype(dt)
            for g in range(len(vg)):
                Ng[g, f] += (ai * vg[g, c0:c1, None].astype(dt)).T @ aj
            for l in range(L):
                S[l, f] += (ai * y0[l, c0:c1, None].astype(dt)).T @ aj
    return Ng[gid], S


def marginal_stats(d: np.ndarray, y: np.ndarray, bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-fold row counts (L, F, K) and label sums (L, F, K) of single atoms (columns of d)."""
    L = y.shape[0]
    F = len(bounds) - 1
    v = np.isfinite(y).astype(np.float64)
    y0 = np.where(np.isfinite(y), y, 0.0)
    N = np.zeros((L, F, d.shape[1]))
    S = np.zeros((L, F, d.shape[1]))
    for f in range(F):
        seg = d[int(bounds[f]):int(bounds[f + 1])].astype(np.float64)
        N[:, f] = v[:, int(bounds[f]):int(bounds[f + 1])] @ seg
        S[:, f] = y0[:, int(bounds[f]):int(bounds[f + 1])] @ seg
    return N, S


def _mean(s: np.ndarray, n: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def screen_tile(Nab: np.ndarray, Sab: np.ndarray, nA: np.ndarray, sA: np.ndarray, nB: np.ndarray, sB: np.ndarray, ok_pair: np.ndarray,
                lab: ScreenLabels, d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Coarse filters for one tile (spec 10.2 step 4). Shapes: Nab/Sab (L,F,bi,bj); nA/sA (L,F,bi); nB/sB (L,F,bj); ok_pair (bi,bj).

    Returns per-label arrays (L,bi,bj): n, mean, t, pos_folds, min_lift, passed.
    """
    n = Nab.sum(1)
    s = Sab.sum(1)
    mean = _mean(s, n)
    t = mean * np.sqrt(n / lab.hold[:, None, None]) / lab.sd[:, None, None]       # hold-length deflation: ~ n / L independent trades
    fold_pos = ((Nab >= d["min_rows_fold"]) & (Sab > 0)).sum(1)
    nA_t, sA_t = nA.sum(1)[:, :, None], sA.sum(1)[:, :, None]
    nB_t, sB_t = nB.sum(1)[:, None, :], sB.sum(1)[:, None, :]
    lift_a = mean - _mean(sA_t - s, nA_t - n)            # child B over parent A: mean(A&B) - mean(A&~B)
    lift_b = mean - _mean(sB_t - s, nB_t - n)            # child A over parent B
    min_lift = np.fmin(lift_a, lift_b)                   # NaN (empty remainder) fails below because NaN > 0 is False
    with np.errstate(invalid="ignore"):
        passed = (ok_pair[None] & (n >= d["min_rows_screen"]) & (t >= d["t_screen"]) & (fold_pos >= d["min_pos_folds"]) & (min_lift > 0))
    return {"n": n, "mean": mean, "t": t, "pos_folds": fold_pos, "min_lift": min_lift, "passed": passed}


def coarse_screen(aset: AtomSet, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice,
                  atom_ids: Optional[Sequence[int]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Screen all legal atom pairs on the train rows. Returns (survivors, info). One survivor row = (pair, direction)."""
    d = cfg["discovery"]
    t0 = time.time()
    ids = np.array(list(range(len(aset.atoms))) if atom_ids is None else list(atom_ids), dtype=np.int64)
    atoms = [aset.atoms[i] for i in ids]
    lab = build_screen_labels(label_table, cfg, rows)
    n_rows = lab.y.shape[1]
    bounds = fold_bounds(n_rows, int(d["n_folds"]))
    legal = legal_matrix(atoms)
    tile = int(d["tile"])
    blocks = [np.arange(s, min(s + tile, len(ids))) for s in range(0, len(ids), tile)]
    dense = {bi: _dense(aset, ids[b], rows) for bi, b in enumerate(blocks)} if len(blocks) <= 4 else None
    getd = (lambda bi: dense[bi]) if dense is not None else (lambda bi: _dense(aset, ids[blocks[bi]], rows))
    nA = np.zeros((len(lab.names), int(d["n_folds"]), len(ids)))
    sA = np.zeros_like(nA)
    for bi, b in enumerate(blocks):
        nA[:, :, b], sA[:, :, b] = marginal_stats(getd(bi), lab.y, bounds)
    recs: List[pd.DataFrame] = []
    n_legal = int(np.triu(legal, 1).sum())
    n_tiles = 0
    for bi, bI in enumerate(blocks):
        di = getd(bi)
        for bj in range(bi, len(blocks)):
            bJ = blocks[bj]
            ok = legal[np.ix_(bI, bJ)] & (bI[:, None] < bJ[None, :])
            if not ok.any():
                continue
            n_tiles += 1
            Nab, Sab = tile_stats(di, None if bj == bi else getd(bj), lab.y, bounds, int(d["chunk_rows"]), d["matmul_dtype"])
            r = screen_tile(Nab, Sab, nA[:, :, bI], sA[:, :, bI], nA[:, :, bJ], sA[:, :, bJ], ok, lab, d)
            for direction in DIRECTIONS:
                sel = np.array([x == direction for x in lab.directions])
                cnt = r["passed"][sel].sum(0)
                ii, jj = np.nonzero(cnt >= int(d["min_labels_pass"]))
                if ii.size == 0:
                    continue
                row = {"atom_a": ids[bI[ii]], "atom_b": ids[bJ[jj]], "direction": direction, "n_labels_pass": cnt[ii, jj]}
                score = np.zeros(ii.size)
                for l in np.flatnonzero(sel):
                    nm = lab.names[l]
                    row[f"{nm}_n"] = r["n"][l][ii, jj]
                    row[f"{nm}_mean"] = r["mean"][l][ii, jj]
                    row[f"{nm}_t"] = r["t"][l][ii, jj]
                    row[f"{nm}_pos_folds"] = r["pos_folds"][l][ii, jj]
                    row[f"{nm}_min_lift"] = r["min_lift"][l][ii, jj]
                    row[f"{nm}_pass"] = r["passed"][l][ii, jj]
                    score += np.where(r["passed"][l][ii, jj], r["t"][l][ii, jj], 0.0)
                row["score"] = score
                recs.append(pd.DataFrame(row))
    surv = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["atom_a", "atom_b", "direction", "n_labels_pass", "score"])
    n_before = len(surv)
    surv = surv.sort_values("score", ascending=False, kind="stable").head(int(d["max_survivors"])).reset_index(drop=True)
    if len(surv):
        surv.insert(2, "expr_a", [aset.atoms[i].expr() for i in surv["atom_a"]])
        surv.insert(3, "expr_b", [aset.atoms[i].expr() for i in surv["atom_b"]])
    info = {"n_atoms": int(len(ids)), "n_legal_pairs": n_legal, "n_tiles": n_tiles, "n_train_rows": int(n_rows),
            "n_survivors": int(len(surv)), "n_survivors_before_cap": int(n_before), "truncated": bool(n_before > len(surv)),
            "survivors_by_direction": {k: int((surv["direction"] == k).sum()) for k in DIRECTIONS} if len(surv) else {k: 0 for k in DIRECTIONS},
            "labels": lab.names, "label_sd": [float(x) for x in lab.sd], "label_hold": [float(x) for x in lab.hold],
            "seconds": round(time.time() - t0, 2)}
    return surv, info
