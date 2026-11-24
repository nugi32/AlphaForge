"""Engine A, depth 3-4 (spec 10.2 step 6): Apriori-style extension of surviving pairs, and the shared N-atom helpers.

Input: the survivors of the depth-2 coarse screen (``05_discovery/survivors_pairs.parquet``). One step adds ONE atom to every surviving
pair (then to every surviving triple), and screens the new rule with the SAME coarse filters as depth 2 (``pairs.screen_tile``):
minimum valid rows, hold-deflated coarse t, positive folds, and a positive lift over the PARENT rule (the rule without the new atom).

Statistics use the same matrix mechanism as depth 2, with the left matrix = dense masks of the parent rules (a beam of the best-scoring
ones) and the right matrix = dense masks of all atoms, processed in tiles of ``deep.apriori.tile`` columns on both sides so that no
``(parents x atoms)`` matrix is ever materialised. Only TRAIN rows are read. Canonical rule = the sorted tuple of atom ids; the same
atom set reached from different parents is merged (best score kept), so a triple is reported once.

Nothing here is evidence: the deep stage only widens the search. Exact statistics, filters, exits, the null test and validation run
exactly as for depth 2 (stage 6 onward). Engine B lives in ``discovery/trees.py`` and feeds the same stage 6.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from edge_research.atoms.atoms import Atom, AtomSet, compatible
from edge_research.discovery.pairs import (DIRECTIONS, ScreenLabels, family_hold, fold_bounds, marginal_stats, screen_tile, tile_stats)
from edge_research.labels.builder import net_R


# ---------------------------------------------------------------------------------------------------------------- N-atom rule helpers
def atoms_of(row: Any) -> List[int]:
    """Atom ids of one rule row or dict: the ``atoms`` field when present, else the depth-2 ``atom_a`` / ``atom_b`` pair."""
    a = row.get("atoms") if isinstance(row, dict) else getattr(row, "atoms", None)
    if a is None or (isinstance(a, float) and np.isnan(a)):      # NaN: a depth-2 row of a mixed frame (pairs x deep columns)
        return [int(getattr(row, "atom_a")), int(getattr(row, "atom_b"))]
    if isinstance(a, str):                       # parquet round trip of a list can come back as a JSON-ish string
        return [int(x) for x in a.strip("[]").split(",") if x.strip()]
    return [int(x) for x in a]


def rule_key(atoms: Sequence[int]) -> str:
    """Canonical text key of an atom set (sorted, so the same set built from different parents has one key)."""
    return ",".join(str(int(i)) for i in sorted(int(x) for x in atoms))


def rule_compatible(atom_objs: Sequence[Atom]) -> bool:
    """True when every pair of the rule's atoms may appear together (the ``atoms.compatible`` rule, applied to all pairs)."""
    for i in range(len(atom_objs)):
        for j in range(i + 1, len(atom_objs)):
            if not compatible(atom_objs[i], atom_objs[j]):
                return False
    return True


def rule_mask(dense, atom_ids: Sequence[int]) -> np.ndarray:
    """AND of the masks of ``atom_ids`` (``dense`` maps an atom id to a boolean row array)."""
    out: Optional[np.ndarray] = None
    for i in atom_ids:
        m = dense(int(i))
        out = m if out is None else (out & m)
    if out is None:
        raise ValueError("a rule needs at least one atom")
    return out


def rule_exprs(aset: AtomSet, atom_ids: Sequence[int]) -> List[str]:
    return [aset.atoms[int(i)].expr() for i in atom_ids]


def rule_expr(aset: AtomSet, atom_ids: Sequence[int]) -> str:
    return " AND ".join(rule_exprs(aset, atom_ids))


def deep_scope_note(cfg: Dict[str, Any]) -> str:
    """One sentence stating what the null test (stage 9) actually repeats, so the report never implies more than was done (spec 13.3)."""
    d = cfg.get("deep", {})
    ap_on = bool(d.get("apriori", {}).get("enabled")) and bool(d.get("apriori", {}).get("include_in_null"))
    tr_on = bool(d.get("trees", {}).get("enabled")) and bool(d.get("trees", {}).get("include_in_null"))
    if ap_on and tr_on:
        return "the null test repeats the FULL deep stage (Apriori expansion AND tree engine) inside every replicate."
    if ap_on:
        return "the null test repeats the Apriori expansion inside every replicate; the tree engine is NOT part of the null statistic (deep.trees.include_in_null is false)."
    if tr_on:
        return "the null test repeats the tree engine inside every replicate; the Apriori expansion is NOT part of the null statistic (deep.apriori.include_in_null is false)."
    return ("the null test covers the depth-2 search only: neither the Apriori expansion nor the tree engine runs inside the null replicates "
            "(deep.apriori.include_in_null and deep.trees.include_in_null are false), so G1 does not price the deep search. Set them to true for an exact null "
            "when the machine can afford it.")


def normalise_rules(frame: pd.DataFrame, aset: AtomSet, source: str) -> pd.DataFrame:
    """Every candidate-rule frame (depth-2 pairs, Apriori survivors, tree leaves) as ``atoms`` / ``direction`` / ``score`` / ``source``.

    Depth-2 rows (``atom_a`` / ``atom_b``) are converted to the same atom-list form, so stage 6 has one input shape.
    """
    cols = ["atoms", "direction", "score", "source"]
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=cols + ["n_atoms"])
    recs: List[Dict[str, Any]] = []
    for r in frame.itertuples(index=False):
        ids = atoms_of(r)
        score = float(getattr(r, "score", 0.0) or 0.0)
        recs.append({"atoms": sorted(ids), "direction": str(r.direction), "score": score, "source": str(getattr(r, "source", source) or source)})
    out = pd.DataFrame(recs, columns=cols)
    for extra in ("hits", "family"):                     # Engine B credibility columns (spec 10.7): carried when the source frame has them
        if extra in frame.columns:
            out[extra] = [getattr(r, extra, None) for r in frame.itertuples(index=False)]
    out["n_atoms"] = [len(x) for x in out["atoms"]]
    # dedup on the canonical rule key: the atoms column holds Python lists, which pandas cannot hash
    out["_key"] = [rule_key(ids) + "|" + str(d) for ids, d in zip(out["atoms"], out["direction"])]
    out = out.drop_duplicates(["_key"], keep="first").drop(columns=["_key"]).reset_index(drop=True)
    # a rule that repeats an atom, or mixes atoms of one feature family, is illegal: drop it here, once, for every source
    keep = []
    for ids in out["atoms"]:
        objs = [aset.atoms[i] for i in ids]
        keep.append(len(set(ids)) == len(ids) and rule_compatible(objs))
    out = out[np.array(keep, dtype=bool)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------------------------------------------- Apriori expansion
def direction_screen_labels(label_table: pd.DataFrame, cfg: Dict[str, Any], direction: str, rows: slice) -> ScreenLabels:
    """The screening labels of ONE direction (``build_screen_labels`` restricted): only a rule of that direction can be screened with them."""
    d = cfg["discovery"]
    names, fams, holds, ys = [], [], [], []
    for fam in d["screen_families"]:
        s = net_R(label_table, direction, fam) if d["target"] == "net" else label_table[f"{direction}_{fam}_gross"]
        names.append(f"{direction}_{fam}")
        fams.append(fam)
        holds.append(family_hold(fam, cfg))
        ys.append(np.array(s.to_numpy(np.float64)[rows], copy=True))
    y = np.vstack(ys)
    sd = np.array([np.nanstd(r, ddof=1) for r in y])
    return ScreenLabels(names, [direction] * len(names), fams, np.array(holds, dtype=np.float64), y, sd)


def _tiles(n: int, size: int) -> List[np.ndarray]:
    return [np.arange(s, min(s + size, n), dtype=np.int64) for s in range(0, n, size)]


def _atom_block(aset: AtomSet, ids: Sequence[int], rows: slice) -> np.ndarray:
    """Dense uint8 block (rows x len(ids)) of the masks of ``ids``."""
    if not len(ids):
        return np.zeros((0, 0), dtype=np.uint8)
    return np.stack([aset.dense(int(i), rows).astype(np.uint8) for i in ids], axis=1)


def _rule_block(aset: AtomSet, parents: Sequence[Sequence[int]], rows: slice) -> np.ndarray:
    """Dense uint8 block (rows x len(parents)) of the AND of every parent's atoms."""
    cols = []
    for ids in parents:
        m: Optional[np.ndarray] = None
        for i in ids:
            a = aset.dense(int(i), rows)
            m = a if m is None else (m & a)
        cols.append(m.astype(np.uint8))
    return np.stack(cols, axis=1)


def expand_one_step(aset: AtomSet, parents: Sequence[Dict[str, Any]], direction: str, lab: ScreenLabels, bounds: np.ndarray,
                    cfg: Dict[str, Any], rows: slice, tile: int, dense_budget_bytes: int = 256 * 1024 * 1024
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """One Apriori step for one direction: every parent x every atom, coarse-screened with the depth-2 filters.

    ``parents``: dicts with ``atoms`` (sorted tuple of ids) and ``score``. Returns (survivors, counters) where a survivor is
    ``{atoms, direction, score, n_labels_pass}`` and ``score`` is the sum of the coarse t over the passing screening labels.

    ``parents`` and ``atoms`` are both tiled (``tile`` columns), and the dense atom block is cached across left tiles only while it fits
    ``dense_budget_bytes`` (otherwise it is rebuilt per left tile, trading bytes for time).
    """
    d = cfg["discovery"]
    cnt = {"n_cells_screened": 0}
    p_tiles = _tiles(len(parents), int(tile))
    a_tiles = _tiles(len(aset.atoms), int(tile))
    n_rows = lab.y.shape[1]
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cache_ok = n_rows * len(aset.atoms) <= int(dense_budget_bytes)
    out: List[Dict[str, Any]] = []
    for pids in p_tiles:
        dpl = _rule_block(aset, [[int(x) for x in parents[int(p)]["atoms"]] for p in pids], rows)
        nA, sA = marginal_stats(dpl, lab.y, bounds)
        for bi, rids in enumerate(a_tiles):
            if bi not in cache:
                dr = _atom_block(aset, [int(j) for j in rids], rows)
                nB, sB = marginal_stats(dr, lab.y, bounds)
                if cache_ok:
                    cache[bi] = (dr, nB, sB)
            else:
                dr, nB, sB = cache[bi]
            ok = np.zeros((len(pids), len(rids)), dtype=bool)
            for a, pid in enumerate(pids):
                pa = [int(x) for x in parents[int(pid)]["atoms"]]
                pobjs = [aset.atoms[x] for x in pa]
                for b, j in enumerate(rids):
                    aj = int(j)
                    if aj in pa:
                        continue
                    if all(compatible(aset.atoms[aj], x) for x in pobjs):
                        ok[a, b] = True
            if not ok.any():
                continue
            cnt["n_cells_screened"] += int(ok.sum())
            Nab, Sab = tile_stats(dpl, dr, lab.y, bounds, int(d["chunk_rows"]), d["matmul_dtype"])
            r = screen_tile(Nab, Sab, nA, sA, nB, sB, ok, lab, d)
            n_pass = r["passed"].sum(0)
            t_sum = np.where(r["passed"], r["t"], 0.0).sum(0)
            ii, jj = np.nonzero(n_pass >= int(d["min_labels_pass"]))
            for a, b in zip(ii, jj):
                ids = tuple(sorted([int(x) for x in parents[int(pids[a])]["atoms"]] + [int(rids[b])]))
                out.append({"atoms": ids, "direction": direction, "score": float(t_sum[a, b]), "n_labels_pass": int(n_pass[a, b])})
    return out, cnt


def expand_survivors(aset: AtomSet, surv_pairs: pd.DataFrame, label_table: pd.DataFrame, cfg: Dict[str, Any], rows: slice
                     ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Depth-3 and depth-4 survivors from the depth-2 survivors. Returns (frame with ``atoms``/``direction``/``score``/``n_atoms``, info)."""
    ap = cfg["deep"]["apriori"]
    d = cfg["discovery"]
    t0 = time.time()
    info: Dict[str, Any] = {"enabled": bool(ap["enabled"]), "n_pairs_in": int(len(surv_pairs)), "n_survivors": 0, "by_depth": {}, "seconds": 0.0,
                            "truncated": False, "n_cells_screened": 0}
    cols = ["atoms", "direction", "score", "n_labels_pass", "n_atoms"]
    if not ap["enabled"] or len(surv_pairs) == 0:
        return pd.DataFrame(columns=cols), info
    max_atoms = int(ap["max_atoms"])
    beam3, beam4 = int(ap["beam_triple"]), int(ap["beam_quad"])
    frontier: Dict[str, List[Dict[str, Any]]] = {k: [] for k in DIRECTIONS}
    for r in surv_pairs.itertuples(index=False):
        ids = tuple(sorted((int(r.atom_a), int(r.atom_b))))
        frontier[str(r.direction)].append({"atoms": ids, "score": float(getattr(r, "score", 0.0) or 0.0), "direction": str(r.direction)})
    all_rows: List[Dict[str, Any]] = []
    for depth in range(3, max_atoms + 1):
        beam = beam3 if depth == 3 else beam4
        nxt: Dict[str, List[Dict[str, Any]]] = {k: [] for k in DIRECTIONS}
        for direction in DIRECTIONS:
            parents = frontier[direction]
            if not parents:
                continue
            parents = sorted(parents, key=lambda x: (-float(x["score"]), x["atoms"]))[:beam]
            lab = direction_screen_labels(label_table, cfg, direction, rows)
            bounds = fold_bounds(lab.y.shape[1], int(d["n_folds"]))
            found, counters = expand_one_step(aset, parents, direction, lab, bounds, cfg, rows, int(ap["tile"]))
            info["n_cells_screened"] += counters["n_cells_screened"]
            best: Dict[Tuple[Tuple[int, ...], str], Dict[str, Any]] = {}
            for x in found:
                key = (tuple(x["atoms"]), x["direction"])
                if key not in best or float(x["score"]) > float(best[key]["score"]):
                    best[key] = x
            nxt[direction] = list(best.values())
            info["by_depth"][f"d{depth}_{direction}"] = len(nxt[direction])
            all_rows += nxt[direction]
        frontier = nxt
    frame = pd.DataFrame(all_rows, columns=cols[:-1])
    if len(frame):
        frame["n_atoms"] = [len(x) for x in frame["atoms"]]
        cap = int(ap["max_survivors"])
        if len(frame) > cap:
            info["truncated"] = True
            frame = frame.sort_values("score", ascending=False, kind="stable").head(cap).reset_index(drop=True)
        frame = frame.sort_values(["n_atoms", "score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    info["n_survivors"] = int(len(frame))
    info["seconds"] = round(time.time() - t0, 2)
    return frame, info
