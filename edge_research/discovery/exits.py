"""Stage 8 (M6): exit selection on the smoothed exit surface of each filtered candidate (spec 12) and F3 (exit neighbourhood).

Exit selection is part of the search: it happens on TRAIN rows only, from the per-family means that Stage 6 already stored for every
candidate (``candidate_families``). The surface has two parts:
  * time stops ``T{N}``: a 1-D grid over N;   * bar-boundary exits ``B{tp}_{sl}_{H}``: a (tp, sl) grid for each horizon H.
The value of a cell is SMOOTHED (mean over the cell and its grid neighbours: 3 points on the 1-D grid, 3 x 3 in the (tp, sl) plane) and the cell with
the best smoothed value is chosen, never the best raw value. Only cells with at least ``min_episodes_train`` episodes (their own episodes, gap = hold) are eligible.
A barrier exit must beat the best smoothed time stop by ``min_improvement_R``, otherwise the simpler time stop is chosen. Equal smoothed values (a flat
plateau) are broken by: time stop before barrier, then the cell with the fullest neighbourhood (the interior of the plateau), then the shorter hold.
F3: the chosen cell and its direct neighbours (+-2 on the 1-D grid, +-1 in tp and in sl for barriers; up to 5 cells) must have a positive mean R and enough
episodes in at least ``f3_min_positive`` of them. Confirmation on validation with frozen parameters is M8; nothing is re-optimised there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from edge_research.mt5_backtest.engine import parse_family


def exit_grid(names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Grid coordinates, smoothing neighbours and F3 neighbours (as family names) for every exit family name."""
    p = {n: parse_family(n) for n in names}
    t_sorted = sorted([n for n in names if p[n]["kind"] == "T"], key=lambda n: p[n]["n"])
    info: Dict[str, Dict[str, Any]] = {}
    for i, n in enumerate(t_sorted):
        info[n] = {"smooth": [t_sorted[j] for j in (i - 1, i, i + 1) if 0 <= j < len(t_sorted)],
                   "f3": [t_sorted[j] for j in range(i - 2, i + 3) if 0 <= j < len(t_sorted)], "hold": p[n]["n"]}
    bs = [n for n in names if p[n]["kind"] in ("B", "I")]               # intrabar exits share the (tp, sl) grid SHAPE of the barrier exits,
    for kind in ("B", "I"):                                             # but B and I at the same coordinates are separate cells (a dict keyed by
        ks = [n for n in bs if p[n]["kind"] == kind]                    # (tp, sl) would let one kind overwrite the other's smoothing neighbours)
        for n in ks:                                                    #
            q = p[n]
            same_h = [m for m in ks if p[m]["h"] == q["h"]]
            tps = sorted({p[m]["tp"] for m in same_h})
            sls = sorted({p[m]["sl"] for m in same_h})
            it, js = tps.index(q["tp"]), sls.index(q["sl"])
            at = {(p[m]["tp"], p[m]["sl"]): m for m in same_h}
            sm = [at[(tps[a], sls[b])] for a in range(max(0, it - 1), min(len(tps), it + 2)) for b in range(max(0, js - 1), min(len(sls), js + 2))]
            f3 = [at[(tps[it], sls[js])]] + [at[(tps[a], sls[js])] for a in (it - 1, it + 1) if 0 <= a < len(tps)] + [at[(tps[it], sls[b])] for b in (js - 1, js + 1) if 0 <= b < len(sls)]
            info[n] = {"smooth": sm, "f3": f3, "hold": q["h"]}
    return info


def select_exits(fam: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (surface, chosen). ``fam`` = candidate_families (one row per candidate x family: mean, n_episodes, t, ...)."""
    e = cfg["exits"]
    min_ep = int(cfg["candidates"]["min_episodes_train"])
    rows_s: List[Dict[str, Any]] = []
    rows_c: List[Dict[str, Any]] = []
    for cid, g in fam.groupby("cand_id", sort=True):
        g = g.set_index("family")
        names = list(g.index)
        grid = exit_grid(names)
        mean = g["mean"].astype(float)
        eligible = (g["n_episodes"] >= min_ep) & mean.notna()
        smooth = {n: float(np.mean([mean[m] for m in grid[n]["smooth"] if np.isfinite(mean[m])])) if any(np.isfinite(mean[m]) for m in grid[n]["smooth"]) else float("nan") for n in names}
        kinds = {n: parse_family(n)["kind"] for n in names}
        for n in names:
            rows_s.append({"cand_id": int(cid), "family": n, "kind": kinds[n], "hold": grid[n]["hold"], "mean": float(mean[n]), "smooth": smooth[n],
                           "t": float(g.at[n, "t"]), "n_episodes": int(g.at[n, "n_episodes"]), "eligible": bool(eligible[n])})
        el = [n for n in names if eligible[n] and np.isfinite(smooth[n])]
        out: Dict[str, Any] = {"cand_id": int(cid), "direction": g["direction"].iloc[0] if "direction" in g.columns else None}
        if not el:
            out.update(chosen_family=None, pass_f3=False, reason="no eligible exit cell")
            rows_c.append(out)
            continue
        order = lambda n: (-round(smooth[n], 12), 0 if kinds[n] == "T" else 1, -len(grid[n]["smooth"]), grid[n]["hold"], n)       # ties: simpler kind, plateau interior, shorter hold
        best_any = min(el, key=order)
        t_el = [n for n in el if kinds[n] == "T"]
        base_T = min(t_el, key=order) if t_el else None
        chosen, reason = best_any, "best smoothed cell"
        if kinds[best_any] in ("B", "I") and base_T is not None and smooth[best_any] - smooth[base_T] < float(e["min_improvement_R"]):
            chosen, reason = base_T, "barrier does not beat the best time stop by min_improvement_R"
        raw_best = max(el, key=lambda n: (mean[n], n))
        nb = [m for m in grid[chosen]["f3"] if eligible[m] and mean[m] > 0]
        out.update(chosen_family=chosen, kind=kinds[chosen], hold=grid[chosen]["hold"], chosen_mean=float(mean[chosen]), chosen_smooth=smooth[chosen],
                   chosen_t=float(g.at[chosen, "t"]), chosen_n_episodes=int(g.at[chosen, "n_episodes"]), baseline_T=base_T,
                   baseline_T_smooth=smooth[base_T] if base_T else float("nan"), improvement_over_T=(smooth[chosen] - smooth[base_T]) if base_T else float("nan"),
                   raw_best_family=raw_best, raw_best_mean=float(mean[raw_best]), raw_best_is_chosen=bool(raw_best == chosen),
                   f3_positive=len(nb), f3_cells=len(grid[chosen]["f3"]), pass_f3=bool(len(nb) >= int(e["f3_min_positive"])),
                   r_per_bar=float(mean[chosen] / max(grid[chosen]["hold"], 1)), reason=reason)
        rows_c.append(out)
    return pd.DataFrame(rows_s), pd.DataFrame(rows_c)


def build_exits(passed: pd.DataFrame, fam: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Exit selection for the candidates that passed Stage 7. ``chosen`` carries the rule columns, the chosen exit, F3 and the final cluster representative.

    The representative of a Stage 7 cluster is re-picked among the members that pass F3 (best plateau score): a candidate whose exit neighbourhood is
    empty cannot be frozen, so it cannot represent anything.
    """
    info: Dict[str, Any] = {"n_in": int(len(passed))}
    if passed.empty:
        info.update(n_with_exit=0, n_pass_f3=0, n_frozen=0, n_changed_from_raw_best=0)
        return pd.DataFrame(), pd.DataFrame(), info
    ids = set(passed["cand_id"])
    surface, ch = select_exits(fam[fam["cand_id"].isin(ids)], cfg)
    keep = ["cand_id", "atoms", "n_atoms", "rule_key", "rule_expr", "source", "atom_a", "atom_b", "expr_a", "expr_b", "direction", "rank_score", "plateau_score",
            "mono_score", "fcluster", "f_cluster_size", "best_family", "best_t", "best_mean", "f8_simplified", "f8_dropped", "f8_val_mean"]
    chosen = passed[keep].rename(columns={"direction": "direction_c"}).merge(ch.drop(columns=["direction"]), on="cand_id", how="left").rename(columns={"direction_c": "direction"})
    chosen["pass_f3"] = chosen["pass_f3"].fillna(False).astype(bool)
    chosen["frozen_ok"] = chosen["pass_f3"] if bool(cfg["exits"]["require_f3"]) else chosen["chosen_family"].notna()
    chosen["is_representative"] = False
    sel = chosen[chosen["frozen_ok"]].sort_values(["plateau_score", "rank_score", "cand_id"], ascending=[False, False, True], kind="stable")
    first = sel.drop_duplicates("fcluster").index
    chosen.loc[first, "is_representative"] = True
    info.update(n_with_exit=int(chosen["chosen_family"].notna().sum()), n_pass_f3=int(chosen["pass_f3"].sum()), n_frozen=int(chosen["is_representative"].sum()),
                n_changed_from_raw_best=int((~chosen["raw_best_is_chosen"].fillna(True)).sum()),
                chosen_kinds={k: int(v) for k, v in chosen.loc[chosen["frozen_ok"], "kind"].value_counts().items()})
    return surface, chosen, info
