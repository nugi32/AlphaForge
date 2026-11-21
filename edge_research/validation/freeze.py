"""Freeze the representatives of Stage 8 into canonical rules (design 14.1 step 1).

Freezing = writing atom, numeric threshold, direction and exit family into a ``FrozenRule``; nothing is tuned after this point. The atoms'
thresholds are the (rounded) train quantiles stored by Stage 4, so the rule is exactly what the discovery stages evaluated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from edge_research.atoms.atoms import AtomSet
from edge_research.rules import FrozenRule

META_COLS = ["cand_id", "fcluster", "f_cluster_size", "rank_score", "plateau_score", "mono_score", "best_family", "best_t", "best_mean", "chosen_family", "chosen_mean",
             "chosen_smooth", "chosen_t", "chosen_n_episodes", "baseline_T", "improvement_over_T", "f3_positive", "f3_cells", "reason", "f8_simplified", "f8_dropped", "f8_val_mean"]


def rule_id_of(cand_id: int) -> str:
    return f"R{int(cand_id):05d}"


def freeze_rules(chosen: pd.DataFrame, aset: AtomSet, cfg: Dict[str, Any], max_rules: Optional[int] = None, null_flags: Optional[pd.DataFrame] = None) -> List[FrozenRule]:
    """Frozen rules for the cluster representatives that may be frozen (``is_representative`` and ``frozen_ok``), best ``plateau_score`` first.

    The order and the cap use TRAIN statistics only. ``null_flags`` (Stage 9 ``rule_flags``) is copied into ``meta`` when given.
    """
    if chosen is None or len(chosen) == 0:
        return []
    sel = chosen[chosen["is_representative"].astype(bool) & chosen["frozen_ok"].astype(bool)]
    sel = sel.sort_values(["plateau_score", "rank_score", "cand_id"], ascending=[False, False, True], kind="stable")
    if max_rules is not None:
        sel = sel.head(int(max_rules))
    flags = None
    if null_flags is not None and len(null_flags) and "cand_id" in null_flags.columns:
        flags = null_flags.set_index("cand_id")
    out: List[FrozenRule] = []
    for _, r in sel.iterrows():
        ids = [int(x) for x in r["atoms"]] if "atoms" in r.index and r["atoms"] is not None else [int(r["atom_a"]), int(r["atom_b"])]
        atoms = []
        for i in ids:
            f, op, thr = aset.atoms[int(i)].to_rule()
            atoms.append({"feature": f, "op": op, "threshold": float(thr)})
        meta: Dict[str, Any] = {}
        for c in META_COLS:
            if c in r.index:
                v = r[c]
                meta[c] = None if (isinstance(v, float) and not np.isfinite(v)) or v is None else (v.item() if hasattr(v, "item") else v)
        meta["atom_ids"] = ids
        if "exprs" in r.index and r["exprs"] is not None:
            meta["exprs"] = [str(x) for x in r["exprs"]]
        elif "rule_expr" in r.index and isinstance(r["rule_expr"], str):
            meta["exprs"] = [x.strip() for x in r["rule_expr"].split(" AND ")]
        else:
            meta["exprs"] = [str(r["expr_a"]), str(r["expr_b"])] if len(ids) == 2 else [aset.atoms[i].expr() for i in ids]
        if flags is not None and int(r["cand_id"]) in flags.index:
            fr = flags.loc[int(r["cand_id"])]
            for c in ("beats_random", "beats_calendar", "calendar_driven"):
                if c not in flags.columns:
                    continue
                v = fr[c]
                meta[f"null_{c}"] = None if v is None or v != v else (v.item() if hasattr(v, "item") else v)   # NaN = the calendar variant was missing: unknown
        out.append(FrozenRule(rule_id=rule_id_of(int(r["cand_id"])), direction=str(r["direction"]), atoms=atoms, exit=str(r["chosen_family"]),
                              cat_stop_atr=None, meta=meta).validate())
    return out
