"""compare_trades (design 15.9): the engine's trade list against a Strategy Tester trade list. The acceptance criterion is a 100 % match of entry and exit times."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ALIASES = {"entry_time": ["entry_time", "open_time", "time_in", "time"], "exit_time": ["exit_time", "close_time", "time_out"],
           "entry_price": ["entry_price", "open_price", "price_in"], "exit_price": ["exit_price", "close_price", "price_out"],
           "profit": ["profit", "net_profit"], "direction": ["direction", "type", "side"]}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    low = {c.lower(): c for c in df.columns}
    for key, names in ALIASES.items():
        for nm in names:
            if nm in low:
                out[key] = df[low[nm]]
                break
    for need in ("entry_time", "exit_time"):
        if need not in out:
            raise ValueError(f"trade list has no {need} column (accepted names: {ALIASES[need]})")
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["exit_time"] = pd.to_datetime(out["exit_time"])
    if "direction" in out:
        d = out["direction"].astype(str).str.lower()
        out["direction"] = np.where(d.isin(["buy", "long"]), "long", np.where(d.isin(["sell", "short"]), "short", d))
    return out.sort_values("entry_time").reset_index(drop=True)


def compare_trades(engine: pd.DataFrame, mt5: pd.DataFrame, price_tol: float = 1e-4, pnl_rel_tol: float = 0.01) -> Dict[str, Any]:
    """Match trades by entry time. A trade list must not contain two trades with the same entry time (hedging accounts with several rules: compare one rule at a time)."""
    a, b = _normalise(engine), _normalise(mt5)
    for nm, d in (("engine", a), ("mt5", b)):
        if d["entry_time"].duplicated().any():
            raise ValueError(f"{nm} trade list has duplicate entry times; compare one rule at a time")
    m = a.merge(b, on="entry_time", how="outer", suffixes=("_e", "_m"), indicator=True)
    both = m[m["_merge"] == "both"]
    only_e, only_m = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]
    n_e, n_m, n_b = len(a), len(b), len(both)
    exit_ok = (both["exit_time_e"] == both["exit_time_m"])
    out: Dict[str, Any] = {"n_engine": n_e, "n_mt5": n_m, "n_matched": n_b,
                           "entry_match_pct": 100.0 * n_b / max(n_e, n_m, 1), "exit_match_pct": 100.0 * float(exit_ok.sum()) / max(n_b, 1) if n_b else 0.0,
                           "only_engine": [str(t) for t in only_e["entry_time"]], "only_mt5": [str(t) for t in only_m["entry_time"]]}
    bad = pd.Series(False, index=both.index)
    for k in ("entry_price", "exit_price"):
        if f"{k}_e" in both and f"{k}_m" in both:
            d = (both[f"{k}_e"].astype(float) - both[f"{k}_m"].astype(float)).abs()
            out[f"max_abs_{k}_diff"] = float(d.max()) if len(d) else 0.0
            bad |= d > price_tol
    if "direction_e" in both and "direction_m" in both:
        dd = both["direction_e"] != both["direction_m"]
        out["direction_mismatches"] = int(dd.sum())
        bad |= dd
    if "profit_e" in both and "profit_m" in both:
        pe, pm = float(both["profit_e"].astype(float).sum()), float(both["profit_m"].astype(float).sum())
        out.update(profit_engine=pe, profit_mt5=pm, profit_rel_diff=abs(pe - pm) / max(abs(pm), 1e-12))
    first: Optional[Dict[str, Any]] = None
    cand: List[Any] = [(t, "only in the engine") for t in only_e["entry_time"]] + [(t, "only in MT5") for t in only_m["entry_time"]]
    cand += [(t, "exit time differs") for t in both.loc[~exit_ok, "entry_time"]] + [(t, "price or direction differs") for t in both.loc[bad & exit_ok, "entry_time"]]
    if cand:
        t0, why = min(cand, key=lambda c: c[0])
        first = {"entry_time": str(t0), "reason": why}
    out["first_difference"] = first
    out["n_price_mismatch"] = int(bad.sum())
    prof_ok = ("profit_rel_diff" not in out) or out["profit_rel_diff"] <= pnl_rel_tol
    out["accepted"] = bool(out["entry_match_pct"] == 100.0 and out["exit_match_pct"] == 100.0 and not bad.any() and prof_ok and n_b > 0)
    return out
