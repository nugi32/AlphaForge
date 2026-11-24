"""Evaluate recipe expressions on a bar table (the Python reference of the MQL5 helpers in ``mql_helpers``).

Expressions are evaluated VECTORISED: ``C(1)`` is the array of closes of the last closed bar for every decision row i (the Python feature table row i = the EA
decision at the first tick of bar i + 1, which reads shift 1). A shift s therefore reads table row i + 1 - s; before the first row the value is NaN.
This module is the semantics the MQL5 helpers must reproduce: it is checked against the Python feature table (feature parity) in the tests and, on the
real EA, by the feature dump (``parity.check_features``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from edge_research.features.registry import SESSIONS


def _shift(arr: np.ndarray, s: int) -> np.ndarray:
    """out[i] = arr[i + 1 - s] (NaN where the row does not exist)."""
    if s < 1:
        raise ValueError("shift must be >= 1 (the EA acts on closed bars)")
    k = s - 1
    out = np.full(len(arr), np.nan)
    if k < len(arr):
        out[k:] = arr[: len(arr) - k]
    return out


class Namespace:
    def __init__(self, df: pd.DataFrame, cfg: Mapping[str, Any]):
        self.df = df
        self.n = len(df)
        self.atr_col = cfg["features"]["atr_column"]
        self._arr: Dict[str, np.ndarray] = {}

    def arr(self, col: str) -> np.ndarray:
        if col not in self._arr:
            a = self.df[col].to_numpy(np.float64).copy()
            self._arr[col] = a
        return self._arr[col]

    # --- primitives -----------------------------------------------------------------------------------------------------------
    def _price(self, col: str, s: int) -> np.ndarray:
        v = _shift(self.arr(col), s)
        return np.where(v > 0.0, v, np.nan)

    def O(self, s): return self._price("open", s)
    def H(self, s): return self._price("high", s)
    def L(self, s): return self._price("low", s)
    def C(self, s): return self._price("close", s)

    def IND(self, col: str) -> Callable[[int], np.ndarray]:
        return lambda s: _shift(self.arr(col), s)

    def ATRN(self, s):
        a = _shift(self.arr(self.atr_col), s)
        return np.where(a > 0.0, a, np.nan)

    @staticmethod
    def DIV(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ok = np.isfinite(a) & np.isfinite(b) & (b != 0.0)
        with np.errstate(all="ignore"):
            return np.where(ok, a / np.where(ok, b, 1.0), np.nan)

    @staticmethod
    def MAX2(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return np.where(np.isfinite(a) & np.isfinite(b), np.where(a > b, a, b), np.nan)

    @staticmethod
    def MIN2(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return np.where(np.isfinite(a) & np.isfinite(b), np.where(a < b, a, b), np.nan)

    def HHV(self, n, s):
        # max of H over shifts s .. s+n-1  =  rolling max of the high column ending at row i + 1 - s
        h = np.where(self.arr("high") > 0.0, self.arr("high"), np.nan)
        return _shift(pd.Series(h).rolling(int(n)).max().to_numpy(), s)

    def LLV(self, n, s):
        l = np.where(self.arr("low") > 0.0, self.arr("low"), np.nan)
        return _shift(pd.Series(l).rolling(int(n)).min().to_numpy(), s)

    def HOURV(self, s):
        return _shift(self.df.index.hour.to_numpy().astype(np.float64), s)

    def DOWV(self, s):
        return _shift(self.df.index.dayofweek.to_numpy().astype(np.float64), s)

    def SESSIONV(self, s):
        h = self.HOURV(s)
        out = np.where(np.isfinite(h), 0.0, np.nan)
        for i, (lo, hi) in enumerate(SESSIONS.values()):
            out = np.where(np.isfinite(h) & (h >= lo) & (h <= hi), float(i), out)
        return out

    def BRKUP(self, n):
        hh, c = self.HHV(n, 2), self.C(1)
        return np.where(np.isfinite(hh) & np.isfinite(c), (c > hh).astype(np.float64), np.nan)

    def BRKDN(self, n):
        ll, c = self.LLV(n, 2), self.C(1)
        return np.where(np.isfinite(ll) & np.isfinite(c), (c < ll).astype(np.float64), np.nan)

    # --- atom comparisons (same semantics as ``rules.atom_mask``) ------------------------------------------------------------
    @staticmethod
    def LE(v, t): v = np.asarray(v, dtype=np.float64); return np.isfinite(v) & (v <= t)
    @staticmethod
    def GT(v, t): v = np.asarray(v, dtype=np.float64); return np.isfinite(v) & (v > t)
    @staticmethod
    def EQ(v, t): v = np.asarray(v, dtype=np.float64); return np.isfinite(v) & (v == t)

    def env(self, columns: Sequence[str], f: Optional[np.ndarray] = None) -> Dict[str, Any]:
        e: Dict[str, Any] = {k: getattr(self, k) for k in ("O", "H", "L", "C", "ATRN", "DIV", "MAX2", "MIN2", "HHV", "LLV", "HOURV", "DOWV", "SESSIONV",
                                                          "BRKUP", "BRKDN", "LE", "GT", "EQ")}
        for c in columns:
            e[f"IND_{c}"] = self.IND(c)
        return e


def eval_expr(ns: Namespace, expr: str, columns: Sequence[str]) -> np.ndarray:
    """Value of one recipe expression at every decision row (float64, NaN where unavailable)."""
    out = eval(expr, {"__builtins__": {}}, ns.env(columns))        # the expression text is generated by recipes.py, never user text
    return np.broadcast_to(np.asarray(out, dtype=np.float64), (ns.n,)).copy()


def plan_features(plan: Any, df: pd.DataFrame, cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Feature values of the plan's rule at every decision row, from the EMITTED expression strings (name -> float64 array)."""
    ns = Namespace(df, cfg)
    cols = [i.column for i in plan.indicators]
    return {r.name: eval_expr(ns, r.expr, cols) for r in plan.features}


def plan_signal(plan: Any, df: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    """Boolean signal of the plan at every decision row: the EMITTED atom expressions ANDed (what ``EvalSignal`` returns on the closed bar)."""
    ns = Namespace(df, cfg)
    cols = [i.column for i in plan.indicators]
    f = [eval_expr(ns, r.expr, cols) for r in plan.features]
    env = ns.env(cols)
    env["f"] = f
    sig = np.ones(len(df), dtype=bool)
    for ae in plan.atom_exprs:
        sig &= np.broadcast_to(np.asarray(eval(ae, {"__builtins__": {}}, env), dtype=bool), (len(df),))
    return sig
