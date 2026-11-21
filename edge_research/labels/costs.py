"""
Shared cost model (single source of truth for the label table AND the backtest engine).

Semantics follow the old engine (`mt5_backtest/engine.py`):
  Bid = bar open, Ask = Bid + spread.
  Long buys at Ask on the entry bar and sells at Bid on the exit bar  -> pays the spread of the ENTRY bar.
  Short sells at Bid on the entry bar and buys at Ask on the exit bar -> pays the spread of the EXIT bar.
  Slippage: adverse points per fill (2 fills per round trip). Commission: 2 * per-lot-per-side / contract_size (price units).
  Swap: nights * swap_per_lot / contract_size; a negative swap rate is a cost.
All quantities here are in PRICE units per 1 unit of the instrument (not per lot).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostSpec:
    point: float = 0.00001
    contract_size: float = 100000.0
    fixed_spread_points: float = 12.0
    slippage_points: float = 0.0
    commission_per_lot_per_side: float = 0.0
    swap_long_per_lot: float = 0.0
    swap_short_per_lot: float = 0.0
    swap_rollover_hour: int = 0
    swap_triple_weekday: int = 2

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CostSpec":
        c = cfg["costs"]
        return cls(point=c["point"], contract_size=c["contract_size"], fixed_spread_points=c["fixed_spread_points"],
                   slippage_points=c["slippage_points"], commission_per_lot_per_side=c["commission_per_lot_per_side"],
                   swap_long_per_lot=c["swap_long_per_lot"], swap_short_per_lot=c["swap_short_per_lot"],
                   swap_rollover_hour=c["swap_rollover_hour"], swap_triple_weekday=c["swap_triple_weekday"])

    @property
    def fixed_spread_price(self) -> float:
        return self.fixed_spread_points * self.point

    @property
    def slippage_price(self) -> float:
        return self.slippage_points * self.point

    @property
    def commission_price(self) -> float:
        return 2.0 * self.commission_per_lot_per_side / self.contract_size

    def swap_cost_price(self, nights: np.ndarray, is_long: bool) -> np.ndarray:
        rate = self.swap_long_per_lot if is_long else self.swap_short_per_lot
        return -np.asarray(nights, dtype=np.float64) * rate / self.contract_size


def spread_array(index: pd.DatetimeIndex, spread_points: Optional[pd.Series], cost: CostSpec) -> Tuple[np.ndarray, int]:
    """Per-bar spread in price units and the number of bars that fell back to the fixed value.

    Same rule as the old engine: a history value <= 0 is replaced by the last known value; leading unknowns use the fixed spread.
    """
    if spread_points is None:
        return np.full(len(index), cost.fixed_spread_price, dtype=np.float64), len(index)
    aligned = spread_points.reindex(index).to_numpy(dtype=np.float64) * cost.point
    aligned = np.where(aligned > 0, aligned, np.nan)
    aligned = pd.Series(aligned).ffill().to_numpy(dtype=np.float64, copy=True)
    bad = ~np.isfinite(aligned)
    aligned[bad] = cost.fixed_spread_price
    return aligned, int(bad.sum())


def swap_nights(t_in: pd.Timestamp, t_out: pd.Timestamp, rollover_hour: int, triple_weekday: int) -> int:
    """Reference (loop) implementation, ported from the old engine. Weighted rollovers in (t_in, t_out]."""
    nights = 0
    day = t_in.normalize()
    last = t_out.normalize()
    while day <= last:
        instant = day + pd.Timedelta(hours=rollover_hour)
        if t_in < instant <= t_out:
            booked = (instant - pd.Timedelta(days=1)) if rollover_hour == 0 else instant
            wd = booked.weekday()
            if wd < 5:
                nights += 3 if wd == triple_weekday else 1
        day += pd.Timedelta(days=1)
    return nights


def cumulative_rollovers(times: pd.DatetimeIndex, rollover_hour: int, triple_weekday: int) -> np.ndarray:
    """Weighted rollover count up to and including each timestamp. ``nights(t_in, t_out) = cum[t_out] - cum[t_in]``.

    Vectorised equivalent of ``swap_nights`` (verified by test); O(n log n) instead of a Python loop per trade.
    """
    first = times.min().normalize() - pd.Timedelta(days=1)
    last = times.max().normalize() + pd.Timedelta(days=1)
    days = pd.date_range(first, last, freq="D")
    instants = days + pd.Timedelta(hours=rollover_hour)
    booked = instants - pd.Timedelta(days=1) if rollover_hour == 0 else instants
    wd = booked.weekday.to_numpy()
    weight = np.where(wd < 5, np.where(wd == triple_weekday, 3, 1), 0).astype(np.int64)
    cum_w = np.concatenate([[0], np.cumsum(weight)])
    pos = np.searchsorted(instants.to_numpy(), times.to_numpy(), side="right")
    return cum_w[pos].astype(np.int64)
