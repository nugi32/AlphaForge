"""Money accounting on top of the engine's R-based trades (design 15.6, 15.8): position size, profit in money, balance and equity per bar.

Conventions (everything in price units per 1 unit of the instrument, converted with ``contract_size``; the deposit currency is assumed to equal the quote currency, NOTES D-006):
* Fill prices are the SIDE prices a Strategy Tester list shows: a long opens at the ask (open + spread + slippage) and closes at the bid (minus slippage); a short opens
  at the bid (minus slippage) and closes at the ask (+ slippage). ``price_pnl`` = signed exit - entry of these fills, so it already contains spread and slippage.
* ``profit = price_pnl x lots x contract_size + commission + swap`` with commission and swap NEGATIVE when they cost money. ``r_net = profit / (lots x contract_size x ATR_signal)``
  equals the engine's ``net_R`` when ``cost_multiplier`` is 1 (tested).
* Balance changes when a trade is realised: at the exit open for an exit at an open, one bar later (the next open) for an intrabar exit. Equity = balance + floating profit
  at each bar OPEN (long marked at the bid, short at the ask); accrued commission and swap are booked at the close only (a small known simplification).
* ``risk_percent`` sizing: lots = risk% x balance / (stop distance x contract_size), floored to ``lot_step`` and clipped to [lot_min, lot_max]; the stop distance is ``sl x ATR``
  (B / I exits) or ``cat_stop_atr x ATR``. Balance for sizing = initial deposit, or the realised balance at the entry open when ``compounding`` is true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from edge_research.labels.builder import _Market

MONEY_COLUMNS = [
    "rule_id", "signal_time", "entry_time", "exit_time", "entry_bar", "exit_bar", "direction", "hold", "entry_price", "exit_price", "spread_entry", "spread_exit",
    "price_pnl", "gross_profit", "commission", "swap", "nights", "profit", "exit_reason", "atr_signal", "r_net", "mae", "mfe", "lots",
]


@dataclass
class AccountResult:
    trades: pd.DataFrame
    balance: pd.Series
    equity: pd.Series
    stats: Dict[str, Any]


def round_lots(raw: np.ndarray | float, step: float, lo: float, hi: float) -> np.ndarray:
    """Floor to the lot step, then clip to [lo, hi]. ``1e-9`` guards float noise (0.3 / 0.01 = 29.999999999999996)."""
    lots = np.floor(np.asarray(raw, dtype=np.float64) / step + 1e-9) * step
    return np.round(np.clip(lots, lo, hi), 8)


def account_trades(trades: pd.DataFrame, market: _Market, settings: Dict[str, Any], index: pd.DatetimeIndex, rule_ids: Optional[List[str]] = None) -> AccountResult:
    """``trades``: engine trades of one or several rules, concatenated, with a ``rule_id`` column when several (sorted here by entry bar, rule order)."""
    m = market
    n = m.n
    cs = m.cost.contract_size
    t = trades.copy()
    if "rule_id" not in t.columns:
        t["rule_id"] = "rule"
    t["_order"] = t["rule_id"].map({r: k for k, r in enumerate(rule_ids or sorted(t["rule_id"].unique()))}) if len(t) else 0
    t = t.sort_values(["entry_bar", "_order"], kind="stable").reset_index(drop=True) if len(t) else t
    init = float(settings["initial_deposit"])
    stats: Dict[str, Any] = {"n_lot_clipped_min": 0, "n_lot_clipped_max": 0}
    if t.empty:
        bal = pd.Series(init, index=index, name="balance")
        return AccountResult(pd.DataFrame(columns=MONEY_COLUMNS), bal, bal.rename("equity"), stats)

    e, x = t["entry_bar"].to_numpy(), t["exit_bar"].to_numpy()
    is_long = (t["direction"] == "long").to_numpy()
    ib = t["exit_intrabar"].to_numpy(dtype=bool)
    atr = t["atr"].to_numpy()
    slip = m.slip
    sp_e, sp_x = m.spread[e], m.spread[x]
    bid_exit = t["exit_price"].to_numpy()                                                                   # engine: bid-equivalent exit price
    entry_fill = np.where(is_long, m.open[e] + sp_e + slip, m.open[e] - slip)
    exit_fill = np.where(is_long, bid_exit - slip, bid_exit + sp_x + slip)
    price_pnl = np.where(is_long, exit_fill - entry_fill, entry_fill - exit_fill)
    nights = (m.roll[x] - m.roll[e]).astype(np.int64)
    comm_per_lot = m.cost.commission_price * cs                                                            # money per lot, round trip
    swap_per_lot = np.array([m.cost.swap_cost_price(nights[k:k + 1], bool(is_long[k]))[0] * cs for k in range(len(t))])
    profit_per_lot = price_pnl * cs - comm_per_lot - swap_per_lot

    # ---- position size
    mode = settings["sizing_mode"]
    lots = np.full(len(t), float(settings["fixed_lots"]))
    realize = np.where(ib, x + 1, x)                                                                       # bar at whose open the balance changes
    if mode == "risk_percent":
        dist = t["stop_distance"].to_numpy()
        if not np.isfinite(dist).all():
            raise ValueError("risk_percent sizing needs a stop (a B / I exit or cat_stop_atr) for every trade")
        risk = float(settings["risk_percent"]) / 100.0
        step, lo, hi = float(settings["lot_step"]), float(settings["lot_min"]), float(settings["lot_max"])
        pending: List[tuple] = []
        bal = init
        for k in range(len(t)):
            if settings["compounding"]:
                done = [p for p in pending if p[0] <= e[k]]
                bal += sum(p[1] for p in done)
                pending = [p for p in pending if p[0] > e[k]]
            raw = risk * bal / (dist[k] * cs)
            lot = float(round_lots(raw, step, 0.0, hi))
            if lot < lo:
                stats["n_lot_clipped_min"] += 1
                lot = lo
            if raw / step > hi / step + 1e-9:
                stats["n_lot_clipped_max"] += 1
            lots[k] = lot
            pending.append((realize[k], lot * profit_per_lot[k]))
    profit = lots * profit_per_lot
    out = pd.DataFrame({
        "rule_id": t["rule_id"].to_numpy(), "signal_time": t["signal_time"].to_numpy(), "entry_time": t["entry_time"].to_numpy(), "exit_time": t["exit_time"].to_numpy(),
        "entry_bar": e, "exit_bar": x, "direction": t["direction"].to_numpy(), "hold": x - e, "entry_price": entry_fill, "exit_price": exit_fill,
        "spread_entry": sp_e, "spread_exit": sp_x, "price_pnl": price_pnl, "gross_profit": price_pnl * lots * cs,
        "commission": -comm_per_lot * lots, "swap": -swap_per_lot * lots, "nights": nights, "profit": profit,
        "exit_reason": t["exit_reason"].to_numpy(), "atr_signal": atr, "r_net": profit / (lots * cs * atr), "mae": t["mae"].to_numpy(), "mfe": t["mfe"].to_numpy(), "lots": lots,
    })[MONEY_COLUMNS]
    out["_realize"] = realize
    out["_ib"] = ib

    # ---- balance and equity per bar open
    step_bal = np.bincount(np.minimum(realize, n - 1), weights=np.where(realize <= n - 1, profit, 0.0), minlength=n)
    balance = init + np.cumsum(step_bal)
    floating = np.zeros(n)
    for k in range(len(t)):
        last = x[k] if ib[k] else x[k] - 1                                                                  # bars whose OPEN still sees the position
        if last < e[k]:
            continue
        sl = slice(e[k], last + 1)
        if is_long[k]:
            fl = m.open[sl] - entry_fill[k]
        else:
            fl = entry_fill[k] - (m.open[sl] + m.spread[sl])
        floating[sl] += np.nan_to_num(fl) * lots[k] * cs
    equity = balance + floating
    stats["gross_move_money"] = float((t["gross_price"].to_numpy() * lots * cs).sum())                          # price move open -> exit before any cost
    stats["cost_money"] = stats["gross_move_money"] - float(profit.sum())
    return AccountResult(out, pd.Series(balance, index=index, name="balance"), pd.Series(equity, index=index, name="equity"), stats)
