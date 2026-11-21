"""Bar-by-bar model of the generated EA's ``OnTick`` (render.py), written independently of the vectorised engine.

There is no MetaTrader in the sandbox, so the EA's ALGORITHM (not its MQL5 syntax) is checked here: this model follows the EA step by step, with scalar state
and the quantities the EA can see at the first tick of a bar (bars held from the position open time, the ATR of the signal bar at shift held + 1, the close of the
last closed bar). Its trade list is compared with ``run_backtest`` (the engine plus the accounting) in ``tests/test_codegen_m10.py``.

Parity mode (design 16.6): the EA cannot know whether a position will cross a weekend gap, so for the trade parity run ``InpMaxGapHours`` = 0 in the tester and
the engine runs with ``engine.no_weekend_hold = false`` (``parity_cfg``). With the gap check on, the EA and the engine differ by construction (a gap-crossing signal
is skipped by the engine, the EA holds the position and blocks the following signals); the EA log has a ``gap_crossed`` column for that analysis.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from edge_research.codegen.plan import EAPlan
from edge_research.codegen.pyeval import plan_signal
from edge_research.labels.builder import _Market


def parity_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Engine configuration of the trade parity run: no weekend exclusion, no slippage (the tester has none), no cost multiplier."""
    c = copy.deepcopy(cfg)
    c.setdefault("engine", {})
    c["engine"]["no_weekend_hold"] = False
    c["engine"]["cost_multiplier"] = 1.0
    c["costs"]["slippage_points"] = 0
    return c


def simulate_ea(df: pd.DataFrame, cfg: Dict[str, Any], plan: EAPlan, spread_points: Optional[pd.Series] = None, max_gap_hours: float = 0.0,
                signal: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Trades of the EA in the Strategy Tester (Open prices only, tester spread = the research spread). Columns: entry_time, exit_time, direction, entry_price,
    exit_price (side fills), signal_bar, entry_bar, exit_bar, exit_reason, gap_crossed. Positions still open at the last bar are not listed (the engine skips them too)."""
    m = _Market(df, cfg, spread_points, cost_mult=1.0)
    n = m.n
    sig = plan_signal(plan, df, cfg) if signal is None else np.asarray(signal, dtype=bool)
    is_long = plan.settings["direction"] == "long"
    ex = plan.exit
    reenter = plan.settings["reenter_on_exit_bar"]
    times = df.index.to_numpy()
    gap_s = float(max_gap_hours) * 3600.0
    dt = np.diff(times).astype("timedelta64[s]").astype(np.float64)       # dt[k] = time[k + 1] - time[k]
    gap_flag = np.concatenate([[False], dt > gap_s]) if gap_s > 0 else np.zeros(n, dtype=bool)     # gap_flag[k]: gap between bar k - 1 and bar k
    rows: List[Dict[str, Any]] = []
    pos: Optional[Dict[str, Any]] = None
    for b in range(1, n):                                   # b = the bar whose first tick is processed; the decision bar (shift 1) is b - 1
        closed_now = False
        if pos is not None:
            held = b - pos["e"]                              # iBarShift(position open time)
            due = False
            if ex["kind"] == "T":
                due = held >= ex["n"]
            else:                                            # bar boundary
                if held >= 1:
                    if held >= ex["h"]:
                        due = True
                    else:
                        atr0 = m.atr[pos["e"] - 1]           # ATR at shift held + 1 = the signal bar
                        cl = m.close[b - 1]                  # close at shift 1
                        if np.isfinite(atr0):
                            lvl_tp = pos["fill"] + (ex["tp"] if is_long else -ex["tp"]) * atr0
                            lvl_sl = pos["fill"] - (ex["sl"] if is_long else -ex["sl"]) * atr0
                            due = (cl >= lvl_tp or cl <= lvl_sl) if is_long else (cl <= lvl_tp or cl >= lvl_sl)
            if due:
                px = m.open[b] if is_long else m.open[b] + m.spread[b]          # long closes at the bid, short at the ask
                rows.append({"signal_bar": pos["e"] - 1, "entry_bar": pos["e"], "exit_bar": b, "entry_time": times[pos["e"]], "exit_time": times[b],
                             "direction": plan.settings["direction"], "entry_price": pos["fill"], "exit_price": px,
                             "exit_reason": "time" if ex["kind"] == "T" else "barrier_or_time", "gap_crossed": bool(gap_flag[pos["e"]:b + 1].any()) if gap_s > 0 else False})
                pos, closed_now = None, True
        if pos is not None:
            continue
        if closed_now and not reenter:
            continue
        i = b - 1
        if not sig[i]:
            continue
        if gap_s > 0 and gap_flag[b]:
            continue
        if not np.isfinite(m.atr[i]) or not np.isfinite(m.open[b]) or not np.isfinite(m.spread[b]):
            continue
        pos = {"e": b, "fill": m.open[b] + m.spread[b] if is_long else m.open[b]}
    out = pd.DataFrame(rows, columns=["signal_bar", "entry_bar", "exit_bar", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "exit_reason", "gap_crossed"])
    return out
