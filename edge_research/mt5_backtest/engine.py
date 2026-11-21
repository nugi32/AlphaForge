"""
Backtest engine (M2b core, M9 extensions). Event rules, identical to the label table where the label families apply:

* The decision is taken at the CLOSE of signal bar i; the order fills at the OPEN of bar e = i + 1 (open prices only: Bid = open, Ask = open + spread).
* Exit kinds (design 15.4)
    ``T{N}``               time stop: open of bar e + N.
    ``B{tp}_{sl}_{H}``     bar-boundary SL/TP: decision on the CLOSED bar, fill at the next open (``labels.builder.barrier_exit_bars``: the same function, not a copy).
    ``I{tp}_{sl}_{H}``     intrabar (research): the high / low of a bar touches a level. SL is assumed first when one bar touches both. A bar whose OPEN is already beyond
                           a level (gap, bars after the entry bar) fills at the open. The fill price is the level (or the open); the exit bar is the touching bar.
  Optional layers on top of any kind: ``cat_stop_atr`` (wide intrabar SL, pessimistic, fills at the level or at the open on a gap) and ``exit_signals``
  (condition exit: a boolean on the closed bar b, exit at the open of b + 1).
* ``max_open_positions`` (default 1): signals are ignored while the limit is reached; with ``reenter_on_exit_bar=False`` no entry on a bar whose OPEN closed a position.
  An intrabar exit closes DURING its bar, so the next entry is possible at the open of the following bar either way.
* A signal whose window is invalid (leaves the data, crosses a gap, non-finite price / spread / ATR) is skipped and does NOT block the next signal (counted in ``stats``).
  ``close_open_at_end=True`` closes a position that would run past the data at the last bar open (reason ``end``) instead of skipping the signal.
* Costs come from ``labels.builder._Market``; the engine never computes costs itself. ``cost_multiplier`` multiplies spread and slippage (design 15.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from edge_research.labels.builder import (DIRECTIONS, _Market, apply_cat_layer, barrier_exit_bars,
                                          entry_and_levels as _entry_and_levels, intrabar_exit as _intrabar_exit)

_T_RE = re.compile(r"^T(\d+)$")
_B_RE = re.compile(r"^([BI])([0-9p]+)_([0-9p]+)_(\d+)$")

TRADE_COLUMNS = [
    "signal_bar", "entry_bar", "exit_bar", "signal_time", "entry_time", "exit_time", "direction",
    "entry_price", "exit_price", "atr", "gross_price", "cost_price", "gross_R", "cost_R", "net_R", "hold",
    "exit_reason", "exit_intrabar", "mae", "mfe", "stop_distance",
]

ENGINE_DEFAULTS: Dict[str, Any] = {
    "fidelity": "open_prices_only", "reenter_on_exit_bar": False, "conflict_policy": "skip", "max_open_positions": 1,
    "close_open_at_end": False, "cost_multiplier": 1.0, "no_weekend_hold": True, "cat_stop_atr": None, "account_mode": "hedging",
    "sizing_mode": "fixed_lots", "fixed_lots": 0.1, "risk_percent": 1.0, "lot_step": 0.01, "lot_min": 0.01, "lot_max": 100.0,
    "initial_deposit": 10000.0, "compounding": False,
}


def engine_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Engine settings = defaults + ``cfg["engine"]``. An unknown key is an ERROR (it catches typos, design 15.2)."""
    eng = dict(cfg.get("engine", {}) or {})
    unknown = sorted(set(eng) - set(ENGINE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown engine settings: {unknown}")
    out = {**ENGINE_DEFAULTS, **eng}
    if out["conflict_policy"] != "skip":
        raise ValueError(f"unsupported conflict_policy: {out['conflict_policy']!r}")
    if out["fidelity"] != "open_prices_only":
        raise ValueError(f"engine.fidelity must be 'open_prices_only' (the label/engine parity contract is built on it), got {out['fidelity']!r}")
    if out["account_mode"] not in ("hedging", "netting"):
        raise ValueError(f"account_mode must be hedging or netting, got {out['account_mode']!r}")
    if out["sizing_mode"] not in ("fixed_lots", "risk_percent"):
        raise ValueError(f"unknown sizing_mode {out['sizing_mode']!r}")
    if int(out["max_open_positions"]) < 1:
        raise ValueError("max_open_positions must be >= 1")
    if float(out["cost_multiplier"]) < 0:
        raise ValueError("cost_multiplier must be >= 0")
    return out


def parse_family(name: str) -> Dict[str, Any]:
    """Inverse of the family names: ``T6``, ``B1_1p5_12`` (bar boundary) or ``I1_1p5_12`` (intrabar research) -> parameter dict."""
    m = _T_RE.match(name)
    if m:
        return {"kind": "T", "n": int(m.group(1))}
    m = _B_RE.match(name)
    if m:
        return {"kind": m.group(1), "tp": float(m.group(2).replace("p", ".")), "sl": float(m.group(3).replace("p", ".")), "h": int(m.group(4))}
    raise ValueError(f"unknown exit family: {name!r}")


@dataclass
class EngineResult:
    trades: pd.DataFrame
    stats: Dict[str, int]
    market: Optional[_Market] = field(default=None, repr=False)

    def summary(self) -> Dict[str, float]:
        t = self.trades
        if t.empty:
            return {"n_trades": 0, "mean_net_R": float("nan"), "sum_net_R": 0.0, "win_rate": float("nan"), "profit_factor": float("nan")}
        r = t["net_R"].to_numpy()
        gains, losses = r[r > 0].sum(), -r[r < 0].sum()
        return {"n_trades": int(len(r)), "mean_net_R": float(r.mean()), "sum_net_R": float(r.sum()),
                "win_rate": float((r > 0).mean()), "profit_factor": float(gains / losses) if losses > 0 else float("inf")}


def _as_bool_array(sig: Any, n: int, name: str) -> np.ndarray:
    """Boolean signal of length n. 0/1 numerics are accepted; NaN counts as False; anything else is an error."""
    arr = np.asarray(sig)
    if arr.shape != (n,):
        raise ValueError(f"signal {name!r} must have shape ({n},), got {arr.shape}")
    if arr.dtype == bool:
        return arr
    vals = np.nan_to_num(arr.astype(np.float64))
    if not np.isin(vals, (0.0, 1.0)).all():
        raise ValueError(f"signal {name!r} must be boolean or 0/1")
    return vals > 0


# ---------------------------------------------------------------------------------------------------------------- exit logic
# _entry_and_levels / _touch / _to_bid / intrabar exits / the catastrophic layer live in labels.builder (ONE implementation
# shared with the label table) and are imported above.


def compute_exits(m: _Market, i: np.ndarray, is_long: bool, p: Dict[str, Any], cat_k: Optional[float], cond: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Exit bar, bid-equivalent exit price, reason and ``intrabar`` flag for every signal bar in ``i`` (all with i + 1 + hmax <= n - 1)."""
    e = i + 1
    n_sig = len(i)
    kind = p["kind"]
    reason = np.full(n_sig, "time", dtype=object)
    intrabar = np.zeros(n_sig, dtype=bool)
    stop_dist = np.full(n_sig, np.nan)
    if kind == "T":
        x = e + p["n"]
        px = m.open[x].copy()
        hmax = p["n"]
    else:
        hmax = p["h"]
        entry, tp_lvl, sl_lvl = _entry_and_levels(m, i, is_long, p["tp"], p["sl"])
        stop_dist = p["sl"] * m.atr[i]
        if kind == "B":
            x = barrier_exit_bars(m, i, p["tp"], p["sl"], p["h"], is_long)
            px = m.open[x].copy()
            cb = m.close[x - 1]                                                   # the bar whose close triggered the exit (or the last bar of the horizon)
            tp_hit = (cb >= tp_lvl) if is_long else (cb <= tp_lvl)
            sl_hit = (cb <= sl_lvl) if is_long else (cb >= sl_lvl)
            reason = np.where(sl_hit, "sl", np.where(tp_hit, "tp", "time")).astype(object)
        else:                                                                       # intrabar research mode: the SHARED builder implementation
            res_i = _intrabar_exit(m, i, is_long, p["tp"], p["sl"], p["h"])
            x, px, reason, intrabar = res_i["x"], res_i["px"], res_i["reason"], res_i["intrabar"]
    # condition exit: decision on the closed bar b (b >= e), fill at the open of b + 1; it must come before the base exit
    if cond is not None:
        found = np.zeros(n_sig, dtype=bool)
        xb = x.copy()
        for k in range(hmax):
            b = e + k
            xc = b + 1
            before = (xc < xb) | (intrabar & (xc <= xb))                          # an open-based exit wins a tie, an intrabar touch happens after the open
            hit = cond[b] & before & ~found
            if hit.any():
                x = np.where(hit, xc, x)
                px = np.where(hit, m.open[np.minimum(xc, m.n - 1)], px)
                reason = np.where(hit, "cond", reason).astype(object)
                intrabar = np.where(hit, False, intrabar)
                found |= hit
    # catastrophic layer: intrabar, strictly before the exit decided so far (the exit open pre-empts the bar it happens in)
    if cat_k is not None:
        stop_dist = np.where(np.isfinite(stop_dist), stop_dist, float(cat_k) * m.atr[i])
        x, px, reason, intrabar = apply_cat_layer(m, i, is_long, cat_k, hmax, x, px, reason, intrabar)
    return {"x": x, "px": px, "reason": reason, "intrabar": intrabar, "stop_distance": stop_dist}


def _mae_mfe(m: _Market, e: np.ndarray, x: np.ndarray, intrabar: np.ndarray, is_long: np.ndarray, entry_fill: np.ndarray, atr: np.ndarray):
    """Adverse / favourable excursion in ATR units from the bar highs and lows between entry and exit (the exit bar counts when the exit happened inside it)."""
    n = len(e)
    mae, mfe = np.full(n, np.nan), np.full(n, np.nan)
    if m.high is None or m.low is None:
        return mae, mfe
    for k in range(n):
        last = x[k] if intrabar[k] else x[k] - 1
        if last < e[k]:
            continue
        sl = slice(e[k], last + 1)
        if is_long[k]:
            hi, lo = m.high[sl].max(), m.low[sl].min()
            mfe[k], mae[k] = hi - entry_fill[k], entry_fill[k] - lo
        else:
            hi, lo = (m.high[sl] + m.spread[sl]).max(), (m.low[sl] + m.spread[sl]).min()
            mfe[k], mae[k] = entry_fill[k] - lo, hi - entry_fill[k]
    return np.maximum(mae, 0.0) / atr, np.maximum(mfe, 0.0) / atr


# ---------------------------------------------------------------------------------------------------------------- engine
def run_engine(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    signals: Mapping[str, Any],
    family: str,
    spread_points: Optional[pd.Series] = None,
    exit_signals: Optional[Mapping[str, Any]] = None,
    cat_stop_atr: Optional[float] = None,
) -> EngineResult:
    """Run one rule. ``signals`` maps ``"long"`` / ``"short"`` to a boolean array aligned with ``df`` (decision at close of bar i).
    ``exit_signals`` (optional, same shape) are condition exits per direction; ``cat_stop_atr`` overrides ``engine.cat_stop_atr`` (and ``labels.cat_stop_atr`` is ignored here)."""
    unknown = set(signals) - set(DIRECTIONS)
    if unknown:
        raise ValueError(f"unknown signal directions: {sorted(unknown)}")
    p = parse_family(family)
    eng = engine_settings(cfg)
    reenter = bool(eng["reenter_on_exit_bar"])
    max_open = int(eng["max_open_positions"])
    cat_k = cat_stop_atr if cat_stop_atr is not None else eng["cat_stop_atr"]
    if cat_k is not None and float(cat_k) <= 0:
        raise ValueError("cat_stop_atr must be positive")
    if exit_signals is not None:
        bad = set(exit_signals) - set(DIRECTIONS)
        if bad:
            raise ValueError(f"unknown exit_signals directions: {sorted(bad)}")

    m = _Market(df, cfg, spread_points, cost_mult=float(eng["cost_multiplier"]))
    n = m.n
    hmax = p["n"] if p["kind"] == "T" else p["h"]
    last_signal = n - 2 - hmax                                  # need e + hmax = i + 1 + hmax <= n - 1
    sig = {d: _as_bool_array(signals[d], n, d) if d in signals else np.zeros(n, dtype=bool) for d in DIRECTIONS}
    cond = {d: (_as_bool_array(exit_signals[d], n, f"exit_{d}") if exit_signals and d in exit_signals else None) for d in DIRECTIONS}

    stats = {"signals_long": int(sig["long"].sum()), "signals_short": int(sig["short"].sum()),
             "conflict_skipped": 0, "out_of_data": 0, "invalid_window": 0, "overlap_skipped": 0, "trades": 0,
             "n_open_at_end": 0, "closed_at_end": 0, "n_spread_fallback": int(m.spread_fallback)}

    both = sig["long"] & sig["short"]
    stats["conflict_skipped"] = int(both.sum())
    pieces = []
    for d in DIRECTIONS:
        s = sig[d] & ~both
        idx = np.flatnonzero(s)
        out_of_data = idx[idx > last_signal]
        i = idx[idx <= last_signal]
        is_long = d == "long"
        frames = []
        if len(i):
            ex = compute_exits(m, i, is_long, p, cat_k, cond[d])
            x = ex["x"]
            ok = m.window_ok(i, x) if bool(eng["no_weekend_hold"]) else (np.isfinite(m.open[i + 1]) & np.isfinite(m.open[x]) & np.isfinite(m.atr[i])
                                                                          & np.isfinite(m.spread[i + 1]) & np.isfinite(m.spread[x]))
            stats["invalid_window"] += int((~ok).sum())
            i_ok = i[ok]
            sel = {k: v[ok] for k, v in ex.items()}
            g, c = m.result_at(i_ok, sel["x"], sel["px"], is_long)
            frames.append(pd.DataFrame({"signal_bar": i_ok, "exit_bar": sel["x"], "direction": d, "gross_R": g, "cost_R": c, "exit_px": sel["px"],
                                        "exit_reason": sel["reason"], "exit_intrabar": sel["intrabar"], "stop_distance": sel["stop_distance"]}))
        if len(out_of_data):
            if bool(eng["close_open_at_end"]):
                frames.append(_closed_at_end(m, out_of_data, is_long, d, eng, stats))
            else:
                stats["out_of_data"] += len(out_of_data)
                stats["n_open_at_end"] += len(out_of_data)
        pieces.extend(f for f in frames if len(f))

    if not pieces:
        return EngineResult(pd.DataFrame(columns=TRADE_COLUMNS), stats, m)

    cand = pd.concat(pieces, ignore_index=True).sort_values("signal_bar", kind="stable").reset_index(drop=True)
    sb, xb, ib = cand["signal_bar"].to_numpy(), cand["exit_bar"].to_numpy(), cand["exit_intrabar"].to_numpy()
    keep = np.zeros(len(cand), dtype=bool)
    free_from: List[int] = []                                   # per open position: first entry bar that finds it closed
    open_exit_bar: List[int] = []                               # per open position: exit bar if it exits AT AN OPEN, else -1
    for k in range(len(cand)):
        e = sb[k] + 1
        if len(free_from):
            alive = [j for j in range(len(free_from)) if free_from[j] > e or (not reenter and open_exit_bar[j] == e)]
            free_from, open_exit_bar = [free_from[j] for j in alive], [open_exit_bar[j] for j in alive]
            # positions that are closed by now are dropped; the second condition keeps "no entry on the tick that closed a trade" for several positions
        blocked_by_tick = (not reenter) and any(ob == e for ob in open_exit_bar)
        occupied = sum(1 for f in free_from if f > e)
        if occupied >= max_open or blocked_by_tick:
            stats["overlap_skipped"] += 1
            continue
        keep[k] = True
        free_from.append(int(xb[k] + 1 if ib[k] else (xb[k] if reenter else xb[k] + 1)))
        open_exit_bar.append(-1 if ib[k] else int(xb[k]))
    t = cand[keep].reset_index(drop=True)
    stats["trades"] = int(len(t))
    return EngineResult(_finish(df, m, t), stats, m)


def _closed_at_end(m: _Market, signal_bars: np.ndarray, is_long: bool, d: str, eng: Dict[str, Any], stats: Dict[str, int]) -> pd.DataFrame:
    """``close_open_at_end``: a signal whose window runs past the data is closed at the open of the LAST bar (reason ``end``)."""
    i = signal_bars[signal_bars + 1 <= m.n - 2]                # needs an entry bar before the last bar
    x = np.full(len(i), m.n - 1)
    ok = m.window_ok(i, x) if bool(eng["no_weekend_hold"]) else np.ones(len(i), dtype=bool)
    i, x = i[ok], x[ok]
    stats["closed_at_end"] += len(i)
    stats["out_of_data"] += int(len(signal_bars) - len(i))
    if not len(i):
        return pd.DataFrame()
    g, c = m.result(i, x, is_long)
    return pd.DataFrame({"signal_bar": i, "exit_bar": x, "direction": d, "gross_R": g, "cost_R": c, "exit_px": m.open[x], "exit_reason": "end",
                         "exit_intrabar": False, "stop_distance": np.nan})


def _finish(df: pd.DataFrame, m: _Market, t: pd.DataFrame) -> pd.DataFrame:
    i = t["signal_bar"].to_numpy()
    x = t["exit_bar"].to_numpy()
    e = i + 1
    atr = m.atr[i]
    is_long = (t["direction"] == "long").to_numpy()
    sign = np.where(is_long, 1.0, -1.0)
    ib = t["exit_intrabar"].to_numpy(dtype=bool)
    entry_fill = np.where(is_long, m.open[e] + m.spread[e] + m.slip, m.open[e] - m.slip)
    mae, mfe = _mae_mfe(m, e, x, ib, is_long, entry_fill, atr)
    out = pd.DataFrame({
        "signal_bar": i, "entry_bar": e, "exit_bar": x,
        "signal_time": df.index[i], "entry_time": df.index[e], "exit_time": df.index[x],
        "direction": t["direction"].to_numpy(),
        "entry_price": m.open[e], "exit_price": t["exit_px"].to_numpy(), "atr": atr,
        "gross_price": t["gross_R"].to_numpy() * atr, "cost_price": t["cost_R"].to_numpy() * atr,
        "gross_R": t["gross_R"].to_numpy(), "cost_R": t["cost_R"].to_numpy(),
        "net_R": t["gross_R"].to_numpy() - t["cost_R"].to_numpy(), "hold": x - e,
        "exit_reason": t["exit_reason"].to_numpy(), "exit_intrabar": ib, "mae": mae, "mfe": mfe, "stop_distance": t["stop_distance"].to_numpy(),
    })
    # Sanity identity: gross price is the signed open-to-exit move (bid-equivalent prices).
    assert np.allclose(out["gross_price"].to_numpy(), sign * (out["exit_price"].to_numpy() - out["entry_price"].to_numpy()), atol=1e-9)
    return out[TRADE_COLUMNS]
