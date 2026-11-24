"""
M0: synthetic MT5-style bar data with an optional PLANTED edge.

Purpose: test every later stage on data where the truth is known.
  * ``plant_edge=None``  -> null data (no exploitable structure by construction).
  * ``plant_edge=PlantedEdge(...)`` -> after a trigger (session hours AND a short-term drop measured in units of the
    exogenous volatility ``sigma``) a small positive drift is added over the next ``bars`` bars (mean reversion).

Design decision D-007: the trigger uses ``sigma`` (an exogenous volatility process generated BEFORE prices), never the
exported indicators, so planting does not create circularity with indicator values.

Output mirrors the real flow: a bar table with exporter-style column names (float32, datetime index named ``time``)
and a separate per-bar spread series in points (as ``spreadExporter.mq5`` would write it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PlantedEdge:
    """Known edge: long drift after a short-term drop during selected hours."""

    hours: Tuple[int, ...] = (7, 8, 9, 10, 11)
    drop_sigmas: float = 1.5      # trigger when close[i] - close[i-3] < -drop_sigmas * sqrt(3) * sigma[i]
    drift_sigmas: float = 0.8     # total extra move over `bars` bars, in units of sigma[i]
    bars: int = 6
    direction: int = 1            # +1 long edge, -1 short edge


def _session_multiplier(hour: np.ndarray) -> np.ndarray:
    """Volatility by server hour: quiet Asia, active London/NY overlap."""
    mult = np.full(hour.shape, 0.7)
    mult[(hour >= 7) & (hour <= 16)] = 1.2
    mult[(hour >= 13) & (hour <= 16)] = 1.5
    return mult


def _build_index(n_bars: int, timeframe_minutes: int, start: str) -> pd.DatetimeIndex:
    per_day = max(1, 24 * 60 // timeframe_minutes)
    n_days = int(n_bars / (per_day * 5 / 7)) + 10
    idx = pd.date_range(start, periods=n_days * per_day, freq=f"{timeframe_minutes}min")
    idx = idx[idx.weekday < 5]
    return idx[:n_bars].rename("time")


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add exporter-style indicator columns computed from OHLCV (approximations of the MT5 formulas)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out = {}
    out["atr_14"] = _sma(tr, 14)
    out["atr_20"] = _sma(tr, 20)
    delta = c.diff()
    up, dn = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = _wilder(up, 14) / _wilder(dn, 14).replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)
    tp = (h + l + c) / 3
    mad = tp.rolling(14).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True)
    out["cci_14"] = (tp - _sma(tp, 14)) / (0.015 * mad)
    out["mom_10"] = c / c.shift(10) * 100.0          # MT5 iMomentum scale: centred near 100
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k_fast = 100 * (c - ll) / (hh - ll)
    out["stoch_14_3_3_k"] = k_fast.rolling(3).mean().rolling(3).mean()
    out["wpr_14"] = -100 * (hh - c) / (hh - ll)
    for n in (5, 10, 20, 50, 100, 200):
        out[f"ma_{n}_sma"] = _sma(c, n)
        out[f"ma_{n}_ema"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    mid, sd = _sma(c, 20), c.rolling(20).std(ddof=0)
    out["bb_20_2_upper"], out["bb_20_2_middle"], out["bb_20_2_lower"] = mid + 2 * sd, mid, mid - 2 * sd
    out["stddev_20"] = sd
    ema12, ema26 = c.ewm(span=12, adjust=False, min_periods=12).mean(), c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    out["macd_12_26_9_main"] = macd
    out["macd_12_26_9_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    plus_dm = (h.diff()).clip(lower=0).where(h.diff() > (-l.diff()).clip(lower=0), 0.0)
    minus_dm = (-l.diff()).clip(lower=0).where((-l.diff()) > h.diff().clip(lower=0), 0.0)
    atr_w = _wilder(tr, 14)
    pdi, mdi = 100 * _wilder(plus_dm, 14) / atr_w, 100 * _wilder(minus_dm, 14) / atr_w
    out["adx_14"] = _wilder(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan), 14)
    res = df.copy()
    for k, v in out.items():
        res[k] = v
    return res


def make_synthetic_bars(
    n_bars: int = 20000,
    seed: int = 12345,
    timeframe_minutes: int = 60,
    start: str = "2015-01-05 00:00",
    price0: float = 1.10,
    sigma_base: float = 0.0006,
    plant_edge: Optional[PlantedEdge] = None,
    point: float = 0.00001,
    base_spread_points: float = 12.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(bars_df, meta)``; ``meta`` has the spread series, sigma, and the planted trigger mask."""
    rng = np.random.default_rng(seed)
    idx = _build_index(n_bars, timeframe_minutes, start)
    n = len(idx)
    hour = idx.hour.to_numpy()

    # Exogenous clustered volatility: AR(1) log-vol times a session multiplier.
    logvol = np.zeros(n)
    shocks = rng.normal(0.0, 0.12, n)
    for t in range(1, n):
        logvol[t] = 0.985 * logvol[t - 1] + shocks[t]
    sigma = sigma_base * np.sqrt(timeframe_minutes / 60.0) * _session_multiplier(hour) * np.exp(logvol - logvol.std() ** 2 / 2)
    innov = sigma * rng.standard_normal(n)
    # Weekend gaps: first bar after a gap gets a larger shock.
    gap_after = np.zeros(n, dtype=bool)
    gap_after[1:] = (idx[1:] - idx[:-1]) > pd.Timedelta(hours=6)
    innov[gap_after] += 3.0 * sigma[gap_after] * rng.standard_normal(int(gap_after.sum()))

    close = np.empty(n)
    drift_add = np.zeros(n + 64)
    trigger = np.zeros(n, dtype=bool)
    active_until = -1
    close[0] = price0
    for t in range(n):
        prev = close[t - 1] if t > 0 else price0
        close[t] = prev + innov[t] + drift_add[t]
        if plant_edge is not None and t >= 3 and t > active_until and hour[t] in plant_edge.hours:
            move = close[t] - close[t - 3]
            if plant_edge.direction * move < -plant_edge.drop_sigmas * np.sqrt(3.0) * sigma[t]:
                trigger[t] = True
                per_bar = plant_edge.direction * plant_edge.drift_sigmas * sigma[t] / plant_edge.bars
                # entry bar is t+1; drift acts on bars t+1 .. t+bars (close-to-close moves of those bars)
                drift_add[t + 1 : t + 1 + plant_edge.bars] += per_bar
                active_until = t + plant_edge.bars

    open_ = np.empty(n)
    open_[0] = price0
    open_[1:] = close[:-1]
    wick_up = np.abs(rng.normal(0.0, 0.4, n)) * sigma
    wick_dn = np.abs(rng.normal(0.0, 0.4, n)) * sigma
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    tick_volume = np.round(500 + 4e5 * sigma * np.abs(rng.standard_normal(n)) + rng.poisson(50, n)).astype(np.float64)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": tick_volume}, index=idx
    )
    df = add_indicators(df)
    for col in df.columns:
        df[col] = df[col].astype(np.float32)

    spread_pts = base_spread_points * (1.0 + 0.5 * np.exp(logvol - logvol.mean()) - 0.5)
    spread_pts = np.where((hour == 23) | (hour == 0), spread_pts * 2.0, spread_pts)
    spread_pts = np.maximum(1.0, np.round(spread_pts + rng.normal(0, 1.0, n)))
    meta = {
        "spread_points": pd.Series(spread_pts.astype(np.float32), index=idx, name="spread"),
        "sigma": pd.Series(sigma, index=idx, name="sigma"),
        "trigger": pd.Series(trigger, index=idx, name="planted_trigger"),
        "planted": plant_edge,
        "seed": seed,
        "point": point,
    }
    return df, meta


def write_exporter_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the table the way dataExporter.mq5 does ('YYYYMMDD HH:MM' time, comma separated)."""
    out = df.copy()
    out.insert(0, "time", out.index.strftime("%Y%m%d %H:%M"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6f")
    return path
