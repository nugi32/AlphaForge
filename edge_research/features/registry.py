"""
Stage 1: feature registry.

Each ``FeatureSpec`` documents one stationary feature (formula + MT5 recipe) and carries the function that computes it
from the loader table. Only closed-bar information is used: the value at row i depends on rows <= i.

``build_registry(columns, cfg)`` returns the specs whose source columns exist; the rest are reported as skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BOUNDED_OSC = ["rsi_7", "rsi_14", "rsi_21", "stoch_5_3_3_k", "stoch_5_3_3_d", "stoch_14_3_3_k", "stoch_14_3_3_d",
               "wpr_14", "adx_14", "adxwilder_14", "demarker_14", "mfi_14", "rvi_10"]
UNBOUNDED_OSC = ["cci_14", "cci_20", "trix_14"]
MA_PREFIX = re.compile(r"^(ma|dema|tema|frama|ama|vidya)_\d+_?[a-z0-9]*$")
SESSIONS = {"asia": (0, 6), "london": (7, 12), "newyork": (13, 21), "late": (22, 23)}   # server hours, inclusive


@dataclass
class Ctx:
    """Inputs shared by all feature functions."""

    df: pd.DataFrame
    atr: pd.Series                 # float64 ATR at each row (closed bar)
    cfg: Dict[str, Any]

    def col(self, name: str) -> pd.Series:
        return self.df[name].astype(np.float64)


@dataclass
class FeatureSpec:
    name: str
    kind: str
    sources: List[str]
    formula: str
    mt5_recipe: str
    fn: Callable[[Ctx], pd.Series]
    warmup_bars: int = 0
    reproducible_in_tester: bool = True
    family: str = ""
    notes: str = ""
    complexity: int = 1            # 1 = trivial in MQL5 ... 5 = needs custom loops/state

    def to_json(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "fn"}
        return d


def _bars_since(event: np.ndarray, cap: int) -> np.ndarray:
    """Bars since the last True in ``event`` (0 on the event bar), capped; before the first event -> cap."""
    n = len(event)
    pos = np.where(event, np.arange(n), -1)
    last = np.maximum.accumulate(pos)
    out = np.where(last >= 0, np.arange(n) - last, cap)
    return np.minimum(out, cap).astype(np.float64)


def _zscore(x: pd.Series, window: int) -> pd.Series:
    return (x - x.rolling(window).mean()) / x.rolling(window).std(ddof=0).replace(0, np.nan)


def build_registry(columns: Sequence[str], cfg: Dict[str, Any], exclude_sources: Optional[set] = None) -> Tuple[List[FeatureSpec], List[Tuple[str, str]]]:
    """Return (specs, skipped) where skipped is a list of (feature_name, reason)."""
    f = cfg["features"]
    have = set(columns)
    exclude_sources = exclude_sources or set()
    specs: List[FeatureSpec] = []
    skipped: List[Tuple[str, str]] = []
    rw, rv = f["roll_window_vol"], f["roll_window_volume"]

    def add(spec: FeatureSpec) -> None:
        missing = [s for s in spec.sources if s not in have]
        bad = [s for s in spec.sources if s in exclude_sources]
        if missing:
            skipped.append((spec.name, f"missing source columns {missing}"))
        elif bad:
            skipped.append((spec.name, f"source excluded by QA {bad}"))
        else:
            specs.append(spec)

    # --- bounded / unbounded oscillators: used as exported -------------------------------------------------------
    for c in BOUNDED_OSC:
        add(FeatureSpec(c, "osc_bounded", [c], f"{c} as exported", f"indicator buffer of {c}, shift 1",
                        lambda x, c=c: x.col(c), family=c.split("_")[0]))
    for c in UNBOUNDED_OSC:
        add(FeatureSpec(c, "osc_unbounded", [c], f"{c} as exported", f"indicator buffer of {c}, shift 1",
                        lambda x, c=c: x.col(c), family=c.split("_")[0]))

    # --- momentum: iMomentum is centred at mom_center (100) ---------------------------------------------------------
    for c in ("mom_10", "mom_14"):
        add(FeatureSpec(c + "_c", "osc_unbounded", [c], f"{c} - mom_center", f"iMomentum buffer 0 shift 1, minus {f['mom_center']}",
                        lambda x, c=c: x.col(c) - x.cfg["features"]["mom_center"], family="mom",
                        notes="verify scale on real data (suspected bug B3 in the old project)"))

    # --- distance to moving averages in ATR units -----------------------------------------------------------------
    ma_cols = sorted(c for c in columns if MA_PREFIX.match(c))
    for c in ma_cols:
        add(FeatureSpec(f"dist_{c}", "price_dist_atr", [c, "close"], f"(close - {c}) / ATR", f"(iClose shift1 - iMA/{c} shift1) / iATR shift1",
                        lambda x, c=c: (x.col("close") - x.col(c)) / x.atr, family=f"madist_{c.split('_')[0]}_{c.split('_')[-1]}", complexity=2))
    for fast, slow in ((10, 20), (20, 50), (50, 200)):
        for typ in ("sma", "ema"):
            a, b = f"ma_{fast}_{typ}", f"ma_{slow}_{typ}"
            add(FeatureSpec(f"rel_{a}_{b}", "ma_relation", [a, b], f"({a} - {b}) / ATR", f"(iMA {fast} - iMA {slow}) / iATR, shift 1",
                            lambda x, a=a, b=b: (x.col(a) - x.col(b)) / x.atr, family=f"marel_{typ}", complexity=2))
    for n in (20, 50):
        for typ in ("sma", "ema"):
            c = f"ma_{n}_{typ}"
            for k in f["slope_k"]:
                add(FeatureSpec(f"slope_{c}_{k}", "slope", [c], f"({c}[t] - {c}[t-{k}]) / ({k} * ATR)", f"(iMA shift1 - iMA shift {1 + k}) / ({k} * iATR shift1)",
                                lambda x, c=c, k=k: (x.col(c) - x.col(c).shift(k)) / (k * x.atr), warmup_bars=k, family=f"slope_{typ}_{k}", complexity=2))

    # --- Bollinger bands: %B and width -----------------------------------------------------------------------------
    for up in sorted(c for c in columns if re.match(r"^bb_.*_upper$", c)):
        base = up[: -len("_upper")]
        lo = base + "_lower"
        add(FeatureSpec(f"pctb_{base}", "band_position", [up, lo, "close"], "(close - lower) / (upper - lower)", f"iBands {base}: (close - lower)/(upper - lower), shift 1",
                        lambda x, up=up, lo=lo: (x.col("close") - x.col(lo)) / (x.col(up) - x.col(lo)).replace(0, np.nan), family="bb_pctb", complexity=2))
        add(FeatureSpec(f"bbwidth_{base}", "band_position", [up, lo], "(upper - lower) / ATR", f"iBands {base}: (upper - lower)/iATR, shift 1",
                        lambda x, up=up, lo=lo: (x.col(up) - x.col(lo)) / x.atr, family="bb_width", complexity=2))
    for up in sorted(c for c in columns if re.match(r"^envelope_.*_upper$", c)):     # spec 6.3: envelope bands, %B like Bollinger
        base = up[: -len("_upper")]
        lo = base + "_lower"
        add(FeatureSpec(f"env_pctb_{base}", "band_position", [up, lo, "close"], "(close - lower) / (upper - lower)",
                        f"iEnvelopes {base}: (close - lower)/(upper - lower), shift 1",
                        lambda x, up=up, lo=lo: (x.col("close") - x.col(lo)) / (x.col(up) - x.col(lo)).replace(0, np.nan), family="envelope", complexity=2))
        add(FeatureSpec(f"env_width_{base}", "band_position", [up, lo], "(upper - lower) / ATR", f"iEnvelopes {base}: (upper - lower)/iATR, shift 1",
                        lambda x, up=up, lo=lo: (x.col(up) - x.col(lo)) / x.atr, family="envelope", complexity=2))
    for c in sorted(c for c in columns if c.startswith("sar_")):
        add(FeatureSpec(f"dist_{c}", "price_dist_atr", [c, "close"], f"(close - {c}) / ATR", "(iClose - iSAR) / iATR, shift 1",
                        lambda x, c=c: (x.col("close") - x.col(c)) / x.atr, family="sar", complexity=2))

    # --- MACD family -----------------------------------------------------------------------------------------------
    for c in ("macd_12_26_9_main", "macd_12_26_9_signal", "osma_12_26_9", "ao", "ac", "bears_13", "bulls_13"):
        add(FeatureSpec(f"{c}_atr", "osc_unbounded", [c], f"{c} / ATR", f"{c} buffer / iATR, shift 1", lambda x, c=c: x.col(c) / x.atr, family="macd" if "macd" in c or "osma" in c else c.split("_")[0]))
    add(FeatureSpec("macd_hist_atr", "osc_unbounded", ["macd_12_26_9_main", "macd_12_26_9_signal"], "(main - signal) / ATR", "(MACD main - MACD signal) / iATR, shift 1",
                    lambda x: (x.col("macd_12_26_9_main") - x.col("macd_12_26_9_signal")) / x.atr, family="macd"))
    add(FeatureSpec("force_13_z", "osc_unbounded", ["force_13"], f"rolling z-score({rw}) of force_13", "needs rolling mean/std buffer in EA", lambda x: _zscore(x.col("force_13"), x.cfg["features"]["roll_window_vol"]),
                    warmup_bars=rw, family="force", complexity=4))

    # --- volatility ---------------------------------------------------------------------------------------------------
    add(FeatureSpec("atr_ratio", "vol_ratio", ["atr_14"], f"atr_14 / rolling mean(atr_14, {rw})", "iATR shift1 / SMA of iATR over rw bars", lambda x: x.col("atr_14") / x.col("atr_14").rolling(rw).mean(),
                    warmup_bars=rw, family="volatility", complexity=3))
    add(FeatureSpec("stddev_atr", "vol_ratio", ["stddev_20"], "stddev_20 / ATR", "iStdDev shift1 / iATR shift1", lambda x: x.col("stddev_20") / x.atr, family="volatility", complexity=2))
    for c in ("volregime_atr_pctl", "volregime_atr_ratio", "volregime_regime"):
        add(FeatureSpec(c, "vol_ratio", [c], f"{c} as exported", "iCustom VolRegime_ATR", lambda x, c=c: x.col(c), family="volregime", complexity=3, notes="verify percentile is not look-ahead"))

    # --- volume -----------------------------------------------------------------------------------------------------
    for c in ("tick_volume", "volumes"):
        add(FeatureSpec(f"{c}_rel", "vol_ratio", [c], f"{c} / rolling median({rv})", "needs rolling median in EA", lambda x, c=c: x.col(c) / x.col(c).rolling(x.cfg["features"]["roll_window_volume"]).median().replace(0, np.nan),
                        warmup_bars=rv, family="volume", complexity=4))
    for c in ("obv", "ad"):
        add(FeatureSpec(f"{c}_chg6_z", "vol_ratio", [c], f"6-bar change of {c} / rolling std over {rv}", "needs rolling std of changes in EA", lambda x, c=c: (lambda d: d / d.rolling(x.cfg["features"]["roll_window_volume"]).std(ddof=0).replace(0, np.nan))(x.col(c).diff(6)),
                        warmup_bars=rv + 6, family="volume_flow", complexity=4, notes="raw level is non-stationary"))
    add(FeatureSpec("chaikin_z", "vol_ratio", ["chaikin_3_10"], f"rolling z-score({rv}) of chaikin", "needs rolling mean/std in EA", lambda x: _zscore(x.col("chaikin_3_10"), x.cfg["features"]["roll_window_volume"]), warmup_bars=rv, family="volume_flow", complexity=4))

    # --- candles (in ATR) -------------------------------------------------------------------------------------------
    add(FeatureSpec("candle_body", "candle", ["open", "close"], "(close - open) / ATR", "(iClose1 - iOpen1) / iATR1", lambda x: (x.col("close") - x.col("open")) / x.atr, family="candle", complexity=1))
    add(FeatureSpec("candle_upper_wick", "candle", ["open", "high", "close"], "(high - max(open, close)) / ATR", "(iHigh1 - MathMax(iOpen1, iClose1)) / iATR1",
                    lambda x: (x.col("high") - np.maximum(x.col("open"), x.col("close"))) / x.atr, family="candle"))
    add(FeatureSpec("candle_lower_wick", "candle", ["open", "low", "close"], "(min(open, close) - low) / ATR", "(MathMin(iOpen1, iClose1) - iLow1) / iATR1",
                    lambda x: (np.minimum(x.col("open"), x.col("close")) - x.col("low")) / x.atr, family="candle"))
    add(FeatureSpec("candle_range", "candle", ["high", "low"], "(high - low) / ATR", "(iHigh1 - iLow1) / iATR1", lambda x: (x.col("high") - x.col("low")) / x.atr, family="candle_range"))
    add(FeatureSpec("candle_close_pos", "candle", ["high", "low", "close"], "(close - low) / (high - low)", "(iClose1 - iLow1) / (iHigh1 - iLow1)",
                    lambda x: (x.col("close") - x.col("low")) / (x.col("high") - x.col("low")).replace(0, np.nan), family="candle"))
    add(FeatureSpec("gap_atr", "candle", ["open", "close"], "(open - close[t-1]) / ATR", "(iOpen1 - iClose2) / iATR1", lambda x: (x.col("open") - x.col("close").shift(1)) / x.atr, warmup_bars=1, family="gap"))

    # --- returns, N-bar structure -----------------------------------------------------------------------------------------
    for k in f["return_k"]:
        add(FeatureSpec(f"ret_{k}_atr", "return_k", ["close"], f"(close - close[t-{k}]) / ATR", f"(iClose1 - iClose{1 + k}) / iATR1", lambda x, k=k: (x.col("close") - x.col("close").shift(k)) / x.atr, warmup_bars=k, family=f"ret_{k}"))
    for n in f["nbar_hilo"]:
        add(FeatureSpec(f"dist_hh_{n}", "structure", ["high", "close"], f"(highest high over last {n} bars - close) / ATR", f"(iHigh(iHighest(HIGH, {n}, 1)) - iClose1) / iATR1",
                        lambda x, n=n: (x.col("high").rolling(n).max() - x.col("close")) / x.atr, warmup_bars=n, family=f"hh_{n}", complexity=2))
        add(FeatureSpec(f"dist_ll_{n}", "structure", ["low", "close"], f"(close - lowest low over last {n} bars) / ATR", f"(iClose1 - iLow(iLowest(LOW, {n}, 1))) / iATR1",
                        lambda x, n=n: (x.col("close") - x.col("low").rolling(n).min()) / x.atr, warmup_bars=n, family=f"ll_{n}", complexity=2))
    for c in sorted(c for c in columns if c.startswith("ps_")):
        add(FeatureSpec(c, "structure", [c], f"{c} as exported", "iCustom PriceStructure (same inputs/buffers as exporter)", lambda x, c=c: x.col(c), family="ps", complexity=3,
                        notes="needs_verification: timing of custom indicator inside the forming bar"))

    # --- time (server time) -----------------------------------------------------------------------------------------------
    add(FeatureSpec("hour", "time", [], "server hour of bar open", "TimeToStruct(iTime(1)).hour", lambda x: pd.Series(x.df.index.hour.astype(np.float64), index=x.df.index), family="time"))
    add(FeatureSpec("dayofweek", "time", [], "server day of week (Mon=0)", "(TimeToStruct(iTime(1)).day_of_week + 6) % 7", lambda x: pd.Series(x.df.index.dayofweek.astype(np.float64), index=x.df.index), family="time"))

    def _session(x: Ctx) -> pd.Series:
        h = x.df.index.hour.to_numpy()
        code = np.zeros(len(h))
        for i, (lo, hi) in enumerate(SESSIONS.values()):
            code[(h >= lo) & (h <= hi)] = i
        return pd.Series(code, index=x.df.index)
    add(FeatureSpec("session", "time", [], "0 asia(0-6) 1 london(7-12) 2 newyork(13-21) 3 late(22-23), server hours", "derived from hour", _session, family="time"))

    # --- events (bars since) ----------------------------------------------------------------------------------------------
    cap = f["event_cap_bars"]
    add(FeatureSpec("bars_since_rsi14_below30", "event", ["rsi_14"], "bars since rsi_14 crossed below 30 (prev>=30, now<30); cap", "scan back over rsi buffer", 
                    lambda x: pd.Series(_bars_since(((x.col("rsi_14") < 30) & (x.col("rsi_14").shift(1) >= 30)).to_numpy(), x.cfg["features"]["event_cap_bars"]), index=x.df.index), warmup_bars=15, family="event_rsi", complexity=4))
    add(FeatureSpec("bars_since_rsi14_above70", "event", ["rsi_14"], "bars since rsi_14 crossed above 70 (prev<=70, now>70); cap", "scan back over rsi buffer",
                    lambda x: pd.Series(_bars_since(((x.col("rsi_14") > 70) & (x.col("rsi_14").shift(1) <= 70)).to_numpy(), x.cfg["features"]["event_cap_bars"]), index=x.df.index), warmup_bars=15, family="event_rsi", complexity=4))
    for direction, op in (("up", ">"), ("down", "<")):
        def _cross(x: Ctx, op=op) -> pd.Series:
            d = x.col("ma_10_sma") - x.col("ma_20_sma")
            ev = ((d > 0) & (d.shift(1) <= 0)) if op == ">" else ((d < 0) & (d.shift(1) >= 0))
            return pd.Series(_bars_since(ev.to_numpy(), x.cfg["features"]["event_cap_bars"]), index=x.df.index)
        add(FeatureSpec(f"bars_since_ma10_20_cross_{direction}", "event", ["ma_10_sma", "ma_20_sma"], f"bars since sma10 crossed {direction} sma20; cap", "scan back over two iMA buffers", _cross, warmup_bars=21, family="event_ma", complexity=4))
    bn = f["breakout_n"]
    add(FeatureSpec(f"breakout_up_{bn}", "event", ["high", "close"], f"1 if close > highest high of the previous {bn} bars", f"iClose1 > iHigh(iHighest(HIGH, {bn}, 2))",
                    lambda x: (x.col("close") > x.col("high").shift(1).rolling(x.cfg["features"]["breakout_n"]).max()).astype(np.float64).where(x.col("high").shift(1).rolling(x.cfg["features"]["breakout_n"]).max().notna()),
                    warmup_bars=bn + 1, family="breakout", complexity=2))
    add(FeatureSpec(f"breakout_down_{bn}", "event", ["low", "close"], f"1 if close < lowest low of the previous {bn} bars", f"iClose1 < iLow(iLowest(LOW, {bn}, 2))",
                    lambda x: (x.col("close") < x.col("low").shift(1).rolling(x.cfg["features"]["breakout_n"]).min()).astype(np.float64).where(x.col("low").shift(1).rolling(x.cfg["features"]["breakout_n"]).min().notna()),
                    warmup_bars=bn + 1, family="breakout", complexity=2))

    # --- calendar / news: excluded unless explicitly enabled ----------------------------------------------------------------
    cal_cols = sorted(c for c in columns if c.startswith(("cal_", "news_")))
    if f["include_calendar_news"]:
        for c in cal_cols:
            add(FeatureSpec(c, "event", [c], f"{c} as exported", "calendar API (not available in Strategy Tester)", lambda x, c=c: x.col(c), reproducible_in_tester=False, family="calendar", complexity=5, notes="live_only"))
    else:
        skipped.extend((c, "calendar/news excluded by default (live_only)") for c in cal_cols)
    return specs, skipped
