"""Synthetic CSV in the exact format of ``dataExporter.mq5`` v1.03 (all 108 columns), for tests and for trying the pipeline without MT5.

OHLCV comes from the project's synthetic bar generator. The indicator columns are computed from those bars with the textbook formulas (SMA / EMA /
SMMA / LWMA, DEMA, TEMA, SAR, RSI, Stochastic, CCI, WPR, MACD, ATR, Bollinger, StdDev, OBV, A/D, MFI, ...); FrAMA / AMA / VIDYA / the custom indicators are
simple causal stand-ins. They are NOT bit-exact copies of MT5 and must never be used for research results.

The file reproduces the quirks of the real exporter so that the loader is tested against them: ``envelope_20_01_upper`` holds the LOWER band and
``envelope_20_01_lower`` is NaN (buffer 1 / 2 bug), leading spread values of 0, ``real_volume`` = 0, ``volumes`` == ``tick_volume``, NaN for indicators during
warm-up, an empty last line, and the ``YYYYMMDD HH:MM`` time format.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from edge_research.mql5_loader.schema import EXPORTER_HEADER, MA_PERIODS
from edge_research.synth.generator import make_synthetic_bars


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _smma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _lwma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=np.float64)
    return x.rolling(n).apply(lambda v: float(np.dot(v, w) / w.sum()), raw=True)


def _sar(h: np.ndarray, l: np.ndarray, step: float, maximum: float) -> np.ndarray:
    n = len(h)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    long_, af, ep = True, step, h[0]
    sar = l[0]
    out[0] = sar
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if long_:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                long_, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, maximum)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                long_, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, maximum)
        out[i] = sar
    return out


def build_exporter_frame(*a, **kw) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _build_exporter_frame(*a, **kw)


def _build_exporter_frame(n_bars: int = 1500, seed: int = 7, timeframe_minutes: int = 60, with_news: bool = False, spread_zero_rows: int = 40) -> pd.DataFrame:
    """The 108-column table the exporter would write (index = bar open time), including the envelope bug."""
    bars, meta = make_synthetic_bars(n_bars, seed, timeframe_minutes=timeframe_minutes)
    o, h, l, c = (bars[k].astype(np.float64).round(5) for k in ("open", "high", "low", "close"))      # a 5-digit broker quotes 5 decimals
    h, l = np.maximum(h, np.maximum(o, c)), np.minimum(l, np.minimum(o, c))
    tv = bars["tick_volume"].astype(np.float64)
    idx = bars.index
    d = pd.DataFrame(index=idx)
    d["open"], d["high"], d["low"], d["close"], d["tick_volume"] = o, h, l, c, tv
    d["real_volume"] = 0.0
    sp = meta["spread_points"].astype(np.float64).copy()
    sp.iloc[:spread_zero_rows] = 0.0                                   # old history without spread information
    d["spread"] = sp
    rng = np.random.default_rng(seed + 1)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    for n in MA_PERIODS:
        d[f"ma_{n}_sma"] = c.rolling(n).mean()
        d[f"ma_{n}_ema"] = _ema(c, n)
        d[f"ma_{n}_smma"] = _smma(c, n)
        d[f"ma_{n}_lwma"] = _lwma(c, n) if n <= 50 else c.rolling(n).mean().ewm(span=3, adjust=False).mean()
    for n in (10, 20, 50):
        e1 = _ema(c, n)
        e2 = _ema(e1.dropna(), n).reindex(idx)
        e3 = _ema(e2.dropna(), n).reindex(idx)
        d[f"dema_{n}"] = 2 * e1 - e2
        d[f"tema_{n}"] = 3 * e1 - 3 * e2 + e3
    for n in (14, 20):
        d[f"frama_{n}"] = c.ewm(alpha=2.0 / (n + 1), adjust=False, min_periods=n).mean()           # stand-in
    change, volat = (c - c.shift(10)).abs(), c.diff().abs().rolling(10).sum()
    er = (change / volat.replace(0, np.nan)).fillna(0)
    sc = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
    ama = np.full(len(c), np.nan)
    for i in range(10, len(c)):
        ama[i] = (c.iloc[i] if np.isnan(ama[i - 1]) else ama[i - 1] + sc.iloc[i] * (c.iloc[i] - ama[i - 1]))
    d["ama_10_2_30"] = ama
    for n in (9, 12):
        d[f"vidya_{n}"] = c.ewm(alpha=2.0 / (13 + 1), adjust=False, min_periods=n + 12).mean()    # stand-in
    for tag, step in (("002", 0.02), ("001", 0.01), ("003", 0.03)):
        d[f"sar_{tag}_02"] = _sar(h.to_numpy(), l.to_numpy(), step, 0.2)

    up_m, dn_m = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    pdm, mdm = up_m.where(up_m > dn_m, 0.0), dn_m.where(dn_m > up_m, 0.0)
    for name, sm in (("adx_14", lambda x: x.rolling(14).mean()), ("adxwilder_14", lambda x: _smma(x, 14))):
        atr_x = sm(tr)
        pdi, mdi = 100 * sm(pdm) / atr_x, 100 * sm(mdm) / atr_x
        d[name] = sm(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))

    delta = c.diff()
    for n in (7, 14, 21):
        rs = _smma(delta.clip(lower=0), n) / _smma((-delta).clip(lower=0), n).replace(0, np.nan)
        d[f"rsi_{n}"] = 100 - 100 / (1 + rs)
    for k in (5, 14):
        ll, hh = l.rolling(k).min(), h.rolling(k).max()
        fast = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        d[f"stoch_{k}_3_3_k"] = fast.rolling(3).mean()
        d[f"stoch_{k}_3_3_d"] = d[f"stoch_{k}_3_3_k"].rolling(3).mean()
    tp = (h + l + c) / 3
    for n in (14, 20):
        mad = tp.rolling(n).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True)
        d[f"cci_{n}"] = (tp - tp.rolling(n).mean()) / (0.015 * mad.replace(0, np.nan))
    d["wpr_14"] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    for n in (10, 14):
        d[f"mom_{n}"] = c / c.shift(n) * 100.0
    macd = _ema(c, 12) - _ema(c, 26)
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    d["macd_12_26_9_main"], d["macd_12_26_9_signal"], d["osma_12_26_9"] = macd, sig, macd - sig
    dem_h, dem_l = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    d["demarker_14"] = dem_h.rolling(14).mean() / (dem_h.rolling(14).mean() + dem_l.rolling(14).mean()).replace(0, np.nan)
    num, den = (c - o) + 2 * (c - o).shift(1) + 2 * (c - o).shift(2) + (c - o).shift(3), (h - l) + 2 * (h - l).shift(1) + 2 * (h - l).shift(2) + (h - l).shift(3)
    d["rvi_10"] = num.rolling(10).mean() / den.rolling(10).mean().replace(0, np.nan)
    e3 = _ema(_ema(_ema(c, 14).dropna(), 14).dropna(), 14).reindex(idx)
    d["trix_14"] = e3.pct_change() * 1.0
    med = (h + l) / 2
    ao = med.rolling(5).mean() - med.rolling(34).mean()
    d["ao"], d["ac"] = ao, ao - ao.rolling(5).mean()
    d["bears_13"], d["bulls_13"] = l - _ema(c, 13), h - _ema(c, 13)

    d["atr_14"], d["atr_20"] = tr.rolling(14).mean(), tr.rolling(20).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    for tag, dev in (("2", 2.0), ("15", 1.5), ("25", 2.5)):
        d[f"bb_20_{tag}_upper"], d[f"bb_20_{tag}_middle"], d[f"bb_20_{tag}_lower"] = mid + dev * sd, mid, mid - dev * sd
    d["stddev_20"] = sd
    d["envelope_20_01_upper"] = mid * (1 - 0.001)          # the exporter bug: buffer 1 (= LOWER band) is written under the "upper" name
    d["envelope_20_01_lower"] = np.nan                     # ... and buffer 2 does not exist

    direction = np.sign(delta.fillna(0))
    d["obv"] = (direction * tv).cumsum()
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    d["ad"] = (clv.fillna(0) * tv).cumsum()
    mf = tp * tv
    pos, neg = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum(), mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    d["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    d["force_13"] = (delta * tv).rolling(13).mean()
    d["chaikin_3_10"] = _ema(d["ad"], 3) - _ema(d["ad"], 10)
    d["volumes"] = tv

    atr14 = d["atr_14"]
    d["volregime_atr_pctl"] = atr14.rolling(100).apply(lambda v: 100.0 * (v[:-1] < v[-1]).mean(), raw=True)
    d["volregime_atr_ratio"] = atr14 / atr14.rolling(100).mean()
    d["volregime_regime"] = np.select([d["volregime_atr_pctl"] < 33.3, d["volregime_atr_pctl"] < 66.7], [0.0, 1.0], 2.0)
    d.loc[d["volregime_atr_pctl"].isna(), "volregime_regime"] = np.nan

    day = pd.Series(idx.normalize(), index=idx)
    day_h, day_l = h.groupby(day).max(), l.groupby(day).min()
    pdh, pdl = day.map(day_h.shift(1)), day.map(day_l.shift(1))
    a = atr14.replace(0, np.nan)
    d["ps_dist_pdh_atr"], d["ps_dist_pdl_atr"] = (pdh - c) / a, (pdl - c) / a
    sess = pd.Series(idx.hour // 8, index=idx)
    key = day.astype(str) + "_" + sess.astype(str)
    s_h, s_l = h.groupby(key).max(), l.groupby(key).min()
    prev_key = (pd.Series(key.unique()).shift(1)).to_numpy()
    prev_map = dict(zip(key.unique(), prev_key))
    pk = key.map(prev_map)
    d["ps_dist_sess_h_atr"] = (pk.map(s_h) - c) / a
    d["ps_dist_sess_l_atr"] = (pk.map(s_l) - c) / a
    step = 0.005
    d["ps_dist_round_up_atr"], d["ps_dist_round_dn_atr"] = (np.ceil(c / step) * step - c) / a, (np.floor(c / step) * step - c) / a
    gap = (o - prev_c) / a + pd.Series(rng.normal(0.0, 0.15, len(d)), index=idx)           # stand-in: the synthetic open equals the previous close
    d["ps_gap_atr"] = gap
    d["ps_gap_fill_pct"] = np.clip(50.0 + 50.0 * np.tanh(-(c - o) * np.sign(gap) / (a * 0.8)), 0, 100)

    news_cols = ["news_min_to_next", "news_next_importance", "news_min_since_last", "news_last_importance", "news_last_surprise", "news_in_window", "news_count_next_24h"]
    for nc in news_cols:
        d[nc] = np.nan
    if with_news:
        ev = pd.DatetimeIndex(sorted(pd.to_datetime(rng.choice(idx.values, size=max(5, n_bars // 40), replace=False))))
        ref = idx + pd.Timedelta(minutes=timeframe_minutes)
        pos_ = np.searchsorted(ev.values, ref.values, side="right")
        nxt = np.where(pos_ < len(ev), (ev.values[np.minimum(pos_, len(ev) - 1)] - ref.values) / np.timedelta64(1, "m"), np.nan)
        lst = np.where(pos_ > 0, (ref.values - ev.values[np.maximum(pos_ - 1, 0)]) / np.timedelta64(1, "m"), np.nan)
        d["news_min_to_next"], d["news_min_since_last"] = np.round(nxt, 1), np.round(lst, 1)
        d["news_next_importance"] = np.where(np.isfinite(nxt), rng.integers(2, 4, len(d)), np.nan)
        d["news_last_importance"] = np.where(np.isfinite(lst), rng.integers(2, 4, len(d)), np.nan)
        d["news_last_surprise"] = np.where(np.isfinite(lst), rng.normal(0, 1, len(d)), np.nan)
        d["news_in_window"] = ((nxt <= 30) | (lst <= 30)).astype(float)
        d["news_count_next_24h"] = rng.integers(0, 4, len(d)).astype(float)

    d["volumes"] = tv
    d = d.reset_index()
    d["time"] = d["time"].dt.strftime("%Y%m%d %H:%M")
    return d[EXPORTER_HEADER]


def write_exporter_csv(path: str | Path, n_bars: int = 1500, seed: int = 7, timeframe_minutes: int = 60, **kw) -> Path:
    """Write the sample file the way the exporter does: 6 decimals, 'NaN' for missing values, spread / volumes as integers, trailing empty line."""
    df = build_exporter_frame(n_bars, seed, timeframe_minutes, **kw)
    int_cols = ["tick_volume", "real_volume", "spread", "volumes"]
    out = df.copy()
    for c in out.columns:
        if c == "time":
            continue
        if c in int_cols:
            out[c] = out[c].round().astype("int64").astype(str)
        else:
            out[c] = out[c].map(lambda v: "NaN" if pd.isna(v) else f"{v:.6f}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n")                                                  # the exporter leaves an empty last line
    return path
