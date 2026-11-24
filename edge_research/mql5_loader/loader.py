"""Load the single CSV written by ``dataExporter.mq5`` and prepare it for the pipeline.

One file carries everything: OHLCV, the per-bar spread, every indicator column and the news columns. There is no second spread file.

``load_mql5_export`` = existing robust CSV reader (encodings, stray quotes, duplicate timestamps) + header check against the exporter manifest +
``prepare_table`` (envelope repair, integrity checks). ``prepare_table`` is idempotent and is applied wherever the pipeline reads the bar table
(stage 0-2 and every later stage that reloads the ingested parquet), so all stages see the same repaired data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_research.ingestion.loader import load_csv_and_validate
from edge_research.mql5_loader.schema import ENVELOPE_COLUMNS, EXPORTER_BY_NAME, header_report

_FILENAME_RE = re.compile(r"^(?P<symbol>[A-Za-z0-9._#\-]+?)_(?P<tf>MN1|M1|M5|M15|M30|H1|H4|D1|W1)_export_(?P<date>\d{8})")
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200}

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


@dataclass
class Finding:
    level: str
    check: str
    message: str


@dataclass
class MQL5Table:
    df: pd.DataFrame                              # bar table, datetime index, float32 columns, envelope repaired
    spread: Optional[pd.Series]                   # per-bar spread in points taken from the SAME file (None if the export has no spread column)
    header: Dict[str, List[str]]                  # header vs exporter manifest
    findings: List[Finding] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

    @property
    def failed(self) -> List[Finding]:
        return [f for f in self.findings if f.level == FAIL]


# ------------------------------------------------------------------------------------------------------------------ file name / header
def parse_export_filename(path: str | Path) -> Tuple[Optional[str], Optional[str]]:
    """(symbol, timeframe) from ``<SYMBOL>_<TF>_export_<YYYYMMDD>*.csv`` (the exporter's own file name), else (None, None)."""
    m = _FILENAME_RE.match(Path(path).name)
    return (m["symbol"], m["tf"]) if m else (None, None)


def read_header(csv_path: str | Path) -> List[str]:
    """Column names of the CSV without loading it."""
    cols = pd.read_csv(csv_path, nrows=0, quoting=3, encoding_errors="replace").columns
    return [str(c).replace('"', "").strip() for c in cols]


def infer_timeframe(index: pd.DatetimeIndex) -> Optional[str]:
    """Timeframe from the median bar spacing (used when the file name does not carry it)."""
    if len(index) < 3:
        return None
    minutes = float(np.median(np.diff(index.values).astype("timedelta64[m]").astype(np.float64)))
    best = min(TIMEFRAME_MINUTES.items(), key=lambda kv: abs(np.log(kv[1] / max(minutes, 1.0))))
    return best[0] if abs(np.log(best[1] / max(minutes, 1.0))) < 0.2 else None


def infer_digits(csv_path: str | Path, nrows: int = 5000) -> Optional[int]:
    """Number of price decimals seen in the ``close`` column (trailing zeros of the 6-decimal export are not significant)."""
    try:
        s = pd.read_csv(csv_path, usecols=["close"], dtype=str, nrows=nrows)["close"].dropna()
    except Exception:
        return None
    if s.empty:
        return None
    return int(min(6, max(len(v.split(".")[1].rstrip("0")) if "." in v else 0 for v in s)))


# ------------------------------------------------------------------------------------------------------------------ repairs
def repair_envelope(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fix the envelope columns written by the exporter (see schema docstring). Idempotent; changes nothing on a healthy file.

    iEnvelopes: buffer 0 = upper, 1 = lower. The exporter reads buffer 1 into ``..._upper`` and buffer 2 (non-existent, NaN) into ``..._lower``.
    Detection: ``..._lower`` is entirely NaN while ``..._upper`` lies BELOW the SMA(20) (it is really the lower band). Repair: the exported "upper" column
    becomes the lower band and the upper band is rebuilt from the symmetry of iEnvelopes (upper = 2 * SMA - lower, exact because both bands are
    SMA * (1 +/- deviation)).
    """
    up, lo = ENVELOPE_COLUMNS
    if up not in df.columns or lo not in df.columns or "ma_20_sma" not in df.columns:
        return df, []
    lower_all_nan = bool(df[lo].isna().all())
    if not lower_all_nan:
        return df, []
    ma = df["ma_20_sma"].to_numpy(np.float64)
    first = df[up].to_numpy(np.float64)
    ok = np.isfinite(ma) & np.isfinite(first) & (ma > 0)
    if not ok.any():
        return df, []
    ratio = float(np.median(first[ok] / ma[ok]))
    out = df.copy()
    if ratio < 1.0:                                       # the "upper" column really holds the lower band
        lower, upper = first, np.where(np.isfinite(first), 2.0 * ma - first, np.nan)
        note = (f"envelope: '{up}' held the LOWER band (median value / SMA20 = {ratio:.6f} < 1) and '{lo}' was all NaN; "
                f"moved it to '{lo}' and rebuilt '{up}' = 2 * ma_20_sma - lower (fix the exporter too: see schema.ENVELOPE_MQL5_FIX)")
    else:                                                 # upper is fine, only the lower band is missing
        upper, lower = first, np.where(np.isfinite(first), 2.0 * ma - first, np.nan)
        note = f"envelope: '{lo}' was all NaN; rebuilt it as 2 * ma_20_sma - '{up}'"
    out[up] = upper.astype(np.float32)
    out[lo] = lower.astype(np.float32)
    return out, [note]


# ------------------------------------------------------------------------------------------------------------------ integrity checks
def _scale(df: pd.DataFrame) -> float:
    return float(df["close"].abs().median()) if "close" in df.columns and len(df) else 1.0


def integrity_report(df: pd.DataFrame, tol: float = 1e-4) -> List[Finding]:
    """Cross-checks between exported columns. They prove the exporter parameters are what the manifest says and catch exporter bugs early.

    ``tol`` is relative to the median close (a 6-decimal export of float32 data cannot be tighter than ~1e-6).
    """
    out: List[Finding] = []
    sc = _scale(df)
    c = lambda name: df[name].to_numpy(np.float64)                                            # noqa: E731

    if {"open", "high", "low", "close"} <= set(df.columns):
        o, h, l, cl = c("open"), c("high"), c("low"), c("close")
        bad = int(np.sum((h < np.maximum(o, cl) - tol * sc) | (l > np.minimum(o, cl) + tol * sc)))
        out.append(Finding(OK if bad == 0 else FAIL, "ohlc", "OHLC consistent" if bad == 0 else f"{bad} bar(s) with high < max(open, close) or low > min(open, close)"))

    for tag, dev in (("2", 2.0), ("15", 1.5), ("25", 2.5)):
        base = f"bb_20_{tag}"
        need = [f"{base}_upper", f"{base}_middle", f"{base}_lower"]
        if not all(n in df.columns for n in need):
            continue
        up, mid, lo = (c(n) for n in need)
        if "ma_20_sma" in df.columns:
            d = np.nanmax(np.abs(mid - c("ma_20_sma"))) if np.isfinite(mid).any() else np.nan
            ok = bool(d <= tol * sc) if np.isfinite(d) else True
            out.append(Finding(OK if ok else WARN, f"{base}_middle", f"{base}_middle equals ma_20_sma (max abs diff {d:.2e})" if ok else
                               f"{base}_middle differs from ma_20_sma (max abs diff {d:.2e}): the band does not use SMA(20) on close"))
        if "stddev_20" in df.columns:
            sd = c("stddev_20")
            m = np.isfinite(up) & np.isfinite(mid) & np.isfinite(sd) & (sd > 1e-9 * sc)
            if m.any():
                k = (up[m] - mid[m]) / sd[m]
                # the export has 6 decimals and is held as float32: allow that rounding noise (relative to each row's stddev) plus a small margin
                bound = (4e-6 + 4.8e-7 * np.abs(mid[m])) / sd[m] + 5e-3
                ok = bool(np.all(np.abs(k - dev) <= bound))
                out.append(Finding(OK if ok else WARN, f"{base}_deviation", f"{base}: (upper - middle) / stddev_20 = {np.median(k):.3f} (expected {dev})" if ok else
                                   f"{base}: (upper - middle) / stddev_20 ranges {np.nanmin(k):.3f}..{np.nanmax(k):.3f}, expected {dev}: the deviation parameter differs from the manifest"))
        sym = np.nanmax(np.abs((up - mid) - (mid - lo))) if np.isfinite(up).any() else 0.0
        out.append(Finding(OK if sym <= tol * sc else WARN, f"{base}_symmetry", f"{base} symmetric around the middle band" if sym <= tol * sc else f"{base} not symmetric (max {sym:.2e})"))

    up_n, lo_n = ENVELOPE_COLUMNS
    if up_n in df.columns and lo_n in df.columns and "ma_20_sma" in df.columns:
        up, lo, ma = c(up_n), c(lo_n), c("ma_20_sma")
        m = np.isfinite(up) & np.isfinite(lo) & np.isfinite(ma)
        if not m.any():
            out.append(Finding(WARN, "envelope", "envelope bands are all NaN after repair"))
        else:
            ok = bool(np.all((up[m] > ma[m]) & (lo[m] < ma[m])))
            out.append(Finding(OK if ok else FAIL, "envelope", f"envelope upper > SMA20 > lower, deviation {float(np.median(up[m] / ma[m] - 1)) * 100:.3f}% of SMA" if ok else
                               "envelope bands do not bracket SMA20 (repair failed)"))

    if "volumes" in df.columns and "tick_volume" in df.columns:
        same = bool(np.array_equal(df["volumes"].to_numpy(), df["tick_volume"].to_numpy()))
        out.append(Finding(INFO, "volumes", "volumes == tick_volume (iVolume written twice; the duplicate is collapsed by feature clustering)" if same else
                           "volumes differs from tick_volume"))
    if "real_volume" in df.columns:
        zero = bool((df["real_volume"].fillna(0) == 0).all())
        out.append(Finding(INFO, "real_volume", "real_volume is always 0 (no exchange volume for this symbol); the feature is dropped by QA as constant" if zero else "real_volume present"))
    if "spread" in df.columns:
        s = c("spread")
        pos = s[np.isfinite(s) & (s > 0)]
        out.append(Finding(OK if len(pos) else WARN, "spread",
                           (f"spread > 0 on {len(pos) / max(len(s), 1):.1%} of bars (median {np.median(pos):.1f} pts); rows with 0 use the last known spread, then the configured fixed spread"
                            if len(pos) else "spread is 0 on every bar: the configured fixed spread is used for all bars")))
    else:
        out.append(Finding(WARN, "spread", "no spread column (export older than dataExporter v1.03): the configured fixed spread is used for every bar"))

    news = [n for n in df.columns if n.startswith("news_")]
    if news and all(df[n].isna().all() for n in news):
        out.append(Finding(WARN, "news", "all news_* columns are NaN: the calendar CSV was not loaded in the tester (run the exporter once on a live chart first)"))
    return out


def prepare_table(df: pd.DataFrame, tol: float = 1e-4) -> Tuple[pd.DataFrame, List[Finding], List[str]]:
    """Idempotent preparation: envelope repair + integrity findings."""
    df, repairs = repair_envelope(df)
    return df, integrity_report(df, tol), repairs


def extract_spread(df: pd.DataFrame) -> Optional[pd.Series]:
    """The per-bar spread (points) taken from the data table itself."""
    if "spread" not in df.columns:
        return None
    return pd.Series(df["spread"].to_numpy(np.float32), index=df.index, name="spread")


# ------------------------------------------------------------------------------------------------------------------ entry point
def load_mql5_export(csv_path: str | Path, tol: float = 1e-4) -> MQL5Table:
    """Single entry point: file -> ``MQL5Table`` (bars, spread, header report, findings, repairs)."""
    path = Path(csv_path)
    header = header_report(read_header(path))
    df = load_csv_and_validate(path)
    df, findings, repairs = prepare_table(df, tol)
    symbol, tf = parse_export_filename(path)
    tf = tf or infer_timeframe(df.index)
    findings = [Finding(FAIL if header["missing_required"] else OK, "header",
                        f"required exporter columns missing: {header['missing_required']}" if header["missing_required"] else "all required exporter columns present")] + findings
    if header["missing"] and not header["missing_required"]:
        findings.append(Finding(INFO, "header", f"optional exporter columns absent (older export): {header['missing']}"))
    if header["unlisted"]:
        findings.append(Finding(WARN, "header", f"columns not in the exporter manifest (exporter newer than schema.py?): {header['unlisted']}"))
    if header["duplicated"]:
        findings.append(Finding(FAIL, "header", f"duplicated column names: {header['duplicated']}"))
    return MQL5Table(df, extract_spread(df), header, findings, repairs, symbol, tf)
