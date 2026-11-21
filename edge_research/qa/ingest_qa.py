"""
Stage 0: data QA on the table returned by the (unchanged) loader.

Every check returns a ``CheckResult`` with status ``pass | warn | fail | skip``. A ``fail`` on a critical check stops the
run (``QAReport.critical_failed``). The report is written to ``00_data_qa/`` (report.md, alignment_checks.csv, gap_report.csv).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    critical: bool = False


@dataclass
class QAReport:
    checks: List[CheckResult] = field(default_factory=list)
    alignment: pd.DataFrame = field(default_factory=pd.DataFrame)
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded_columns: Dict[str, str] = field(default_factory=dict)   # column -> reason
    flagged_columns: Dict[str, str] = field(default_factory=dict)    # column -> note (kept, but flagged)
    scales: pd.DataFrame = field(default_factory=pd.DataFrame)       # per-column min / median / max (spec 5.1 scale summary)
    warmup_rows: int = 0
    data_hash: str = ""

    @property
    def critical_failed(self) -> bool:
        return any(c.status == FAIL and c.critical for c in self.checks)

    def add(self, name: str, status: str, detail: str, critical: bool = False) -> None:
        self.checks.append(CheckResult(name, status, detail, critical))


def data_hash(df: pd.DataFrame) -> str:
    """Hash of values + index (stable across runs; used by the checkpoint fingerprint)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df.index, index=False).to_numpy().tobytes())
    h.update(str(list(df.columns)).encode())
    for col in df.columns:
        h.update(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:32]


def check_time_index(df: pd.DataFrame, rep: QAReport) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        rep.add("time_index", FAIL, "index is not a DatetimeIndex", critical=True)
        return
    dup = int(df.index.duplicated().sum())
    mono = bool(df.index.is_monotonic_increasing)
    if dup or not mono:
        rep.add("time_index", FAIL, f"monotonic={mono}, duplicates={dup}", critical=True)
    else:
        rep.add("time_index", PASS, "monotonic increasing, no duplicates")


def check_bar_count(df: pd.DataFrame, rep: QAReport, expected_per_year: float, tolerance: float) -> None:
    span_years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400)
    if span_years < 0.25:
        rep.add("bar_count", SKIP, f"span {span_years:.2f}y too short for a per-year expectation")
        return
    per_year = len(df) / span_years
    ok = abs(per_year - expected_per_year) / expected_per_year <= tolerance
    rep.add("bar_count", PASS if ok else WARN, f"{per_year:.0f} bars/year vs expected ~{expected_per_year:.0f}")


def find_gaps(df: pd.DataFrame, interval_minutes: int, gap_factor: float) -> pd.DataFrame:
    """Gaps longer than gap_factor x interval, split into weekend gaps and other gaps."""
    dt = df.index.to_series().diff()
    long_gap = dt > pd.Timedelta(minutes=interval_minutes * gap_factor)
    g = pd.DataFrame({"gap_start": df.index[:-1][long_gap.to_numpy()[1:]], "gap_end": df.index[1:][long_gap.to_numpy()[1:]]})
    if g.empty:
        return g.assign(hours=[], kind=[])
    g["hours"] = (g["gap_end"] - g["gap_start"]).dt.total_seconds() / 3600.0
    weekend = (g["gap_start"].dt.weekday >= 4) & (g["hours"] <= 72)
    g["kind"] = np.where(weekend, "weekend", "other")
    return g.reset_index(drop=True)


def check_gaps(df: pd.DataFrame, rep: QAReport, interval_minutes: int, gap_factor: float) -> None:
    rep.gaps = find_gaps(df, interval_minutes, gap_factor)
    other = rep.gaps[rep.gaps["kind"] == "other"] if len(rep.gaps) else rep.gaps
    if len(other):
        rep.add("gaps", WARN, f"{len(other)} non-weekend gaps (max {other['hours'].max():.1f}h); {len(rep.gaps) - len(other)} weekend gaps")
    else:
        rep.add("gaps", PASS, f"no non-weekend gaps; {len(rep.gaps)} weekend gaps")


def check_columns(df: pd.DataFrame, rep: QAReport, nan_threshold: float) -> None:
    nan_frac = df.isna().mean()
    num = df.select_dtypes(include=[np.number])
    scale_rows = []
    for c in num.columns:                                            # scale summary (spec 5.1): min / median / max per column
        v = num[c].to_numpy()
        fin = v[np.isfinite(v)]
        scale_rows.append({"column": c, "min": float(fin.min()) if fin.size else float("nan"),
                           "median": float(np.nanmedian(v)) if fin.size else float("nan"),
                           "max": float(fin.max()) if fin.size else float("nan"), "nan_pct": float(nan_frac[c])})
    rep.scales = pd.DataFrame(scale_rows)
    for col in df.columns:
        vals = df[col].to_numpy()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0 or np.nanmin(finite) == np.nanmax(finite):
            rep.excluded_columns[col] = "constant or all-NaN"
        elif nan_frac[col] > nan_threshold:
            # Warm-up NaN is trimmed separately; this is the share AFTER ignoring the leading NaN block.
            first = df[col].first_valid_index()
            tail = df.loc[first:, col] if first is not None else df[col]
            if tail.isna().mean() > nan_threshold:
                rep.excluded_columns[col] = f"{tail.isna().mean():.1%} NaN after warm-up"
    rep.add("columns", PASS if not rep.excluded_columns else WARN, f"{len(rep.excluded_columns)} columns excluded: {sorted(rep.excluded_columns)[:8]}")
    for col in df.columns:
        if col.startswith("ps_"):
            rep.flagged_columns[col] = "needs_verification: custom indicator may expose a forming bar (parity test required)"
        if col.startswith(("cal_", "news_")):
            rep.flagged_columns[col] = "live_only: not reproducible in Strategy Tester"
        if col.startswith("mom_"):
            med = float(np.nanmedian(df[col].to_numpy()))
            if abs(med - 100.0) < 20.0:
                rep.flagged_columns[col] = f"centre ~100 (median {med:.1f}); transform to mom-100"


def compute_warmup(df: pd.DataFrame, required: List[str]) -> int:
    """Rows to drop at the start so every required column is valid (first row where all are finite)."""
    ok = np.isfinite(df[required].to_numpy(np.float64)).all(axis=1)
    return int(np.argmax(ok)) if ok.any() else len(df)


ALIGNMENT_CANDIDATES = ("ma_5_sma", "ma_10_sma", "ma_20_sma")


def check_alignment(df: pd.DataFrame, rep: QAReport, tolerance: float) -> None:
    """Recompute SMA(n) from close and compare with the exported column at shifts -1, 0, +1. Must match at shift 0."""
    rows = []
    for col in ALIGNMENT_CANDIDATES:
        if col not in df.columns or "close" not in df.columns:
            continue
        n = int(col.split("_")[1])
        mine = df["close"].astype(np.float64).rolling(n).mean()
        exported = df[col].astype(np.float64)
        for shift in (-1, 0, 1):
            diff = (mine.shift(shift) - exported).abs()
            scale = float(df["close"].abs().median())
            rows.append({"column": col, "shift": shift, "max_abs_diff": float(diff.max()), "median_abs_diff": float(diff.median()),
                         "match": bool(diff.dropna().max() <= tolerance * scale)})
    rep.alignment = pd.DataFrame(rows)
    if rep.alignment.empty:
        rep.add("alignment", SKIP, "no SMA column (ma_5/10/20_sma) available to test alignment")
        return
    at0 = rep.alignment[rep.alignment["shift"] == 0]
    if at0["match"].all():
        rep.add("alignment", PASS, "exported SMAs match close-based recomputation at shift 0", critical=True)
    else:
        others = rep.alignment[(rep.alignment["shift"] != 0) & rep.alignment["match"]]
        hint = f" (matches at shift {sorted(others['shift'].unique().tolist())}: exporter shifts bars)" if len(others) else ""
        rep.add("alignment", FAIL, "SMA mismatch at shift 0" + hint, critical=True)


def check_spread(spread_points: Optional[pd.Series], df: pd.DataFrame, rep: QAReport) -> None:
    if spread_points is None:
        rep.add("spread", WARN, "no spread file: fixed spread from config will be used for every bar")
        return
    aligned = spread_points.reindex(df.index)
    missing = float((~(aligned > 0)).mean())
    rep.add("spread", PASS if missing < 0.01 else WARN,
            f"coverage {1 - missing:.1%}; median {float(aligned.median()):.1f} pts, p95 {float(aligned.quantile(0.95)):.1f} pts")


def run_qa(df: pd.DataFrame, cfg: Dict[str, Any], spread_points: Optional[pd.Series] = None) -> QAReport:
    rep = QAReport()
    q, d = cfg["qa"], cfg["data"]
    check_time_index(df, rep)
    if rep.critical_failed:
        return rep
    check_bar_count(df, rep, q["expected_bars_per_year"], q["bars_per_year_tolerance"])
    check_gaps(df, rep, d["timeframe_minutes"], q["gap_factor"])
    check_columns(df, rep, q["nan_column_threshold"])
    check_alignment(df, rep, q["alignment_tolerance"])
    check_spread(spread_points, df, rep)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]        # spec 5.1: report the warm-up block explicitly
    rep.warmup_rows = compute_warmup(df, num_cols) if num_cols else 0
    rep.add("warmup", PASS if rep.warmup_rows < len(df) else WARN,
            f"{rep.warmup_rows} leading rows are not fully finite; stage 1 starts the feature table at the first valid row")
    rep.data_hash = data_hash(df)
    return rep


def write_report(rep: QAReport, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 0 data QA report", "", f"data_hash: `{rep.data_hash}`", "", "| check | status | detail |", "|---|---|---|"]
    for c in rep.checks:
        lines.append(f"| {c.name}{' (critical)' if c.critical else ''} | {c.status} | {c.detail} |")
    lines += ["", f"critical_failed: **{rep.critical_failed}**", f"warmup_rows: **{rep.warmup_rows}**", ""]
    if rep.excluded_columns:
        lines += ["## Excluded columns", *[f"- `{k}`: {v}" for k, v in sorted(rep.excluded_columns.items())], ""]
    if rep.flagged_columns:
        lines += ["## Flagged columns", *[f"- `{k}`: {v}" for k, v in sorted(rep.flagged_columns.items())], ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    rep.alignment.to_csv(out / "alignment_checks.csv", index=False)
    rep.gaps.to_csv(out / "gap_report.csv", index=False)
    rep.scales.to_csv(out / "column_stats.csv", index=False)             # spec 5.1: per-column min / median / max
    return out / "report.md"
