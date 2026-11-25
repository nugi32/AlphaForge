"""One-command runner: exporter CSV -> config derived from the file -> coverage audit -> stages 0..N of the stock pipeline.

The configuration is layered (later wins): ``config/default.yaml`` < values derived from the file (timeframe, digits / point, fallback spread) <
``config/mql5_loader.yaml`` < the user's ``--config`` < ``config/MT5_TESTER/<SYMBOL>/<TF>.yaml`` (costs / engine) < explicit CLI overrides.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from edge_research.config import DEFAULT_CONFIG_PATH, _deep_merge, symbol_config, validate_config
from edge_research.mql5_loader.coverage import assert_all_processed, audit_header, audit_markdown
from edge_research.mql5_loader.extension import codegen_config
from edge_research.mql5_loader.loader import TIMEFRAME_MINUTES, MQL5Table, infer_digits, load_mql5_export, read_header

LOADER_CONFIG_PATH = DEFAULT_CONFIG_PATH.parent / "mql5_loader.yaml"

# max_gap_hours: a gap larger than this between two bars is a weekend / holiday (a trade window crossing it is excluded when engine.no_weekend_hold)
# expected_bars_per_year: 24 x 5 trading days at the timeframe (H1 ~ 6000, the default.yaml reference)
_TF_FIXED = {"D1": (96.0, 260), "W1": (240.0, 52), "MN1": (840.0, 12)}


def timeframe_settings(tf: str) -> Dict[str, Any]:
    m = TIMEFRAME_MINUTES[tf]
    max_gap, per_year = _TF_FIXED[tf] if tf in _TF_FIXED else (max(6.0, 1.5 * m / 60.0), int(round(6000 * 60 / m)))
    return {"data": {"timeframe": tf, "timeframe_minutes": m, "max_gap_hours": float(max_gap)}, "qa": {"expected_bars_per_year": int(per_year)}}


def derive_settings(table: MQL5Table, csv_path: str | Path, digits: Optional[int] = None, timeframe: Optional[str] = None) -> Dict[str, Any]:
    """Config values read from the file: timeframe, price digits / point, and a data-driven fallback spread."""
    tf = timeframe or table.timeframe
    if tf not in TIMEFRAME_MINUTES:
        raise ValueError(f"timeframe not recognised (file name {Path(csv_path).name!r}, spacing of the bars): pass it explicitly, one of {sorted(TIMEFRAME_MINUTES)}")
    out = timeframe_settings(tf)
    dg = digits if digits is not None else infer_digits(csv_path)
    if dg is not None:
        if digits is None and dg < 5 and float(table.df["close"].median()) < 20.0 and table.symbol and not table.symbol.upper().endswith(("JPY", "XAU", "XAG")):
            dg = max(dg, 5)                    # a short file can miss the 5th decimal (e.g. 14 daily bars); majors are quoted with 5 digits
        out["costs"] = {"digits": int(dg), "point": float(10.0 ** (-dg))}
    if table.spread is not None:
        pos = table.spread[table.spread > 0]
        if len(pos) >= 20 and len(pos) >= 0.05 * len(table.spread):
            out.setdefault("costs", {})["fixed_spread_points"] = float(np.median(pos))        # used only for bars whose history spread is 0 and before the first known one
    return out


def build_config(table: MQL5Table, csv_path: str | Path, user_config: Optional[str | Path] = None, symbol: Optional[str] = None, digits: Optional[int] = None,
                 timeframe: Optional[str] = None, gate_news: bool = False, allow_unprocessed: bool = False, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = _deep_merge(cfg, derive_settings(table, csv_path, digits, timeframe))
    with open(LOADER_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg = _deep_merge(cfg, codegen_config())                                       # exporter indicator handles for the EA generator
    if user_config:
        with open(user_config, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    sym, tf = symbol or table.symbol or "SYMBOL", cfg["data"]["timeframe"]
    cfg = symbol_config(cfg, sym, tf)
    cfg = _deep_merge(cfg, {"backtest": {"symbol": str(sym).upper()}})              # label of the chart / report; per-pair values come from config/MT5_TESTER/<SYMBOL>/<TF>.yaml
    if gate_news:
        cfg["features"]["include_calendar_news"] = False
    if allow_unprocessed:
        cfg["data"]["allow_unprocessed"] = True
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def backtest_outputs(run_dir: str | Path) -> Dict[str, Any]:
    """What the backtest stage left in a run: per segment the equity + balance PNG and the report (None when stage 11 produced nothing, e.g. no accepted rule)."""
    bt = Path(run_dir) / "11_backtest"
    out: Dict[str, Any] = {}
    for seg in sorted(d for d in bt.iterdir() if d.is_dir()) if bt.is_dir() else []:
        pngs = sorted(seg.glob("*.png"))
        out[seg.name] = {"chart": str(pngs[0]) if pngs else None, "report": str(seg / "report.md") if (seg / "report.md").exists() else None}
    return out or {"status": "no backtest output (stage 11 did not run or no rule was accepted)"}


def audit_file(csv_path: str | Path, cfg: Optional[Dict[str, Any]] = None, user_config: Optional[str | Path] = None, **kw: Any) -> Dict[str, Any]:
    """Header-only audit (plus a full load for the integrity findings): which columns are processed, which are not."""
    table = load_mql5_export(csv_path)
    cfg = cfg or build_config(table, csv_path, user_config, **kw)
    audit = audit_header(["time"] + list(table.df.columns), cfg)
    return {"table": table, "cfg": cfg, "audit": audit, "markdown": audit_markdown(audit)}


class _LogOnce(logging.Filter):
    """Drops a log record whose level and text were already seen: the exporter CSV is read twice per run (coverage audit, then stage 0 ingest), which printed every NaN warning twice."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set = set()

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.levelno, record.getMessage())
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


def run_pipeline(csv_path: str | Path, out_dir: str | Path, through: int = 2, **kw: Any) -> Dict[str, Any]:
    """Run stages 0..``through`` on ONE exporter CSV (see :func:`_run_pipeline`); repeated ingest warnings are printed once."""
    lg = logging.getLogger("edge_research.ingestion.loader")
    once = _LogOnce()
    lg.addFilter(once)
    try:
        return _run_pipeline(csv_path, out_dir, through, **kw)
    finally:
        lg.removeFilter(once)


def _run_pipeline(csv_path: str | Path, out_dir: str | Path, through: int = 2, user_config: Optional[str | Path] = None, symbol: Optional[str] = None, digits: Optional[int] = None,
                 timeframe: Optional[str] = None, gate_news: bool = False, allow_unprocessed: bool = False, max_replicates: Optional[int] = None) -> Dict[str, Any]:
    """Run stages 0..``through`` on ONE exporter CSV. ``through`` is the number of the last stage function (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13)."""
    from edge_research import pipeline as P

    fn = getattr(P, f"run_stages_0_{through}", None)
    if fn is None:
        raise ValueError(f"no run_stages_0_{through} in the pipeline")
    a = audit_file(csv_path, None, user_config, symbol=symbol, digits=digits, timeframe=timeframe, gate_news=gate_news, allow_unprocessed=allow_unprocessed)
    table, cfg, audit = a["table"], a["cfg"], a["audit"]
    if not cfg["data"].get("allow_unprocessed", False):
        assert_all_processed(audit)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    sym = symbol or table.symbol or "SYMBOL"
    kwargs: Dict[str, Any] = {}
    if through >= 9 and max_replicates is not None:
        kwargs["max_replicates_this_call"] = max_replicates
    result = fn(csv_path, out, cfg, sym, None, **kwargs)                              # spread_csv=None: the spread comes from the data table
    pair_cfg = Path(DEFAULT_CONFIG_PATH.parent / "MT5_TESTER" / str(sym).upper() / f"{cfg['data']['timeframe']}.yaml")
    summary = {"symbol": sym, "timeframe": cfg["data"]["timeframe"], "csv": str(csv_path), "rows": int(len(table.df)), "repairs": table.repairs,
               "pair_config": str(pair_cfg) if pair_cfg.exists() else None, "backtest_outputs": backtest_outputs(out) if through >= 11 else None,
               "spread_source": "spread column of the same csv", "findings": [f"{f.level}: {f.check}: {f.message}" for f in table.findings if f.level in ("warn", "fail")],
               "pipeline": result}
    (out / "mql5_loader_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
