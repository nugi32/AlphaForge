"""V2 configuration loading: one YAML file, deep-merged over config/default.yaml."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load default config, then merge an optional user YAML file, then optional dict overrides."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    d = cfg["data"]
    total = d["train_pct"] + d["val_pct"] + d["holdout_pct"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train/val/holdout fractions must sum to 1.0, got {total}")
    if cfg["labels"]["atr_column"] != cfg["features"]["atr_column"]:
        raise ValueError("labels.atr_column and features.atr_column must match (same ATR for scale and distances)")
    fidelity = cfg.get("engine", {}).get("fidelity", "open_prices_only")
    if fidelity != "open_prices_only":
        raise ValueError(f"engine.fidelity must be 'open_prices_only' (the pipeline and the parity contract are built on it), got {fidelity!r}")
    validate_backtest_section(cfg.get("backtest", {}) or {})


BACKTEST_KEYS = {"segments", "account_currency", "symbol", "chart"}
BACKTEST_SEGMENTS = {"train", "val", "all"}          # the holdout is opened only by the validation stage


def validate_backtest_section(bt: Dict[str, Any]) -> None:
    """``backtest:`` section (what a backtest run writes and over which segments). An unknown key is an error (typos)."""
    unknown = sorted(set(bt) - BACKTEST_KEYS)
    if unknown:
        raise ValueError(f"unknown backtest settings: {unknown}")
    segs = bt.get("segments")
    if segs is not None:
        if not isinstance(segs, (list, tuple)) or not segs or set(segs) - BACKTEST_SEGMENTS:
            raise ValueError(f"backtest.segments must be a non-empty list drawn from {sorted(BACKTEST_SEGMENTS)} (the holdout is opened only by the validation stage), got {segs!r}")
    from edge_research.mt5_backtest.plots import chart_settings
    chart_settings({"backtest": bt})                 # validates the chart keys


def tester_overrides(symbol: str, timeframe: str, base_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Per-symbol / per-timeframe tester overrides from ``config/MT5_TESTER/<SYMBOL>/<TF>.yaml`` (design 18: one global
    ``costs:`` block is not enough for real markets). The file may set ``costs:`` and ``engine:`` keys only; a missing file
    means no override. Deep-merged over the global config by :func:`symbol_config`."""
    root = (Path(base_dir) / "MT5_TESTER") if base_dir is not None else DEFAULT_CONFIG_PATH.parent / "MT5_TESTER"
    path = root / str(symbol).upper() / f"{str(timeframe).upper()}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        over = yaml.safe_load(fh) or {}
    bad = set(over) - {"costs", "engine", "backtest"}
    if bad:
        raise ValueError(f"{path}: only 'costs', 'engine' and 'backtest' sections may be overridden per symbol/TF, got {sorted(bad)}")
    return over


def symbol_config(cfg: Dict[str, Any], symbol: str, timeframe: str, base_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """The global config with the ``config/MT5_TESTER/<SYMBOL>/<TF>.yaml`` overrides (costs / engine) merged in.
    ``base_dir`` relocates the MT5_TESTER tree (tests, scripts pointing at another checkout)."""
    over = tester_overrides(symbol, timeframe, base_dir)
    return _deep_merge(cfg, over) if over else cfg


def config_hash(cfg: Dict[str, Any], section: Optional[str] = None) -> str:
    """Stable short hash of the whole config or of one section (used for stage fingerprints)."""
    payload = cfg if section is None else cfg[section]
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
