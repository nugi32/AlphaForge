"""Build the float32 feature table from the loader table and the registry (Stage 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_research.features.registry import Ctx, FeatureSpec, build_registry


@dataclass
class FeatureResult:
    table: pd.DataFrame                   # float32, same index as the input
    specs: List[FeatureSpec]
    skipped: List[Tuple[str, str]]
    first_valid_row: int                  # rows before this have at least one NaN feature (warm-up)
    observed_warmup: Dict[str, int]       # per feature: leading NaN rows


def build_feature_table(df: pd.DataFrame, cfg: Dict[str, Any], exclude_sources: Optional[set] = None) -> FeatureResult:
    atr_col = cfg["features"]["atr_column"]
    if atr_col not in df.columns:
        raise KeyError(f"ATR column {atr_col!r} not found; it is required to normalise features")
    atr = df[atr_col].astype(np.float64)
    atr = atr.where(atr > 0)                      # non-positive ATR -> NaN, never divided by
    ctx = Ctx(df=df, atr=atr, cfg=cfg)
    specs, skipped = build_registry(list(df.columns), cfg, exclude_sources)
    cols: Dict[str, np.ndarray] = {}
    for spec in specs:
        vals = np.array(spec.fn(ctx).to_numpy(np.float64), dtype=np.float64, copy=True)
        vals[~np.isfinite(vals)] = np.nan          # inf (division by ~0) -> NaN
        cols[spec.name] = vals.astype(np.float32)
    table = pd.DataFrame(cols, index=df.index)
    observed = {c: int(np.argmax(table[c].notna().to_numpy())) if table[c].notna().any() else len(table) for c in table.columns}
    first_valid = int(np.argmax(table.notna().all(axis=1).to_numpy())) if table.notna().all(axis=1).any() else len(table)
    return FeatureResult(table, specs, skipped, first_valid, observed)


def write_features(res: FeatureResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_parquet(out / "feature_table.parquet", compression="zstd")
    reg = [dict(s.to_json(), observed_warmup=res.observed_warmup.get(s.name)) for s in res.specs]
    (out / "feature_registry.json").write_text(json.dumps({"features": reg, "skipped": res.skipped, "first_valid_row": res.first_valid_row}, indent=2), encoding="utf-8")
