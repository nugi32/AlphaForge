"""Features for the exporter columns that the stock registry never reads (found by the coverage audit).

Stock registry (``features/registry.py``) turns most exporter columns into stationary features, but these columns were loaded and then ignored:

====================  ==============================================================================================================
column                feature(s) added here
====================  ==============================================================================================================
spread                ``spread_atr`` (spread cost relative to the bar scale), ``spread_rel`` (spread relative to its rolling median)
real_volume           ``real_volume_rel`` (relative to its rolling median; constant 0 for FX, then dropped by QA)
atr_20                ``atr_ratio_14_20`` (ATR term structure: fast / slow volatility)
ama_10_2_30           ``dist_ama_10_2_30`` (distance of the close to the adaptive MA in ATR units; the stock regex skips its two-part parameters)
bb_20_*_middle        ``bb_mid_dist_bb_20_*`` (distance of the close to the band's own middle line in ATR units)
====================  ==============================================================================================================

The envelope columns are handled by the stock registry (``env_pctb_*`` / ``env_width_*``) once the loader has repaired them.

``extension_specs`` is called from ``build_registry`` when ``cfg['data']['source'] == 'mql5_loader'``; a feature whose source column is missing or excluded
by QA is skipped by the registry's own ``add`` exactly like every other feature.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from edge_research.features.registry import Ctx, FeatureSpec


def _rolling_median_ratio(s: pd.Series, window: int) -> pd.Series:
    return s / s.rolling(window).median().replace(0, np.nan)


def extension_specs(columns: Sequence[str], cfg: Dict[str, Any]) -> List[FeatureSpec]:
    f = cfg["features"]
    rv = int(f["roll_window_volume"])
    have = set(columns)
    specs: List[FeatureSpec] = []

    if "spread" in have:
        def _spread(x: Ctx) -> pd.Series:
            s = x.col("spread")
            return s.where(s > 0)                                       # 0 = no spread in the broker history: unknown, not "free"
        specs.append(FeatureSpec(
            "spread_atr", "cost", ["spread"], "spread[points] * point / ATR (NaN when the history spread is 0)", "iSpread(shift 1) * _Point / iATR shift 1",
            lambda x: _spread(x) * float(x.cfg["costs"]["point"]) / x.atr, family="spread", complexity=3,
            notes="liquidity / cost regime; the export has history spreads only where the broker stored them"))
        specs.append(FeatureSpec(
            "spread_rel", "cost", ["spread"], f"spread / rolling median({rv}) of spread", "needs rolling median in EA",
            lambda x: _rolling_median_ratio(_spread(x), int(x.cfg["features"]["roll_window_volume"])), warmup_bars=rv, family="spread", complexity=4,
            notes="spread spikes (rollover, news) relative to the normal spread"))
    if "real_volume" in have:
        specs.append(FeatureSpec(
            "real_volume_rel", "vol_ratio", ["real_volume"], f"real_volume / rolling median({rv})", "needs rolling median in EA",
            lambda x: _rolling_median_ratio(x.col("real_volume").where(x.col("real_volume") > 0), int(x.cfg["features"]["roll_window_volume"])),
            warmup_bars=rv, family="volume", complexity=4, notes="exchange volume; always 0 for spot FX, then removed by QA as constant"))
    if "atr_20" in have and "atr_14" in have:
        specs.append(FeatureSpec(
            "atr_ratio_14_20", "vol_ratio", ["atr_14", "atr_20"], "atr_14 / atr_20", "iATR(14) shift 1 / iATR(20) shift 1",
            lambda x: x.col("atr_14") / x.col("atr_20").where(x.col("atr_20") > 0), family="volatility", complexity=2,
            notes="volatility term structure: > 1 = short-term volatility above the slower measure"))
    if "ama_10_2_30" in have:
        specs.append(FeatureSpec(
            "dist_ama_10_2_30", "price_dist_atr", ["ama_10_2_30", "close"], "(close - ama_10_2_30) / ATR", "(iClose shift1 - iAMA(10,2,30) shift1) / iATR shift1",
            lambda x: (x.col("close") - x.col("ama_10_2_30")) / x.atr, family="madist_ama_10_2_30", complexity=2))
    for mid in sorted(c for c in columns if re.fullmatch(r"bb_\d+_\d+_middle", c)):
        base = mid[: -len("_middle")]
        specs.append(FeatureSpec(
            f"bb_mid_dist_{base}", "price_dist_atr", [mid, "close"], "(close - middle band) / ATR", f"(iClose shift1 - iBands {base} buffer 0 shift1) / iATR shift1",
            lambda x, mid=mid: (x.col("close") - x.col(mid)) / x.atr, family="bb_mid", complexity=2,
            notes="equals dist_ma_20_sma when the band uses SMA(20) on close (the loader integrity check verifies it); clustering keeps one of the two"))
    return specs


def build_extended_registry(columns: Sequence[str], cfg: Dict[str, Any], exclude_sources: set | None = None) -> Tuple[List[FeatureSpec], List[Tuple[str, str]]]:
    """The registry the pipeline uses in MQL5 loader mode (stock features + the features above)."""
    from edge_research.config import _deep_merge
    from edge_research.features.registry import build_registry

    cfg2 = _deep_merge(cfg, {"data": {"source": "mql5_loader"}})
    return build_registry(list(columns), cfg2, exclude_sources)
