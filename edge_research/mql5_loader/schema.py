"""Column manifest of ``mql5/dataExporter.mq5`` (v1.03): the single source of truth for what the exporter writes.

The manifest mirrors ``InitializeCSV`` of the exporter column by column, in the same order. For every indicator column it records the MQL5 call that
creates the handle and the buffer that holds the value, so that:

* the loader can check a CSV header against the exporter (missing / unknown / reordered columns),
* the coverage audit can prove that every exported column is read by at least one feature,
* the code generator creates each handle with EXACTLY the parameters of the exporter (``codegen_overrides``),
* a test can parse ``dataExporter.mq5`` and verify that this manifest still matches the MQL5 source.

Two buffer numbers are kept per column. ``exporter_buffer`` is what the exporter READS today; ``buffer`` is what the indicator really holds. They differ
only for the ``envelope_*`` columns: ``iEnvelopes`` has buffer 0 = upper and 1 = lower (there is no buffer 2), but the exporter reads buffers 1 and 2 and
labels them upper and lower. The CSV column ``envelope_20_01_upper`` therefore holds the LOWER band and ``envelope_20_01_lower`` is always NaN. The loader
repairs this (see ``loader.repair_envelope``); fixing the MQL5 side as well is recommended (see ``ENVELOPE_MQL5_FIX``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

EXPORTER_VERSION = "1.03"

MA_PERIODS = (5, 10, 20, 50, 100, 200)
MA_MODES = (("sma", "MODE_SMA"), ("ema", "MODE_EMA"), ("smma", "MODE_SMMA"), ("lwma", "MODE_LWMA"))

ENVELOPE_MQL5_FIX = (
    "In GetVolatilityValues() read iEnvelopes buffers 0 (upper) and 1 (lower) instead of 1 and 2: "
    "GetIndicatorValue(g_handles.envelope_20_01, bar_index, value, 0) for the upper band and "
    "GetIndicatorValue(g_handles.envelope_20_01, bar_index, value, 1) for the lower band."
)


@dataclass(frozen=True)
class ExporterColumn:
    name: str
    group: str                                   # bar | ma | adaptive_ma | trend | oscillator | volatility | volume | custom_volregime | custom_pricestructure | news
    create: Optional[str] = None                 # MQL5 handle expression (``TF`` = EA timeframe); None for bar data and iCustom columns
    buffer: Optional[int] = None                 # buffer that really holds the value
    exporter_buffer: Optional[int] = None        # buffer the exporter reads (differs from ``buffer`` only for the envelope columns)
    note: str = ""


def _build() -> List[ExporterColumn]:
    cols: List[ExporterColumn] = []

    def add(name: str, group: str, create: Optional[str] = None, buffer: Optional[int] = None, exporter_buffer: Optional[int] = None, note: str = "") -> None:
        if create is not None and buffer is None:
            buffer = 0
        if exporter_buffer is None:
            exporter_buffer = buffer
        cols.append(ExporterColumn(name, group, create, buffer, exporter_buffer, note))

    add("time", "bar")
    for c in ("open", "high", "low", "close", "tick_volume", "real_volume"):
        add(c, "bar")
    add("spread", "bar", note="iSpread in points of the closed bar; 0 = the broker history has no spread for that bar")

    for n in MA_PERIODS:
        for mode, const in MA_MODES:
            add(f"ma_{n}_{mode}", "ma", f"iMA(_Symbol,TF,{n},0,{const},PRICE_CLOSE)")

    for n in (10, 20, 50):
        add(f"dema_{n}", "adaptive_ma", f"iDEMA(_Symbol,TF,{n},0,PRICE_CLOSE)")
    for n in (10, 20, 50):
        add(f"tema_{n}", "adaptive_ma", f"iTEMA(_Symbol,TF,{n},0,PRICE_CLOSE)")
    for n in (14, 20):
        add(f"frama_{n}", "adaptive_ma", f"iFrAMA(_Symbol,TF,{n},0,PRICE_CLOSE)")
    add("ama_10_2_30", "adaptive_ma", "iAMA(_Symbol,TF,10,2,30,0,PRICE_CLOSE)")
    for n in (9, 12):
        add(f"vidya_{n}", "adaptive_ma", f"iVIDyA(_Symbol,TF,{n},12,0,PRICE_CLOSE)")

    for tag, step in (("002", "0.02"), ("001", "0.01"), ("003", "0.03")):
        add(f"sar_{tag}_02", "trend", f"iSAR(_Symbol,TF,{step},0.2)")
    add("adx_14", "trend", "iADX(_Symbol,TF,14)")
    add("adxwilder_14", "trend", "iADXWilder(_Symbol,TF,14)")

    for n in (7, 14, 21):
        add(f"rsi_{n}", "oscillator", f"iRSI(_Symbol,TF,{n},PRICE_CLOSE)")
    for k in (5, 14):
        sto = f"iStochastic(_Symbol,TF,{k},3,3,MODE_SMA,STO_LOWHIGH)"
        add(f"stoch_{k}_3_3_k", "oscillator", sto, 0)
        add(f"stoch_{k}_3_3_d", "oscillator", sto, 1)
    for n in (14, 20):
        add(f"cci_{n}", "oscillator", f"iCCI(_Symbol,TF,{n},PRICE_TYPICAL)")
    add("wpr_14", "oscillator", "iWPR(_Symbol,TF,14)")
    for n in (10, 14):
        add(f"mom_{n}", "oscillator", f"iMomentum(_Symbol,TF,{n},PRICE_CLOSE)")
    macd = "iMACD(_Symbol,TF,12,26,9,PRICE_CLOSE)"
    add("macd_12_26_9_main", "oscillator", macd, 0)
    add("macd_12_26_9_signal", "oscillator", macd, 1)
    add("osma_12_26_9", "oscillator", "iOsMA(_Symbol,TF,12,26,9,PRICE_CLOSE)")
    add("demarker_14", "oscillator", "iDeMarker(_Symbol,TF,14)")
    add("rvi_10", "oscillator", "iRVI(_Symbol,TF,10)")
    add("trix_14", "oscillator", "iTriX(_Symbol,TF,14,PRICE_CLOSE)")
    add("ac", "oscillator", "iAC(_Symbol,TF)")
    add("ao", "oscillator", "iAO(_Symbol,TF)")
    add("bears_13", "oscillator", "iBearsPower(_Symbol,TF,13)")
    add("bulls_13", "oscillator", "iBullsPower(_Symbol,TF,13)")

    for n in (14, 20):
        add(f"atr_{n}", "volatility", f"iATR(_Symbol,TF,{n})")
    for tag, dev in (("2", "2.0"), ("15", "1.5"), ("25", "2.5")):
        bb = f"iBands(_Symbol,TF,20,0,{dev},PRICE_CLOSE)"
        add(f"bb_20_{tag}_upper", "volatility", bb, 1)
        add(f"bb_20_{tag}_middle", "volatility", bb, 0)
        add(f"bb_20_{tag}_lower", "volatility", bb, 2)
    add("stddev_20", "volatility", "iStdDev(_Symbol,TF,20,0,MODE_SMA,PRICE_CLOSE)")
    env = "iEnvelopes(_Symbol,TF,20,0,MODE_SMA,PRICE_CLOSE,0.1)"
    add("envelope_20_01_upper", "volatility", env, 0, 1, "exporter reads buffer 1 = the LOWER band (bug); repaired by the loader")
    add("envelope_20_01_lower", "volatility", env, 1, 2, "exporter reads buffer 2 (does not exist) = always NaN; rebuilt by the loader")

    add("obv", "volume", "iOBV(_Symbol,TF,VOLUME_TICK)")
    add("ad", "volume", "iAD(_Symbol,TF,VOLUME_TICK)")
    add("mfi_14", "volume", "iMFI(_Symbol,TF,14,VOLUME_TICK)")
    add("force_13", "volume", "iForce(_Symbol,TF,13,MODE_SMA,VOLUME_TICK)")
    add("chaikin_3_10", "volume", "iChaikin(_Symbol,TF,3,10,MODE_EMA,VOLUME_TICK)")
    add("volumes", "volume", note="iVolume (tick volume) of the closed bar, written a second time next to the volume indicators")

    for name, buf in (("volregime_atr_pctl", 0), ("volregime_atr_ratio", 5), ("volregime_regime", 6)):
        add(name, "custom_volregime", None, buf, note="iCustom VolRegime_ATR")
    for i, name in enumerate(("ps_dist_pdh_atr", "ps_dist_pdl_atr", "ps_dist_sess_h_atr", "ps_dist_sess_l_atr",
                              "ps_dist_round_up_atr", "ps_dist_round_dn_atr", "ps_gap_atr", "ps_gap_fill_pct")):
        add(name, "custom_pricestructure", None, i, note="iCustom PriceStructure")

    for name in ("news_min_to_next", "news_next_importance", "news_min_since_last", "news_last_importance", "news_last_surprise", "news_in_window", "news_count_next_24h"):
        add(name, "news", note="economic calendar CSV joined inside the tester (InpAddNewsColumns)")
    return cols


EXPORTER_COLUMNS: List[ExporterColumn] = _build()
EXPORTER_BY_NAME: Dict[str, ExporterColumn] = {c.name: c for c in EXPORTER_COLUMNS}
EXPORTER_HEADER: List[str] = [c.name for c in EXPORTER_COLUMNS]
NEWS_COLUMNS: List[str] = [c.name for c in EXPORTER_COLUMNS if c.group == "news"]
ENVELOPE_COLUMNS: Tuple[str, str] = ("envelope_20_01_upper", "envelope_20_01_lower")

# Columns that exist in older exports (before v1.03 added ``spread`` / v1.02 added ``news_*``) and may legitimately be absent.
OPTIONAL_COLUMNS = frozenset(["spread"] + NEWS_COLUMNS + [c.name for c in EXPORTER_COLUMNS if c.group.startswith("custom_")])


def codegen_overrides() -> Dict[str, Dict[str, object]]:
    """``codegen.indicators`` entries for every indicator column, transcribed from the exporter (``verified``: the source is the exporter itself)."""
    return {c.name: {"create": c.create, "buffer": int(c.buffer), "verified": True}
            for c in EXPORTER_COLUMNS if c.create is not None}


def header_report(columns: List[str]) -> Dict[str, List[str]]:
    """Compare a CSV header with the exporter manifest."""
    cols = [str(c).replace('"', "").strip() for c in columns]
    have, want = set(cols), set(EXPORTER_HEADER)
    return {
        "missing": [c for c in EXPORTER_HEADER if c not in have],
        "missing_required": [c for c in EXPORTER_HEADER if c not in have and c not in OPTIONAL_COLUMNS],
        "unlisted": [c for c in cols if c not in want],
        "duplicated": sorted({c for c in cols if cols.count(c) > 1}),
        "reordered": [] if [c for c in cols if c in want] == [c for c in EXPORTER_HEADER if c in have] else ["column order differs from the exporter"],
    }
