"""Feature -> MQL5 expression (design 16.1 / 16.3). One expression per registry feature, written over a small vocabulary of primitives:

    O(s) H(s) L(s) C(s)      bar prices at shift s (shift 1 = the last CLOSED bar = the decision bar of the Python table row)
    IND_<column>(s)          buffer of the exported indicator ``column`` at shift s
    ATRN(s)                  ATR at shift s, NaN unless > 0 (the Python table divides by NaN for a non-positive ATR)
    DIV(a, b)                a / b, NaN when b == 0 or an input is NaN (MQL5 stops the EA on a real division by zero)
    MAX2 MIN2                NaN-propagating max / min
    HHV(n, s) LLV(n, s)      highest high / lowest low of the n bars at shifts s .. s+n-1 (NaN when the history is too short)
    HOURV(s) DOWV(s) SESSIONV(s)   server hour / day of week (Mon = 0) / session code of the bar at shift s
    BRKUP(n) BRKDN(n)        breakout flags of the closed bar against the n bars before it

The expression text is valid MQL5 AND valid Python: ``pyeval`` evaluates the very same string on the bar table, which is how the recipes are
checked against the Python feature table (tests/test_codegen_m10.py). A feature without an exact translation raises ``CodegenRefusal`` with the reason
(``REFUSED_KINDS``); it is never approximated (design 16.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from edge_research.codegen import CodegenRefusal
from edge_research.codegen.indicators import PRICE_COLUMNS
from edge_research.features.registry import BOUNDED_OSC, UNBOUNDED_OSC, FeatureSpec

A1 = "ATRN(1)"


@dataclass(frozen=True)
class Recipe:
    name: str
    expr: str
    sources: Tuple[str, ...]          # exported indicator columns read through IND_<column> (price columns are primitives)
    helpers: Tuple[str, ...]          # preamble helper functions the expression needs (codegen/mql_helpers.py)


def _src(col: str, shift: str) -> str:
    if col in PRICE_COLUMNS:
        return f"{PRICE_COLUMNS[col]}({shift})"
    return f"IND_{col}({shift})"


def _r(name: str, expr: str, cols: List[str], helpers: Tuple[str, ...] = ()) -> Recipe:
    return Recipe(name, expr, tuple(sorted({c for c in cols if c not in PRICE_COLUMNS})), helpers)


# features whose exact translation needs rolling state or a scan that this generator version does not implement
REFUSED_PATTERNS: List[Tuple[str, str]] = [
    (r"^atr_ratio$", "needs a rolling mean of the ATR buffer over roll_window_vol bars (state not implemented in the generator yet)"),
    (r"^force_13_z$", "needs a rolling z-score of a buffer (not implemented in the generator yet)"),
    (r"^(tick_volume|volumes)_rel$", "needs a rolling median of volume (not implemented in the generator yet)"),
    (r"^(obv|ad)_chg6_z$", "needs a rolling std of 6-bar changes (not implemented in the generator yet)"),
    (r"^chaikin_z$", "needs a rolling z-score of a buffer (not implemented in the generator yet)"),
    (r"^bars_since_", "a bars-since event scan over an indicator buffer (not implemented in the generator yet)"),
    (r"^volregime_", "custom indicator (iCustom): binary name, inputs and buffer must be given by the user and the bar timing verified (design 16.6)"),
    (r"^ps_", "custom indicator (iCustom): binary name, inputs and buffer must be given by the user and the bar timing verified (design 16.6)"),
    (r"^(cal_|news_)", "calendar / news (live_only): the calendar API does not work in the Strategy Tester (design 16.7)"),
]


def refusal_reason(name: str) -> Optional[str]:
    for rx, why in REFUSED_PATTERNS:
        if re.match(rx, name):
            return why
    return None


def recipe_for(spec: FeatureSpec, cfg: Dict[str, Any]) -> Recipe:
    """Exact MQL5 expression of a registry feature, or ``CodegenRefusal``."""
    n = spec.name
    f = cfg["features"]
    if not spec.reproducible_in_tester:
        raise CodegenRefusal(f"feature {n}: not reproducible in the Strategy Tester (live_only)")
    why = refusal_reason(n)
    if why:
        raise CodegenRefusal(f"feature {n}: {why}")
    if n in BOUNDED_OSC or n in UNBOUNDED_OSC:
        return _r(n, f"IND_{n}(1)", [n])
    m = re.fullmatch(r"(mom_\d+)_c", n)
    if m:
        return _r(n, f"IND_{m[1]}(1) - {float(f['mom_center'])!r}", [m[1]])
    m = re.fullmatch(r"dist_((?:ma|dema|tema|frama|ama|vidya)_\d+_?[a-z0-9]*|sar_.+)", n)
    if m:
        return _r(n, f"DIV(C(1) - IND_{m[1]}(1), {A1})", [m[1]])
    m = re.fullmatch(r"rel_(ma_\d+_[a-z]+)_(ma_\d+_[a-z]+)", n)
    if m:
        return _r(n, f"DIV(IND_{m[1]}(1) - IND_{m[2]}(1), {A1})", [m[1], m[2]])
    m = re.fullmatch(r"slope_(ma_\d+_[a-z]+)_(\d+)", n)
    if m:
        k = int(m[2])
        return _r(n, f"DIV(IND_{m[1]}(1) - IND_{m[1]}({1 + k}), {float(k)!r} * {A1})", [m[1]])
    m = re.fullmatch(r"pctb_(bb_.+)", n)
    if m:
        up, lo = m[1] + "_upper", m[1] + "_lower"
        return _r(n, f"DIV(C(1) - IND_{lo}(1), IND_{up}(1) - IND_{lo}(1))", [up, lo])
    m = re.fullmatch(r"bbwidth_(bb_.+)", n)
    if m:
        up, lo = m[1] + "_upper", m[1] + "_lower"
        return _r(n, f"DIV(IND_{up}(1) - IND_{lo}(1), {A1})", [up, lo])
    m = re.fullmatch(r"(macd_12_26_9_main|macd_12_26_9_signal|osma_12_26_9|ao|ac|bears_13|bulls_13)_atr", n)
    if m:
        return _r(n, f"DIV(IND_{m[1]}(1), {A1})", [m[1]])
    if n == "macd_hist_atr":
        return _r(n, f"DIV(IND_macd_12_26_9_main(1) - IND_macd_12_26_9_signal(1), {A1})", ["macd_12_26_9_main", "macd_12_26_9_signal"])
    if n == "stddev_atr":
        return _r(n, f"DIV(IND_stddev_20(1), {A1})", ["stddev_20"])
    candle = {
        "candle_body": f"DIV(C(1) - O(1), {A1})",
        "candle_upper_wick": f"DIV(H(1) - MAX2(O(1), C(1)), {A1})",
        "candle_lower_wick": f"DIV(MIN2(O(1), C(1)) - L(1), {A1})",
        "candle_range": f"DIV(H(1) - L(1), {A1})",
        "candle_close_pos": "DIV(C(1) - L(1), H(1) - L(1))",
        "gap_atr": f"DIV(O(1) - C(2), {A1})",
    }
    if n in candle:
        return _r(n, candle[n], [])
    m = re.fullmatch(r"ret_(\d+)_atr", n)
    if m:
        return _r(n, f"DIV(C(1) - C({1 + int(m[1])}), {A1})", [])
    m = re.fullmatch(r"dist_hh_(\d+)", n)
    if m:
        return _r(n, f"DIV(HHV({m[1]}, 1) - C(1), {A1})", [], ("HHV",))
    m = re.fullmatch(r"dist_ll_(\d+)", n)
    if m:
        return _r(n, f"DIV(C(1) - LLV({m[1]}, 1), {A1})", [], ("LLV",))
    if n == "hour":
        return _r(n, "HOURV(1)", [], ("HOURV",))
    if n == "dayofweek":
        return _r(n, "DOWV(1)", [], ("DOWV",))
    if n == "session":
        return _r(n, "SESSIONV(1)", [], ("HOURV", "SESSIONV"))
    m = re.fullmatch(r"breakout_(up|down)_(\d+)", n)
    if m:
        return _r(n, f"BRK{'UP' if m[1] == 'up' else 'DN'}({m[2]})", [], ("HHV" if m[1] == "up" else "LLV", "BRKUP" if m[1] == "up" else "BRKDN"))
    from edge_research.mql5_loader.extension import extension_recipe          # loader-mode features (envelope, AMA / band-middle distance, ATR ratio, spread)
    ext = extension_recipe(spec, cfg)
    if ext is not None:
        return ext
    raise CodegenRefusal(f"feature {n}: no MQL5 recipe in the generator (kind {spec.kind}); add one to codegen/recipes.py with a parity test")
