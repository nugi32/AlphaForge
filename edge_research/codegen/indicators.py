"""Source column -> MQL5 indicator handle (design 16.2: the EA must create each indicator with EXACTLY the parameters of ``dataExporter.mq5``).

The original ``dataExporter.mq5`` is not part of this project, so every default below is an ASSUMPTION derived from the column name and the
MT5 default applied price. They are all ``verified=False``: the generated EA lists them in its header and the review checklist asks to compare each
line with the exporter. A column whose parameters cannot be read from its name (``sar_*``, ``frama_*``, custom indicators) has NO default: the
generator refuses unless ``codegen.indicators`` in the config gives the creation call, e.g.
``{"sar_002_02": {"create": "iSAR(_Symbol,TF,0.02,0.2)", "buffer": 0}}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from edge_research.codegen import CodegenRefusal

PRICE_COLUMNS = {"open": "O", "high": "H", "low": "L", "close": "C"}
MA_MODES = {"sma": "MODE_SMA", "ema": "MODE_EMA", "smma": "MODE_SMMA", "lwma": "MODE_LWMA"}


@dataclass(frozen=True)
class IndicatorSpec:
    column: str
    create: str            # MQL5 expression returning an int handle (``TF`` is the EA timeframe constant)
    buffer: int
    verified: bool = False
    source: str = "assumed from the column name"


def _dev(s: str) -> str:
    """Band deviation from a column-name token: '2' -> 2.0, '2p5' -> 2.5, and the exporter's two-digit form '15' -> 1.5, '25' -> 2.5 (tenths, no separator)."""
    if "p" in s:
        return repr(float(s.replace("p", ".")))
    return repr(float(s) / 10.0 if len(s) > 1 else float(s))


# (regex, builder(match) -> (create, buffer)); the first matching pattern wins
_PATTERNS: List[Tuple[str, Callable[[Any], Tuple[str, int]]]] = [
    (r"^rsi_(\d+)$", lambda m: (f"iRSI(_Symbol,TF,{m[1]},PRICE_CLOSE)", 0)),
    (r"^cci_(\d+)$", lambda m: (f"iCCI(_Symbol,TF,{m[1]},PRICE_TYPICAL)", 0)),
    (r"^atr_(\d+)$", lambda m: (f"iATR(_Symbol,TF,{m[1]})", 0)),
    (r"^ma_(\d+)_(sma|ema|smma|lwma)$", lambda m: (f"iMA(_Symbol,TF,{m[1]},0,{MA_MODES[m[2]]},PRICE_CLOSE)", 0)),
    (r"^dema_(\d+)$", lambda m: (f"iDEMA(_Symbol,TF,{m[1]},0,PRICE_CLOSE)", 0)),
    (r"^tema_(\d+)$", lambda m: (f"iTEMA(_Symbol,TF,{m[1]},0,PRICE_CLOSE)", 0)),
    (r"^mom_(\d+)$", lambda m: (f"iMomentum(_Symbol,TF,{m[1]},PRICE_CLOSE)", 0)),
    (r"^stoch_(\d+)_(\d+)_(\d+)_(k|d)$", lambda m: (f"iStochastic(_Symbol,TF,{m[1]},{m[2]},{m[3]},MODE_SMA,STO_LOWHIGH)", 0 if m[4] == "k" else 1)),
    (r"^wpr_(\d+)$", lambda m: (f"iWPR(_Symbol,TF,{m[1]})", 0)),
    (r"^adx_(\d+)$", lambda m: (f"iADX(_Symbol,TF,{m[1]})", 0)),
    (r"^adxwilder_(\d+)$", lambda m: (f"iADXWilder(_Symbol,TF,{m[1]})", 0)),
    (r"^demarker_(\d+)$", lambda m: (f"iDeMarker(_Symbol,TF,{m[1]})", 0)),
    (r"^mfi_(\d+)$", lambda m: (f"iMFI(_Symbol,TF,{m[1]},VOLUME_TICK)", 0)),
    (r"^rvi_(\d+)$", lambda m: (f"iRVI(_Symbol,TF,{m[1]})", 0)),
    (r"^trix_(\d+)$", lambda m: (f"iTriX(_Symbol,TF,{m[1]},PRICE_CLOSE)", 0)),
    (r"^stddev_(\d+)$", lambda m: (f"iStdDev(_Symbol,TF,{m[1]},0,MODE_SMA,PRICE_CLOSE)", 0)),
    (r"^bb_(\d+)_(\d+(?:p\d+)?)_(upper|middle|lower)$", lambda m: (f"iBands(_Symbol,TF,{m[1]},0,{_dev(m[2])},PRICE_CLOSE)", {"middle": 0, "upper": 1, "lower": 2}[m[3]])),
    (r"^macd_(\d+)_(\d+)_(\d+)_(main|signal)$", lambda m: (f"iMACD(_Symbol,TF,{m[1]},{m[2]},{m[3]},PRICE_CLOSE)", 0 if m[4] == "main" else 1)),
    (r"^osma_(\d+)_(\d+)_(\d+)$", lambda m: (f"iOsMA(_Symbol,TF,{m[1]},{m[2]},{m[3]},PRICE_CLOSE)", 0)),
    (r"^ao$", lambda m: ("iAO(_Symbol,TF)", 0)),
    (r"^ac$", lambda m: ("iAC(_Symbol,TF)", 0)),
    (r"^bears_(\d+)$", lambda m: (f"iBearsPower(_Symbol,TF,{m[1]})", 0)),
    (r"^bulls_(\d+)$", lambda m: (f"iBullsPower(_Symbol,TF,{m[1]})", 0)),
]
_COMPILED = [(re.compile(p), fn) for p, fn in _PATTERNS]
_CREATE_RE = re.compile(r"^i[A-Za-z0-9]+\(")
_FORBIDDEN_CHARS = set(';{}"\'\\\n\r')


def _is_single_call(text: str) -> bool:
    """One indicator call and nothing else: the parenthesis opened first closes at the very end, no statement separators, quotes, braces or comments."""
    if not _CREATE_RE.match(text) or any(c in _FORBIDDEN_CHARS for c in text) or "//" in text or "/*" in text:
        return False
    depth, opened = 0, False
    for k, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        opened = opened or ch == "("
        if depth < 0 or (opened and depth == 0 and k < len(text) - 1):
            return False
    return opened and depth == 0


def indicator_for(column: str, overrides: Optional[Mapping[str, Any]] = None) -> IndicatorSpec:
    """The handle of a source column. Raises ``CodegenRefusal`` when neither an override nor a name pattern defines it."""
    overrides = overrides or {}
    if column in overrides:
        o = overrides[column]
        if not isinstance(o, Mapping) or set(o) - {"create", "buffer", "verified"} or "create" not in o:
            raise CodegenRefusal(f"codegen.indicators[{column!r}] must be {{create: 'iXxx(...)', buffer: int}}")
        if not _is_single_call(str(o["create"]).strip()):
            raise CodegenRefusal(f"codegen.indicators[{column!r}].create must be one indicator call like iSAR(_Symbol,TF,0.02,0.2)")
        return IndicatorSpec(column, str(o["create"]).strip(), int(o.get("buffer", 0)), bool(o.get("verified", False)), "config override")
    for rx, fn in _COMPILED:
        m = rx.match(column)
        if m:
            create, buf = fn(m)
            return IndicatorSpec(column, create, buf)
    raise CodegenRefusal(f"no MQL5 recipe for the source column {column!r}: its indicator parameters cannot be read from the name; give codegen.indicators[{column!r}] in the config")
