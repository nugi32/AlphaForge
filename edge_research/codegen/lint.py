"""Static sanity checks of the generated MQL5 text (there is no MQL5 compiler in the sandbox: MetaEditor must still compile the file).

Catches what a generator can get wrong without a compiler: unbalanced brackets, unfilled template fields, non-ASCII text, a helper / indicator getter that is called but not
defined (or defined after its first use: MQL5 needs declarations before use), duplicate definitions.
"""

from __future__ import annotations

import re
from typing import Dict, List

_KNOWN_FUNCS = {  # MQL5 / CTrade names the template uses (anything else that looks like a call must be defined in the file)
    "iOpen", "iHigh", "iLow", "iClose", "iTime", "iBarShift", "Bars", "MathIsValidNumber", "MathArcsin", "CopyBuffer", "IndicatorRelease", "PrintFormat", "Print", "StringFormat",
    "TimeToString", "TimeToStruct", "DoubleToString", "EnumToString", "FileOpen", "FileClose", "FileDelete", "FileSize", "FileSeek", "FileWriteString", "MQLInfoInteger",
    "PositionsTotal", "PositionGetTicket", "PositionGetString", "PositionGetInteger", "PositionGetDouble", "SymbolInfoInteger", "SymbolInfoDouble", "NormalizeDouble",
    "HistoryDealSelect", "HistoryDealGetInteger", "HistoryDealGetDouble", "HistoryDealGetString", "HistorySelectByPosition", "HistoryDealsTotal", "HistoryDealGetTicket",
    "ArrayResize", "Sleep", "GetLastError", "SetExpertMagicNumber", "SetDeviationInPoints", "SetTypeFillingBySymbol", "PositionClose", "Buy", "Sell", "ResultRetcode",
    "ResultRetcodeDescription", "if", "for", "while", "return", "int", "double", "bool", "string", "datetime", "long", "ulong", "const", "sizeof",
}
_DEF_RE = re.compile(r"^(?:bool|double|int|void|string|datetime)\s+([A-Za-z_]\w*)\s*\(", re.M)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _strip(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', src)


def lint_mql5(src: str) -> List[str]:
    issues: List[str] = []
    if not src.isascii():
        issues.append("non-ASCII characters in the source")
    left = sorted(set(re.findall(r"@@[A-Z_]+@@", src)))
    if left:
        issues.append(f"unfilled template fields {left}")
    code = _strip(src)
    for o, c in ("()", "{}", "[]"):
        if code.count(o) != code.count(c):
            issues.append(f"unbalanced {o}{c}: {code.count(o)} vs {code.count(c)}")
    depth = 0
    for ch in code:
        depth += (ch in "({[") - (ch in ")}]")
        if depth < 0:
            issues.append("a closing bracket without an opening one")
            break
    defs: Dict[str, int] = {}
    for m in _DEF_RE.finditer(code):
        if m.group(1) in defs:
            issues.append(f"function {m.group(1)} defined twice")
        defs[m.group(1)] = m.start()
    first_call: Dict[str, int] = {}
    for m in _CALL_RE.finditer(code):
        first_call.setdefault(m.group(1), m.start())
    for name, pos in first_call.items():
        if name in _KNOWN_FUNCS or name in ("OnInit", "OnDeinit", "OnTick", "OnTradeTransaction"):
            continue
        if name.startswith(("i", "I")) and re.match(r"^i[A-Z]", name):          # indicator creation calls (iRSI, iMA, ...)
            continue
        if name in defs:
            if defs[name] > pos:
                issues.append(f"{name} is used before its definition")
        elif name not in ("MathMax", "MathMin"):
            issues.append(f"{name}() is called but not defined")
    return issues
