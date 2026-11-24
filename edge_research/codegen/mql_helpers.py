"""MQL5 source of the primitives and helpers the recipe expressions use (see ``recipes`` for the vocabulary).

Everything returns a double; ``QNaN()`` marks an unavailable value (history too short, indicator buffer not ready, ATR not positive). Any NaN makes every atom
False, like the Python table (``atoms`` semantics: a non-finite feature value makes the atom False).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from edge_research.features.registry import SESSIONS

# name -> (dependencies, MQL5 source). ``TF`` is the EA timeframe constant.
BASE_HELPERS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "QNaN": ((), "double QNaN(){ return MathArcsin(2.0); }   // NaN: asin is undefined outside [-1, 1]"),
    "O": (("QNaN",), "double O(const int s){ double v=iOpen(_Symbol,TF,s); return v>0.0 ? v : QNaN(); }"),
    "H": (("QNaN",), "double H(const int s){ double v=iHigh(_Symbol,TF,s); return v>0.0 ? v : QNaN(); }"),
    "L": (("QNaN",), "double L(const int s){ double v=iLow(_Symbol,TF,s); return v>0.0 ? v : QNaN(); }"),
    "C": (("QNaN",), "double C(const int s){ double v=iClose(_Symbol,TF,s); return v>0.0 ? v : QNaN(); }"),
    "DIV": (("QNaN",), "double DIV(const double a,const double b){ if(!MathIsValidNumber(a) || !MathIsValidNumber(b) || b==0.0) return QNaN(); return a/b; }"),
    "MAX2": (("QNaN",), "double MAX2(const double a,const double b){ if(!MathIsValidNumber(a) || !MathIsValidNumber(b)) return QNaN(); return a>b ? a : b; }"),
    "MIN2": (("QNaN",), "double MIN2(const double a,const double b){ if(!MathIsValidNumber(a) || !MathIsValidNumber(b)) return QNaN(); return a<b ? a : b; }"),
    "LE": ((), "bool LE(const double v,const double t){ return MathIsValidNumber(v) && v<=t; }"),
    "GT": ((), "bool GT(const double v,const double t){ return MathIsValidNumber(v) && v>t; }"),
    "EQ": ((), "bool EQ(const double v,const double t){ return MathIsValidNumber(v) && v==t; }"),
}


def _session_source() -> str:
    lines = ["double SESSIONV(const int s)", "{", "   double h=HOURV(s);", "   if(!MathIsValidNumber(h)) return QNaN();"]
    for i, (lo, hi) in enumerate(SESSIONS.values()):
        lines.append(f"   if(h>={lo}.0 && h<={hi}.0) return {i}.0;")
    lines += ["   return 0.0;", "}"]
    return "\n".join(lines)


FEATURE_HELPERS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "HHV": (("QNaN", "H"), """double HHV(const int n,const int s)   // highest high of the n bars at shifts s .. s+n-1
{
   if(Bars(_Symbol,TF) < s+n) return QNaN();
   double r=-DBL_MAX;
   for(int k=s;k<s+n;k++){ double v=H(k); if(!MathIsValidNumber(v)) return QNaN(); if(v>r) r=v; }
   return r;
}"""),
    "LLV": (("QNaN", "L"), """double LLV(const int n,const int s)   // lowest low of the n bars at shifts s .. s+n-1
{
   if(Bars(_Symbol,TF) < s+n) return QNaN();
   double r=DBL_MAX;
   for(int k=s;k<s+n;k++){ double v=L(k); if(!MathIsValidNumber(v)) return QNaN(); if(v<r) r=v; }
   return r;
}"""),
    "HOURV": (("QNaN",), """double HOURV(const int s)   // server hour of the open time of the bar at shift s
{
   datetime t=iTime(_Symbol,TF,s);
   if(t<=0) return QNaN();
   MqlDateTime d; TimeToStruct(t,d);
   return (double)d.hour;
}"""),
    "DOWV": (("QNaN",), """double DOWV(const int s)   // day of week of the bar at shift s, Monday = 0 (MQL5: Sunday = 0)
{
   datetime t=iTime(_Symbol,TF,s);
   if(t<=0) return QNaN();
   MqlDateTime d; TimeToStruct(t,d);
   return (double)((d.day_of_week+6)%7);
}"""),
    "SESSIONV": (("QNaN", "HOURV"), _session_source()),
    "BRKUP": (("QNaN", "C", "HHV"), """double BRKUP(const int n)   // 1 if the closed bar closes above the highest high of the n bars before it
{
   double hh=HHV(n,2), c=C(1);
   if(!MathIsValidNumber(hh) || !MathIsValidNumber(c)) return QNaN();
   return c>hh ? 1.0 : 0.0;
}"""),
    "BRKDN": (("QNaN", "C", "LLV"), """double BRKDN(const int n)   // 1 if the closed bar closes below the lowest low of the n bars before it
{
   double ll=LLV(n,2), c=C(1);
   if(!MathIsValidNumber(ll) || !MathIsValidNumber(c)) return QNaN();
   return c<ll ? 1.0 : 0.0;
}"""),
}

ALL_HELPERS = {**BASE_HELPERS, **FEATURE_HELPERS}


def helper_closure(names: List[str]) -> List[str]:
    """The helpers needed by ``names`` including dependencies, in an order where every helper comes after its dependencies."""
    out: List[str] = []

    def visit(n: str) -> None:
        if n in out:
            return
        for d in ALL_HELPERS[n][0]:
            visit(d)
        out.append(n)

    for n in names:
        if n not in ALL_HELPERS:
            raise KeyError(f"unknown MQL5 helper {n!r}")
        visit(n)
    return out
