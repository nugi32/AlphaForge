"""Rule + config -> ``EAPlan`` (design 16.1, 16.2). The plan holds every decision of the generator as data (expressions, handles, exit parameters, settings);
``render.render_ea`` only lays it out as text and ``pyeval`` / ``ea_sim`` evaluate the same plan in Python, so what is tested is what is emitted.

The generator refuses (``CodegenRefusal``, with ALL reasons found) when
* the rule is research only (intrabar exit ``I...``) or flagged ``live_only``,
* an atom's feature has no exact recipe (rolling state, events, custom indicators, calendar) or is not in the feature registry,
* a source column has no verified-or-assumed indicator mapping (``codegen.indicators`` in the config),
* the engine settings ask for something the EA does not do (``max_open_positions`` != 1).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from edge_research.codegen import CODEGEN_VERSION, CodegenRefusal
from edge_research.codegen.indicators import IndicatorSpec, indicator_for
from edge_research.codegen.mql_helpers import BASE_HELPERS, helper_closure
from edge_research.codegen.recipes import Recipe, recipe_for
from edge_research.mt5_backtest.engine import engine_settings, parse_family
from edge_research.rules import FrozenRule

TIMEFRAMES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


@dataclass(frozen=True)
class SpecLite:
    """The three registry fields the recipes read (``feature_registry.json`` rows or ``FeatureSpec`` objects both fit)."""

    name: str
    kind: str = ""
    reproducible_in_tester: bool = True


@dataclass
class EAPlan:
    rule: FrozenRule
    ea_name: str
    features: List[Recipe]                    # in ``rule.features()`` order: the EA array f[i]
    atom_exprs: List[str]                     # one boolean MQL5 expression per atom, all ANDed
    indicators: List[IndicatorSpec]
    helpers: List[str]                        # preamble helpers in dependency order
    exit: Dict[str, Any]                      # parse_family(rule.exit)
    settings: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        return [r.name for r in self.features]


def mql_double(x: float) -> str:
    """Decimal literal that parses back to the same double in MQL5 and Python."""
    s = repr(float(x))
    if "e" in s or "E" in s:
        mant, exp = s.lower().split("e")
        if "." not in mant:
            mant += ".0"
        s = f"{mant}e{int(exp)}"
    return s


def _lookup(registry: Mapping[str, Any], name: str) -> Optional[SpecLite]:
    row = registry.get(name)
    if row is None:
        return None
    if isinstance(row, SpecLite):
        return row
    if isinstance(row, Mapping):
        return SpecLite(name, str(row.get("kind", "")), bool(row.get("reproducible_in_tester", True)))
    return SpecLite(name, getattr(row, "kind", ""), bool(getattr(row, "reproducible_in_tester", True)))


def magic_for(rule_id: str) -> int:
    return 7_000_000 + zlib.crc32(rule_id.encode("utf-8")) % 1_000_000


def build_plan(rule: FrozenRule, cfg: Dict[str, Any], registry: Mapping[str, Any], provenance: Optional[Dict[str, Any]] = None) -> EAPlan:
    """``registry``: feature name -> spec-like (``FeatureSpec``, ``SpecLite`` or a row of ``feature_registry.json``)."""
    reasons: List[str] = []
    warnings: List[str] = []
    rule.validate()
    if rule.research_only:
        reasons.append(f"exit {rule.exit} is intrabar research mode (no MT5 equivalent in a bar-open tester, NOTES D-037)")
    if rule.meta.get("live_only"):
        reasons.append("the rule is flagged live_only (calendar / news): not reproducible in the Strategy Tester (design 16.7)")
    if rule.exit_condition:
        reasons.append("the rule carries a condition exit: the generator does not translate exit conditions yet (NOTES Q-008)")
    eng = engine_settings(cfg)
    if int(eng["max_open_positions"]) != 1:
        reasons.append(f"engine.max_open_positions = {eng['max_open_positions']}: the EA keeps one position per magic number (design 16.5)")
    tf = str(cfg["data"]["timeframe"]).upper()
    if tf not in TIMEFRAMES:
        reasons.append(f"timeframe {tf!r} is not supported by the generator ({sorted(TIMEFRAMES)})")
    elif int(cfg["data"].get("timeframe_minutes", TIMEFRAMES[tf])) != TIMEFRAMES[tf]:
        reasons.append(f"data.timeframe {tf} and data.timeframe_minutes {cfg['data'].get('timeframe_minutes')} disagree")
    overrides = (cfg.get("codegen") or {}).get("indicators") or {}

    recipes: List[Recipe] = []
    for name in rule.features():
        spec = _lookup(registry, name)
        if spec is None:
            reasons.append(f"feature {name}: not in the feature registry")
            continue
        try:
            recipes.append(recipe_for(spec, cfg))
        except CodegenRefusal as e:
            reasons.extend(e.reasons)

    atr_col = cfg["features"]["atr_column"]
    cols = set()
    for r in recipes:
        cols.update(r.sources)
        if "ATRN(" in r.expr:
            cols.add(atr_col)
    cols.add(atr_col)                                  # always: the entry requires a valid ATR (the engine skips a signal with an invalid ATR) and the barrier levels use the ATR of the signal bar
    inds: List[IndicatorSpec] = []
    for c in sorted(cols):
        try:
            inds.append(indicator_for(c, overrides))
        except CodegenRefusal as e:
            reasons.extend(e.reasons)
    if reasons:
        raise CodegenRefusal(reasons)

    names = rule.features()
    idx = {n: i for i, n in enumerate(names)}
    atom_exprs: List[str] = []
    for a in rule.atoms:
        fn = {"<=": "LE", ">": "GT", "==": "EQ"}[a["op"]]
        atom_exprs.append(f"{fn}(f[{idx[a['feature']]}], {mql_double(a['threshold'])})")

    helper_names = ["LE", "GT", "EQ", "QNaN", "O", "H", "L", "C"]
    for r in recipes:
        helper_names.extend(r.helpers)
        for tok in ("DIV", "MAX2", "MIN2"):
            if tok + "(" in r.expr:
                helper_names.append(tok)
    helpers = helper_closure(sorted(set(helper_names), key=helper_names.index))

    mg = cfg["data"].get("max_gap_hours")
    settings = {
        "timeframe": tf, "magic": magic_for(rule.rule_id), "lots": float(eng["fixed_lots"]), "reenter_on_exit_bar": bool(eng["reenter_on_exit_bar"]),
        "max_gap_hours": float(mg) if (mg is not None and bool(eng["no_weekend_hold"])) else 0.0,
        "assumed_spread_points": float(cfg["costs"]["fixed_spread_points"]), "slippage_points": float(cfg["costs"]["slippage_points"]),
        "atr_column": atr_col, "direction": rule.direction, "cat_stop_atr": rule.cat_stop_atr,
    }
    unverified = [i for i in inds if not i.verified]
    if unverified:
        warnings.append(f"{len(unverified)} indicator handle(s) are ASSUMED from column names, not read from dataExporter.mq5: compare each with the exporter before trusting the EA")
    if eng["sizing_mode"] != "fixed_lots":
        warnings.append(f"engine.sizing_mode = {eng['sizing_mode']}: the generated EA trades fixed lots (InpLots); risk-percent sizing is not generated")
    if rule.cat_stop_atr is not None:
        warnings.append("catastrophic stop is a server-side SL: the engine models it intrabar, which the Open-prices-only tester cannot reproduce; verify in 1-minute OHLC or tick mode")
    if int(cfg["costs"]["slippage_points"]) != 0 or float(eng["cost_multiplier"]) != 1.0:
        warnings.append("engine slippage / cost_multiplier are not part of the EA; the tester applies its own spread and no slippage")
    return EAPlan(rule, f"ER_{rule.rule_id}", recipes, atom_exprs, inds, helpers, parse_family(rule.exit), settings, warnings, dict(provenance or {}))
