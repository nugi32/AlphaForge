"""Stage 12 (M10): MQL5 files for the rules ACCEPTED by stage 10, plus the refusal list and the review checklist (design 16)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from edge_research.codegen import CODEGEN_VERSION, CodegenRefusal
from edge_research.codegen.lint import lint_mql5
from edge_research.codegen.plan import EAPlan, SpecLite, build_plan
from edge_research.codegen.render import render_ea
from edge_research.config import config_hash
from edge_research.rules import FrozenRule, load_rules

CHECKLIST = """# Review checklist (design 16.8) - tick every line before any live use

For each generated EA `ER_<rule>.mq5`:

- [ ] Compile in MetaEditor: 0 errors, 0 warnings. (The generator has no MQL5 compiler; `codegen.lint` only checks the text.)
- [ ] Indicator handles: every `ASSUMED` line of the header equals the call in `dataExporter.mq5` (period, method, applied price, buffer). Wrong parameters = wrong signals.
- [ ] Bar shift: all values are read at shift 1 (the last closed bar); the EA acts on the first tick of a new bar only.
- [ ] Direction, exit (T / B), catastrophic stop, one position per magic number.
- [ ] Feature parity: run once with `InpDumpFeatures = true` over the research period, then `scripts/check_parity.py --features ...`.
- [ ] Trade parity: Strategy Tester, Open prices only, same period / spread / deposit as the config, `InpMaxGapHours = 0`; compare with `scripts/check_parity.py --trades ...`
      (the engine side runs with `engine.no_weekend_hold = false`, see `codegen/ea_sim.py`).
- [ ] Stress run in tick or 1-minute OHLC mode: no unexpected stops, no double entries; explain any P&L difference.
- [ ] Custom indicators (`ps_*`, `volregime_*`): same binary, inputs and buffer as the exporter; values on closed bars (these rules are refused until `codegen.indicators` gives the call).
- [ ] Costs: spread / commission / swap of the tester account equal the config `costs`.
- [ ] Account-level risk management (daily loss limit, equity stop) is NOT in the EA: add it.
"""


def load_registry(run: Path) -> Dict[str, SpecLite]:
    d = json.loads((run / "01_features" / "feature_registry.json").read_text(encoding="utf-8"))
    return {f["name"]: SpecLite(f["name"], f.get("kind", ""), bool(f.get("reproducible_in_tester", True))) for f in d["features"]}


def plan_summary(p: EAPlan) -> Dict[str, Any]:
    return {"rule_id": p.rule.rule_id, "ea_name": p.ea_name, "direction": p.rule.direction, "exit": p.rule.exit, "cat_stop_atr": p.rule.cat_stop_atr,
            "features": [{"name": r.name, "expr": r.expr} for r in p.features], "atoms": p.atom_exprs,
            "indicators": [{"column": i.column, "create": i.create, "buffer": i.buffer, "verified": i.verified, "source": i.source} for i in p.indicators],
            "settings": p.settings, "warnings": p.warnings, "provenance": p.provenance, "generator_version": CODEGEN_VERSION}


def generate_eas(run_dir: str | Path, cfg: Dict[str, Any], rules: Optional[List[FrozenRule]] = None, out_dir: Optional[str | Path] = None,
                 only: Optional[List[str]] = None) -> Dict[str, Any]:
    """Write ``ER_<rule>.mq5`` and ``ER_<rule>.plan.json`` for every rule the generator can translate exactly; list the others in ``refused.json`` with ALL reasons."""
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run / "12_ea"
    out.mkdir(parents=True, exist_ok=True)
    if rules is None:
        rules = load_rules(run / "10_validation" / "accepted_rules.json")
    if only:
        rules = [r for r in rules if r.rule_id in set(only)]
    registry = load_registry(run)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prov = {"run": run.name, "data_hash": manifest.get("data_hash"), "config_hash": config_hash(cfg), "generator": CODEGEN_VERSION}
    generated: List[str] = []
    refused: Dict[str, List[str]] = {}
    lint_issues: Dict[str, List[str]] = {}
    for rule in rules:
        try:
            plan = build_plan(rule, cfg, registry, prov)
        except CodegenRefusal as e:
            refused[rule.rule_id] = e.reasons
            continue
        src = render_ea(plan)
        (out / f"{plan.ea_name}.mq5").write_text(src, encoding="ascii", newline="\n")
        (out / f"{plan.ea_name}.plan.json").write_text(json.dumps(plan_summary(plan), indent=2, default=float), encoding="utf-8")
        issues = lint_mql5(src)
        if issues:
            lint_issues[rule.rule_id] = issues
        generated.append(rule.rule_id)
    (out / "refused.json").write_text(json.dumps(refused, indent=2), encoding="utf-8")
    (out / "REVIEW_CHECKLIST.md").write_text(CHECKLIST, encoding="utf-8")
    info = {"n_rules": len(rules), "generated": generated, "refused": refused, "lint_issues": lint_issues, "generator_version": CODEGEN_VERSION}
    (out / "ea_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
