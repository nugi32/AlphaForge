"""Canonical rule file (JSON): the single description of a frozen rule that the backtest, the validation stage and the MQL5 generator all read (design 15.2, 17).

A rule is an AND of atoms ``feature <= t`` / ``feature > t`` / ``feature == v`` with ONE direction and ONE exit. The atom semantics are identical to ``atoms.atoms``:
a non-finite feature value makes every atom False (``>`` is "finite and not <=").
An OPTIONAL ``exit_condition`` (same atom form) closes an open position at the open of the bar AFTER a bar on which all its atoms
hold (design 15.4 condition exit; the engine layer ``exit_signals``). No discovery stage searches for one yet (NOTES Q-008), but the
format, the backtest path and the codegen refusal are in place so a frozen rule can carry one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from edge_research.mt5_backtest.engine import parse_family

OPS = ("<=", ">", "==")
RULE_VERSION = 1


@dataclass
class FrozenRule:
    rule_id: str
    direction: str                                  # "long" | "short"
    atoms: List[Dict[str, Any]]                     # [{"feature": str, "op": "<="|">"|"==", "threshold": float}]
    exit: str                                       # family name: T6, B2_1p5_12 (I... is research only)
    cat_stop_atr: Optional[float] = None
    exit_condition: Optional[List[Dict[str, Any]]] = None   # same atom form: AND of these closes the position early (design 15.4)
    meta: Dict[str, Any] = field(default_factory=dict)   # free: t, expectancy, folds, live_only, ... (never read by the engine)

    def validate(self) -> "FrozenRule":
        if self.direction not in ("long", "short"):
            raise ValueError(f"rule {self.rule_id}: direction must be long or short")
        if not self.atoms:
            raise ValueError(f"rule {self.rule_id}: no atoms")
        for a in self.atoms:
            if set(a) != {"feature", "op", "threshold"} or a["op"] not in OPS or not isinstance(a["feature"], str):
                raise ValueError(f"rule {self.rule_id}: bad atom {a!r}")
            if not math.isfinite(float(a["threshold"])):
                raise ValueError(f"rule {self.rule_id}: non-finite threshold in {a!r}")
        for a in self.exit_condition or []:
            if set(a) != {"feature", "op", "threshold"} or a["op"] not in OPS or not isinstance(a["feature"], str):
                raise ValueError(f"rule {self.rule_id}: bad exit-condition atom {a!r}")
            if not math.isfinite(float(a["threshold"])):
                raise ValueError(f"rule {self.rule_id}: non-finite threshold in exit condition {a!r}")
        parse_family(self.exit)
        if self.cat_stop_atr is not None and not float(self.cat_stop_atr) > 0:
            raise ValueError(f"rule {self.rule_id}: cat_stop_atr must be positive")
        return self

    @property
    def research_only(self) -> bool:
        return parse_family(self.exit)["kind"] == "I"

    def features(self) -> List[str]:
        return sorted({a["feature"] for a in self.atoms})

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["version"] = RULE_VERSION
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FrozenRule":
        d = dict(d)
        ver = d.pop("version", RULE_VERSION)
        if ver != RULE_VERSION:
            raise ValueError(f"unsupported rule version {ver}")
        unknown = set(d) - {"rule_id", "direction", "atoms", "exit", "cat_stop_atr", "exit_condition", "meta"}
        if unknown:
            raise ValueError(f"unknown rule fields: {sorted(unknown)}")
        return FrozenRule(rule_id=d["rule_id"], direction=d["direction"], atoms=[dict(a) for a in d["atoms"]], exit=d["exit"],
                          cat_stop_atr=d.get("cat_stop_atr"), exit_condition=d.get("exit_condition"),
                          meta=dict(d.get("meta", {}))).validate()


def save_rules(rules: List[FrozenRule], path: str | Path) -> None:
    ids = [r.rule_id for r in rules]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate rule ids")
    Path(path).write_text(json.dumps({"rules": [r.validate().to_dict() for r in rules]}, indent=2, sort_keys=True, default=float))


def load_rules(path: str | Path) -> List[FrozenRule]:
    d = json.loads(Path(path).read_text())
    rules = [FrozenRule.from_dict(x) for x in d["rules"]]
    if len({r.rule_id for r in rules}) != len(rules):
        raise ValueError("duplicate rule ids")
    return rules


def atom_mask(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if op == "<=":
        return x <= threshold
    if op == ">":
        return np.isfinite(x) & ~(x <= threshold)
    if op == "==":
        return x == threshold
    raise ValueError(f"unknown op {op!r}")


def rule_signal(rule: FrozenRule, table: pd.DataFrame) -> np.ndarray:
    """Boolean signal (decision at the close of each row) of the AND of all atoms."""
    missing = [f for f in rule.features() if f not in table.columns]
    if missing:
        raise KeyError(f"rule {rule.rule_id}: features missing from the table: {missing}")
    m = np.ones(len(table), dtype=bool)
    for a in rule.atoms:
        m &= atom_mask(table[a["feature"]].to_numpy(np.float64), a["op"], float(a["threshold"]))
    return m


def exit_condition_signal(rule: FrozenRule, table: pd.DataFrame) -> Optional[np.ndarray]:
    """Boolean condition-exit signal (decision at the close of each row) = the AND of the rule's ``exit_condition`` atoms; None when
    the rule carries none. The engine closes the position at the open of the bar AFTER a true bar (design 15.4)."""
    if not rule.exit_condition:
        return None
    feats = sorted({a["feature"] for a in rule.exit_condition})
    missing = [f for f in feats if f not in table.columns]
    if missing:
        raise KeyError(f"rule {rule.rule_id}: exit-condition features missing from the table: {missing}")
    m = np.ones(len(table), dtype=bool)
    for a in rule.exit_condition:
        m &= atom_mask(table[a["feature"]].to_numpy(np.float64), a["op"], float(a["threshold"]))
    return m
