"""MQL5 recipes for the loader-mode features and the exporter's indicator handles for the code generator.

* ``extension_recipe``: exact MQL5 expression for features that the stock ``codegen/recipes.py`` does not know (envelope, AMA distance, band middle distance,
  ATR ratio). It only uses the primitives of the stock vocabulary (``IND_<column>``, ``C``, ``ATRN``, ``DIV``), so ``pyeval`` evaluates the very same text and
  the parity test compares it with the Python feature table. Features without an exact translation raise ``CodegenRefusal`` with the reason (never an
  approximation).
* ``codegen_config``: ``codegen.indicators`` entries transcribed from the exporter (``schema.codegen_overrides``), merged into the config.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from edge_research.codegen import CodegenRefusal
from edge_research.codegen.recipes import Recipe
from edge_research.features.registry import FeatureSpec
from edge_research.mql5_loader.schema import codegen_overrides

A1 = "ATRN(1)"

_REFUSED = {
    "spread_atr": "needs iSpread (the history spread of the closed bar) and _Point: not in the generator's primitive vocabulary yet",
    "spread_rel": "needs a rolling median of the spread buffer (state not implemented in the generator yet)",
    "real_volume_rel": "needs a rolling median of the real volume (state not implemented in the generator yet)",
}


def extension_recipe(spec: FeatureSpec, cfg: Dict[str, Any]) -> Optional[Recipe]:
    """Recipe of a loader-mode feature, ``None`` if the name is not one of them (the caller then raises its own refusal)."""
    n = spec.name
    if n in _REFUSED:
        raise CodegenRefusal(f"feature {n}: {_REFUSED[n]}")
    if n == "atr_ratio_14_20":
        return Recipe(n, "DIV(IND_atr_14(1), IND_atr_20(1))", ("atr_14", "atr_20"), ())
    if n == "dist_ama_10_2_30":
        return Recipe(n, f"DIV(C(1) - IND_ama_10_2_30(1), {A1})", ("ama_10_2_30",), ())
    m = re.fullmatch(r"bb_mid_dist_(bb_\d+_\d+)", n)
    if m:
        mid = m[1] + "_middle"
        return Recipe(n, f"DIV(C(1) - IND_{mid}(1), {A1})", (mid,), ())
    m = re.fullmatch(r"env_(pctb|width)_(envelope_\d+_\d+)", n)
    if m:
        up, lo = m[2] + "_upper", m[2] + "_lower"
        expr = f"DIV(C(1) - IND_{lo}(1), IND_{up}(1) - IND_{lo}(1))" if m[1] == "pctb" else f"DIV(IND_{up}(1) - IND_{lo}(1), {A1})"
        return Recipe(n, expr, (lo, up), ())
    return None


def codegen_config() -> Dict[str, Any]:
    """Config fragment with the exporter's indicator handles (merge it over the base config)."""
    return {"codegen": {"indicators": codegen_overrides()}}
