"""Glue between the stock pipeline and the MQL5 loader mode (``data.source: mql5_loader``).

``prepare_run_inputs`` is called by ``pipeline.run_stages_0_2`` and ``pipeline.load_run_inputs`` right after the bar table is read, so stage 0-2 and every
later stage see the SAME repaired table and the SAME spread series, taken from the one CSV.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from edge_research.mql5_loader.coverage import assert_all_processed, audit_header, audit_markdown
from edge_research.mql5_loader.loader import FAIL, extract_spread, prepare_table


class LoaderCheckFailed(RuntimeError):
    pass


def is_loader_mode(cfg: Dict[str, Any]) -> bool:
    return cfg.get("data", {}).get("source") == "mql5_loader"


def prepare_run_inputs(df: pd.DataFrame, cfg: Dict[str, Any], spread: Optional[pd.Series], report_dir: Optional[str | Path] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Repair + check the bar table and take the spread from it.

    * the envelope repair and the integrity checks run (idempotent);
    * an explicit ``spread`` argument (the old second file) still wins, otherwise the spread comes from the ``spread`` column of the SAME table
      (``data.spread_from_data``, default true in loader mode);
    * with ``report_dir`` the coverage audit runs first and the findings are written to ``mql5_loader_report.json`` / ``column_coverage_before.md``:
      a column that no feature reads stops the run (``data.allow_unprocessed`` overrides) and a FAIL finding stops it too.
    """
    d = cfg.get("data", {})
    df, findings, repairs = prepare_table(df)
    if spread is None and bool(d.get("spread_from_data", True)):
        spread = extract_spread(df)
    if report_dir is not None:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        audit = audit_header(["time"] + list(df.columns), cfg)
        (out / "column_coverage_before.md").write_text(audit_markdown(audit), encoding="utf-8")
        (out / "mql5_loader_report.json").write_text(json.dumps({"repairs": repairs, "findings": [asdict(f) for f in findings], "unprocessed": audit.unprocessed,
                                                                 "spread_source": "data column 'spread'" if spread is not None else "none (fixed spread from config)"}, indent=2), encoding="utf-8")
        failed = [f for f in findings if f.level == FAIL]
        if failed:
            raise LoaderCheckFailed("MQL5 loader integrity check failed: " + "; ".join(f"{f.check}: {f.message}" for f in failed))
        if not bool(d.get("allow_unprocessed", False)):
            assert_all_processed(audit)
    return df, spread
