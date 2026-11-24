"""
Coverage audit: proof that every column of the exporter is turned into at least one feature.

Two moments:

* ``audit_header`` (before the run, header only): builds the extended registry for the file's columns and lists, per column, the features that read it. A column
  nobody reads is an ``unprocessed`` column; ``assert_all_processed`` raises ``UnprocessedColumnsError`` for it. This is the guard against the failure
  "the exporter gained an indicator, the CSV loads fine, the pipeline silently ignores it".
* ``write_run_coverage`` (after stage 1): per column, how many of its features were really built, survived the QA exclusion, and were kept as a cluster
  representative for the atoms. A column whose features were all skipped by QA (constant / mostly NaN) is reported as such, with the reason.

Gated columns (``news_*`` / ``cal_*`` when ``features.include_calendar_news`` is false) are reported as gated, not as unprocessed: the base pipeline excludes
them on purpose (live-only data).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from edge_research.mql5_loader.features import build_extended_registry
from edge_research.mql5_loader.schema import EXPORTER_BY_NAME

EXEMPT = {"time": "bar timestamp: becomes the index of every table"}
GATED_PREFIXES = ("news_", "cal_")


class UnprocessedColumnsError(RuntimeError):
    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        super().__init__(
            f"{len(self.columns)} column(s) of the exporter file are read by NO feature and would be silently ignored: {self.columns}. "
            "Add a feature for them in edge_research/mql5_loader/features.py (and a recipe in extension.py), or pass --allow-unprocessed."
        )


@dataclass
class ColumnCoverage:
    column: str
    group: str
    features: List[str]
    status: str                      # processed | gated | exempt | unprocessed
    note: str = ""


@dataclass
class CoverageAudit:
    rows: List[ColumnCoverage] = field(default_factory=list)

    @property
    def unprocessed(self) -> List[str]:
        return [r.column for r in self.rows if r.status == "unprocessed"]

    def by_column(self) -> Dict[str, ColumnCoverage]:
        return {r.column: r for r in self.rows}


def audit_header(columns: Sequence[str], cfg: Dict[str, Any]) -> CoverageAudit:
    cols = [str(c).replace('"', "").strip() for c in columns]
    specs, _ = build_extended_registry(cols, cfg)
    readers: Dict[str, List[str]] = {c: [] for c in cols}
    for s in specs:
        for src in s.sources:
            if src in readers:
                readers[src].append(s.name)
    gate_on = bool(cfg["features"].get("include_calendar_news", False))
    audit = CoverageAudit()
    for c in cols:
        meta = EXPORTER_BY_NAME.get(c)
        group = meta.group if meta else "unlisted"
        if c in EXEMPT:
            audit.rows.append(ColumnCoverage(c, group, [], "exempt", EXEMPT[c]))
        elif readers[c]:
            audit.rows.append(ColumnCoverage(c, group, sorted(readers[c]), "processed"))
        elif c.startswith(GATED_PREFIXES) and not gate_on:
            audit.rows.append(ColumnCoverage(c, group, [], "gated", "features.include_calendar_news is false: calendar / news columns are excluded on purpose"))
        else:
            audit.rows.append(ColumnCoverage(c, group, [], "unprocessed", "no feature reads this column"))
    return audit


def assert_all_processed(audit: CoverageAudit) -> None:
    if audit.unprocessed:
        raise UnprocessedColumnsError(audit.unprocessed)


def audit_markdown(audit: CoverageAudit) -> str:
    n = {s: sum(1 for r in audit.rows if r.status == s) for s in ("processed", "gated", "exempt", "unprocessed")}
    lines = ["# Exporter column coverage (before the run)", "",
             f"{len(audit.rows)} columns: {n['processed']} processed, {n['gated']} gated, {n['exempt']} exempt, **{n['unprocessed']} unprocessed**", "",
             "| column | group | status | features reading it |", "|---|---|---|---|"]
    for r in audit.rows:
        feats = ", ".join(r.features[:6]) + (f" (+{len(r.features) - 6})" if len(r.features) > 6 else "")
        lines.append(f"| `{r.column}` | {r.group} | {r.status} | {feats or r.note} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ after stage 1
def write_run_coverage(run_dir: str | Path, cfg: Dict[str, Any], columns: Sequence[str]) -> pd.DataFrame:
    """Per column: features expected / built / kept, QA exclusion reason, and the verdict. Writes ``01_features/column_coverage.{csv,md}``.

    Verdicts: ``used`` (a feature of the column is a cluster representative and goes to the atoms), ``redundant`` (built, but every feature was merged into a
    cluster represented by another feature), ``qa_excluded`` (the column was removed by stage 0 QA, e.g. constant / all NaN), ``gated``, ``exempt``,
    ``unprocessed`` (a bug: nothing reads the column).
    """
    run = Path(run_dir)
    audit = audit_header(columns, cfg)
    reg = json.loads((run / "01_features" / "feature_registry.json").read_text(encoding="utf-8"))
    built = {f["name"] for f in reg["features"]}
    kept = set(json.loads((run / "01_features" / "kept_features.json").read_text(encoding="utf-8")))
    qa_file = run / "00_data_qa" / "excluded_columns.json"
    excluded: Dict[str, str] = json.loads(qa_file.read_text(encoding="utf-8")) if qa_file.exists() else {}
    clusters = pd.read_csv(run / "01_features" / "family_clusters.csv")
    rep_of = {r.feature: clusters.loc[(clusters["cluster"] == r.cluster) & clusters["representative"], "feature"].iloc[0] for r in clusters.itertuples()}
    table = pd.read_parquet(run / "01_features" / "feature_table.parquet")
    nan_pct = table.isna().mean()

    rows = []
    for r in audit.rows:
        feats_built = [f for f in r.features if f in built]
        feats_kept = [f for f in feats_built if f in kept]
        if r.status in ("exempt", "gated", "unprocessed"):
            verdict, why = r.status, r.note
        elif r.column in excluded:
            verdict, why = "qa_excluded", f"stage 0 QA removed the column: {excluded[r.column]}"
        elif feats_kept:
            verdict, why = "used", "kept as cluster representative: " + ", ".join(feats_kept[:4])
        elif feats_built:
            reps = sorted({rep_of[f] for f in feats_built if f in rep_of})
            verdict, why = "redundant", "merged into cluster representative(s): " + ", ".join(reps[:4])
        else:
            verdict, why = "qa_excluded", "every feature of the column was skipped (a co-source column was excluded by QA)"
        rows.append({"column": r.column, "group": r.group, "verdict": verdict, "features_expected": len(r.features), "features_built": len(feats_built),
                     "features_kept": len(feats_kept), "min_feature_nan_pct": round(float(min((nan_pct[f] for f in feats_built), default=float("nan"))) * 100, 1) if feats_built else None,
                     "detail": why})
    out = pd.DataFrame(rows)
    out.to_csv(run / "01_features" / "column_coverage.csv", index=False)
    counts = out["verdict"].value_counts().to_dict()
    lines = ["# Exporter column coverage (after stage 1)", "", "Verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), "",
             "| column | group | verdict | built | kept | detail |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r.column}` | {r.group} | {r.verdict} | {r.features_built}/{r.features_expected} | {r.features_kept} | {r.detail} |" for r in out.itertuples()]
    (run / "01_features" / "column_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
