"""A run in which no rule survives the search must still finish (stages 10-13), and the loader runner must pass its options on correctly."""

import logging

import pandas as pd

from edge_research import pipeline as P
from edge_research.mql5_loader import runner
from edge_research.mql5_loader.sample import write_exporter_csv
from edge_research.reporting.run_summary import _read_csv, gate_log


def test_null_flags_empty_or_missing_file_is_none(tmp_path):
    (tmp_path / "09_null").mkdir()
    assert P._read_null_flags(tmp_path) is None                                  # missing
    (tmp_path / "09_null" / "rule_flags.csv").write_text("\n")                   # what a stage writes when no rule passed: no column at all
    assert P._read_null_flags(tmp_path) is None
    (tmp_path / "09_null" / "rule_flags.csv").write_text("rule_id,flag\nA,1\n")
    assert list(P._read_null_flags(tmp_path)["rule_id"]) == ["A"]


def test_gate_log_of_a_gate_table_without_columns_is_an_empty_frame_with_the_standard_columns(tmp_path):
    (tmp_path / "10_validation").mkdir()
    assert gate_log(tmp_path) is None                                            # stage 10 not run
    (tmp_path / "10_validation" / "gate_table.csv").write_text("\n")
    t = gate_log(tmp_path)
    assert t is not None and len(t) == 0 and "rule_id" in t.columns and "fail_reasons" in t.columns
    assert _read_csv(tmp_path / "10_validation" / "gate_table.csv") is None


def test_loader_runner_forwards_max_replicates_under_the_pipeline_name(tmp_path, monkeypatch):
    csv = write_exporter_csv(tmp_path / "EURUSD_H1_export_20261101.csv", 3000, 7, 60)
    seen = {}

    def fake_0_9(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call=None):
        seen["n"] = max_replicates_this_call
        return {"status": {}}

    monkeypatch.setattr(P, "run_stages_0_9", fake_0_9)
    runner.run_pipeline(csv, tmp_path / "run", 9, max_replicates=7)
    assert seen["n"] == 7


def test_ingest_warnings_are_printed_once_per_run(caplog):
    lg = logging.getLogger("edge_research.ingestion.loader")
    once = runner._LogOnce()
    lg.addFilter(once)
    try:
        with caplog.at_level(logging.WARNING, logger=lg.name):
            for _ in range(2):
                lg.warning("news_*: all 7 news columns are NaN")
            lg.warning("another warning")
    finally:
        lg.removeFilter(once)
    assert [r.getMessage() for r in caplog.records] == ["news_*: all 7 news columns are NaN", "another warning"]
