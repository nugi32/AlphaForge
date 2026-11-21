import json

import pytest

from edge_research.config import load_config
from edge_research.pipeline import QAFailed, run_stages_0_2
from edge_research.splits import make_splits, max_hold_bars
from edge_research.synth.generator import make_synthetic_bars, write_exporter_csv


def test_splits_have_embargo_and_order():
    cfg = load_config()
    s = make_splits(10000, cfg)
    assert max_hold_bars(cfg) == 24 and s.embargo == 30
    assert s.train.stop + s.embargo <= s.val.start and s.val.stop + s.embargo <= s.holdout.start
    assert s.holdout.stop == 10000 and s.train.start == 0
    with pytest.raises(ValueError):
        make_splits(100, cfg)


def _write(tmp_path, n=6000, seed=9):
    df, meta = make_synthetic_bars(n, seed=seed)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    return csv, tmp_path / "s_spread.csv"


def test_end_to_end_and_resume(tmp_path):
    csv, sp = _write(tmp_path)
    cfg, run = load_config(), tmp_path / "run"
    out1 = run_stages_0_2(csv, run, cfg, "SYNTH", sp)
    assert out1["status"] == {"00_data_qa": "ran", "01_features": "ran", "02_labels": "ran"}
    for rel in ["00_data_qa/report.md", "01_features/feature_table.parquet", "01_features/feature_registry.json",
                "01_features/family_clusters.csv", "02_labels/label_table.parquet", "02_labels/label_spec.json", "manifest.json"]:
        assert (run / rel).exists(), rel
    man = json.loads((run / "manifest.json").read_text())
    assert man["holdout_accesses"] == [] and man["data_hash"] and man["seed"] == cfg["seed"]
    out2 = run_stages_0_2(csv, run, cfg, "SYNTH", sp)                       # same data + config -> everything skipped
    assert set(out2["status"].values()) == {"skipped"}
    cfg2 = load_config(overrides={"labels": {"time_stops": [1, 2, 3, 6, 12, 24, 48]}})
    out3 = run_stages_0_2(csv, run, cfg2, "SYNTH", sp)                      # label config changed -> only labels rerun
    assert out3["status"] == {"00_data_qa": "skipped", "01_features": "skipped", "02_labels": "ran"}


def test_qa_failure_stops_the_run(tmp_path):
    df, _ = make_synthetic_bars(3000, seed=9)
    for c in ("ma_5_sma", "ma_10_sma", "ma_20_sma"):
        df[c] = df[c].shift(1)                                              # simulate an exporter that shifts bars
    csv = write_exporter_csv(df, tmp_path / "bad.csv")
    with pytest.raises(QAFailed):
        run_stages_0_2(csv, tmp_path / "run", load_config(), "SYNTH")
