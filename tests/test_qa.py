import numpy as np
import pandas as pd

from edge_research.config import load_config
from edge_research.qa.ingest_qa import compute_warmup, data_hash, find_gaps, run_qa, write_report
from edge_research.synth.generator import make_synthetic_bars


def _cfg():
    return load_config()


def test_clean_synthetic_data_passes(tmp_path):
    df, meta = make_synthetic_bars(8000, seed=2)
    rep = run_qa(df, _cfg(), meta["spread_points"])
    assert not rep.critical_failed
    status = {c.name: c.status for c in rep.checks}
    assert status["alignment"] == "pass" and status["time_index"] == "pass"
    assert "mom_10" in rep.flagged_columns
    assert write_report(rep, tmp_path).exists()


def test_alignment_detects_shifted_exporter():
    df, _ = make_synthetic_bars(3000, seed=2)
    shifted = df.copy()
    shifted["ma_5_sma"] = shifted["ma_5_sma"].shift(1)      # exporter exposes the previous bar's value
    shifted["ma_10_sma"] = shifted["ma_10_sma"].shift(1)
    shifted["ma_20_sma"] = shifted["ma_20_sma"].shift(1)
    rep = run_qa(shifted, _cfg())
    assert rep.critical_failed
    assert any("shift" in c.detail for c in rep.checks if c.name == "alignment")


def test_duplicate_or_unsorted_index_stops():
    df, _ = make_synthetic_bars(500, seed=2)
    bad = pd.concat([df, df.iloc[:5]])
    assert run_qa(bad, _cfg()).critical_failed


def test_gap_classification():
    df, _ = make_synthetic_bars(1000, seed=2)
    cut = pd.concat([df.iloc[:300], df.iloc[320:]])           # remove 20 bars mid-week or so
    gaps = find_gaps(cut, 60, 2.0)
    assert (gaps["kind"] == "other").sum() >= 1
    assert (gaps["kind"] == "weekend").sum() >= 5


def test_constant_column_excluded_and_warmup():
    df, _ = make_synthetic_bars(1000, seed=2)
    df["const"] = np.float32(1.0)
    rep = run_qa(df, _cfg())
    assert "const" in rep.excluded_columns
    assert compute_warmup(df, ["close", "ma_200_sma"]) == 199


def test_hash_changes_with_data():
    df, _ = make_synthetic_bars(500, seed=2)
    h1 = data_hash(df)
    df2 = df.copy()
    df2.iloc[10, 0] += 1e-3
    assert h1 == data_hash(df) and h1 != data_hash(df2)
