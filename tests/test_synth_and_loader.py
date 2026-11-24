import numpy as np
import pandas as pd

from edge_research.config import load_config
from edge_research.ingestion.loader import ingest_pipeline, load_from_parquet
from edge_research.synth.generator import PlantedEdge, make_synthetic_bars, write_exporter_csv


def test_config_loads():
    cfg = load_config()
    assert cfg["labels"]["atr_column"] == "atr_14"


def test_reproducible_by_seed():
    a, _ = make_synthetic_bars(2000, seed=7)
    b, _ = make_synthetic_bars(2000, seed=7)
    c, _ = make_synthetic_bars(2000, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_shape_and_no_weekends():
    df, meta = make_synthetic_bars(3000, seed=1)
    assert len(df) == 3000 and df.index.is_monotonic_increasing and df.index.is_unique
    assert (df.index.weekday < 5).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()
    assert 90 < df["mom_10"].dropna().median() < 110  # centred near 100 like MT5 iMomentum


def test_planted_edge_has_positive_forward_drift():
    df, meta = make_synthetic_bars(40000, seed=3, plant_edge=PlantedEdge())
    trig = meta["trigger"].to_numpy()
    assert trig.sum() > 100
    o = df["open"].to_numpy(np.float64)
    atr = df["atr_14"].to_numpy(np.float64)
    idx = np.where(trig)[0]
    idx = idx[(idx + 8 < len(df)) & np.isfinite(atr[idx])]
    fwd = (o[idx + 7] - o[idx + 1]) / atr[idx]          # open[e+6] - open[e], e = i+1
    all_i = np.arange(50, len(df) - 8)
    base = ((o[all_i + 7] - o[all_i + 1]) / atr[all_i]).mean()
    assert fwd.mean() > base + 0.1


def test_null_data_has_no_triggers():
    _, meta = make_synthetic_bars(5000, seed=3, plant_edge=None)
    assert meta["trigger"].sum() == 0


def test_csv_passes_unchanged_loader(tmp_path):
    df, _ = make_synthetic_bars(1500, seed=5)
    csv = write_exporter_csv(df, tmp_path / "synth.csv")
    loaded, parquet = ingest_pipeline(csv, tmp_path / "pq", "SYNTH", "H1")
    assert len(loaded) == len(df)
    assert parquet.exists()
    back = load_from_parquet(parquet)
    assert list(back.columns) == list(df.columns)
    np.testing.assert_allclose(back["close"].to_numpy(), df["close"].to_numpy(), rtol=1e-5)
