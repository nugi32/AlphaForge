import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.features.cluster import cluster_features
from edge_research.features.leakcheck import check_no_lookahead
from edge_research.features.registry import _bars_since, build_registry
from edge_research.features.table import build_feature_table, write_features
from edge_research.synth.generator import make_synthetic_bars


@pytest.fixture(scope="module")
def data():
    df, _ = make_synthetic_bars(3000, seed=11)
    return df, load_config()


def test_table_shape_and_dtypes(data):
    df, cfg = data
    res = build_feature_table(df, cfg)
    assert res.table.shape[0] == len(df) and res.table.shape[1] >= 50
    assert (res.table.dtypes == np.float32).all()
    assert np.isfinite(res.table.to_numpy()[res.first_valid_row :]).all()
    assert res.first_valid_row <= 210                  # ma_200 warm-up dominates


def test_every_feature_is_leak_free(data):
    df, cfg = data
    bad = check_no_lookahead(lambda d: build_feature_table(d, cfg).table, df, cut=1500, seed=1)
    assert bad == []


def test_leakcheck_catches_a_leaky_function(data):
    df, cfg = data
    leaky = lambda d: pd.DataFrame({"fwd": d["close"].shift(-1)}, index=d.index)   # uses the next bar
    assert check_no_lookahead(leaky, df, cut=1500) == ["fwd"]


def test_candle_and_distance_values_match_hand_computation(data):
    df, cfg = data
    t = build_feature_table(df, cfg).table
    i = 400
    atr = float(df["atr_14"].iloc[i])
    assert t["candle_body"].iloc[i] == pytest.approx((float(df["close"].iloc[i]) - float(df["open"].iloc[i])) / atr, rel=1e-4)
    assert t["dist_ma_20_sma"].iloc[i] == pytest.approx((float(df["close"].iloc[i]) - float(df["ma_20_sma"].iloc[i])) / atr, rel=1e-4)
    assert t["ret_3_atr"].iloc[i] == pytest.approx((float(df["close"].iloc[i]) - float(df["close"].iloc[i - 3])) / atr, rel=1e-4)
    assert t["mom_10_c"].iloc[i] == pytest.approx(float(df["mom_10"].iloc[i]) - 100.0, abs=1e-4)


def test_bars_since_semantics():
    ev = np.array([0, 0, 1, 0, 0, 1, 0], dtype=bool)
    assert _bars_since(ev, cap=50).tolist() == [50, 50, 0, 1, 2, 0, 1]
    assert _bars_since(ev, cap=1).tolist() == [1, 1, 0, 1, 1, 0, 1]


def test_calendar_columns_excluded_by_default(data):
    df, cfg = data
    cols = list(df.columns) + ["cal_next_impact", "news_flag"]
    specs, skipped = build_registry(cols, cfg)
    assert not any(s.name.startswith(("cal_", "news_")) for s in specs)
    assert {n for n, _ in skipped} >= {"cal_next_impact", "news_flag"}


def test_clustering_reduces_duplicates_and_keeps_time(data):
    df, cfg = data
    res = build_feature_table(df, cfg)
    train = res.table.iloc[res.first_valid_row : 1800]
    kept, cmap = cluster_features(train, res.specs, cfg["features"]["corr_cluster_threshold"])
    assert 8 <= len(kept) < res.table.shape[1]
    assert {"hour", "dayofweek", "session"} <= set(kept)
    assert cmap["representative"].sum() == len(kept)
    # a representative has the lowest complexity of its cluster
    by = {s.name: s for s in res.specs}
    for cid, g in cmap[cmap.cluster >= 0].groupby("cluster"):
        rep = g[g.representative].feature.iloc[0]
        assert by[rep].complexity == min(by[m].complexity for m in g.feature)


def test_write_features(tmp_path, data):
    df, cfg = data
    write_features(build_feature_table(df, cfg), tmp_path)
    assert (tmp_path / "feature_table.parquet").exists() and (tmp_path / "feature_registry.json").exists()
