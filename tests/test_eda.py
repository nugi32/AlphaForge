import json

import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.eda.univariate import cluster_mean_se, make_bins, period_codes, resolve_t_min, run_univariate
from edge_research.features.cluster import cluster_features
from edge_research.features.table import build_feature_table
from edge_research.labels.builder import build_labels
from edge_research.pipeline import NoGo, run_stages_0_3
from edge_research.splits import make_splits
from edge_research.synth.generator import PlantedEdge, make_synthetic_bars, write_exporter_csv

CFG = load_config()
STRONG = PlantedEdge(drop_sigmas=1.0, drift_sigmas=3.0, hours=tuple(range(6, 14)))


def _prep(seed, edge=None, n=20000, cfg=CFG):
    df, meta = make_synthetic_bars(n, seed=seed, plant_edge=edge)
    f = build_feature_table(df, cfg)
    lab = build_labels(df, cfg, meta["spread_points"]).table
    sp = make_splits(len(df), cfg)
    tr = f.table.iloc[max(sp.train.start, f.first_valid_row): sp.train.stop]
    kept, _ = cluster_features(tr, f.specs, cfg["features"]["corr_cluster_threshold"])
    return f, lab, sp, kept, {s.name: s.kind for s in f.specs}


def _run(seed, edge=None, n=20000, cfg=CFG):
    f, lab, sp, kept, kinds = _prep(seed, edge, n, cfg)
    return run_univariate(f.table, lab, kinds, kept, cfg, sp, f.first_valid_row), (f, lab, sp, kept, kinds)


def test_cluster_se_matches_brute_force_and_handles_nan():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 15)
    y = rng.normal(size=g.size) + np.repeat(rng.normal(size=20), 15)          # clustered noise
    y[::7] = np.nan
    m, se, n = cluster_mean_se(y, g, 20)
    v = np.isfinite(y)
    sums = np.array([(y[v & (g == k)] - y[v].mean()).sum() for k in range(20)])
    assert n == v.sum() and m == pytest.approx(y[v].mean())
    assert se == pytest.approx(np.sqrt(20 / 19 * (sums**2).sum()) / v.sum())
    assert np.isnan(cluster_mean_se(np.array([1.0]), np.array([0]), 1)[1])


def test_make_bins_equal_frequency_categorical_and_nan():
    x = np.r_[np.arange(1000, dtype=float), np.nan]
    b, edges, cat = make_bins(x, "numeric", 10, 10)
    assert not cat and len(edges) == 10 and b[-1] == -1 and (np.bincount(b[b >= 0]) == 100).all()
    assert edges[0] == (0.0, 99.0) and edges[-1][1] == 999.0
    b2, e2, cat2 = make_bins(np.array([0.0, 1.0, 1.0, 0.0, np.nan]), "event", 10, 10)
    assert cat2 and e2 == [(0.0, 0.0), (1.0, 1.0)] and list(b2) == [0, 1, 1, 0, -1]
    _, e3, cat3 = make_bins(np.tile(np.arange(24.0), 50), "time", 10, 10)
    assert cat3 and len(e3) == 24                                              # hour is categorical even with > 10 levels


def test_period_codes_and_t_min_resolution():
    idx = pd.date_range("2020-01-01", periods=24 * 400, freq="1h")            # about 1.1 years -> quarters
    _, unit = period_codes(idx, 3, 50)
    assert unit == "quarter"
    idx2 = pd.date_range("2018-01-01", periods=24 * 365 * 4, freq="1h")
    assert period_codes(idx2, 3, 50)[1] == "year"
    assert resolve_t_min({"t_min": "auto", "alpha": 0.05}, 20000) == pytest.approx(4.56, abs=0.02)
    assert resolve_t_min({"t_min": 3.0, "alpha": 0.05}, 20000) == 3.0


def test_planted_edge_is_visible_and_hits_sit_on_planted_structure():
    res, _ = _run(1, STRONG)
    v = res.verdict
    assert v["verdict"] == "GO" and v["n_consistent_hits"] > 0
    assert v["n_cells_tested"] > 10000 and v["train_rows"] > 10000
    hits = res.hits
    assert (hits["direction"] == "long").all()                                 # edge was planted long; no short hits
    hh = hits[hits["feature"] == "hour"]
    # Hour hits sit in the planted window 6-13, or a few hours earlier because label windows (up to 24 bars) reach into it.
    assert len(hh) > 0 and hh["bin_lo"].between(3, 13).all()
    assert hits["lift_gross_R"].gt(0).all() and hits["frac_pos_periods"].ge(CFG["eda"]["min_frac_positive_periods"]).all()


def test_short_edge_gives_short_hits_only():
    res, _ = _run(1, PlantedEdge(direction=-1, drop_sigmas=1.0, drift_sigmas=3.0, hours=tuple(range(6, 14))))
    assert res.verdict["verdict"] == "GO" and (res.hits["direction"] == "short").all()


def test_null_data_gives_no_go_on_fixed_seeds():
    bad = []
    for seed in (1, 2, 3, 4, 5, 6):
        res, _ = _run(seed)
        if res.verdict["verdict"] != "NO_GO":
            bad.append((seed, res.verdict["features_with_hits"]))
    assert not bad, bad                                                        # calibration: 0 of 16 seeds failed when tuned (see NOTES D-017)


def test_stage3_uses_train_rows_only():
    res, (f, lab, sp, kept, kinds) = _run(1, STRONG)
    lab2, f2 = lab.copy(), f.table.copy()
    lab2.iloc[sp.train.stop:] = lab2.iloc[sp.train.stop:].to_numpy() * -7.0 + 3.0       # destroy validation + holdout labels
    f2.iloc[sp.train.stop:] = f2.iloc[sp.train.stop:].to_numpy() * 5.0
    res2 = run_univariate(f2, lab2, kinds, kept, CFG, sp, f.first_valid_row)
    pd.testing.assert_frame_equal(res.univariate, res2.univariate)
    assert res.verdict == res2.verdict


def test_univariate_table_internal_consistency():
    res, _ = _run(2)
    u = res.univariate
    assert {"feature", "bin", "label", "n", "mean_R", "se", "ci_lo", "ci_hi", "lift_R", "lift_gross_R", "t_lift_gross", "frac_pos_periods"} <= set(u.columns)
    assert (u["ci_hi"] - u["ci_lo"]).dropna().gt(0).all()
    base = res.baseline.set_index("label")
    feat = u.loc[~u["categorical"], "feature"].iloc[0]
    one = u[(u["feature"] == feat) & (u["label"] == "long_T6")]
    assert len(one) == CFG["eda"]["n_bins"]
    assert one["n"].sum() <= base.loc["long_T6", "n"]
    wmean = (one["mean_R"] * one["n"]).sum() / one["n"].sum()
    assert wmean == pytest.approx(base.loc["long_T6", "mean_net_R"], abs=0.02)   # bins partition (almost) all train rows
    assert set(res.time_profile["by"]) == {"hour", "dayofweek"} and len(res.summary_md) > 200


def test_pipeline_stage3_end_to_end_resume_and_stop_flag(tmp_path):
    df, meta = make_synthetic_bars(8000, seed=9)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    run = tmp_path / "run"
    out = run_stages_0_3(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")
    assert out["status"]["03_eda"] == "ran" and out["eda"]["verdict"] in ("NO_GO", "GO", "GO_WEAK", "GO_REGIME_CAUTION")
    for rel in ("univariate.csv", "baseline.csv", "time_profile.csv", "hits.csv", "summary.md", "verdict.json"):
        assert (run / "03_eda" / rel).exists(), rel
    assert json.loads((run / "manifest.json").read_text())["holdout_accesses"] == []
    out2 = run_stages_0_3(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")
    assert set(out2["status"].values()) == {"skipped"}
    cfg2 = load_config(overrides={"eda": {"min_n_cell": 300}})                   # only the eda section changed -> only stage 3 reruns
    out3 = run_stages_0_3(csv, run, cfg2, "SYNTH", tmp_path / "s_spread.csv")
    assert out3["status"] == {"00_data_qa": "skipped", "01_features": "skipped", "02_labels": "skipped", "03_eda": "ran"}
    if out3["eda"]["verdict"] == "NO_GO":
        with pytest.raises(NoGo):
            run_stages_0_3(csv, run, load_config(overrides={"eda": {"min_n_cell": 300, "stop_on_no_go": True}}), "SYNTH", tmp_path / "s_spread.csv")
