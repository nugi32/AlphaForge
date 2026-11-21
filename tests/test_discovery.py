"""M4b: episodes, pair statistics (vs brute force), coarse screen (vs brute force), planted edge and null behaviour."""

import copy
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from edge_research.atoms.atoms import build_atoms, compatible, family_map
from edge_research.config import load_config
from edge_research.discovery.episodes import episode_ids, rule_stats
from edge_research.discovery.pairs import (build_screen_labels, coarse_screen, family_hold, fold_bounds, legal_matrix, marginal_stats,
                                           tile_stats)
from edge_research.eda.univariate import cluster_mean_se
from edge_research.features.cluster import cluster_features
from edge_research.features.table import build_feature_table
from edge_research.labels.builder import build_labels
from edge_research.splits import make_splits
from edge_research.synth.generator import PlantedEdge, make_synthetic_bars

# The deep engines (stage 5b) have their own dedicated test file; they are OFF in the shared CFG so the end-to-end
# pipeline tests below stay fast: on the strong synthetic world the Apriori expansion alone hits its 100k cap and
# stage 6 would take many minutes on the deep candidates.
DEEP_OFF = {"deep": {"apriori": {"enabled": False}, "trees": {"enabled": False}}}
CFG = load_config(overrides=DEEP_OFF)
STRONG = PlantedEdge(drop_sigmas=1.0, drift_sigmas=3.0, hours=tuple(range(6, 14)))


@lru_cache(maxsize=8)
def _world(seed: int, n: int, strong: bool):
    df, _ = make_synthetic_bars(n, seed=seed, plant_edge=STRONG if strong else None)
    f = build_feature_table(df, CFG)
    sp = make_splits(len(df), CFG)
    tr = slice(max(sp.train.start, f.first_valid_row), sp.train.stop)
    kept, cm = cluster_features(f.table.iloc[tr], f.specs, CFG["features"]["corr_cluster_threshold"])
    kinds = {s.name: s.kind for s in f.specs}
    aset = build_atoms(f.table, kinds, family_map(cm), kept, CFG, tr)
    return aset, build_labels(df, CFG).table, tr


def _cfg(**over):
    c = copy.deepcopy(CFG)
    c["discovery"].update(over)
    return c


# ---------------------------------------------------------------- episodes
def test_episode_ids_hand_examples():
    np.testing.assert_array_equal(episode_ids(np.array([2, 3, 4, 10, 11, 30]), 1), [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(episode_ids(np.array([2, 5, 8, 20]), 3), [0, 0, 0, 1])      # distance == g merges
    np.testing.assert_array_equal(episode_ids(np.array([2, 6]), 3), [0, 1])                   # distance g + 1 does not
    assert episode_ids(np.array([], dtype=int), 3).size == 0
    with pytest.raises(ValueError):
        episode_ids(np.array([3, 3]), 1)


def test_rule_stats_matches_manual_cluster_formula_and_ignores_nan_labels():
    y = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 7.0])
    trig = np.zeros(10, bool)
    trig[[0, 1, 2, 8, 9]] = True
    r = rule_stats(trig, y, g=2)
    assert r["n_trigger"] == 5 and r["n_episodes"] == 2                                       # {0,1,2} and {8,9}
    mean = np.mean([1, 2, 3, 5, 7])
    c1, c2 = (1 + 2 + 3 - 3 * mean), (5 + 7 - 2 * mean)
    se = np.sqrt(2 / 1 * (c1**2 + c2**2)) / 5
    assert r["mean"] == pytest.approx(mean) and r["se"] == pytest.approx(se) and r["t"] == pytest.approx(mean / se)
    assert r["pct_positive"] == 1.0
    # a trigger without a label is not an observation and must not bridge two episodes
    y2 = y.copy()
    y2[[1, 2]] = np.nan
    r2 = rule_stats(trig, y2, g=1)
    assert r2["n_trigger"] == 3 and r2["n_episodes"] == 2
    # the row restriction slices both arrays
    assert rule_stats(trig, y, g=2, rows=slice(0, 3))["n_trigger"] == 3


def test_rule_stats_empty_and_single_episode():
    r = rule_stats(np.zeros(5, bool), np.ones(5), 2)
    assert r["n_trigger"] == 0 and np.isnan(r["t"])
    one = rule_stats(np.array([True, True, True]), np.array([1.0, 2.0, 3.0]), 5)
    assert one["n_episodes"] == 1 and np.isnan(one["se"]) and np.isnan(one["t"])
    m, se, n = cluster_mean_se(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), 1)
    assert n == 3 and np.isnan(se)


# ---------------------------------------------------------------- pair statistics vs brute force
def _random_problem(seed=0, n=530, k=9, n_labels=3):
    rng = np.random.default_rng(seed)
    d = (rng.random((n, k)) < rng.uniform(0.1, 0.7, k)).astype(np.uint8)
    y = rng.normal(0, 1, (n_labels, n))
    y[0, -25:] = np.nan                                                    # windows leaving the data
    y[1, rng.random(n) < 0.05] = np.nan                                    # scattered gaps; a different validity mask
    y[2] = y[0] + 0.0                                                      # identical validity mask: shares a count product
    return d, y


@pytest.mark.parametrize("dtype,tol", [("float64", 1e-9), ("float32", 2e-4)])
def test_tile_stats_equal_brute_force_per_fold_with_nan_labels(dtype, tol):
    d, y = _random_problem()
    bounds = fold_bounds(d.shape[0], 5)
    di, dj = d[:, :4], d[:, 4:]
    N, S = tile_stats(di, dj, y, bounds, chunk_rows=37, dtype=dtype)             # chunk size that does not divide the folds
    assert N.shape == (3, 5, 4, 5)
    for l in range(3):
        for f in range(5):
            sl = slice(bounds[f], bounds[f + 1])
            v = np.isfinite(y[l, sl])
            for a in range(4):
                for b in range(5):
                    m = (di[sl, a] == 1) & (dj[sl, b] == 1) & v
                    assert N[l, f, a, b] == m.sum()
                    assert S[l, f, a, b] == pytest.approx(y[l, sl][m].sum(), rel=tol, abs=tol)
    # square tile (dj=None) equals the explicit self-product, and the diagonal is the marginal
    Ns, Ss = tile_stats(di, None, y, bounds, 64, dtype)
    Ne, Se = tile_stats(di, di, y, bounds, 64, dtype)
    np.testing.assert_array_equal(Ns, Ne)
    nm, sm = marginal_stats(di, y, bounds)
    for a in range(4):
        np.testing.assert_allclose(Ns[:, :, a, a], nm[:, :, a])
        np.testing.assert_allclose(Ss[:, :, a, a], sm[:, :, a], rtol=tol, atol=tol)


def test_four_cells_from_three_numbers_are_exact():
    d, y = _random_problem(seed=3)
    bounds = fold_bounds(d.shape[0], 5)
    N, S = tile_stats(d[:, :3], d[:, 3:6], y, bounds, 100, "float64")
    nA, sA = marginal_stats(d[:, :3], y, bounds)
    nB, sB = marginal_stats(d[:, 3:6], y, bounds)
    v = np.isfinite(y[1])
    a, b = d[:, 1] == 1, d[:, 4] == 1
    n_ab, s_ab = N[1].sum(0)[1, 1], S[1].sum(0)[1, 1]
    assert nA[1].sum(0)[1] - n_ab == (a & ~b & v).sum()                    # A & ~B
    assert nB[1].sum(0)[1] - n_ab == (~a & b & v).sum()                    # ~A & B
    assert sA[1].sum(0)[1] - s_ab == pytest.approx(y[1][a & ~b & v].sum())


def test_legal_matrix_equals_the_scalar_compatibility_rule_for_all_pairs():
    aset, _, _ = _world(1, 12000, False)
    L = legal_matrix(aset.atoms)
    for i, a in enumerate(aset.atoms):
        for j, b in enumerate(aset.atoms):
            assert L[i, j] == compatible(a, b), (a.expr(), b.expr())
    assert not L.diagonal().any() and (L == L.T).all()


def test_family_hold_and_screen_labels_are_train_only_net_values():
    assert family_hold("T3", CFG) == 3 and family_hold("T12", CFG) == 12 and family_hold("B2_1_12", CFG) == 12
    with pytest.raises(KeyError):
        family_hold("T99", CFG)
    aset, lab_table, tr = _world(1, 12000, False)
    lab = build_screen_labels(lab_table, CFG, tr)
    assert lab.names == ["long_T3", "long_T12", "long_B2_1_12", "short_T3", "short_T12", "short_B2_1_12"]
    assert lab.y.shape == (6, tr.stop - tr.start)
    expect = (lab_table["long_T3_gross"] - lab_table["long_T3_cost"]).to_numpy(np.float64)[tr]
    np.testing.assert_allclose(lab.y[0], expect, equal_nan=True, rtol=1e-6)
    g = build_screen_labels(lab_table, _cfg(target="gross"), tr)
    np.testing.assert_allclose(g.y[0], lab_table["long_T3_gross"].to_numpy(np.float64)[tr], equal_nan=True, rtol=1e-6)


# ---------------------------------------------------------------- coarse screen vs an independent brute force
def _brute_screen(aset, lab_table, cfg, rows, ids):
    """Per pair and per label, straight from boolean masks; no matrix products, no tiles."""
    d = cfg["discovery"]
    lab = build_screen_labels(lab_table, cfg, rows)
    n = lab.y.shape[1]
    bounds = fold_bounds(n, d["n_folds"])
    dense = {i: aset.dense(i, rows) for i in ids}
    out = {}
    for x, i in enumerate(ids):
        for j in ids[x + 1:]:
            if not compatible(aset.atoms[i], aset.atoms[j]):
                continue
            a, b = dense[i], dense[j]
            per = []
            for l in range(len(lab.names)):
                y, v = lab.y[l], np.isfinite(lab.y[l])
                m = a & b & v
                nn = int(m.sum())
                mean = y[m].mean() if nn else np.nan
                t = mean * np.sqrt(nn / lab.hold[l]) / lab.sd[l] if nn else np.nan
                pos = 0
                for f in range(d["n_folds"]):
                    sl = slice(bounds[f], bounds[f + 1])
                    mf = m[sl]
                    pos += int(mf.sum() >= d["min_rows_fold"] and y[sl][mf].sum() > 0)
                ra, rb = a & ~b & v, ~a & b & v
                la = mean - (y[ra].mean() if ra.any() else np.nan)
                lb = mean - (y[rb].mean() if rb.any() else np.nan)
                lift = np.fmin(la, lb)
                ok = bool(nn >= d["min_rows_screen"] and t >= d["t_screen"] and pos >= d["min_pos_folds"] and lift > 0)
                per.append((nn, mean, t, pos, lift, ok))
            out[(i, j)] = per
    return lab, out


def test_coarse_screen_equals_brute_force_pair_by_pair():
    aset, lab_table, tr = _world(1, 12000, True)
    cfg = _cfg(min_rows_screen=100, t_screen=1.0, min_pos_folds=3, min_labels_pass=1, tile=16, chunk_rows=500, matmul_dtype="float64")
    ids = list(range(0, len(aset.atoms), 6))
    surv, info = coarse_screen(aset, lab_table, cfg, tr, atom_ids=ids)
    lab, ref = _brute_screen(aset, lab_table, cfg, tr, ids)
    assert info["n_tiles"] > 3 and info["n_atoms"] == len(ids)                     # several tiles, including off-diagonal ones
    expected = set()
    for (i, j), per in ref.items():
        for direction in ("long", "short"):
            if sum(p[5] for p, dr in zip(per, lab.directions) if dr == direction) >= cfg["discovery"]["min_labels_pass"]:
                expected.add((i, j, direction))
    got = {(int(r.atom_a), int(r.atom_b), r.direction) for r in surv.itertuples()}
    assert expected and got == expected, (len(expected), len(got))
    for r in surv.itertuples():                                                    # reported numbers equal the brute-force numbers
        per = ref[(int(r.atom_a), int(r.atom_b))]
        for l, nm in enumerate(lab.names):
            if lab.directions[l] != r.direction:
                continue
            nn, mean, t, pos, lift, ok = per[l]
            assert getattr(r, f"{nm}_n") == nn and getattr(r, f"{nm}_pos_folds") == pos and bool(getattr(r, f"{nm}_pass")) == ok
            assert getattr(r, f"{nm}_mean") == pytest.approx(mean, rel=1e-9, abs=1e-12)
            assert getattr(r, f"{nm}_t") == pytest.approx(t, rel=1e-9, abs=1e-12)
            assert getattr(r, f"{nm}_min_lift") == pytest.approx(lift, rel=1e-8, abs=1e-12)


def test_survivors_never_contain_illegal_pairs_or_duplicate_pairs():
    aset, lab_table, tr = _world(1, 12000, True)
    surv, _ = coarse_screen(aset, lab_table, _cfg(), tr)
    assert len(surv) > 0
    assert (surv["atom_a"] < surv["atom_b"]).all()
    assert not surv.duplicated(["atom_a", "atom_b", "direction"]).any()
    for r in surv.head(300).itertuples():
        assert compatible(aset.atoms[int(r.atom_a)], aset.atoms[int(r.atom_b)])


# ---------------------------------------------------------------- planted edge, null, train-only, cap
def test_strong_planted_long_edge_is_found_and_only_long():
    aset, lab_table, tr = _world(1, 20000, True)
    surv, info = coarse_screen(aset, lab_table, CFG, tr)
    assert info["survivors_by_direction"]["short"] == 0 and info["survivors_by_direction"]["long"] > 0
    top = surv.head(20)
    feats = {aset.atoms[int(a)].feature for a in top["atom_a"]} | {aset.atoms[int(b)].feature for b in top["atom_b"]}
    assert "session" in feats                                                      # hours 6-13 = the session level the generator plants
    assert (top["n_labels_pass"] >= 2).all() and (surv["score"].diff().dropna() <= 1e-12).all()


def test_null_data_yields_no_survivors_on_several_seeds():
    for seed in (1, 2, 3):
        aset, lab_table, tr = _world(seed, 20000, False)
        surv, info = coarse_screen(aset, lab_table, CFG, tr)
        assert info["n_survivors"] == 0, (seed, info["n_survivors"])


def test_screen_reads_train_rows_only():
    aset, lab_table, tr = _world(1, 12000, True)
    a, _ = coarse_screen(aset, lab_table, CFG, tr)
    vals = lab_table.to_numpy(np.float32, copy=True)
    rng = np.random.default_rng(0)
    after = np.arange(len(vals)) >= tr.stop
    vals[after] = rng.normal(0, 5, (int(after.sum()), vals.shape[1])).astype(np.float32)
    wrecked = pd.DataFrame(vals, index=lab_table.index, columns=lab_table.columns)
    b, _ = coarse_screen(aset, wrecked, CFG, tr)
    assert len(a) > 0
    assert a.equals(b)


def test_survivor_cap_is_applied_and_flagged():
    aset, lab_table, tr = _world(1, 12000, True)
    full, i0 = coarse_screen(aset, lab_table, CFG, tr)
    capped, i1 = coarse_screen(aset, lab_table, _cfg(max_survivors=5), tr)
    assert len(full) > 5 and len(capped) == 5 and i1["truncated"] and not i0["truncated"]
    assert capped["score"].tolist() == full["score"].head(5).tolist()


# ---------------------------------------------------------------- pipeline stage 05
def test_pipeline_stage5_end_to_end_resume_and_invalidation(tmp_path):
    import pandas as pd
    from edge_research.pipeline import run_stages_0_5
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(12000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    run = tmp_path / "run"
    out = run_stages_0_5(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")
    assert out["status"]["05_discovery"] == "ran" and out["n_survivors"] > 0
    surv = pd.read_parquet(run / "05_discovery" / "survivors_pairs.parquet")
    assert len(surv) == out["n_survivors"] and set(surv["direction"]) == {"long"}
    assert (run / "05_discovery" / "summary.md").exists() and (run / "05_discovery" / "screen_info.json").exists()
    assert set(run_stages_0_5(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")["status"].values()) == {"skipped"}
    cfg2 = load_config(overrides={"discovery": {"min_rows_screen": 800}})                   # only discovery config changed -> only stage 5 reruns
    out3 = run_stages_0_5(csv, run, cfg2, "SYNTH", tmp_path / "s_spread.csv")
    assert out3["status"] == {"00_data_qa": "skipped", "01_features": "skipped", "02_labels": "skipped", "03_eda": "skipped",
                              "04_atoms": "skipped", "05_discovery": "ran"}
    assert out3["n_survivors"] <= out["n_survivors"]
    cfg3 = load_config(overrides={"atoms": {"quantile_pcts": [25, 50, 75]}})                # atoms change -> stage 4 and 5 rerun
    out4 = run_stages_0_5(csv, run, cfg3, "SYNTH", tmp_path / "s_spread.csv")
    assert out4["status"]["04_atoms"] == "ran" and out4["status"]["05_discovery"] == "ran"
