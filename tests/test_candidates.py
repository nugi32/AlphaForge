"""M5: exact statistics, lift over parents, minimum episodes, dedup, stage 06 (against independent brute-force computations)."""

import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.discovery.candidates import build_candidates, build_family_labels, evaluate_pair, leader_clusters
from edge_research.discovery.episodes import lift_stats, rule_stats
from edge_research.discovery.pairs import coarse_screen, family_hold, fold_bounds
from edge_research.labels.builder import family_list
from edge_research.synth.generator import make_synthetic_bars
from test_discovery import CFG, DEEP_OFF, STRONG, _cfg, _world


def _ccfg(**over):
    import copy
    c = copy.deepcopy(CFG)
    c["candidates"].update(over)
    return c


# ---------------------------------------------------------------- independent references
def _episodes_loop(idx, g):
    """Plain python episode assignment: a new episode starts when the gap to the previous trigger is > g."""
    out, e, prev = [], -1, None
    for i in idx:
        if prev is None or i - prev > g:
            e += 1
        out.append(e)
        prev = i
    return np.array(out, dtype=int)


def _cluster_ols_dummy(y, d, cl):
    """Cluster-robust OLS of y on [1, d] (sandwich, CR1 factor G/(G-1)): coefficient and SE of d. No shared code with lift_stats."""
    X = np.column_stack([np.ones(len(y)), d.astype(float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((2, 2))
    G = len(np.unique(cl))
    if G < 2:
        return float("nan"), float("nan")
    for g in np.unique(cl):
        s = X[cl == g].T @ u[cl == g]
        meat += np.outer(s, s)
    V = G / (G - 1) * XtX_inv @ meat @ XtX_inv
    return beta[1], float(np.sqrt(V[1, 1]))



def _brute_rule(trig, y, g, bounds):
    idx = np.flatnonzero(trig & np.isfinite(y))
    if idx.size == 0:
        return None
    ep = _episodes_loop(idx, g)
    G = ep.max() + 1
    yy = y[idx]
    mean = yy.mean()
    cs = np.array([(yy[ep == k] - mean).sum() for k in range(G)])
    se = np.sqrt(G / (G - 1) * (cs**2).sum()) / len(yy) if G > 1 else np.nan
    fold = [yy[(idx >= bounds[f]) & (idx < bounds[f + 1])] for f in range(len(bounds) - 1)]
    return {"n": len(idx), "G": G, "mean": mean, "se": se, "fold_mean": [f.mean() if f.size else np.nan for f in fold], "fold_n": [f.size for f in fold]}


# ---------------------------------------------------------------- rule_stats extras
def test_rule_stats_fold_means_and_gross_use_the_same_observations():
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    gross = y + 0.5
    trig = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1], bool)
    bounds = np.array([0, 3, 6, 10])
    r = rule_stats(trig, y, 1, bounds=bounds, gross=gross)
    assert r["fold_n"] == [2, 2, 3]                                      # row 2 has no label -> not an observation of fold 0
    assert r["fold_mean"] == pytest.approx([1.5, (4 + 6) / 2, (7 + 9 + 10) / 3])      # fold 1: rows 3, 5; fold 2: rows 6, 8, 9
    assert r["mean_gross"] == pytest.approx(r["mean"] + 0.5)
    e = rule_stats(np.zeros(10, bool), y, 1, bounds=bounds, gross=gross)
    assert e["fold_n"] == [0, 0, 0] and np.isnan(e["fold_mean"]).all() and np.isnan(e["mean_gross"])
    # a restriction of the rows applies to bounds, trigger, label and gross together
    r2 = rule_stats(trig, y, 1, rows=slice(3, 10), bounds=np.array([0, 3, 7]), gross=gross)
    assert r2["fold_n"] == [2, 3] and r2["fold_mean"] == pytest.approx([(4 + 6) / 2, (7 + 9 + 10) / 3])      # rows 3..9 -> restricted trigger rows 0,2 | 3,5,6


# ---------------------------------------------------------------- lift over a parent
def test_lift_stats_hand_example_and_edge_cases():
    # parent rows 0..7, episodes (g=1): {0,1} {3,4} {6,7}; child = rows 0,1,3 ; rest = 4,6,7
    y = np.array([3.0, 5.0, 0.0, 4.0, 1.0, 0.0, 0.0, 2.0])
    p = np.array([1, 1, 0, 1, 1, 0, 1, 1], bool)
    c = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
    r = lift_stats(p, c, y, 1)
    m1, m0 = (3.0 + 5.0 + 4.0) / 3, (1.0 + 0.0 + 2.0) / 3
    assert r["n_child"] == 3 and r["n_rest"] == 3 and r["lift"] == pytest.approx(m1 - m0)
    # cluster sums of residuals: ep0 child rows (3-4)+(5-4)=0 ; ep1 child row 4-4=0, rest row (1-1)=0 ; ep2 rest rows (0-1)+(2-1)=0 -> degenerate zero SE
    assert r["se"] == pytest.approx(0.0) and np.isnan(r["t"])
    y2 = np.array([3.0, 6.0, 0.0, 4.0, 1.0, 0.0, 0.0, 2.5])
    r2 = lift_stats(p, c, y2, 1)
    ref_lift, ref_se = _cluster_ols_dummy(y2[p], c[p], _episodes_loop(np.flatnonzero(p), 1))
    assert r2["lift"] == pytest.approx(ref_lift) and r2["se"] == pytest.approx(ref_se) and r2["t"] == pytest.approx(ref_lift / ref_se)
    # empty remainder, empty child, a single cluster: NaN, never a number
    assert np.isnan(lift_stats(p, p, y2, 1)["t"]) and np.isnan(lift_stats(p, np.zeros(8, bool), y2, 1)["t"])
    assert np.isnan(lift_stats(p, c, y2, 100)["t"])
    # a row with no label is no observation and does not bridge episodes
    y3 = y2.copy()
    y3[[3, 4]] = np.nan
    r3 = lift_stats(p, c, y3, 1)
    assert r3["n_child"] == 2 and r3["n_rest"] == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lift_stats_equals_cluster_robust_ols_on_random_data(seed):
    rng = np.random.default_rng(seed)
    n = 600
    y = rng.normal(0.1, 1.0, n)
    y[rng.random(n) < 0.05] = np.nan
    p = rng.random(n) < 0.45
    c = p & (rng.random(n) < 0.5)
    g = int(rng.integers(1, 6))
    r = lift_stats(p, c, y, g)
    obs = np.flatnonzero(p & np.isfinite(y))
    ref_lift, ref_se = _cluster_ols_dummy(y[obs], c[obs], _episodes_loop(obs, g))
    assert r["lift"] == pytest.approx(ref_lift, rel=1e-9) and r["se"] == pytest.approx(ref_se, rel=1e-9)
    assert r["n_child"] + r["n_rest"] == len(obs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lift_stats_block_clusters_equal_cluster_robust_ols(seed):
    rng = np.random.default_rng(seed + 10)
    n = 900
    y = rng.normal(0.1, 1.0, n)
    y[rng.random(n) < 0.05] = np.nan
    p = rng.random(n) < 0.6
    c = p & (rng.random(n) < 0.5)
    r = lift_stats(p, c, y, 3, block=20)
    obs = np.flatnonzero(p & np.isfinite(y))
    ref_lift, ref_se = _cluster_ols_dummy(y[obs], c[obs], obs // 20)
    assert r["lift"] == pytest.approx(ref_lift, rel=1e-9) and r["se"] == pytest.approx(ref_se, rel=1e-9)
    r2 = lift_stats(p, c, y, 3, rows=slice(100, 700), block=20)                      # blocks count from the start of the restricted rows
    obs2 = np.flatnonzero(p[100:700] & np.isfinite(y[100:700]))
    l2, s2 = _cluster_ols_dummy(y[100:700][obs2], c[100:700][obs2], obs2 // 20)
    assert r2["lift"] == pytest.approx(l2, rel=1e-9) and r2["se"] == pytest.approx(s2, rel=1e-9)
    with pytest.raises(ValueError):
        lift_stats(p, c, y, 3, block=0)


def test_dense_parent_is_one_episode_but_block_clusters_stay_defined():
    """D-027: a parent with support above ~1/g chains the whole segment into one episode; the episode SE is then undefined."""
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.normal(0, 1, n)
    p = rng.random(n) < 0.8                                   # dense parent: gaps are almost always <= g
    c = p & (rng.random(n) < 0.3)
    assert np.isnan(lift_stats(p, c, y, 12)["t"])             # episode clusters: one cluster -> NaN
    r = lift_stats(p, c, y, 12, block=60)
    assert np.isfinite(r["t"]) and np.isfinite(r["se"])


# ---------------------------------------------------------------- leader clustering by Jaccard
def _jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def test_leader_clusters_hand_example_boundary_and_order():
    n = 100
    a = np.zeros(n, bool); a[:50] = True
    b = np.zeros(n, bool); b[:40] = True                     # J(a,b) = 40/50 = 0.8
    c = np.zeros(n, bool); c[60:90] = True                   # disjoint
    d = np.zeros(n, bool); d[:30] = True                     # J(a,d) = 0.6, J(b,d) = 0.75
    masks = [a, c, b, d]
    cid, rep = leader_clusters(lambda i: masks[i], 4, 0.8)    # J == 0.8 joins (>=)
    assert cid.tolist() == [0, 1, 0, 2] and rep.tolist() == [True, True, False, True]
    cid, rep = leader_clusters(lambda i: masks[i], 4, 0.6)    # J(a,d) = 0.6 exactly: joins the best-ranked representative (a)
    assert cid.tolist() == [0, 1, 0, 0] and rep.tolist() == [True, True, False, False]
    cid, rep = leader_clusters(lambda i: masks[i], 4, 0.9)
    assert cid.tolist() == [0, 1, 2, 3] and rep.all()
    cid, rep = leader_clusters(lambda i: np.zeros(n, bool), 3, 0.5)    # empty masks are never merged
    assert cid.tolist() == [0, 1, 2]


@pytest.mark.parametrize("thr", [0.3, 0.6, 0.85])
def test_leader_clusters_properties_against_brute_force(thr):
    rng = np.random.default_rng(int(thr * 100))
    base = [rng.random(400) < p for p in (0.3, 0.1, 0.5, 0.05)]
    masks = []
    for _ in range(60):                                      # noisy copies of a few bases -> many true overlaps
        m = base[rng.integers(0, len(base))].copy()
        m ^= rng.random(400) < rng.choice([0.0, 0.02, 0.1])
        masks.append(m)
    cid, rep = leader_clusters(lambda i: masks[i], len(masks), thr)
    reps = [i for i in range(len(masks)) if rep[i]]
    for i in range(len(masks)):
        earlier = [r for r in reps if r < i]
        similar = [r for r in earlier if _jac(masks[i], masks[r]) >= thr]
        if rep[i]:
            assert not similar                              # a new representative is below the threshold against all earlier ones
        else:
            assert similar and cid[i] == cid[similar[0]]    # joins the BEST-RANKED similar representative
    assert sorted(set(cid.tolist())) == list(range(len(reps)))


# ---------------------------------------------------------------- exact family statistics vs brute force
def test_evaluate_pair_equals_brute_force_for_every_family():
    aset, lab_table, tr = _world(1, 12000, True)
    surv, _ = coarse_screen(aset, lab_table, CFG, tr)
    fl = build_family_labels(lab_table, CFG, "long", tr)
    n = fl.y.shape[1]
    bounds = fold_bounds(n, 5)
    assert fl.names == [f for f, _ in family_list(CFG)] and len(fl.names) == 42        # 6 T + 18 B + 18 I (intrabar_families: true)
    assert fl.hold.tolist() == [family_hold(f, CFG) for f in fl.names]
    cfg = _ccfg(min_episodes_train=50)
    for r in surv.head(3).itertuples():
        a, b = aset.dense(int(r.atom_a), tr), aset.dense(int(r.atom_b), tr)
        ev = evaluate_pair(a, b, fl, bounds, cfg["candidates"])
        for k, x in enumerate(ev["families"]):
            ref = _brute_rule(a & b, fl.y[k], int(fl.hold[k]), bounds)
            assert x["n_trigger"] == ref["n"] and x["n_episodes"] == ref["G"]
            assert x["mean"] == pytest.approx(ref["mean"], rel=1e-9) and x["se"] == pytest.approx(ref["se"], rel=1e-9)
            assert x["t"] == pytest.approx(ref["mean"] / ref["se"], rel=1e-9)
            assert x["fold_n"] == ref["fold_n"] and x["fold_mean"] == pytest.approx(ref["fold_mean"], nan_ok=True)
            obs = (a & b) & np.isfinite(fl.y[k])
            assert x["mean_gross"] == pytest.approx(fl.gross[k][obs].mean(), rel=1e-9)
            assert x["cost"] == pytest.approx(x["mean_gross"] - x["mean"], abs=1e-12) and x["cost"] >= -1e-9      # cost is a deduction
            if x["pass_t"] and np.isfinite(x["lift_a_t"]):                                 # lift against both parents, independent OLS
                for parent, key in ((a, "lift_a"), (b, "lift_b")):
                    o = np.flatnonzero(parent & np.isfinite(fl.y[k]))
                    lift, se = _cluster_ols_dummy(fl.y[k][o], (a & b)[o], o // (cfg["candidates"]["lift_block_mult"] * int(fl.hold[k])))
                    assert x[key] == pytest.approx(lift, rel=1e-8) and x[key + "_t"] == pytest.approx(lift / se, rel=1e-8)
            pe = x["n_episodes"] >= cfg["candidates"]["min_episodes_train"]
            assert x["pass_episodes"] == pe and x["pass_t"] == (pe and x["t"] >= cfg["candidates"]["t_exact"])


def test_family_labels_use_the_whole_label_set_and_net_target():
    aset, lab_table, tr = _world(1, 12000, False)
    fl = build_family_labels(lab_table, CFG, "short", tr)
    k = fl.names.index("B3_1p5_12")
    expect = (lab_table["short_B3_1p5_12_gross"] - lab_table["short_B3_1p5_12_cost"]).to_numpy(np.float64)[tr]
    np.testing.assert_allclose(fl.y[k], expect, equal_nan=True, rtol=1e-6)
    np.testing.assert_allclose(fl.gross[k], lab_table["short_B3_1p5_12_gross"].to_numpy(np.float64)[tr], equal_nan=True, rtol=1e-6)
    fg = build_family_labels(lab_table, _cfg(target="gross"), "short", tr)
    np.testing.assert_allclose(fg.y[k], fg.gross[k], equal_nan=True)


# ---------------------------------------------------------------- candidates on planted edge, null, dedup, filters
@pytest.fixture(scope="module")
def strong():
    aset, lab_table, tr = _world(1, 20000, True)
    surv, _ = coarse_screen(aset, lab_table, CFG, tr)
    cand, fam, info = build_candidates(surv, aset, lab_table, CFG, tr)
    return aset, lab_table, tr, surv, cand, fam, info


def test_planted_edge_survives_and_is_long_only(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    assert len(cand) > 0 and (cand["direction"] == "long").all()
    reps = cand[cand["is_representative"]]
    feats = lambda r: {aset.atoms[int(r.atom_a)].feature, aset.atoms[int(r.atom_b)].feature}
    planted = [r for r in reps.itertuples() if "session" in feats(r) and (feats(r) & {"ret_3_atr", "ret_6_atr", "cci_14", "rsi_14", "candle_body"})]
    assert planted, "no representative pairs the session atom with a price-drop atom"
    assert (cand["best_mean"] > 0).all() and (cand["best_t"] >= CFG["candidates"]["t_exact"]).all()
    assert (cand["best_mean_gross"] >= cand["best_mean"] - 1e-12).all()                  # costs are deducted


def test_every_candidate_obeys_every_filter_on_its_passing_families(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    c = CFG["candidates"]
    assert (cand["n_families_pass"] >= c["min_families_pass"]).all()
    passing = fam[fam["passed"]]
    assert (passing["n_episodes"] >= c["min_episodes_train"]).all() and (passing["t"] >= c["t_exact"]).all()
    assert (passing["lift_a_t"] >= c["lift_t_min"]).all() and (passing["lift_b_t"] >= c["lift_t_min"]).all()
    assert passing.groupby("cand_id").size().reindex(cand["cand_id"]).tolist() == cand["n_families_pass"].tolist()
    assert fam.groupby("cand_id").size().eq(len(fam["family"].unique())).all()          # all families of the direction are reported
    f = info["funnel"]
    assert f["survivors"] >= f["episodes"] >= f["t"] >= f["lift"] == len(cand)


def test_clusters_are_consistent_with_jaccard_on_trigger_rows(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    thr = CFG["candidates"]["dedup_jaccard"]
    trig = lambda i: aset.dense(int(cand.at[i, "atom_a"]), tr) & aset.dense(int(cand.at[i, "atom_b"]), tr)
    assert cand["cluster"].min() == 0 and cand["cluster"].nunique() == info["n_clusters"] < len(cand)      # duplicates were merged
    assert cand.groupby("cluster")["is_representative"].sum().eq(1).all()
    assert cand.groupby("cluster")["cluster_size"].first().sum() == len(cand)
    reps = cand.index[cand["is_representative"]].tolist()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(cand), 60, replace=False):                                  # members: similar to their representative's cluster leader...
        r = cand.index[(cand["cluster"] == cand.at[i, "cluster"]) & cand["is_representative"]][0]
        assert r <= i                                                                   # the representative is the best-ranked member
        if i != r:
            assert _jac(trig(i), trig(r)) >= thr - 1e-12
    for x in range(min(len(reps), 25)):                                                 # ...and representatives are mutually distinct
        for y in range(x):
            assert _jac(trig(reps[x]), trig(reps[y])) < thr
    assert cand["rank_score"].is_monotonic_decreasing


def test_tightening_any_filter_never_adds_candidates_and_extremes_empty_the_list(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    base = set(zip(cand["atom_a"], cand["atom_b"], cand["direction"]))
    for over in ({"min_episodes_train": 300}, {"t_exact": 4.0}, {"lift_t_min": 3.0}, {"min_families_pass": 4}):
        c2, _, i2 = build_candidates(surv, aset, lab_table, _ccfg(**over), tr)
        got = set(zip(c2["atom_a"], c2["atom_b"], c2["direction"]))
        assert got <= base and len(got) < len(base), over
    for over in ({"min_episodes_train": 10**6}, {"t_exact": 1e9}, {"lift_t_min": 1e9}, {"min_families_pass": 99}):
        c2, f2, i2 = build_candidates(surv, aset, lab_table, _ccfg(**over), tr)
        assert len(c2) == 0 and len(f2) == 0 and i2["n_clusters"] == 0, over
    c3, _, _ = build_candidates(surv, aset, lab_table, _ccfg(dedup_jaccard=1.01), tr)      # nothing can merge -> every candidate is its own idea
    assert len(c3) == len(cand) and c3["is_representative"].all() and c3["cluster"].nunique() == len(c3)
    c4, _, _ = build_candidates(surv, aset, lab_table, _ccfg(dedup_jaccard=0.2), tr)
    assert c4["cluster"].nunique() < cand["cluster"].nunique()


def test_null_data_gives_no_candidate_even_when_the_screen_is_relaxed():
    for seed in (1, 2, 3):
        aset, lab_table, tr = _world(seed, 20000, False)
        surv, _ = coarse_screen(aset, lab_table, CFG, tr)
        cand, fam, info = build_candidates(surv, aset, lab_table, CFG, tr)
        assert len(cand) == 0 and info["n_clusters"] == 0 and len(fam) == 0
        relaxed = _cfg(min_rows_screen=300, t_screen=1.0, min_pos_folds=3, min_labels_pass=1)
        surv2, _ = coarse_screen(aset, lab_table, relaxed, tr)
        assert len(surv2) > 50                                                          # the exact filters have something to reject
        cand2, _, info2 = build_candidates(surv2, aset, lab_table, CFG, tr)
        assert len(cand2) == 0, (seed, info2["funnel"])


def test_candidates_read_train_rows_only(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    vals = lab_table.to_numpy(np.float32, copy=True)
    after = np.arange(len(vals)) >= tr.stop
    vals[after] = np.random.default_rng(0).normal(0, 5, (int(after.sum()), vals.shape[1])).astype(np.float32)
    wrecked = pd.DataFrame(vals, index=lab_table.index, columns=lab_table.columns)
    c2, f2, _ = build_candidates(surv, aset, wrecked, CFG, tr)
    pd.testing.assert_frame_equal(cand, c2)
    pd.testing.assert_frame_equal(fam, f2)


def test_empty_survivors_give_empty_outputs():
    aset, lab_table, tr = _world(1, 12000, False)
    empty = pd.DataFrame(columns=["atom_a", "atom_b", "expr_a", "expr_b", "direction", "n_labels_pass", "score"])
    cand, fam, info = build_candidates(empty, aset, lab_table, CFG, tr)
    assert len(cand) == 0 and len(fam) == 0 and info["n_candidates"] == 0 and info["funnel"]["survivors"] == 0


def test_short_direction_is_evaluated_with_short_labels():
    from edge_research.synth.generator import PlantedEdge
    # a planted SHORT edge must come out as short candidates only (labels are never mixed between directions)
    import test_discovery as td
    cfg = td.CFG
    df, _ = make_synthetic_bars(20000, seed=5, plant_edge=PlantedEdge(drop_sigmas=1.0, drift_sigmas=3.0, hours=tuple(range(6, 14)), direction=-1))
    from edge_research.atoms.atoms import build_atoms, family_map
    from edge_research.features.cluster import cluster_features
    from edge_research.features.table import build_feature_table
    from edge_research.labels.builder import build_labels
    from edge_research.splits import make_splits
    f = build_feature_table(df, cfg)
    sp = make_splits(len(df), cfg)
    tr = slice(max(sp.train.start, f.first_valid_row), sp.train.stop)
    kept, cm = cluster_features(f.table.iloc[tr], f.specs, cfg["features"]["corr_cluster_threshold"])
    aset = build_atoms(f.table, {s.name: s.kind for s in f.specs}, family_map(cm), kept, cfg, tr)
    lab = build_labels(df, cfg).table
    surv, _ = coarse_screen(aset, lab, cfg, tr)
    cand, fam, info = build_candidates(surv, aset, lab, cfg, tr)
    assert len(cand) > 0 and (cand["direction"] == "short").all() and set(fam["direction"]) == {"short"}


# ---------------------------------------------------------------- pipeline stage 06
def test_pipeline_stage6_end_to_end_resume_and_invalidation(tmp_path, monkeypatch):
    from edge_research import pipeline
    from edge_research.pipeline import MissingUpstream, run_stage_6, run_stages_0_6
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(12000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    run = tmp_path / "run"
    with pytest.raises(MissingUpstream):
        run_stage_6(tmp_path / "nothing", CFG)
    sp_csv = tmp_path / "s_spread.csv"
    out = run_stages_0_6(csv, run, CFG, "SYNTH", sp_csv)
    assert out["status"]["06_candidates"] == "ran" and out["n_candidates"] > 0 and 0 < out["n_clusters"] <= out["n_candidates"]
    d6 = run / "06_candidates"
    for f in ("candidates.parquet", "candidate_families.parquet", "clusters.csv", "candidates_info.json", "summary.md"):
        assert (d6 / f).exists(), f
    cand = pd.read_parquet(d6 / "candidates.parquet")
    assert len(cand) == out["n_candidates"] and cand["cluster"].nunique() == out["n_clusters"]
    assert len(pd.read_csv(d6 / "clusters.csv")) == out["n_clusters"]
    assert "independent clusters" in (d6 / "summary.md").read_text(encoding="utf-8")
    assert set(run_stages_0_6(csv, run, CFG, "SYNTH", sp_csv)["status"].values()) == {"skipped"}
    cfg2 = load_config(overrides={"candidates": {"t_exact": 3.5}, **DEEP_OFF})              # only the candidates section -> only stage 6 reruns (DEEP_OFF keeps the deep section identical to CFG)
    out2 = run_stages_0_6(csv, run, cfg2, "SYNTH", sp_csv)
    assert {k: v for k, v in out2["status"].items() if v == "ran"} == {"06_candidates": "ran"} and out2["n_candidates"] <= out["n_candidates"]
    cfg3 = load_config(overrides={"candidates": {"t_exact": 3.5}, "discovery": {"min_rows_screen": 800}, **DEEP_OFF})      # discovery change -> stage 5, 5b and 6
    out3 = run_stages_0_6(csv, run, cfg3, "SYNTH", sp_csv)
    assert out3["status"]["05_discovery"] == "ran" and out3["status"]["06_candidates"] == "ran" and out3["status"]["04_atoms"] == "skipped"
    # a new Stage 5 run (here: a code version bump) invalidates Stage 6 even though its own config and code did not change
    monkeypatch.setitem(pipeline.STAGE_VERSIONS, "05_discovery", "99")
    out4 = run_stages_0_6(csv, run, cfg3, "SYNTH", sp_csv)
    assert out4["status"]["05_discovery"] == "ran" and out4["status"]["06_candidates"] == "ran"


def test_min_episodes_boundary_is_inclusive_and_best_family_is_the_max_t_passing_family(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    r = cand.iloc[0]
    one = fam[(fam["cand_id"] == r["cand_id"])]
    best = one[one["passed"]].sort_values("t", ascending=False).iloc[0]
    assert r["best_family"] == best["family"] and r["best_t"] == pytest.approx(best["t"]) and r["best_n_episodes"] == best["n_episodes"]
    assert r["rank_score"] == pytest.approx(np.median(np.where(np.isfinite(one["t"]), one["t"], -np.inf)))      # median over ALL 24 families (breadth counts)
    fl = build_family_labels(lab_table, CFG, "long", tr)
    a, b = aset.dense(int(r["atom_a"]), tr), aset.dense(int(r["atom_b"]), tr)
    bounds = fold_bounds(fl.y.shape[1], 5)
    k = fl.names.index(best["family"])
    n_ep = int(best["n_episodes"])
    at = evaluate_pair(a, b, fl, bounds, {**CFG["candidates"], "min_episodes_train": n_ep})["families"][k]
    above = evaluate_pair(a, b, fl, bounds, {**CFG["candidates"], "min_episodes_train": n_ep + 1})["families"][k]
    assert at["pass_episodes"] and not above["pass_episodes"] and not above["pass_t"] and not above["passed"]


def test_directions_are_never_merged_and_both_direction_pairs_are_flagged(strong, monkeypatch):
    """The same pair surviving in both directions gives two clusters (never merged) and is flagged as a suspected volatility marker."""
    import edge_research.discovery.candidates as cm
    aset, lab_table, tr, surv, cand, fam, info = strong

    def fake_eval(trig, parents, fl, bounds, c, n_years=None):
        fams = [{"family": f, "g": 3, "n_trigger": 500, "n_episodes": 300, "mean": 0.2, "mean_gross": 0.3, "cost": 0.1, "se": 0.05, "t": 4.0, "pct_positive": 0.6,
                 "fold_mean": [0.2] * 5, "fold_n": [100] * 5, "lift_a": 0.1, "lift_a_t": 2.0, "lift_b": 0.1, "lift_b_t": 2.0, "lift_min": 0.1, "lift_min_t": 2.0,
                 "pass_episodes": True, "pass_t": True, "pass_lift": True, "passed": True} for f in fl.names[:3]]
        return {"families": fams, "n_pass_episodes": 3, "n_pass_t": 3, "n_pass": 3}

    monkeypatch.setattr(cm, "evaluate_rule", fake_eval)       # stage 6 evaluates N-atom rules (M11); patching evaluate_pair would not take effect
    two = pd.concat([surv.head(1).assign(direction="long"), surv.head(1).assign(direction="short")], ignore_index=True)
    c2, f2, i2 = cm.build_candidates(two, aset, lab_table, CFG, tr)
    assert sorted(c2["direction"]) == ["long", "short"] and c2["cluster"].nunique() == 2 and c2["is_representative"].all()
    assert c2["both_directions"].all() and i2["n_both_directions"] == 2
    assert i2["clusters_by_direction"] == {"long": 1, "short": 1}
    one = cm.build_candidates(surv.head(1), aset, lab_table, CFG, tr)[0]
    assert not one["both_directions"].any()


def test_rank_rewards_breadth_over_a_single_high_t(strong):
    aset, lab_table, tr, surv, cand, fam, info = strong
    narrow = cand[cand["n_families_pass"] == cand["n_families_pass"].min()]
    broad = cand[cand["n_families_pass"] == cand["n_families_pass"].max()]
    assert cand["n_families_pass"].max() >= 10 > cand["n_families_pass"].min() == 2
    assert broad["rank_score"].max() > narrow["rank_score"].max()                       # the broadest rules outrank the narrowest ones
    assert cand["n_families_pass"].iloc[:50].mean() > cand["n_families_pass"].mean()    # the top of the ranking is broader than the average candidate
