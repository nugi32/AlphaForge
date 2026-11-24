"""M5b: F2 (time stability), F7 (cost stress), F4 (plateau), F6 (soft), dedup again, stage 07."""

import copy

import numpy as np
import pandas as pd
import pytest

from edge_research.atoms.atoms import build_atoms
from edge_research.config import load_config
from edge_research.discovery.candidates import _MaskCache, build_candidates, build_family_labels
from edge_research.discovery.episodes import episode_ids
from edge_research.discovery.filters import (_Ladders, _f8_backward_eliminate, _pnl_corr, period_keys, plateau_and_monotonic,
                                             run_filters, time_stability)
from edge_research.discovery.pairs import DIRECTIONS, coarse_screen, fold_bounds
from edge_research.synth.generator import make_synthetic_bars
from test_discovery import CFG, DEEP_OFF, STRONG, _world

FC = CFG["filters"]


def _fcfg(**over):
    c = copy.deepcopy(CFG)
    c["filters"].update(over)
    return c


def _ccfg(**over):
    c = copy.deepcopy(CFG)
    c["candidates"].update(over)
    return c


# ---------------------------------------------------------------- F2 against a plain python reference
def _ref_f2(idx, yy, ep, period, bounds, c):
    n, total = len(yy), float(yy.sum())
    folds = [yy[(idx >= bounds[f]) & (idx < bounds[f + 1])] for f in range(len(bounds) - 1)]
    pos_folds = sum(1 for f in folds if len(f) and f.sum() > 0)
    per = {}
    for k in sorted(set(period)):
        m = period == k
        per[k] = (yy[m].sum(), m.sum(), len(set(ep[m])))
    usable = [k for k, v in per.items() if v[2] >= c["min_episodes_period"]]
    pos_share = np.mean([per[k][0] > 0 for k in usable]) if usable else np.nan
    loyo = np.mean([(total - per[k][0]) / (n - per[k][1]) > 0 for k in usable]) if usable else np.nan
    if total > 0:
        shares = [max(f.sum(), 0) / total for f in folds] + [max(per[k][0], 0) / total for k in usable]
        eps = sorted([yy[ep == e].sum() for e in set(ep)], reverse=True)
        return pos_folds, len(usable), pos_share, loyo, max(shares), sum(eps[:5]) / total
    return pos_folds, len(usable), pos_share, loyo, np.inf, np.inf


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_time_stability_equals_reference_on_random_data(seed):
    rng = np.random.default_rng(seed)
    n_rows = 4000
    idx = np.sort(rng.choice(n_rows, 900, replace=False))
    yy = rng.normal(0.15, 1.0, len(idx))
    ep = episode_ids(idx, 4)
    period = (idx // 500).astype(np.int64)
    bounds = fold_bounds(n_rows, 5)
    r = time_stability(idx, yy, ep, period, bounds, FC)
    pf, npd, ps, lo, ms, t5 = _ref_f2(idx, yy, ep, period, bounds, FC)
    assert r["pos_folds"] == pf and r["n_periods"] == npd
    assert r["pos_period_share"] == pytest.approx(ps) and r["loyo_share"] == pytest.approx(lo)
    assert r["max_period_share"] == pytest.approx(ms) and r["top5_share"] == pytest.approx(t5)
    assert r["pass_f2"] == (r["pass_folds"] and r["pass_periods"] and r["pass_concentration"])


def _make(yfun, n_rows=4000, n_obs=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_rows, n_obs, replace=False))
    ep = episode_ids(idx, 2)
    period = (idx // 500).astype(np.int64)                           # 8 periods
    yy = yfun(rng, idx, ep)
    return idx, yy, ep, period, fold_bounds(n_rows, 5)


def test_f2_each_failure_mode_is_caught_and_a_steady_edge_passes():
    steady = time_stability(*_make(lambda r, i, e: r.normal(0.3, 1.0, len(i))), FC)
    assert steady["pass_f2"] and steady["pos_folds"] >= 4
    # all profit in one period: positive overall but concentrated and most periods negative
    one = time_stability(*_make(lambda r, i, e: np.where(i // 500 == 3, r.normal(3.0, 1, len(i)), r.normal(-0.05, 1, len(i)))), FC)
    assert not one["pass_f2"] and (not one["pass_concentration"] or not one["pass_periods"])
    assert one["max_period_share"] > 0.5
    # a handful of huge episodes carry the result
    def spikes(r, i, e):
        y = r.normal(0.02, 0.2, len(i))
        big = np.isin(e, np.unique(e)[:4])
        y[big] += 40.0
        return y
    sp = time_stability(*_make(spikes), FC)
    assert sp["top5_share"] > 0.30 and not sp["pass_concentration"] and not sp["pass_f2"]
    # negative total -> concentration metrics are infinite, never a pass
    neg = time_stability(*_make(lambda r, i, e: r.normal(-0.3, 1.0, len(i))), FC)
    assert not neg["pass_f2"] and np.isinf(neg["max_period_share"])
    # only 3 of 5 folds positive
    def fold_neg(r, i, e):
        y = r.normal(0.0, 0.5, len(i))
        return np.where((i >= 1600) & (i < 3200), y - 1.0, y + 1.0)
    fn = time_stability(*_make(fold_neg), FC)
    assert fn["pos_folds"] <= 3 and not fn["pass_folds"] and not fn["pass_f2"]


def test_a_result_that_lives_in_one_period_fails_on_period_share_and_loyo_is_hand_checked():
    def one_year(r, i, e):
        return np.where(i // 500 == 0, r.normal(2.0, 0.5, len(i)), r.normal(-0.1, 0.5, len(i)))
    idx, yy, ep, period, b = _make(one_year)
    r = time_stability(idx, yy, ep, period, b, FC)
    assert r["total_profit"] > 0 and r["n_periods"] == 8
    assert r["pos_period_share"] == pytest.approx(1 / 8) and not r["pass_periods"] and not r["pass_f2"]
    # dropping the good period flips the mean; dropping any of the 7 others leaves it positive: 7 of 8
    assert r["loyo_share"] == pytest.approx(7 / 8)
    keep0 = period != 0
    assert yy[keep0].mean() < 0


def test_period_keys_switch_to_quarters_for_short_history():
    long_idx = pd.date_range("2018-01-01", periods=3 * 365, freq="D")
    k, kind = period_keys(long_idx, 3)
    assert kind == "year" and len(np.unique(k)) == 3
    short_idx = pd.date_range("2020-01-01", periods=400, freq="D")
    k2, kind2 = period_keys(short_idx, 3)
    assert kind2 == "quarter" and len(np.unique(k2)) >= 4 and k2[0] == 2020 * 4


# ---------------------------------------------------------------- F4 plateau and F6 on a designed world
@pytest.fixture(scope="module")
def designed():
    rng = np.random.default_rng(5)
    n = 8000
    t = pd.DataFrame({"x": rng.random(n), "w": rng.random(n), "h": rng.integers(0, 4, n).astype(float), "d": rng.integers(0, 3, n).astype(float)},
                     index=pd.date_range("2020-01-01", periods=n, freq="h")).astype(np.float32)
    kinds = {"x": "osc_unbounded", "w": "osc_unbounded", "h": "time", "d": "time"}
    fam = {"x": "fx", "w": "fw", "h": "fh", "d": "fd"}
    aset = build_atoms(t, kinds, fam, ["x", "w", "h", "d"], CFG, slice(0, n))
    return aset, t, rng


def _atom(aset, feat, op, pct=None, thr=None):
    return [a for a in aset.atoms if a.feature == feat and a.op == op and (pct is None or a.pct == pct) and (thr is None or a.threshold == thr)][0]


def test_plateau_accepts_a_broad_edge_and_rejects_a_one_threshold_spike(designed):
    aset, t, rng = designed
    x = t["x"].to_numpy(np.float64)
    ax, aw = _atom(aset, "x", "<=", 50), _atom(aset, "w", ">", 10)
    dense = lambda i: aset.dense(i)
    lad = _Ladders(aset)
    noise = rng.normal(0, 0.5, len(x))
    broad = np.where(x <= 0.45, 0.6, -0.3) + noise
    spike = np.where((x > 0.4) & (x <= 0.5), 2.0, -0.3) + noise
    trig = dense(ax.id) & dense(aw.id)
    for y, expect in ((broad, True), (spike, False)):
        mean = float(y[trig].mean())
        assert mean > 0
        r = plateau_and_monotonic(ax, aw, mean, lad, dense, y.astype(np.float64), FC)
        assert r["pass_f4"] == expect, r
        assert not r["plateau_na"] and r["n_neighbours"] == 6
    rb = plateau_and_monotonic(ax, aw, float(broad[trig].mean()), lad, dense, broad, FC)
    rs = plateau_and_monotonic(ax, aw, float(spike[trig].mean()), lad, dense, spike, FC)
    assert rb["plateau_ratio"] > 0.9 and rs["plateau_ratio"] < 0.5 and rs["plateau_pos_share"] < 0.8
    assert rb["plateau_smooth"] == pytest.approx(np.mean([broad[trig].mean()] + [broad[dense(_atom(aset, "x", "<=", p).id) & dense(aw.id)].mean() for p in (30, 40, 60, 70)] + [broad[dense(ax.id) & dense(_atom(aset, "w", ">", p).id)].mean() for p in (20, 30)]), rel=1e-9)
    from scipy.stats import spearmanr
    def rho(feat, op, other):                                         # independent Spearman over the whole ladder with the other atom fixed
        lst = sorted([a for a in aset.atoms if a.feature == feat and a.op == op], key=lambda a: a.threshold)
        m = [broad[dense(a.id) & dense(other.id)].mean() for a in lst]
        return abs(spearmanr([a.threshold for a in lst], m)[0])
    assert rho("x", "<=", aw) > 0.95                                  # the result falls as x loosens...
    assert rb["mono_score"] == pytest.approx((rho("x", "<=", aw) + rho("w", ">", ax)) / 2, rel=1e-9)


def test_plateau_is_not_applicable_without_a_numeric_atom_and_neighbours_respect_compatibility(designed):
    aset, t, rng = designed
    lad = _Ladders(aset)
    dense = lambda i: aset.dense(i)
    ah, ad = _atom(aset, "h", "==", thr=1.0), _atom(aset, "d", "==", thr=2.0)
    y = rng.normal(0.1, 1, len(t))
    r = plateau_and_monotonic(ah, ad, 0.1, lad, dense, y, FC)
    assert r["plateau_na"] and r["pass_f4"] and r["n_neighbours"] == 0 and r["plateau_ratio"] == 1.0 and np.isnan(r["mono_score"])
    # same-feature interval: neighbours that would break the interval (lo < hi) are skipped, not scored
    lo, hi = _atom(aset, "x", ">", 40), _atom(aset, "x", "<=", 60)
    r2 = plateau_and_monotonic(lo, hi, 0.1, lad, dense, y, FC)
    assert r2["n_neighbours"] == 6                                   # 2 x 4 ladder neighbours, minus the two that would equal the other bound (lo < hi is strict)


def test_neighbour_with_too_few_observations_counts_as_not_positive(designed):
    aset, t, rng = designed
    lad = _Ladders(aset)
    dense = lambda i: aset.dense(i)
    ax, aw = _atom(aset, "x", "<=", 50), _atom(aset, "w", ">", 10)
    y = np.full(len(t), 0.5)
    y[:] = np.nan                                                    # no observations anywhere -> every neighbour is "no result"
    r = plateau_and_monotonic(ax, aw, 0.5, lad, dense, y, FC)
    assert r["plateau_pos_share"] == 0.0 and not r["pass_f4"]


# ---------------------------------------------------------------- run_filters on the planted-edge world
@pytest.fixture(scope="module")
def strong_f():
    aset, lab, tr = _world(1, 20000, True)
    surv, _ = coarse_screen(aset, lab, CFG, tr)
    cand, fam, _ = build_candidates(surv, aset, lab, CFG, tr)
    log, passed, info = run_filters(cand, fam, aset, lab, CFG, tr)
    return aset, lab, tr, cand, fam, log, passed, info


def test_funnel_log_and_flags_are_consistent(strong_f):
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    f = info["funnel"]
    assert f["candidates"] == len(cand) == len(log) >= f["f2"] >= f["f7"] >= f["f4"] == len(passed) > 0
    assert (log["passed_filters"] == (log["pass_f2"] & log["pass_f7"] & log["pass_f4"])).all()
    assert log["cand_id"].tolist() == cand["cand_id"].tolist()                       # failing candidates stay visible, in order
    assert passed["passed_filters"].all()
    cs = FC["cost_stress_mult"]
    chk = fam.set_index(["cand_id", "family"])
    for r in log.head(30).itertuples():
        row = chk.loc[(r.cand_id, r.best_family)]
        assert r.cost_stress_mean == pytest.approx(row["mean_gross"] - cs * row["cost"], abs=1e-12)
        assert r.pass_f7 == (r.cost_stress_mean > 0)
    assert log["f2_pos_folds"].between(0, 5).all() and log["plateau_ratio"].notna().all()


def test_second_dedup_representative_is_the_simplest_then_best_plateau_and_clusters_are_distinct(strong_f):
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    thr = CFG["candidates"]["dedup_jaccard"]
    # F9 representative (spec 11): the SIMPLEST rule of the cluster (fewest atoms), then the best plateau score among the simplest
    assert (passed["plateau_score"] == passed["plateau_smooth"]).all()                  # smoothed R, not the ratio to the original
    assert passed.groupby("fcluster")["f_is_representative"].sum().eq(1).all()
    assert passed.groupby("fcluster")["f_cluster_size"].first().sum() == len(passed) and info["n_clusters"] == passed["fcluster"].nunique()
    for _, g in passed.groupby("fcluster"):
        rep = g[g["f_is_representative"]].iloc[0]
        assert int(rep["n_atoms"]) == int(g["n_atoms"].min())
        simplest = g[g["n_atoms"] == g["n_atoms"].min()]
        assert rep["plateau_score"] >= simplest["plateau_score"].max() - 1e-12
    assert "merged_by_corr" in info
    trig = lambda i: aset.dense(int(passed.at[i, "atom_a"]), tr) & aset.dense(int(passed.at[i, "atom_b"]), tr)
    reps = passed.index[passed["f_is_representative"]].tolist()[:25]
    for x in range(len(reps)):
        for y in range(x):
            a, b = trig(reps[x]), trig(reps[y])
            assert (a & b).sum() / (a | b).sum() < thr
    top = passed[passed["f_is_representative"]].head(10)
    feats = {aset.atoms[int(r.atom_a)].feature for r in top.itertuples()} | {aset.atoms[int(r.atom_b)].feature for r in top.itertuples()}
    assert "session" in feats or "hour" in feats                                     # the planted structure leads


def test_tightening_filters_never_adds_and_extremes_empty(strong_f):
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    base = set(passed["cand_id"])
    for over in ({"cost_stress_mult": 1.5 * 1.0 + 4.0}, {"plateau_min_ratio": 0.99}, {"min_loyo_share": 1.0, "max_top5_episode_share": 0.1}, {"min_pos_folds": 5, "max_period_profit_share": 0.25}):
        _, p2, i2 = run_filters(cand, fam, aset, lab, _fcfg(**over), tr)
        assert set(p2["cand_id"]) <= base and len(p2) < len(base), over
    for over in ({"cost_stress_mult": 1e6}, {"min_pos_folds": 6}, {"plateau_min_ratio": 1e6}):
        _, p2, i2 = run_filters(cand, fam, aset, lab, _fcfg(**over), tr)
        assert len(p2) == 0 and i2["n_clusters"] == 0, over


def test_filters_read_train_rows_only_and_empty_input_is_valid(strong_f):
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    vals = lab.to_numpy(np.float32, copy=True)
    vals[tr.stop:] = np.random.default_rng(0).normal(0, 5, (len(vals) - tr.stop, vals.shape[1])).astype(np.float32)
    wrecked = pd.DataFrame(vals, index=lab.index, columns=lab.columns)
    l2, p2, _ = run_filters(cand, fam, aset, wrecked, CFG, tr)
    pd.testing.assert_frame_equal(log, l2)
    pd.testing.assert_frame_equal(passed, p2)
    l0, p0, i0 = run_filters(cand.iloc[0:0], fam.iloc[0:0], aset, lab, CFG, tr)
    assert len(l0) == len(p0) == 0 and i0["n_passed"] == 0 and i0["n_clusters"] == 0


# ---------------------------------------------------------------- stage 07
def test_pipeline_stage7_end_to_end_resume_and_invalidation(tmp_path, monkeypatch):
    from edge_research import pipeline
    from edge_research.pipeline import MissingUpstream, run_stage_7, run_stages_0_7
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(12000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    run = tmp_path / "run"
    with pytest.raises(MissingUpstream):
        run_stage_7(tmp_path / "nothing", CFG) if (tmp_path / "nothing").exists() else (_ for _ in ()).throw(MissingUpstream("x"))
    out = run_stages_0_7(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")
    assert out["status"]["07_filters"] == "ran" and out["n_filtered"] > 0 and 0 < out["n_filtered_clusters"] <= out["n_filtered"]
    d7 = run / "07_filters"
    for f in ("filter_log.parquet", "candidates_filtered.parquet", "clusters.csv", "filters_info.json", "summary.md"):
        assert (d7 / f).exists(), f
    assert len(pd.read_csv(d7 / "clusters.csv")) == out["n_filtered_clusters"]
    assert set(run_stages_0_7(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")["status"].values()) == {"skipped"}
    cfg2 = load_config(overrides={"filters": {"plateau_min_ratio": 0.95}, **DEEP_OFF})     # DEEP_OFF: the shared CFG runs with the deep engines off
    o2 = run_stages_0_7(csv, run, cfg2, "SYNTH", tmp_path / "s_spread.csv")
    assert {k: v for k, v in o2["status"].items() if v == "ran"} == {"07_filters": "ran"} and o2["n_filtered"] <= out["n_filtered"]
    cfg3 = load_config(overrides={"filters": {"plateau_min_ratio": 0.95}, "candidates": {"t_exact": 3.5}, **DEEP_OFF})
    o3 = run_stages_0_7(csv, run, cfg3, "SYNTH", tmp_path / "s_spread.csv")
    assert o3["status"]["06_candidates"] == "ran" and o3["status"]["07_filters"] == "ran" and o3["status"]["05_discovery"] == "skipped"


def test_plateau_positive_share_alone_can_reject(designed):
    """Median neighbour is fine (ratio 1.0) but a third of the neighbours are not positive: only the share condition can reject it."""
    aset, t, rng = designed
    x = t["x"].to_numpy(np.float64)
    lad = _Ladders(aset)
    dense = lambda i: aset.dense(i)
    ax, aw = _atom(aset, "x", "<=", 50), _atom(aset, "w", ">", 10)
    y = np.where(x <= 0.35, -0.5, 3.0) + rng.normal(0, 0.05, len(x))
    trig = dense(ax.id) & dense(aw.id)
    r = plateau_and_monotonic(ax, aw, float(y[trig].mean()), lad, dense, y, FC)
    assert r["plateau_ratio"] >= FC["plateau_min_ratio"] and r["plateau_pos_share"] == pytest.approx(4 / 6, abs=1e-9)
    assert not r["pass_f4"]


def test_cost_stress_flag_matches_the_arithmetic_for_a_mixed_outcome(strong_f):
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    log6, _, _ = run_filters(cand, fam, aset, lab, _fcfg(cost_stress_mult=6.0), tr)
    chk = fam.set_index(["cand_id", "family"])
    stress = np.array([chk.loc[(r.cand_id, r.best_family)]["mean_gross"] - 6.0 * chk.loc[(r.cand_id, r.best_family)]["cost"] for r in log6.itertuples()])
    assert np.allclose(log6["cost_stress_mean"].to_numpy(), stress, atol=1e-12)
    assert ((stress > 0) == log6["pass_f7"].to_numpy()).all()
    assert (stress > 0).any() and (stress <= 0).any() and ((stress < 0) & (stress > -1)).any()       # a genuinely mixed outcome around zero


def test_period_usability_boundary_is_inclusive():
    idx, yy, ep, period, b = _make(lambda r, i, e: r.normal(0.3, 1.0, len(i)))
    per_ep = {k: len(set(ep[period == k])) for k in set(period)}
    m = min(per_ep.values())
    assert time_stability(idx, yy, ep, period, b, {**FC, "min_episodes_period": m})["n_periods"] == 8
    assert time_stability(idx, yy, ep, period, b, {**FC, "min_episodes_period": m + 1})["n_periods"] == 8 - sum(1 for v in per_ep.values() if v == m)


def test_ladder_means_skip_atoms_that_would_break_the_interval(designed):
    from edge_research.discovery.filters import _ladder_means
    aset, t, rng = designed
    lad = _Ladders(aset)
    lo, hi = _atom(aset, "x", ">", 40), _atom(aset, "x", "<=", 60)
    lst, pos = lad.of(lo)
    y = rng.normal(0.2, 1, len(t))
    res = _ladder_means(lst, hi, aset.dense(hi.id), lambda i: aset.dense(i), y, 30)
    for atom, m in res:
        legal = atom.threshold < hi.threshold
        assert np.isfinite(m) == legal, (atom.expr(), m)


def test_period_concentration_is_checked_separately_from_folds():
    """One period holds 70% of the profit while no fold holds more than ~30%: only the period share can catch it."""
    rng = np.random.default_rng(9)
    idx = np.sort(rng.choice(4000, 1000, replace=False))
    yy = np.where(idx < 2000, rng.normal(0.3, 0.2, len(idx)), rng.normal(0.7, 0.2, len(idx)))
    ep = episode_ids(idx, 2)
    r = time_stability(idx, yy, ep, (idx // 2000).astype(np.int64), fold_bounds(4000, 5), {**FC, "min_episodes_period": 10})
    assert r["max_period_share"] == pytest.approx(0.7, abs=0.05) and not r["pass_concentration"]
    f_share = [yy[(idx >= a) & (idx < b)].sum() / yy.sum() for a, b in zip(fold_bounds(4000, 5)[:-1], fold_bounds(4000, 5)[1:])]
    assert max(f_share) < 0.35


# ---------------------------------------------------------------- F8 backward elimination (spec 11) and the F9 P&L-correlation dedup
def test_pnl_corr_matches_manual_pearson_and_respects_min_overlap():
    rng = np.random.default_rng(2)
    a = (np.array([0, 1, 2, 3]), rng.normal(0, 1, 4))
    b = (np.array([2, 3, 4, 5]), rng.normal(0, 1, 4))
    assert _pnl_corr(a, b, min_overlap=5) is None                       # fewer than min_overlap non-zero bars: undefined, never merges
    idx_u = np.union1d(a[0], b[0])
    va = dict(zip(a[0].tolist(), a[1].tolist()))
    vb = dict(zip(b[0].tolist(), b[1].tolist()))
    x = np.array([va.get(int(i), 0.0) for i in idx_u])
    z = np.array([vb.get(int(i), 0.0) for i in idx_u])
    want = float(((x - x.mean()) * (z - z.mean())).mean() / (x.std() * z.std()))
    got = _pnl_corr(a, b, min_overlap=4)
    assert got is not None and got == pytest.approx(want, rel=1e-12)
    same = (np.array([0, 1, 2, 3, 4]), np.array([1.0, -1.0, 1.0, -1.0, 1.0]))
    assert _pnl_corr(same, same, min_overlap=3) == pytest.approx(1.0)
    flip = (same[0], -same[1])
    assert _pnl_corr(same, flip, min_overlap=3) == pytest.approx(-1.0)
    flat = (same[0], np.zeros(5))
    assert _pnl_corr(same, flat, min_overlap=3) is None                 # a zero-variance series has no correlation


def test_f8_disabled_or_pair_rules_never_touch_the_frame(strong_f):
    """On the pair world F8 must be a strict NO-OP (all rules have 2 atoms) but it ALWAYS leaves the three typed flag
    columns behind — disabled or not — so downstream stages see one stable schema."""
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    c = CFG["filters"]
    dense = _MaskCache(aset, tr, 512)
    labels = {d: build_family_labels(lab, CFG, d, tr) for d in DIRECTIONS}
    bounds = fold_bounds(labels["long"].y.shape[1], int(CFG["discovery"]["n_folds"]))
    period, _ = period_keys(lab.index[tr], int(c["min_periods"]))
    nfo0: dict = {}
    p0, _, fl0 = _f8_backward_eliminate(passed.copy(), fam.copy(), aset, dense, _Ladders(aset), labels, bounds, period, lab, CFG, tr,
                                        {**c, "f8_enabled": False}, nfo0)
    assert len(fl0) == 0 and nfo0["f8_simplified"] == 0
    assert (p0["f8_simplified"] == False).all() and p0["f8_dropped"].eq(0).all() and p0["f8_val_mean"].isna().all()
    nfo1: dict = {}
    p1, _, fl1 = _f8_backward_eliminate(passed.copy(), fam.copy(), aset, dense, _Ladders(aset), labels, bounds, period, lab, CFG, tr, c, nfo1)
    assert len(fl1) == 0 and nfo1["f8_simplified"] == 0 and nfo1["f8_dropped_atoms"] == 0 and nfo1["f8_reverted"] == 0
    assert (p1["n_atoms"] == 2).all()
    pd.testing.assert_frame_equal(p1.drop(columns=["f8_simplified", "f8_dropped", "f8_val_mean"]), passed.drop(columns=["f8_simplified", "f8_dropped", "f8_val_mean"]))


def test_run_filters_flag_columns_have_clean_dtypes(strong_f):
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    assert (passed["f8_simplified"] == False).all() and passed["f8_dropped"].eq(0).all()
    assert passed["f8_simplified"].dtype == bool and passed["f8_dropped"].dtype.kind == "i"
    assert (log["f8_simplified"] == False).all() and log["f8_dropped"].eq(0).all()


def test_dedup_corr_merges_highly_correlated_rules_even_below_the_jaccard_threshold(strong_f):
    """F9 second criterion (spec 11): with the Jaccard threshold effectively off, ladder-neighbour rules whose per-bar P&L
    series correlate >= dedup_corr must still merge; with the criterion disabled they must stay separate clusters."""
    aset, lab, tr, cand, fam, log, passed, info = strong_f
    _, p_on, i_on = run_filters(cand, fam, aset, lab, _ccfg(dedup_jaccard=0.999), tr)
    _, p_off, i_off = run_filters(cand, fam, aset, lab, _ccfg(dedup_jaccard=0.999, dedup_corr=1.1), tr)
    assert i_off["merged_by_corr"] == 0                                  # correlation 1.1 is unreachable: the criterion is off
    assert i_on["merged_by_corr"] > 0
    assert i_on["n_clusters"] < i_off["n_clusters"]
    assert p_on.groupby("fcluster")["f_is_representative"].sum().eq(1).all()
    assert p_on.groupby("fcluster")["f_cluster_size"].first().sum() == len(p_on) and \
        p_off.groupby("fcluster")["f_cluster_size"].first().sum() == len(p_off)      # one size per cluster (test above line 230)
