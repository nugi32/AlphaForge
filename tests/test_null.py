"""M7: null test over the whole search: rotation, identical procedure, p_global / false discovery by hand, resume, calendar flag, stage 09."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.discovery.candidates import build_candidates
from edge_research.discovery.filters import run_filters
from edge_research.discovery.pairs import coarse_screen
from edge_research.nulltest import null as nl
from edge_research.nulltest.null import (calendar_driven, estimate_week_bars, load_distribution, null_summary, p_global, rotate_labels, rotation_offsets,
                                         run_null, search_statistic)
from edge_research.splits import max_hold_bars
from edge_research.synth.generator import make_synthetic_bars
from test_discovery import CFG, DEEP_OFF, STRONG, _world


def _ncfg(**over):
    c = copy.deepcopy(CFG)
    c["nulltest"].update(over)
    return c


# ---------------------------------------------------------------- rotation
def test_rotate_labels_moves_whole_rows_inside_the_segment_only():
    idx = pd.date_range("2020-01-01", periods=50, freq="h")
    lab = pd.DataFrame({"a": np.arange(50, dtype=np.float32), "b": np.arange(50, dtype=np.float32) * 10}, index=idx)
    rows = slice(10, 40)
    r = nl.rotate_labels(lab, rows, 7)
    seg = lab.iloc[rows].to_numpy()
    np.testing.assert_array_equal(r.iloc[rows].to_numpy(), np.roll(seg, 7, axis=0))
    np.testing.assert_array_equal(r.iloc[:10].to_numpy(), lab.iloc[:10].to_numpy())             # outside the segment: untouched
    np.testing.assert_array_equal(r.iloc[40:].to_numpy(), lab.iloc[40:].to_numpy())
    assert (r["b"] == r["a"] * 10).all()                                                        # columns of one row stay together
    assert sorted(r["a"].iloc[rows]) == sorted(lab["a"].iloc[rows])                             # a permutation of the same rows
    pd.testing.assert_frame_equal(nl.rotate_labels(lab, rows, 0), lab)
    assert lab["a"].iloc[10] == 10                                                              # the input is not modified
    pd.testing.assert_frame_equal(nl.rotate_labels(lab, rows, 30), lab)                         # a full turn is the identity


def test_rotation_offsets_range_step_prefix_and_errors():
    rng = lambda: np.random.default_rng(3)
    o = rotation_offsets(12000, 60, 40, rng())
    assert len(o) == 40 and o.min() >= 60 and o.max() <= 12000 - 60 and len(set(o)) == 40
    cal = rotation_offsets(12000, 60, 40, rng(), step=120)
    assert (cal % 120 == 0).all() and cal.min() >= 120 and cal.max() <= 11940
    np.testing.assert_array_equal(rotation_offsets(12000, 60, 10, rng()), o[:10])               # prefix-stable: R can be extended later
    np.testing.assert_array_equal(rotation_offsets(12000, 60, 10, rng(), step=120), cal[:10])
    small = rotation_offsets(300, 50, 12, rng(), step=100)                                       # pool {100, 200}: repeats only because the pool is smaller than n
    assert len(small) == 12 and set(small) <= {100, 200}
    with pytest.raises(ValueError):
        rotation_offsets(100, 60, 5, rng())
    with pytest.raises(ValueError):
        rotation_offsets(1000, 60, 5, rng(), step=5000)


def test_estimate_week_bars_for_h1_with_weekends():
    idx = pd.date_range("2021-01-04", periods=24 * 7 * 30, freq="h")
    idx = idx[idx.dayofweek < 5]
    assert estimate_week_bars(idx) == 120


# ---------------------------------------------------------------- identical procedure
@pytest.fixture(scope="module")
def world():
    aset, lab, tr = _world(1, 12000, True)
    return aset, lab, tr


def test_search_statistic_is_the_stage_5_to_7_search_and_offset_zero_is_the_real_data(world):
    aset, lab, tr = world
    real = search_statistic(aset, lab, CFG, tr)
    surv, _ = coarse_screen(aset, lab, CFG, tr)
    cand, fam, cinfo = build_candidates(surv, aset, lab, CFG, tr)
    log, passed, _ = run_filters(cand, fam, aset, lab, CFG, tr)
    assert real["n_surv"] == len(surv) and real["n_cand"] == len(cand) and real["n_pass_f2"] == int(log["pass_f2"].sum()) and real["n_pass_all"] == len(passed)
    ok = set(log.loc[log["pass_f2"], "cand_id"])
    f = fam[fam["cand_id"].isin(ok) & (fam["n_episodes"] >= CFG["candidates"]["min_episodes_train"])]
    assert real["S"] == pytest.approx(f["t"].max()) and real["S_cont"] == pytest.approx(cinfo["max_t_exact"]) and real["S_cont"] >= real["S"] - 1e-12
    assert real["S"] > 0
    assert search_statistic(aset, rotate_labels(lab, tr, 0), CFG, tr) == real                  # the rotation is the ONLY thing that differs
    off = search_statistic(aset, rotate_labels(lab, tr, 777), CFG, tr)
    assert off["n_pass_f2"] < real["n_pass_f2"] and off["n_surv"] < real["n_surv"]
    assert real["deep_used"] == "none" and isinstance(real["deep_note"], str) and real["deep_note"]        # the scope note travels with the statistic
    s = null_summary(real, _dist("random", [1.0] * 5), _ncfg(min_replicates=3))
    assert s["real"]["deep_used"] == "none" and s["real"]["deep_note"] == real["deep_note"]


def test_null_data_has_nothing_to_find_and_a_rotated_null_neither():
    aset, lab, tr = _world(1, 20000, False)
    real = search_statistic(aset, lab, CFG, tr)
    assert real["S"] == 0.0 and real["n_cand"] == 0 and real["n_pass_f2"] == 0
    assert search_statistic(aset, rotate_labels(lab, tr, 1234), CFG, tr)["n_pass_f2"] == 0


# ---------------------------------------------------------------- p_global, thresholds, false discovery, G1 (hand numbers)
def _dist(variant, S, S_cont=None, n_pass=None):
    n = len(S)
    return pd.DataFrame({"variant": variant, "rep": range(n), "offset": range(n), "S": S, "S_cont": S if S_cont is None else S_cont, "n_surv": 0, "n_cand": 0,
                         "n_pass_f2": [0] * n if n_pass is None else n_pass, "n_pass_all": 0, "seconds": 0.0})


def test_p_global_and_summary_by_hand():
    assert p_global(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]), 8.5) == pytest.approx(3 / 11)                 # 9 and 10 are >= 8.5
    assert p_global(np.array([1.0, 2.0]), 2.0) == pytest.approx(2 / 3)                                          # equality counts as "as good"
    assert p_global(np.array([1.0, 2.0, 3.0]), 99.0) == pytest.approx(1 / 4)                                    # smallest possible: 1 / (R + 1)
    cfg = _ncfg(min_replicates=10, quantile=0.9)
    real = {"S": 12.0, "S_cont": 12.0, "n_surv": 500, "n_cand": 100, "n_pass_f2": 80, "n_pass_all": 70}
    d = pd.concat([_dist("random", list(np.arange(0, 10, 1.0)) + [0.0] * 10, n_pass=[4] * 10 + [0] * 10), _dist("calendar", [20.0] * 20)])
    s = null_summary(real, d, cfg)
    rv = s["variants"]["random"]
    assert rv["n_replicates"] == 20 and not rv["below_minimum"]
    assert rv["S_threshold"] == float(np.quantile(d[d.variant == "random"]["S"], 0.9, method="higher")) == 8.0
    assert rv["p_global_S"] == pytest.approx(1 / 21) and rv["null_pass_f2_mean"] == pytest.approx(2.0) and rv["false_discovery_estimate"] == pytest.approx(2.0 / 80)
    assert rv["null_with_any_pass"] == pytest.approx(0.5)
    assert s["g1_pass"] and "passes" in s["g1_reason"]
    assert s["variants"]["calendar"]["p_global_S"] == pytest.approx(1.0)                                         # every calendar null is better than the real data
    # each failure reason on its own
    assert not null_summary({**real, "n_pass_f2": 0}, d, cfg)["g1_pass"]
    assert not null_summary({**real, "S": 5.0}, d, cfg)["g1_pass"]
    heavy = _dist("random", list(np.arange(0, 10, 1.0)) + [0.0] * 10, n_pass=[100] * 20)
    assert "false discovery" in null_summary(real, heavy, cfg)["g1_reason"] and not null_summary(real, heavy, cfg)["g1_pass"]
    few = null_summary(real, _dist("random", [1.0] * 5), _ncfg(min_replicates=50))
    assert few["variants"]["random"]["below_minimum"] and not few["g1_pass"] and "replicates" in few["g1_reason"]
    assert not null_summary(real, _dist("calendar", [1.0] * 20), cfg)["g1_pass"]                                 # no random variant: cannot pass


def test_g1_boundaries_alpha_equality_and_minimum_replicates():
    real = {"S": 12.0, "S_cont": 12.0, "n_surv": 5, "n_cand": 5, "n_pass_f2": 10, "n_pass_all": 10}
    eq = _dist("random", [12.0] + [0.0] * 18)                                                                   # R = 19, one null as good: p = 2/20 = 0.1 = 1 - q exactly
    assert null_summary(real, eq, _ncfg(min_replicates=10, quantile=0.9))["g1_pass"]                            # equality passes (the float 1 - 0.9 must not break it)
    assert not null_summary(real, _dist("random", [12.0] * 2 + [0.0] * 17), _ncfg(min_replicates=10, quantile=0.9))["g1_pass"]
    low = null_summary(real, _dist("random", [1.0] * 5), _ncfg(min_replicates=50, quantile=0.5))               # p = 1/6 passes alpha 0.5; ONLY the replicate minimum fails
    assert low["variants"]["random"]["p_global_S"] <= 0.5 and not low["g1_pass"] and "replicates" in low["g1_reason"]


def test_week_bars_fallback_and_tie():
    idx = pd.date_range("2021-01-04", periods=240, freq="h")                                                   # 10 days, 24x7: counts 168 and 72, nothing repeats -> ratio fallback
    assert abs(estimate_week_bars(idx) - 168) <= 2
    wk = pd.date_range("2021-01-04", periods=24 * 5 * 4 + 24 * 2 * 4, freq="h")
    wk = wk[wk.dayofweek < 5]
    assert len(wk) == 480
    wk = wk.delete(list(np.arange(240, 242)) + list(np.arange(360, 362)))                                      # weeks 3 and 4 lose 2 bars: counts 120, 120, 118, 118 -> a tie, the larger wins
    assert estimate_week_bars(wk) == 120


def test_calendar_driven_flag():
    assert calendar_driven(8.0, 5.0, 9.0) == {"beats_random": True, "beats_calendar": False, "calendar_driven": True}
    assert calendar_driven(8.0, 5.0, 6.0)["calendar_driven"] is False and calendar_driven(8.0, 5.0, 6.0)["beats_calendar"]
    assert calendar_driven(4.0, 5.0, 3.0) == {"beats_random": False, "beats_calendar": True, "calendar_driven": False}
    assert calendar_driven(5.0, 5.0, 1.0)["beats_random"] is False                                                # strictly greater
    flags = nl.rule_flags(pd.DataFrame({"is_representative": [True, True, False], "best_t": [8.0, 4.0, 99.0], "chosen_t": [1.0, 1.0, 1.0], "expr_a": "a", "expr_b": "b", "chosen_family": "T6"}),
                          {"variants": {"random": {"S_threshold": 5.0}, "calendar": {"S_threshold": 9.0}}})
    assert flags["calendar_driven"].tolist() == [True, False] and len(flags) == 2
    # without the calendar variant (partial run, config without it) the calendar comparison is UNKNOWN, never a silent calendar_driven = beats_random
    un = nl.rule_flags(pd.DataFrame({"is_representative": [True, True], "best_t": [8.0, 4.0], "chosen_t": [1.0, 1.0], "expr_a": "a", "expr_b": "b", "chosen_family": "T6"}),
                       {"variants": {"random": {"S_threshold": 5.0}}})
    assert un["beats_random"].tolist() == [True, False]
    assert un["beats_calendar"].isna().all() and un["calendar_driven"].isna().all()


def test_null_base_fingerprint_binds_the_search_procedure_version(monkeypatch):
    a = nl.null_base_fingerprint("fp7", 11)
    assert a != nl.null_base_fingerprint("fp8", 11) and a != nl.null_base_fingerprint("fp7", 12)
    assert a == hashlib.sha256(f"fp7|11|{nl.SEARCH_VERSION}".encode()).hexdigest()[:16]
    monkeypatch.setattr(nl, "SEARCH_VERSION", "9")
    assert nl.null_base_fingerprint("fp7", 11) != a        # a procedure change invalidates every stored replicate and real.json


# ---------------------------------------------------------------- run, resume, fingerprints
def test_run_null_resume_and_fingerprint(world, tmp_path, monkeypatch):
    aset, lab, tr = world
    cfg = _ncfg(replicates=3, variants=["random", "calendar"])
    calls = []
    orig = nl.search_statistic

    def counting(a, l, c, r, *extra):        # the search statistic gained an optional feature_table argument (M11)
        calls.append(1)
        return orig(a, l, c, r)

    monkeypatch.setattr(nl, "search_statistic", counting)
    out = tmp_path / "n"
    r1 = run_null(aset, lab, cfg, tr, out, "fpA", max_replicates_this_call=4)
    assert not r1["complete"] and r1["n_computed"] == 4 and r1["n_done"] == 4 and len(calls) == 1 + 4         # real + 4 replicates
    r2 = run_null(aset, lab, cfg, tr, out, "fpA")
    assert r2["complete"] and r2["n_done"] == 6 and r2["n_computed"] == 2 and len(calls) == 1 + 4 + 2           # only the missing ones
    r3 = run_null(aset, lab, cfg, tr, out, "fpA")
    assert r3["n_computed"] == 0 and len(calls) == 7
    d = load_distribution(out, "fpA", cfg)
    assert len(d) == 6 and set(d["variant"]) == {"random", "calendar"} and (d.groupby("variant").size() == 3).all()
    cal = d[d["variant"] == "calendar"]["offset"]
    assert (cal % r3["week_bars"] == 0).all() and d["offset"].min() >= r3["embargo"] and r3["embargo"] == max_hold_bars(cfg)
    n_tr = tr.stop - tr.start
    for vi, (v, step) in enumerate((("random", None), ("calendar", r3["week_bars"]))):                           # documented seeds: seed + 1000 x (variant index + 1)
        exp = rotation_offsets(n_tr, r3["embargo"], 3, np.random.default_rng(int(cfg["seed"]) + 1000 * (vi + 1)), step=step)
        assert d[d["variant"] == v].sort_values("rep")["offset"].tolist() == exp.tolist()
    ref = [orig(aset, rotate_labels(lab, tr, int(o)), cfg, tr)["S"] for o in d[d["variant"] == "random"]["offset"]]
    assert d[d["variant"] == "random"]["S"].tolist() == pytest.approx(ref)                                        # stored replicates are the real computation
    run_null(aset, lab, cfg, tr, out, "fpB")                                                                      # another fingerprint: nothing is reused
    assert len(calls) == 7 + 1 + 6
    assert len(load_distribution(out, "fpA", cfg)) == 0 or len(load_distribution(out, "fpB", cfg)) == 6
    ext = _ncfg(replicates=5, variants=["random", "calendar"])                                                    # extending R keeps the first 3 of each variant
    n_before = len(calls)
    run_null(aset, lab, ext, tr, out, "fpB")
    assert len(calls) - n_before == 4 and len(load_distribution(out, "fpB", ext)) == 10


def test_planted_edge_beats_its_null_and_null_data_cannot_pass(world, tmp_path):
    aset, lab, tr = world
    cfg = _ncfg(replicates=6, variants=["random", "calendar"], min_replicates=6, quantile=0.8)
    res = run_null(aset, lab, cfg, tr, tmp_path / "p", "fp")
    d = load_distribution(tmp_path / "p", "fp", cfg)
    s = null_summary(res["real"], d, cfg)
    rv = s["variants"]["random"]
    assert res["real"]["S"] > rv["S_null_max"] and rv["p_global_S"] == pytest.approx(1 / 7)
    assert res["real"]["n_pass_f2"] > 20 * rv["null_pass_f2_mean"] and rv["false_discovery_estimate"] < 0.5
    assert s["g1_pass"]                                                                                           # quantile 0.8 -> alpha 0.2 >= 1/7
    na, nlab, ntr = _world(1, 20000, False)
    nres = run_null(na, nlab, _ncfg(replicates=3, variants=["random"], min_replicates=3), ntr, tmp_path / "q", "fp")
    ns = null_summary(nres["real"], load_distribution(tmp_path / "q", "fp", cfg), _ncfg(min_replicates=3))
    assert nres["real"]["S"] == 0.0 and not ns["g1_pass"] and "no rule passes" in ns["g1_reason"]


# ---------------------------------------------------------------- stage 09
def test_pipeline_stage9(tmp_path):
    from edge_research.pipeline import run_stage_9, run_stages_0_8
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(12000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    sps, run = tmp_path / "s_spread.csv", tmp_path / "run"
    cfg = load_config(overrides={"nulltest": {"replicates": 3, "min_replicates": 3, "quantile": 0.8}, **DEEP_OFF})   # the deep engines are tested separately (M11); here the depth-2 search must stay fast
    run_stages_0_8(csv, run, cfg, "SYNTH", sps)
    p1 = run_stage_9(run, cfg, max_replicates_this_call=2)
    assert p1["status"] == "partial" and not p1["complete"] and (run / "09_null" / "summary.md").exists()
    assert "09_null" not in json.loads((run / "manifest.json").read_text())["stages"]
    p2 = run_stage_9(run, cfg)
    assert p2["status"] == "ran" and p2["complete"] and p2["n_done"] == 6
    d9 = run / "09_null"
    for f in ("null_distribution.csv", "rule_flags.csv", "null_info.json", "summary.md"):
        assert (d9 / f).exists(), f
    assert len(pd.read_csv(d9 / "null_distribution.csv")) == 6
    assert run_stage_9(run, cfg)["status"] == "skipped"
    cfg2 = load_config(overrides={"nulltest": {"replicates": 5, "min_replicates": 3, "quantile": 0.8}, **DEEP_OFF})    # only the null settings changed
    n_files = len(list((d9 / "replicates").glob("*.json")))
    p3 = run_stage_9(run, cfg2)
    assert p3["status"] == "ran" and p3["n_done"] == 10 and len(list((d9 / "replicates").glob("*.json"))) == n_files + 4      # 3 reused per variant, +2 each
    info = json.loads((d9 / "null_info.json").read_text(encoding="utf-8"))
    assert info["summary"]["real"]["deep_used"] == "none" and info["summary"]["real"]["deep_note"]         # the scope reaches the report
    assert "Deep scope" in (d9 / "summary.md").read_text(encoding="utf-8")
