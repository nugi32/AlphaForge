"""M8: frozen rules, validation gates G1-G8 (pure logic with hand values + engine integration), holdout protocol, stage 10 / 11 wiring, outer walk-forward."""

import copy
import json
import os
import shutil
import stat

import numpy as np
import pandas as pd
import pytest

from edge_research.atoms.atoms import load_atoms
from edge_research.config import load_config
from edge_research.labels.builder import net_R
from edge_research.rules import FrozenRule
from edge_research.synth.generator import make_synthetic_bars, write_exporter_csv
from edge_research.validation import holdout as hm
from edge_research.validation import walkforward as wf
from edge_research.validation.demo import evaluate_demo_gate, load_demo_statement, update_gate_table
from edge_research.validation.freeze import freeze_rules, rule_id_of
from edge_research.validation.gates import GATE_COLS, apply_gates, build_gate_table, lift_over_parents, t_critical, trade_clusters, trade_stats
from test_discovery import CFG, STRONG, _world


# ================================================================ pure logic
def test_t_critical_matches_student_t_tables_and_degenerate_cases():
    assert t_critical(0.05, 1, 11) == pytest.approx(1.812461, abs=1e-5)      # t(0.95, df=10)
    assert t_critical(0.05, 5, 11) == pytest.approx(2.763769, abs=1e-5)      # alpha / K = 0.01, df = 10
    assert t_critical(0.05, 5, 11) > t_critical(0.05, 1, 11)                 # more rules tested -> higher bar
    assert t_critical(0.05, 5, 41) < t_critical(0.05, 5, 11)                 # more clusters -> lower bar
    assert t_critical(0.05, 0, 11) == t_critical(0.05, 1, 11)                # K is at least 1
    assert t_critical(0.05, 3, 1) == float("inf") and t_critical(0.05, 3, 0) == float("inf")


def test_trade_clusters_gap_boundary_and_original_order():
    e = np.array([10, 12, 40, 14, 100])                                      # sorted: 10 12 14 40 100 (gaps 2, 2, 26, 60)
    np.testing.assert_array_equal(trade_clusters(e, 3), [0, 0, 1, 0, 2])
    np.testing.assert_array_equal(trade_clusters(np.array([0, 3, 7]), 3), [0, 0, 1])      # distance == gap merges, gap + 1 does not
    np.testing.assert_array_equal(trade_clusters(np.array([5, 5, 5]), 0), [0, 0, 0])      # equal entry bars (several positions) share a cluster
    assert trade_clusters(np.array([], dtype=int), 3).size == 0


def test_trade_stats_hand_example_and_empty():
    tr = pd.DataFrame({"entry_bar": [0, 1, 50, 51, 100], "r_net": [1.0, 3.0, 2.0, 2.0, -1.0]})
    s = trade_stats(tr, 3)                                                   # clusters {0,1} {50,51} {100}
    mean = 7 / 5
    c = np.array([(1 - mean) + (3 - mean), (2 - mean) * 2, (-1 - mean)])
    se = np.sqrt(3 / 2 * (c**2).sum()) / 5
    assert s["n_trades"] == 5 and s["n_clusters"] == 3
    assert s["mean_R"] == pytest.approx(mean) and s["se_R"] == pytest.approx(se) and s["t"] == pytest.approx(mean / se)
    e = trade_stats(tr.iloc[:0], 3)
    assert e["n_trades"] == 0 and e["n_clusters"] == 0 and np.isnan(e["mean_R"]) and np.isnan(e["t"])
    one = trade_stats(tr.iloc[:2], 3)                                        # a single cluster: no SE, no t
    assert one["n_clusters"] == 1 and np.isnan(one["t"])


V = dict(lift_t_min=1.5, decay_min_ratio=0.5, min_trades_val=50, min_profit_factor=1.15, max_equity_drawdown_pct=30.0, require_g1=True)


def _row(**o):
    r = dict(lift_a_t_train=2.0, lift_b_t_train=2.0, lift_a_t_val=2.0, lift_b_t_val=2.0, val_mean_R=0.5, val_t=3.0, t_crit_bonferroni=3.0, decay_ratio=0.6, val_trades=50,
             stress_mean_R=0.1, stress_net_profit=10.0, val_profit_factor=1.2, val_equity_dd_pct=10.0, g2_time_stability=True, g3_exit_plateau=True)
    r.update(o)
    return r


def _gate(row, v=V, lt=1.5, g1=True):
    return apply_gates(row, dict(v), lt, g1)


def test_apply_gates_all_pass_and_every_boundary():
    r = _gate(_row())
    assert all(r[c] for c in GATE_COLS) and r["accepted"] and r["eligible_for_holdout"] and r["fail_reasons"] == "" and r["g1_null"] is True
    # G4: both parents, train and validation, inclusive threshold; NaN never passes
    assert _gate(_row(lift_a_t_val=1.5))["g4_lift"] and not _gate(_row(lift_a_t_val=1.4999))["g4_lift"]
    assert not _gate(_row(lift_b_t_val=1.4999))["g4_lift"]
    assert _gate(_row(lift_a_t_train=1.5))["g4_lift"] and not _gate(_row(lift_a_t_train=1.4999))["g4_lift"]
    assert not _gate(_row(lift_b_t_train=1.4999))["g4_lift"]
    assert not _gate(_row(lift_a_t_val=np.nan))["g4_lift"] and not _gate(_row(lift_b_t_train=np.nan))["g4_lift"]
    assert not _gate(_row(lift_a_t_train=1.4), lt=1.5)["g4_lift"] and _gate(_row(lift_a_t_train=1.4), lt=1.4)["g4_lift"]      # train threshold is the argument, validation the config
    assert not _gate(_row(lift_a_t_val=1.4), v={**V, "lift_t_min": 1.5})["g4_lift"] and _gate(_row(lift_a_t_val=1.4), v={**V, "lift_t_min": 1.4})["g4_lift"]
    # G5: mean > 0 (strict), t >= critical (inclusive), decay >= ratio (inclusive)
    assert not _gate(_row(val_mean_R=0.0))["g5_oos"] and _gate(_row(val_mean_R=1e-9))["g5_oos"] and not _gate(_row(val_mean_R=np.nan))["g5_oos"]
    assert _gate(_row(val_t=3.0, t_crit_bonferroni=3.0))["g5_oos"] and not _gate(_row(val_t=2.999, t_crit_bonferroni=3.0))["g5_oos"]
    assert not _gate(_row(val_t=np.nan))["g5_oos"] and not _gate(_row(t_crit_bonferroni=np.inf))["g5_oos"]
    assert _gate(_row(decay_ratio=0.5))["g5_oos"] and not _gate(_row(decay_ratio=0.4999))["g5_oos"] and not _gate(_row(decay_ratio=np.nan))["g5_oos"]
    # G6
    assert _gate(_row(val_trades=50))["g6_trades"] and not _gate(_row(val_trades=49))["g6_trades"]
    # G7: strictly positive stressed mean AND stressed net profit
    assert not _gate(_row(stress_mean_R=0.0))["g7_cost_stress"] and not _gate(_row(stress_net_profit=0.0))["g7_cost_stress"] and not _gate(_row(stress_mean_R=np.nan))["g7_cost_stress"]
    assert _gate(_row(stress_mean_R=1e-9, stress_net_profit=0.01))["g7_cost_stress"]
    # G8: PF inclusive, infinite PF (no losing trade) passes, NaN fails; drawdown inclusive upper bound
    assert _gate(_row(val_profit_factor=1.15))["g8_quality"] and not _gate(_row(val_profit_factor=1.1499))["g8_quality"]
    assert _gate(_row(val_profit_factor=np.inf))["g8_quality"] and not _gate(_row(val_profit_factor=np.nan))["g8_quality"]
    assert _gate(_row(val_equity_dd_pct=30.0))["g8_quality"] and not _gate(_row(val_equity_dd_pct=30.01))["g8_quality"] and not _gate(_row(val_equity_dd_pct=np.nan))["g8_quality"]


def test_apply_gates_accepted_needs_g2_and_g3_and_g1_only_for_holdout():
    assert not _gate(_row(g2_time_stability=False))["accepted"] and "g2_time_stability" in _gate(_row(g2_time_stability=False))["fail_reasons"]
    assert not _gate(_row(g3_exit_plateau=False))["accepted"]
    r = _gate(_row(), g1=False)                                              # null test failed: still accepted by G2-G8, never eligible
    assert r["accepted"] and not r["eligible_for_holdout"] and r["fail_reasons"] == "g1_null" and r["g1_null"] is False
    r = _gate(_row(), g1=None)                                               # null test not run
    assert r["accepted"] and not r["eligible_for_holdout"] and r["g1_null"] is None and r["fail_reasons"] == "g1_null"
    r = _gate(_row(), v={**V, "require_g1": False}, g1=None)
    assert r["accepted"] and r["eligible_for_holdout"] and r["fail_reasons"] == ""
    r = _gate(_row(val_trades=10, lift_a_t_val=0.0), g1=True)                # reasons list every failed gate, in gate order
    assert r["fail_reasons"] == "g4_lift;g6_trades" and not r["eligible_for_holdout"]


def test_lift_over_parents_matches_brute_force_group_means_and_sandwich_se():
    aset, lab, tr = _world(1, 12000, True)
    y = net_R(lab, "long", "T6").to_numpy(np.float64)
    rows = slice(tr.start, tr.stop)
    ok = np.isfinite(y)
    pair = None
    for i in range(0, 60):
        for j in range(i + 1, 60):
            a, b = aset.dense(i), aset.dense(j)
            m = np.zeros(len(y), bool)
            m[rows] = True
            if (a & b & ok & m).sum() > 80 and (a & ~b & ok & m).sum() > 80 and (b & ~a & ok & m).sum() > 80:
                pair = (i, j)
                break
        if pair:
            break
    assert pair is not None
    i, j = pair
    a, b = aset.dense(i), aset.dense(j)
    sel = np.zeros(len(y), bool)
    sel[rows] = True
    hold, mult = 6, 5
    out = lift_over_parents(aset, [i, j], y, hold, rows, mult)
    child = a & b & ok & sel
    for tag, parent, other in (("a", a, b), ("b", b, a)):
        rest = parent & ~other & ok & sel
        assert out[f"lift_{tag}"] == pytest.approx(y[child].mean() - y[rest].mean())
        assert out[f"n_rest_{tag}"] == int(rest.sum())
        # independent sandwich estimator: clusters = blocks of mult x hold rows counted from the start of the segment, observations = parent rows with a finite label
        pos = np.flatnonzero(parent[rows] & ok[rows])
        blocks = pos // (mult * hold)
        v = y[rows][pos]
        c_mask = (a & b)[rows][pos]
        n1, n0 = c_mask.sum(), (~c_mask).sum()
        m1, m0 = v[c_mask].mean(), v[~c_mask].mean()
        infl = []
        for g in np.unique(blocks):
            s = blocks == g
            infl.append(((v[s & c_mask] - m1).sum() / n1) - ((v[s & ~c_mask] - m0).sum() / n0))
        G = len(infl)
        se = np.sqrt(G / (G - 1) * (np.array(infl) ** 2).sum())
        assert out[f"lift_{tag}_t"] == pytest.approx((m1 - m0) / se, rel=1e-9)
    assert out["n_child"] == int(child.sum())
    # the validation rows are a different segment: shifting the segment changes the answer (the rows argument is honoured)
    other_rows = slice(tr.start, tr.start + (tr.stop - tr.start) // 2)
    assert lift_over_parents(aset, [i, j], y, hold, other_rows, mult)["n_child"] < out["n_child"]


def _rule(rid="R1", thr=1.0, direction="long", exit_="T6", cat=None, meta=None):
    return FrozenRule(rid, direction, [{"feature": "f", "op": "<=", "threshold": thr}, {"feature": "g", "op": ">", "threshold": 2.0}], exit_, cat, meta=meta or {}).validate()


def test_rule_sha_covers_what_the_engine_reads_and_nothing_else():
    base = hm.rule_sha(_rule())
    assert hm.rule_sha(_rule(meta={"anything": 1})) == base and hm.rule_sha(_rule(rid="OTHER")) == base
    assert hm.rule_sha(_rule(thr=1.0000001)) != base and hm.rule_sha(_rule(direction="short")) != base
    assert hm.rule_sha(_rule(exit_="T12")) != base and hm.rule_sha(_rule(cat=3.0)) != base


def test_verdict_of_boundaries():
    assert hm.verdict_of(19, 1.0, 10.0, 20) == "INCONCLUSIVE" and hm.verdict_of(20, 1.0, 10.0, 20) == "PASS"
    assert hm.verdict_of(20, 0.0, 10.0, 20) == "FAIL" and hm.verdict_of(20, 1.0, 0.0, 20) == "FAIL" and hm.verdict_of(20, 1.0, -1.0, 20) == "FAIL"
    assert hm.verdict_of(20, float("nan"), 5.0, 20) == "FAIL" and hm.verdict_of(0, float("nan"), 0.0, 20) == "INCONCLUSIVE"


def test_holdout_state_transitions_and_verdict_binding():
    r1, r2 = _rule("R1"), _rule("R2", thr=5.0)
    assert hm.holdout_state({"holdout_accesses": []})["status"] == "untouched" and hm.holdout_state({})["status"] == "untouched"
    start = {"event": "start", "time": "t0", "rule_shas": {"R1": hm.rule_sha(r1)}, "rule_ids": ["R1"]}
    m_started = {"holdout_accesses": [start]}
    m_open = {"holdout_accesses": [start, {"event": "complete"}]}
    assert hm.holdout_state(m_started)["status"] == "started" and hm.holdout_state(m_open)["status"] == "opened"
    res = {"rules": {"R1": {"verdict": "PASS"}}}
    assert hm.holdout_verdict_for({"holdout_accesses": []}, [r1], None) == {"R1": "NOT OPENED"}
    assert hm.holdout_verdict_for(m_open, [r1], res) == {"R1": "PASS"}
    assert hm.holdout_verdict_for(m_open, [_rule("R1", thr=1.5)], res) == {"R1": "BURNED"}          # same id, changed threshold: the verdict does not carry over
    assert hm.holdout_verdict_for(m_open, [r1, r2], res) == {"R1": "PASS", "R2": "BURNED"}          # a rule that was not opened cannot be opened later
    assert hm.holdout_verdict_for(m_started, [r1], None) == {"R1": "BURNED (access incomplete)"}


def test_walkforward_bar_windows_geometry():
    assert wf.bar_windows(100, 40, 10, 10) == [(0, 40, 50), (10, 50, 60), (20, 60, 70), (30, 70, 80), (40, 80, 90), (50, 90, 100)]
    assert len(wf.bar_windows(99, 40, 10, 10)) == 5 and wf.bar_windows(49, 40, 10, 10) == []        # the last window must fit completely
    assert wf.bar_windows(50, 40, 10, 10) == [(0, 40, 50)]
    with pytest.raises(ValueError):
        wf.bar_windows(100, 0, 10, 10)
    tp, vp, hp = wf.window_pcts(300, 100)
    assert (tp, vp, hp) == (0.75, 0.25, 0.0)


def test_walkforward_calendar_windows_geometry():
    idx = pd.date_range("2020-01-01", periods=26280, freq="h")              # 1095 days
    w = wf.calendar_windows(idx, 1.0, 0.5, 0.5)
    assert w == [(0, 8766, 13149), (4383, 13149, 17532), (8766, 17532, 21915)]
    assert w[0][0] == 0 and idx[w[1][0]] - idx[0] == pd.Timedelta(days=182.625)
    assert wf.calendar_windows(idx, 3.0, 1.0, 0.5) == []
    with pytest.raises(ValueError):
        wf.calendar_windows(idx, 1.0, 0.0, 0.5)


def test_walkforward_calendar_windows_keep_a_window_that_ends_at_the_last_bar():
    h = 13149                                                                # train 1.0 + test 0.5 years = 547.875 days = 13149 hours
    on_edge = pd.date_range("2020-01-01", periods=h + 1, freq="h")           # last timestamp == the test end: the window fits
    assert wf.calendar_windows(on_edge, 1.0, 0.5, 0.5) == [(0, 8766, 13149)]
    short = pd.date_range("2020-01-01", periods=h, freq="h")                 # last timestamp one hour before the test end: it does not
    assert wf.calendar_windows(short, 1.0, 0.5, 0.5) == []
    late = pd.date_range("2020-01-01", periods=h + 30 * 24, freq="h")        # data continues 30 days past the window: still included (and no second window fits)
    assert wf.calendar_windows(late, 1.0, 0.5, 0.5) == [(0, 8766, 13149)]


def test_apply_holdout_verdicts_before_the_opening_marks_every_rule_not_opened():
    from edge_research.pipeline import apply_holdout_verdicts
    r1, r2 = _rule("R1"), _rule("R2", thr=5.0)
    t = pd.DataFrame({"rule_id": ["R1", "R2"], "eligible_for_holdout": [True, False]})
    untouched = {"holdout_accesses": []}
    assert list(apply_holdout_verdicts(t, [r1, r2], untouched, None)["g10_holdout"]) == ["NOT OPENED", "NOT OPENED"]      # nothing is decided before the opening, eligible or not
    opened = {"holdout_accesses": [{"event": "start", "rule_shas": {"R1": hm.rule_sha(r1)}}, {"event": "complete"}]}
    out = apply_holdout_verdicts(t, [r1, r2], opened, {"rules": {"R1": {"verdict": "FAIL"}}})
    assert list(out["g10_holdout"]) == ["FAIL", "NOT ELIGIBLE"]
    assert list(t.columns) == ["rule_id", "eligible_for_holdout"]            # the input table is not modified
    assert apply_holdout_verdicts(t.iloc[:0], [], opened, None).empty


def test_walkforward_aggregate_hand_example():
    wins = [dict(n_rules=2, n_trades=2, oos_mean_R=1.0, is_mean_R=2.0), dict(n_rules=0, n_trades=0, oos_mean_R=np.nan, is_mean_R=np.nan),
            dict(n_rules=1, n_trades=2, oos_mean_R=-0.5, is_mean_R=1.0)]
    tr = pd.DataFrame({"r_net": [2.0, 0.0, -1.0, 0.0], "profit": [200.0, 0.0, -100.0, 0.0], "exit_time": pd.date_range("2020-01-01", periods=4, freq="D")})
    a = wf.aggregate_windows(wins, tr, 10000.0)
    assert a["n_windows"] == 3 and a["n_windows_with_rules"] == 2 and a["share_windows_with_rules"] == pytest.approx(2 / 3) and a["n_trades"] == 4
    assert a["oos_mean_R"] == pytest.approx(0.25) and a["oos_net_profit"] == pytest.approx(100.0) and a["is_mean_R"] == pytest.approx(1.5)
    assert a["oos_is_ratio"] == pytest.approx(0.25 / 1.5) and a["share_windows_positive"] == pytest.approx(0.5)
    assert a["window_mean_R_std"] == pytest.approx(np.std([1.0, -0.5], ddof=1)) and a["positive_oos"] is True
    assert a["oos_equity_dd_abs"] == pytest.approx(100.0) and a["oos_equity_dd_pct"] == pytest.approx(100 / 10200 * 100)
    neg = wf.aggregate_windows(wins, tr.assign(r_net=[-1.0, 0.0, -1.0, 0.0], profit=[-100.0, 0.0, -100.0, 0.0]), 10000.0)
    assert neg["positive_oos"] is False and neg["oos_is_ratio"] == pytest.approx(-0.5 / 1.5)
    empty = wf.aggregate_windows([dict(n_rules=0, n_trades=0, oos_mean_R=np.nan, is_mean_R=np.nan)], pd.DataFrame(), 10000.0)
    assert empty["n_trades"] == 0 and empty["positive_oos"] is False and empty["share_windows_with_rules"] == 0.0


def test_walkforward_drawdown_uses_exit_time_order_not_row_order():
    tr = pd.DataFrame({"r_net": [1.0, -1.0], "profit": [100.0, -100.0], "exit_time": pd.to_datetime(["2020-01-02", "2020-01-01"])})     # the loss comes first in time
    wins = [dict(n_rules=1, n_trades=2, oos_mean_R=0.0, is_mean_R=1.0)]
    a = wf.aggregate_windows(wins, tr, 1000.0)
    assert a["oos_equity_dd_abs"] == pytest.approx(100.0) and a["oos_equity_dd_pct"] == pytest.approx(10.0)     # 1000 -> 900 -> 1000


def test_config_validation_section_and_zero_holdout_split():
    v = CFG["validation"]
    assert v["alpha"] == 0.05 and v["min_trades_val"] == 50 and v["cost_stress_mult"] == 1.5 and v["open_holdout"] is False and v["require_g1"] is True
    from edge_research.splits import make_splits
    c = copy.deepcopy(CFG)
    c["data"].update(train_pct=0.75, val_pct=0.25, holdout_pct=0.0)
    s = make_splits(8000, c)
    assert s.holdout.stop - s.holdout.start == 0 and s.val.stop == 8000 and s.train.stop == 6000 - s.embargo and s.val.start == 6000 + s.embargo
    d = make_splits(8000, CFG)                                              # the normal split is unchanged
    assert d.holdout.stop == 8000 and d.holdout.stop - d.holdout.start > 0


def test_freeze_orders_by_train_plateau_caps_and_keeps_only_frozen_representatives():
    aset, lab, tr = _world(1, 12000, True)
    n = 6
    chosen = pd.DataFrame({"cand_id": np.arange(n), "atom_a": np.arange(n), "atom_b": np.arange(n) + 1, "expr_a": [f"a{i}" for i in range(n)], "expr_b": [f"b{i}" for i in range(n)],
                           "direction": ["long"] * n, "chosen_family": ["T6"] * n, "plateau_score": [1.0, 3.0, 2.0, 9.0, 5.0, 4.0], "rank_score": [0.0] * n,
                           "is_representative": [True, True, True, True, True, False], "frozen_ok": [True, True, False, False, True, True], "chosen_smooth": [0.5] * n, "best_t": [3.0] * n})
    rules = freeze_rules(chosen, aset, CFG)
    assert [r.rule_id for r in rules] == ["R00004", "R00001", "R00000"]       # plateau 5 (id 4), 3 (id 1), 1 (id 0); id 3 and 2 are not frozen_ok, id 5 is not a representative
    capped = freeze_rules(chosen, aset, CFG, max_rules=1)
    assert [r.rule_id for r in capped] == ["R00004"]
    assert freeze_rules(chosen.iloc[:0], aset, CFG) == [] and freeze_rules(None, aset, CFG) == []
    r = rules[0]
    assert r.direction == "long" and r.exit == "T6" and len(r.atoms) == 2 and r.meta["atom_ids"] == [4, 5] and r.meta["exprs"] == ["a4", "b4"]
    assert r.atoms[0]["feature"] == aset.atoms[4].feature and r.atoms[0]["threshold"] == float(aset.atoms[4].threshold)
    assert rule_id_of(7) == "R00007"
    flags = pd.DataFrame({"cand_id": [4], "beats_random": [True], "beats_calendar": [False], "calendar_driven": [True]})
    r2 = freeze_rules(chosen, aset, CFG, null_flags=flags)
    by = {x.rule_id: x for x in r2}
    assert by["R00004"].meta["null_beats_random"] is True and "null_beats_random" not in by["R00001"].meta


# ================================================================ integration: a small planted-edge world
LOOSE_V = dict(max_rules=8, require_g1=False, alpha=0.5, decay_min_ratio=0.0, lift_t_min=-1e9, min_trades_val=1, min_profit_factor=0.0, max_equity_drawdown_pct=100.0, min_holdout_trades=1)


def _vcfg(loose=False, **v):
    c = copy.deepcopy(CFG)
    if loose:
        c["validation"].update(LOOSE_V)
        c["candidates"]["lift_t_min"] = -1e9                                  # train side of G4 (only stage 10 sees this; no earlier stage is rerun)
    c["validation"].update(v)
    return c


def _world_dir(tmp_path_factory, name):
    """A persistent folder when M8_CACHE is set (mutation-check harness: the slow, M8-independent stages 0-8 are built once), otherwise a temp folder."""
    root = os.environ.get("M8_CACHE")
    if root:
        d = __import__("pathlib").Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path_factory.mktemp(name)


@pytest.fixture(scope="module")
def m8_base(tmp_path_factory):
    from edge_research.pipeline import run_stages_0_8
    d = _world_dir(tmp_path_factory, "m8")
    df, meta = make_synthetic_bars(12000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, d / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(d / "s_spread.csv", index=False)
    run = d / "run"
    out = run_stages_0_8(csv, run, CFG, "SYNTH", d / "s_spread.csv")
    assert out["n_frozen"] > 0
    return run, csv, d / "s_spread.csv"


@pytest.fixture()
def run_copy(m8_base, tmp_path):
    run, csv, sps = m8_base
    dst = tmp_path / "run"
    shutil.copytree(run, dst)
    return dst, sps


def test_stage10_runs_resumes_and_invalidates(run_copy):
    from edge_research.pipeline import run_stage_10
    run, sps = run_copy
    cfg = _vcfg(loose=True)
    r = run_stage_10(run, cfg, sps)
    d = run / "10_validation"
    assert r["status"] == "ran" and 1 <= r["n_frozen"] <= 8 and r["g1_pass"] is None
    for f in ("frozen_rules.json", "accepted_rules.json", "gate_table.csv", "gates.md", "validation_info.json"):
        assert (d / f).exists(), f
    t = pd.read_csv(d / "gate_table.csv")
    assert len(t) == r["n_frozen"] and (t["k_tested"] == len(t)).all()
    assert (t["accepted"] == t[GATE_COLS].all(axis=1)).all()                                       # accepted = all of G2-G8, row by row
    assert r["n_accepted"] == int(t["accepted"].sum()) and r["n_eligible"] == int(t["eligible_for_holdout"].sum()) and r["n_eligible"] >= 1
    assert set(t["lift_train_source"]) <= {"stored", "recomputed"} and (t["g10_holdout"] == "NOT OPENED").all()
    assert (t["t_crit_bonferroni"] > 1.0).all() and (t["val_trades"] >= 0).all()
    has = t["val_trades"] > 0
    assert (t.loc[has, "stress_mean_R"] < t.loc[has, "val_mean_R"]).all()                          # the stressed run has costlier fills on the same trades
    from edge_research.rules import load_rules
    frozen, acc = load_rules(d / "frozen_rules.json"), load_rules(d / "accepted_rules.json")
    assert [x.rule_id for x in frozen] == list(t["rule_id"]) and {x.rule_id for x in acc} == set(t.loc[t["accepted"], "rule_id"])
    assert "Holdout: untouched" in (d / "gates.md").read_text()
    m = json.loads((run / "manifest.json").read_text())
    assert m["stages"]["10_validation"]["status"] == "done" and m["holdout_accesses"] == []         # stage 10 never touches the holdout by itself
    assert run_stage_10(run, cfg, sps)["status"] == "skipped"
    c2 = _vcfg(loose=True, min_trades_val=10**6)                                                   # validation settings are part of the fingerprint
    r2 = run_stage_10(run, c2, sps)
    assert r2["status"] == "ran" and r2["n_accepted"] == 0 and r2["n_eligible"] == 0
    assert pd.read_csv(d / "gate_table.csv")["g6_trades"].sum() == 0
    assert len(load_rules(d / "accepted_rules.json")) == 0
    c3 = _vcfg(loose=True, max_rules=3)
    assert run_stage_10(run, c3, sps)["n_frozen"] == 3
    c4 = copy.deepcopy(c3)
    c4["exits"]["min_improvement_R"] = 0.5                                                         # an upstream section changed: the stage fingerprint changes
    assert run_stage_10(run, c4, sps)["status"] == "ran"


def test_stage10_without_null_test_is_not_eligible_when_g1_required(run_copy):
    from edge_research.pipeline import run_stage_10
    run, sps = run_copy
    r = run_stage_10(run, _vcfg(loose=True, require_g1=True), sps)
    assert r["g1_pass"] is None and r["n_eligible"] == 0
    t = pd.read_csv(run / "10_validation" / "gate_table.csv")
    assert t["g1_null"].isna().all() and t["fail_reasons"].str.contains("g1_null").all()


def test_evaluate_rule_stress_is_exactly_the_engine_cost_multiplier(run_copy):
    from edge_research.pipeline import load_run_inputs
    from edge_research.validation.gates import evaluate_rule
    run, sps = run_copy
    cfg = _vcfg(loose=True, max_rules=2)
    aset = load_atoms(run / "04_atoms")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    df, ft, spread, splits = load_run_inputs(run, cfg, sps)
    rule = freeze_rules(chosen, aset, cfg, max_rules=1)[0]
    base = evaluate_rule(df, cfg, rule, ft, spread, splits.val)
    st = evaluate_rule(df, cfg, rule, ft, spread, splits.val, 1.5)
    c2 = copy.deepcopy(cfg)
    c2["engine"]["cost_multiplier"] = 1.5
    ref = evaluate_rule(df, c2, rule, ft, spread, splits.val)
    assert st == ref and st["mean_R"] < base["mean_R"] and base["n_trades"] == st["n_trades"]             # same trades, costlier fills
    c3 = copy.deepcopy(cfg)
    c3["engine"]["cost_multiplier"] = 2.0                                                              # a configured multiplier is multiplied, not replaced
    assert evaluate_rule(df, c3, rule, ft, spread, splits.val, 1.5) == evaluate_rule(df, copy.deepcopy(cfg) | {"engine": {**cfg["engine"], "cost_multiplier": 3.0}}, rule, ft, spread, splits.val)
    assert cfg["engine"].get("cost_multiplier", 1.0) == 1.0                                              # the caller's config is not modified


def test_gate_table_recomputes_the_train_lift_when_stage6_stored_none(run_copy):
    from edge_research.pipeline import load_run_inputs
    run, sps = run_copy
    cfg = _vcfg(loose=True, max_rules=4)
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    fam = pd.read_parquet(run / "06_candidates" / "candidate_families.parquet")
    filt = pd.read_parquet(run / "07_filters" / "candidates_filtered.parquet")
    df, ft, spread, splits = load_run_inputs(run, cfg, sps)
    rules = freeze_rules(chosen, aset, cfg, max_rules=4)
    stored = build_gate_table(rules, chosen, filt, fam, aset, labels, df, ft, spread, splits.val, cfg, None)
    assert (stored["lift_train_source"] == "stored").all()
    fam2 = fam.copy()
    fam2[["lift_a_t", "lift_b_t"]] = np.nan
    rec = build_gate_table(rules, chosen, filt, fam2, aset, labels, df, ft, spread, splits.val, cfg, None)
    assert (rec["lift_train_source"] == "recomputed").all() and np.isfinite(rec["lift_a_t_train"]).all()
    r0 = rules[0]
    y = net_R(labels, r0.direction, r0.exit).to_numpy(np.float64)
    from edge_research.discovery.pairs import family_hold
    ref = lift_over_parents(aset, r0.meta["atom_ids"], y, family_hold(r0.exit, cfg), slice(*aset.train), cfg["candidates"]["lift_block_mult"])
    assert rec["lift_a_t_train"].iloc[0] == pytest.approx(ref["lift_a_t"]) and rec["lift_b_t_train"].iloc[0] == pytest.approx(ref["lift_b_t"])
    assert rec["lift_a_t_train"].iloc[0] == pytest.approx(stored["lift_a_t_train"].iloc[0], abs=1.0)   # same estimator, only the block alignment may differ from Stage 6 (a sanity bound)


def test_gate_table_reads_g2_g3_from_the_stage_tables_and_the_validation_lift_from_validation_rows(run_copy):
    from edge_research.pipeline import load_run_inputs
    run, sps = run_copy
    cfg = _vcfg(loose=True, max_rules=4)
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    fam = pd.read_parquet(run / "06_candidates" / "candidate_families.parquet")
    filt = pd.read_parquet(run / "07_filters" / "candidates_filtered.parquet")
    df, ft, spread, splits = load_run_inputs(run, cfg, sps)
    rules = freeze_rules(chosen, aset, cfg, max_rules=4)
    ids = [r.meta["cand_id"] for r in rules]
    base = build_gate_table(rules, chosen, filt, fam, aset, labels, df, ft, spread, splits.val, cfg, None)
    assert base["g2_time_stability"].all() and base["g3_exit_plateau"].all()                        # frozen rules passed F2 / F3 / F4 upstream
    filt2, chosen2 = filt.copy(), chosen.copy()
    filt2.loc[filt2["cand_id"] == ids[0], "pass_f2"] = False
    filt2.loc[filt2["cand_id"] == ids[1], "pass_f4"] = False
    chosen2.loc[chosen2["cand_id"] == ids[2], "pass_f3"] = False
    t = build_gate_table(rules, chosen2, filt2, fam, aset, labels, df, ft, spread, splits.val, cfg, None).set_index("rule_id")
    r = [x.rule_id for x in rules]
    assert list(t["g2_time_stability"]) == [False, True, True, True] and list(t["g3_exit_plateau"]) == [True, False, False, True]
    assert list(t["accepted"][:3]) == [False, False, False] and "g2_time_stability" in t.at[r[0], "fail_reasons"] and "g3_exit_plateau" in t.at[r[2], "fail_reasons"]
    for i in range(4):                                                                              # the other columns do not depend on those flags
        assert t.at[r[i], "val_mean_R"] == base.set_index("rule_id").at[r[i], "val_mean_R"] if base.set_index("rule_id").at[r[i], "val_trades"] else True
    # the validation lift columns are the F5 estimate on the VALIDATION rows (child count by brute force)
    r0 = rules[0]
    y = net_R(labels, r0.direction, r0.exit).to_numpy(np.float64)
    a, b = aset.dense(r0.meta["atom_ids"][0]), aset.dense(r0.meta["atom_ids"][1])
    m = np.zeros(len(y), bool)
    m[splits.val] = True
    assert base["lift_n_child_val"].iloc[0] == int((a & b & np.isfinite(y) & m).sum())


def test_open_holdout_refuses_an_empty_segment_duplicates_and_logs_nothing(run_copy):
    from edge_research.pipeline import load_run_inputs
    run, sps = run_copy
    cfg = _vcfg(loose=True, max_rules=2)
    aset = load_atoms(run / "04_atoms")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    df, ft, spread, splits = load_run_inputs(run, cfg, sps)
    rules = freeze_rules(chosen, aset, cfg, max_rules=2)
    n = len(df)
    with pytest.raises(hm.HoldoutNotAllowed, match="empty"):
        hm.open_holdout(run, cfg, rules, df, ft, spread, slice(n, n), "x")
    with pytest.raises(hm.HoldoutNotAllowed, match="duplicate"):
        hm.open_holdout(run, cfg, [rules[0], rules[0]], df, ft, spread, splits.holdout, "x")
    assert json.loads((run / "manifest.json").read_text())["holdout_accesses"] == [] and not hm.results_path(run).exists()


def test_gate_table_reads_only_validation_rows_and_never_the_holdout(run_copy, monkeypatch):
    from edge_research.pipeline import load_run_inputs
    from edge_research.validation import gates as gm
    run, sps = run_copy
    cfg = _vcfg(loose=True, max_rules=6)
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    fam = pd.read_parquet(run / "06_candidates" / "candidate_families.parquet")
    filt = pd.read_parquet(run / "07_filters" / "candidates_filtered.parquet")
    df, ft, spread, splits = load_run_inputs(run, cfg, sps)
    rules = freeze_rules(chosen, aset, cfg, max_rules=6)
    seen = []
    real = gm.run_backtest
    monkeypatch.setattr(gm, "run_backtest", lambda *a, **k: (seen.append(a[5]), real(*a, **k))[1])
    t1 = build_gate_table(rules, chosen, filt, fam, aset, labels, df, ft, spread, splits.val, cfg, None)
    assert seen and all(s == splits.val for s in seen) and len(seen) == 2 * len(rules)           # base + stressed run, validation rows only
    h = splits.holdout.start
    rng = np.random.default_rng(0)

    def scramble(x):
        x = x.copy()
        for c in x.columns:
            if pd.api.types.is_float_dtype(x[c]):
                x.iloc[h:, x.columns.get_loc(c)] = (rng.normal(size=len(x) - h) * 1e3).astype(x[c].dtype)
        return x
    t2 = build_gate_table(rules, chosen, filt, fam, aset, scramble(labels), scramble(df), scramble(ft), spread, splits.val, cfg, None)
    pd.testing.assert_frame_equal(t1, t2)                                                          # destroying every holdout row changes nothing


def test_holdout_opens_once_logs_before_reading_locks_results_and_refuses_a_second_time(run_copy, monkeypatch):
    from edge_research.pipeline import run_holdout_stage, run_stage_10
    run, sps = run_copy
    cfg = _vcfg(loose=True)
    s10 = run_stage_10(run, cfg, sps)
    assert s10["n_eligible"] >= 1 and "holdout" not in s10
    mpath = run / "manifest.json"
    order = []
    real = hm.run_backtest

    def spy(*a, **k):
        m = json.loads(mpath.read_text())
        order.append([x["event"] for x in m["holdout_accesses"]])                                   # what the manifest says at the moment the holdout is read
        return real(*a, **k)
    monkeypatch.setattr(hm, "run_backtest", spy)
    out = run_holdout_stage(run, cfg, sps)
    assert out["status"] == "opened" and out["n_rules"] == s10["n_eligible"] and order and all(o == ["start"] for o in order)
    m = json.loads(mpath.read_text())
    assert [x["event"] for x in m["holdout_accesses"]] == ["start", "complete"] and m["holdout_accesses"][0]["rule_ids"] == list(out["verdicts"])
    p = hm.results_path(run)
    assert p.exists() and not (os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))   # read-only
    with pytest.raises(FileExistsError):
        open(p, "x")                                                                                 # the file cannot be created again
    res = hm.read_results(run)
    assert set(res["rules"]) == set(out["verdicts"]) and all(v["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE") for v in res["rules"].values())
    t = pd.read_csv(run / "10_validation" / "gate_table.csv")
    elig = t["eligible_for_holdout"]
    assert set(t.loc[elig, "g10_holdout"]) <= {"PASS", "FAIL", "INCONCLUSIVE"}
    if (~elig).any():
        assert set(t.loc[~elig, "g10_holdout"]) == {"NOT ELIGIBLE"}
    for rid, v in res["rules"].items():
        assert t.set_index("rule_id").at[rid, "g10_holdout"] == v["verdict"]
    snapshot = mpath.read_text()
    calls = len(order)
    with pytest.raises(hm.HoldoutNotAllowed, match="already accessed"):
        run_holdout_stage(run, cfg, sps)
    assert mpath.read_text() == snapshot and len(order) == calls                                    # a refused second opening reads nothing and logs nothing
    # stage 10 rerun after the opening keeps the stored verdicts and does not open anything
    r2 = run_stage_10(run, _vcfg(loose=True, max_equity_drawdown_pct=99.0), sps)
    assert r2["status"] == "ran" and len(order) == calls
    t2 = pd.read_csv(run / "10_validation" / "gate_table.csv")
    for rid, v in res["rules"].items():
        if rid in set(t2["rule_id"]) and t2.set_index("rule_id").at[rid, "eligible_for_holdout"]:
            assert t2.set_index("rule_id").at[rid, "g10_holdout"] == v["verdict"]
    assert "opened once" in (run / "10_validation" / "gates.md").read_text()


def test_holdout_refused_when_nothing_is_eligible_or_stage10_is_stale_and_burns_nothing(run_copy):
    from edge_research.pipeline import run_holdout_stage, run_stage_10
    run, sps = run_copy
    mpath = run / "manifest.json"
    with pytest.raises(hm.HoldoutNotAllowed, match="stage 10"):
        run_holdout_stage(run, _vcfg(loose=True), sps)                                              # stage 10 has not run
    run_stage_10(run, _vcfg(loose=True, min_trades_val=10**6), sps)
    with pytest.raises(hm.HoldoutNotAllowed, match="no rule is eligible"):
        run_holdout_stage(run, _vcfg(loose=True, min_trades_val=10**6), sps)
    with pytest.raises(hm.HoldoutNotAllowed, match="stale"):
        run_holdout_stage(run, _vcfg(loose=True), sps)                                              # a different config than the one stage 10 ran with
    assert json.loads(mpath.read_text())["holdout_accesses"] == [] and not hm.results_path(run).exists()   # none of the refusals used the holdout up


def test_holdout_crash_after_start_counts_as_used(run_copy, monkeypatch):
    from edge_research.pipeline import run_holdout_stage, run_stage_10
    run, sps = run_copy
    cfg = _vcfg(loose=True)
    run_stage_10(run, cfg, sps)
    monkeypatch.setattr(hm, "run_backtest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        run_holdout_stage(run, cfg, sps)
    m = json.loads((run / "manifest.json").read_text())
    assert [x["event"] for x in m["holdout_accesses"]] == ["start"] and hm.holdout_state(m)["status"] == "started" and not hm.results_path(run).exists()
    monkeypatch.undo()
    with pytest.raises(hm.HoldoutNotAllowed, match="already accessed"):
        run_holdout_stage(run, cfg, sps)                                                            # the holdout may have been seen: it is not opened again
    run_stage_10(run, _vcfg(loose=True, max_equity_drawdown_pct=99.0), sps)
    t = pd.read_csv(run / "10_validation" / "gate_table.csv")
    assert set(t.loc[t["eligible_for_holdout"], "g10_holdout"]) == {"BURNED (access incomplete)"}


def test_open_holdout_flag_opens_inside_stage_10_exactly_once(run_copy):
    from edge_research.pipeline import run_stage_10
    run, sps = run_copy
    r = run_stage_10(run, _vcfg(loose=True, open_holdout=True), sps)
    assert r["holdout"] == "opened"
    m = json.loads((run / "manifest.json").read_text())
    assert [x["event"] for x in m["holdout_accesses"]] == ["start", "complete"]
    r2 = run_stage_10(run, _vcfg(loose=True, open_holdout=True, max_equity_drawdown_pct=99.0), sps)                    # open_holdout stays true: still no second opening
    assert "holdout" not in r2 and len(json.loads((run / "manifest.json").read_text())["holdout_accesses"]) == 2


def test_backtest_stage_still_refuses_the_holdout_segment(run_copy):
    from edge_research.pipeline import run_backtest_stage, run_stage_10
    run, sps = run_copy
    cfg = _vcfg(loose=True)
    run_stage_10(run, cfg, sps)
    with pytest.raises(PermissionError):
        run_backtest_stage(run, cfg, run / "10_validation" / "frozen_rules.json", segment="holdout", spread_csv=sps)


def test_stage11_backtests_only_accepted_rules_and_resumes(run_copy):
    from edge_research.pipeline import run_stage_10, run_stage_11
    run, sps = run_copy
    cfg = _vcfg(loose=True)
    s10 = run_stage_10(run, cfg, sps)
    assert s10["n_accepted"] >= 1
    r = run_stage_11(run, cfg, sps)
    assert r["status"] == "ran" and r["n_rules"] == s10["n_accepted"] and set(r["summaries"]) == {"train", "val"}
    for seg in ("train", "val"):
        assert (run / "11_backtest" / seg / "trades.csv").exists() and (run / "11_backtest" / seg / "summary.json").exists()
    assert not (run / "11_backtest" / "holdout").exists()
    t = pd.read_csv(run / "10_validation" / "gate_table.csv").set_index("rule_id")
    tv = pd.read_csv(run / "11_backtest" / "val" / "trades.csv")
    assert set(tv["rule_id"]) <= set(t.index[t["accepted"]])                                         # no unaccepted rule is traded
    if len(tv) and s10["n_accepted"] == 1:
        rid = tv["rule_id"].iloc[0]
        assert len(tv) == int(t.at[rid, "val_trades"]) and tv["r_net"].mean() == pytest.approx(t.at[rid, "val_mean_R"])     # stage 11 val run == the gate's engine run
    assert run_stage_11(run, cfg, sps)["status"] == "skipped"
    run_stage_10(run, _vcfg(loose=True, min_trades_val=10**6), sps)
    r0 = run_stage_11(run, _vcfg(loose=True, min_trades_val=10**6), sps)
    assert r0["status"] == "no_rules" and r0["n_rules"] == 0


# ================================================================ G11 demo forward (offline harness)
def _demo_world():
    idx = pd.date_range("2026-01-05 08:00", periods=4, freq="24h")
    eng = pd.DataFrame({"entry_time": idx, "exit_time": idx + pd.Timedelta(hours=6), "entry_price": [100.0, 101.0, 102.0, 103.0],
                        "exit_price": [101.0, 102.0, 103.0, 104.0], "direction": ["long"] * 4, "profit": [10.0, 10.0, 10.0, 10.0]})
    demo = pd.DataFrame({"time": eng["entry_time"].dt.strftime("%Y.%m.%d %H:%M"), "close_time": eng["exit_time"].dt.strftime("%Y.%m.%d %H:%M"),
                         "type": ["buy"] * 4, "price": eng["entry_price"], "close_price": eng["exit_price"], "profit": eng["profit"]})
    return eng, demo


def test_demo_gate_verdicts_and_statement_loading(tmp_path):
    eng, demo = _demo_world()
    p = tmp_path / "demo_R00001.csv"
    demo.to_csv(p, index=False)
    st = load_demo_statement(p)
    assert len(st) == 4 and st["open_time"].iloc[0] == pd.Timestamp("2026-01-05 08:00")       # MT5 aliases are understood
    r = evaluate_demo_gate(eng, p, min_trades=4)
    assert r["verdict"] == "PASS" and r["n_demo"] == 4 and r["entry_match_pct"] == 100.0 and r["accepted"]
    assert r["window"]["start"].startswith("2026-01-05")
    bad = tmp_path / "bad.csv"
    bad.write_text("time,close_time,type,price,close_price,profit\n" + "\n".join(
        f"{t:%Y.%m.%d %H:%M},{t + pd.Timedelta(hours=7):%Y.%m.%d %H:%M},buy,{pr},{px},{pf}"
        for t, pr, px, pf in zip(eng["entry_time"], eng["entry_price"], eng["exit_price"], eng["profit"])))
    assert evaluate_demo_gate(eng, bad)["verdict"] == "FAIL"                                  # every exit is one hour late
    empty = tmp_path / "empty.csv"
    demo.iloc[:0].to_csv(empty, index=False)
    assert evaluate_demo_gate(eng, empty, min_trades=4)["verdict"] == "NO_TRADES"             # a demo run without trades is a verdict too
    assert evaluate_demo_gate(eng, p, min_trades=10)["verdict"] == "WARN"                     # matched but too few trades
    (tmp_path / "x.csv").write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="column"):
        load_demo_statement(tmp_path / "x.csv")


def test_update_gate_table_writes_g11_column_and_json(run_copy):
    run, _sps = run_copy
    gdir = run / "10_validation"
    gdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rule_id": ["R00001", "R00002"], "g11_demo": ["PENDING", "PENDING"]}).to_csv(gdir / "gate_table.csv", index=False)
    res = {"R00001": {"verdict": "PASS", "n_demo": 12, "min_trades": 10, "window": {"start": "2026-01-05", "end": "2026-02-02"}},
           "R99999": {"verdict": "FAIL", "n_demo": 3, "min_trades": 10, "window": {"start": "2026-01-05", "end": "2026-02-02"},
                      "first_difference": {"reason": "only in the engine"}}}
    updated = update_gate_table(run, res)
    back = pd.read_csv(gdir / "gate_table.csv").set_index("rule_id")
    assert updated == ["R00001", "R99999"] and back.at["R00001", "g11_demo"].startswith("PASS (n=12")
    assert back.at["R99999", "g11_demo"].startswith("FAIL") and "only in the engine" in back.at["R99999", "g11_demo"]
    assert back.at["R00002", "g11_demo"] == "PENDING"                                         # no statement: the rule stays pending
    js = json.loads((gdir / "g11_demo.json").read_text())
    assert set(js["results"]) == {"R00001", "R99999"} and js["updated"] == updated
    assert update_gate_table(run, {}) == []
    with pytest.raises(FileNotFoundError):
        update_gate_table(run.parent, res)


def test_load_run_inputs_accepts_parquet_and_feather_and_rejects_ambiguity(run_copy, tmp_path):
    from edge_research.pipeline import load_run_inputs
    run, sps = run_copy
    files = list((run / "_ingest").glob("*"))
    assert len(files) == 1
    a = load_run_inputs(run, CFG, sps)[0]
    fake = tmp_path / "fake"
    (fake / "_ingest").mkdir(parents=True)
    (fake / "01_features").mkdir()
    shutil.copy(run / "01_features" / "feature_table.parquet", fake / "01_features" / "feature_table.parquet")
    assert files[0].suffix == ".parquet"                                    # pyarrow is installed: the loader writes parquet (feather is the no-pyarrow fallback)
    pd.read_parquet(files[0]).reset_index().to_feather(fake / "_ingest" / "SYMBOL_H1.feather")
    b = load_run_inputs(fake, CFG, sps)[0]
    pd.testing.assert_frame_equal(a, b)
    shutil.copy(fake / "_ingest" / "SYMBOL_H1.feather", fake / "_ingest" / "OTHER_H1.feather")
    with pytest.raises(FileNotFoundError):
        load_run_inputs(fake, CFG, sps)


# ================================================================ walk-forward end to end
def _wcfg(**w):
    c = copy.deepcopy(CFG)
    c["validation"]["walkforward"].update(dict(mode="bars", train_bars=7200, test_bars=1800, step_bars=4000, max_rules_per_window=5, min_window_bars=3000))
    c["validation"]["walkforward"].update(w)
    return c


@pytest.fixture(scope="module")
def wf_world(tmp_path_factory):
    d = _world_dir(tmp_path_factory, "wf")
    df, meta = make_synthetic_bars(16000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, d / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(d / "s_spread.csv", index=False)
    return d, csv, d / "s_spread.csv", df


def test_walkforward_end_to_end_windows_oos_only_holdout_untouched_and_resume(wf_world):
    d, csv, sps, df = wf_world
    cfg = _wcfg()
    for f in list((d / "wf1" / "walkforward").glob("w*/window_result.json")) + list((d / "wf1" / "walkforward").glob("w*/oos_trades.parquet")):
        f.unlink()                                                                                  # (cached folder only) M8 results are always recomputed, stages 0-8 are not
    out = wf.run_walkforward(csv, d / "wf1", cfg, sps, "SYNTH", None, 1)
    assert out["complete"] is False and out["new_windows"] == 1 and out["n_windows"] == 1 and out["n_windows_planned"] == 2          # max_windows bounds the new windows of one call
    full = wf.make_splits(16000, cfg)
    assert out["usable_rows"] == full.holdout.start - full.embargo                                                                      # the full run's holdout is never used
    out2 = wf.run_walkforward(csv, d / "wf1", cfg, sps, "SYNTH")
    assert out2["complete"] is True and out2["new_windows"] == 1 and out2["n_windows"] == 2
    out3 = wf.run_walkforward(csv, d / "wf1", cfg, sps, "SYNTH")
    assert out3["new_windows"] == 0 and (out3["oos_mean_R"] == out2["oos_mean_R"] or (np.isnan(out3["oos_mean_R"]) and np.isnan(out2["oos_mean_R"])))
    base = d / "wf1" / "walkforward"
    for f in ("summary.json", "summary.md", "windows.csv"):
        assert (base / f).exists(), f
    summ = json.loads((base / "summary.json").read_text())
    wins = summ["windows"]
    assert [w["start_row"] for w in wins] == [0, 4000] and [w["end_row"] for w in wins] == [9000, 13000] and all(w["end_row"] <= out["usable_rows"] for w in wins)
    assert [w["test_start_time"][:16] for w in wins] == [pd.Timestamp(df.index[w["train_end_row"]]).strftime("%Y-%m-%d %H:%M") for w in wins]
    assert sum(w["n_rules"] > 0 for w in wins) == out2["n_windows_with_rules"]
    if (base / "oos_trades.csv").exists():
        tr = pd.read_csv(base / "oos_trades.csv", parse_dates=["entry_time", "signal_time"])
        assert len(tr) == out2["n_trades"] == sum(w["n_trades"] for w in wins)
        for w in wins:
            tw = tr[tr["window"] == w["window"]]
            assert (tw["signal_time"] >= pd.Timestamp(w["test_start_time"])).all() and (tw["entry_time"] <= pd.Timestamp(w["end_time"])).all()   # OOS trades only inside the test part
        assert out2["oos_mean_R"] == pytest.approx(tr["r_net"].mean())
        assert out2["n_windows_with_rules"] >= 1
    md = (base / "summary.md").read_text()
    assert "Outer walk-forward" in md and "holdout of the full run is never used" in md and "Verdict:" in md
    # the procedure only ever saw the TRAIN part of a window: the window's atoms (thresholds, support) were built inside it
    for k, w in enumerate(wins):
        aset = load_atoms(d / "wf1" / "walkforward" / f"w{k:03d}" / "run" / "04_atoms")
        n_train = w["train_end_row"] - w["start_row"]
        assert aset.train[1] <= n_train
    # a changed procedure setting invalidates finished windows
    out4 = wf.run_walkforward(csv, d / "wf1", _wcfg(max_rules_per_window=3), sps, "SYNTH", None, 1)
    assert out4["new_windows"] == 1
    # null approximation: only rules whose best-family t beats the FULL-run threshold; flagged in the report
    (d / "wf1" / "09_null").mkdir(exist_ok=True)
    nj = d / "wf1" / "09_null" / "null_info.json"
    nj.write_text(json.dumps({"summary": {"variants": {"random": {"S_threshold": 1e9}}}}))
    hi = wf.run_walkforward(csv, d / "wf1", _wcfg(require_null_approx=True), sps, "SYNTH")
    assert hi["n_windows"] == 2 and hi["n_windows_with_rules"] == 0 and hi["n_trades"] == 0 and hi["null_approximation"] is True and hi["null_threshold"] == 1e9
    assert "approximation" in (d / "wf1" / "walkforward" / "summary.md").read_text() and hi["positive_oos"] is False
    nj.write_text(json.dumps({"summary": {"variants": {"random": {"S_threshold": -1e9}}}}))
    lo = wf.run_walkforward(csv, d / "wf1", _wcfg(require_null_approx=True), sps, "SYNTH")
    assert lo["n_windows_with_rules"] == out2["n_windows_with_rules"] and lo["n_trades"] == out2["n_trades"]


def test_walkforward_null_approximation_needs_stage9_and_flags_the_report(wf_world, tmp_path):
    d, csv, sps, df = wf_world
    with pytest.raises(FileNotFoundError, match="require_null_approx"):
        wf.run_walkforward(csv, tmp_path / "nowf", _wcfg(require_null_approx=True), sps, "SYNTH", None, 0)
    with pytest.raises(ValueError, match="mode"):
        wf.run_walkforward(csv, tmp_path / "bad", _wcfg(mode="weekly"), sps, "SYNTH", None, 0)
    out = wf.run_walkforward(csv, tmp_path / "zero", _wcfg(), sps, "SYNTH", None, 0)                   # max_windows = 0: plan only, nothing runs
    assert out["new_windows"] == 0 and out["n_windows"] == 0 and out["n_windows_planned"] == 2 and out["complete"] is False
