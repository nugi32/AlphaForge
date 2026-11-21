"""M11: Engine A depth 3-4 (Apriori expansion, ``discovery/deep.py``), Engine B (shallow trees, ``discovery/trees.py``), the unified
stage 6 rule input (``normalise_rules``) and the N-atom generalisation of F5 / the gate machinery (``validation/gates.py``).

The world is deliberately small (6000 bars, strong planted edge) and the engine sizes are reduced so the whole module runs in
about a minute on a slow machine; the default-config behaviour is exercised by the end-to-end scripts on the user's machine.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from edge_research.atoms.atoms import Atom, build_atoms, compatible, family_map
from edge_research.config import load_config
from edge_research.discovery.deep import (atoms_of, deep_scope_note, expand_survivors, normalise_rules, rule_expr,
                                          rule_key, rule_mask, rule_compatible)
from edge_research.discovery.episodes import lift_stats
from edge_research.discovery.pairs import DIRECTIONS, coarse_screen, family_hold
from edge_research.discovery.trees import _grow, canonical_atoms, induce_rules, split_table, week_blocks
from edge_research.features.cluster import cluster_features
from edge_research.features.table import build_feature_table
from edge_research.labels.builder import build_labels, net_R
from edge_research.splits import make_splits
from edge_research.synth.generator import make_synthetic_bars
from edge_research.validation.gates import GATE_COLS, apply_gates, lift_over_parents
from test_discovery import CFG, STRONG

DEEP = {"apriori": {"enabled": True, "beam_triple": 200, "beam_quad": 80, "max_atoms": 4, "tile": 16, "max_survivors": 1500,
                    "include_in_null": False},
        "trees": {"enabled": True, "families": ["T6"], "n_trees": 40, "max_depth": 3, "min_leaf_rows": 80, "feature_frac": 0.5,
                  "min_hits": 2, "max_rules": 100, "include_in_null": False, "seed": 7}}


@pytest.fixture(scope="module")
def m11():
    """One small planted world shared by every test: atoms, labels, feature table, depth-2 survivors and both deep engines."""
    df, _ = make_synthetic_bars(6000, seed=4, plant_edge=STRONG)
    f = build_feature_table(df, CFG)
    sp = make_splits(len(df), CFG)
    tr = slice(max(sp.train.start, f.first_valid_row), sp.train.stop)
    kept, cm = cluster_features(f.table.iloc[tr], f.specs, CFG["features"]["corr_cluster_threshold"])
    kinds = {s.name: s.kind for s in f.specs}
    aset = build_atoms(f.table, kinds, family_map(cm), kept, CFG, tr)
    lab = build_labels(df, CFG).table
    cfg = copy.deepcopy(CFG)
    cfg["deep"]["apriori"].update(copy.deepcopy(DEEP["apriori"]))          # leaf-level merge: the other keys of the section stay at their defaults
    cfg["deep"]["trees"].update(copy.deepcopy(DEEP["trees"]))
    surv, _ = coarse_screen(aset, lab, cfg, tr)
    ap, ainfo = expand_survivors(aset, surv, lab, cfg, tr)
    tree, tinfo = induce_rules(aset, f.table, lab, cfg, tr)
    return {"df": df, "feats": f.table, "aset": aset, "lab": lab, "tr": tr, "cfg": cfg, "surv": surv, "ap": ap,
            "ainfo": ainfo, "tree": tree, "tinfo": tinfo}


# ---------------------------------------------------------------- N-atom helpers
def _dense(aset):
    return lambda i: aset.dense(int(i), slice(None))


def test_rule_key_mask_expr_and_normalisation(m11):
    aset, w = m11["aset"], m11
    assert rule_key([3, 1, 2]) == "1,2,3" and rule_key([7]) == "7"
    ids = [int(x) for x in w["ap"].iloc[0]["atoms"]] if len(w["ap"]) else [int(w["surv"].iloc[0]["atom_a"]), int(w["surv"].iloc[0]["atom_b"])]
    dense = _dense(aset)
    m = rule_mask(dense, ids)
    expect = dense(ids[0])
    for i in ids[1:]:
        expect = expect & dense(i)
    assert (m == expect).all() and int(m.sum()) <= min(int(dense(i).sum()) for i in ids)
    assert rule_expr(aset, ids) == " AND ".join(aset.atoms[i].expr() for i in ids)
    # atoms of DIFFERENT features for the legal rules (atoms of one feature family may never share a rule)
    picked, seen = [], set()
    for a in aset.atoms:
        if a.feature not in seen:
            seen.add(a.feature)
            picked.append(int(a.id))
        if len(picked) == 3:
            break
    p1, p2, p3 = picked
    base = {"score": 1.0, "source": "apriori"}
    rows = [{"atoms": [p2, p1], "direction": "long", "score": 2.0, **base},
            {"atoms": [p1, p2], "direction": "long", "score": 0.5, **base},        # the same rule, another score
            {"atoms": [p1, p2], "direction": "short", "score": 1.0, **base},       # other direction: never merged
            {"atoms": [p1, p1, p3], "direction": "long", "score": 1.0, **base},    # a repeated atom is illegal
            {"atoms": float("nan"), "atom_a": p2, "atom_b": p3, "direction": "long", "score": 3.0, **base}]
    out = normalise_rules(pd.DataFrame(rows), aset, "apriori")
    assert len(out) == 3 and set(out["n_atoms"]) == {2}
    got = {(rule_key(ids), str(d)) for ids, d in zip(out["atoms"], out["direction"])}
    assert got == {(rule_key([p1, p2]), "long"), (rule_key([p1, p2]), "short"), (rule_key([p2, p3]), "long")}
    # two atoms of the SAME feature are not compatible together, and normalise drops such a rule
    same = [a for a in aset.atoms if a.kind == "quantile"]
    f2 = next(a.feature for a in same if sum(1 for b in same if b.feature == a.feature) >= 2)
    pair = [a for a in same if a.feature == f2][:2]
    assert not compatible(pair[0], pair[1])
    bad = normalise_rules(pd.DataFrame([{"atoms": sorted([pair[0].id, pair[1].id]), "direction": "long", "score": 1.0, **base}]), aset, "apriori")
    assert len(bad) == 0


def test_normalise_rules_carries_tree_hits_and_family_and_survives_dedup(m11):
    """Engine B credibility columns (spec 10.7): ``hits`` and ``family`` survive normalisation and dedup, and a depth-2 frame
    without them produces no such columns, so build_candidates can write ``tree_hits`` / ``tree_family`` for every source."""
    aset = m11["aset"]
    same = [a for a in aset.atoms if a.kind == "quantile"]
    f2 = next(a.feature for a in same if sum(1 for b in same if b.feature == a.feature) >= 2)
    p1, p2 = (a.id for a in [a for a in same if a.feature == f2][:2])
    other = next(a.id for a in same if a.feature != f2)
    tree = pd.DataFrame([
        {"atoms": [int(p1), int(other)], "direction": "long", "score": 4.0, "hits": 7, "family": "T6", "source": "tree"},
        {"atoms": [int(other), int(p1)], "direction": "long", "score": 1.0, "hits": 2, "family": "T6", "source": "tree"},   # same rule: dedup keeps the FIRST row
        {"atoms": [int(p2), int(other)], "direction": "short", "score": 2.0, "hits": 3, "family": "T24", "source": "tree"}])
    out = normalise_rules(tree, aset, "tree")
    assert len(out) == 2
    assert out["hits"].tolist() == [7, 3] and out["family"].tolist() == ["T6", "T24"]
    assert set(out["source"]) == {"tree"}
    pairs = pd.DataFrame([{"atom_a": int(p1), "atom_b": int(other), "direction": "long", "score": 9.0, "source": "pairs"}])
    out2 = normalise_rules(pairs, aset, "pairs")
    assert len(out2) == 1 and "hits" not in out2.columns and "family" not in out2.columns    # no tree columns invented for depth-2
    distinct = next(a.id for a in same if a.feature != f2 and a.id != other)                  # a rule the tree frame does NOT contain:
    mixed = pd.DataFrame([{"atoms": [int(p2), int(distinct)], "direction": "long", "score": 9.0, "source": "pairs"}])   # otherwise the
    mixed = pd.concat([tree, mixed], ignore_index=True)                                      # keep-first dedup would drop it as a duplicate
    out3 = normalise_rules(mixed, aset, "mixed")
    row = out3[out3["source"] == "pairs"].iloc[0]
    assert pd.isna(row["hits"]) and pd.isna(row["family"])


def test_apriori_survivors_are_legal_unique_and_deterministic(m11):
    w = m11
    ap, ainfo = w["ap"], w["ainfo"]
    assert ainfo["enabled"] and ainfo["n_survivors"] == len(ap) > 0
    assert {"d3_long", "d4_long"} <= set(ainfo["by_depth"])                     # the planted edge is long-only: no short deep survivors here
    assert set(ap["n_atoms"]) <= {3, 4}
    keys = [(rule_key(ids), str(d)) for ids, d in zip(ap["atoms"], ap["direction"])]
    assert len(set(keys)) == len(keys)                                    # the same rule set reached from different parents is merged
    for r in ap.itertuples(index=False):
        ids = [int(x) for x in r.atoms]
        assert len(set(ids)) == len(ids) and rule_compatible([w["aset"].atoms[i] for i in ids])
        assert str(r.direction) in DIRECTIONS
    ap2, ainfo2 = expand_survivors(w["aset"], w["surv"], w["lab"], w["cfg"], w["tr"])
    pd.testing.assert_frame_equal(ap.reset_index(drop=True), ap2.reset_index(drop=True))
    off = copy.deepcopy(w["cfg"])
    off["deep"]["apriori"]["enabled"] = False
    empty, einfo = expand_survivors(w["aset"], w["surv"], w["lab"], off, w["tr"])
    assert empty.empty and einfo["n_survivors"] == 0 and not einfo["enabled"]


def test_tree_engine_rules_are_legal_deterministic_and_scored(m11):
    w = m11
    tree, tinfo, cfg = w["tree"], w["tinfo"], w["cfg"]
    assert tinfo["enabled"] and tinfo["n_trees"] == 40 and tinfo["seconds"] >= 0
    assert len(tree) >= 1 and list(tree.columns) == ["atoms", "direction", "score", "hits", "family", "n_atoms", "source"]
    assert set(tree["source"]) == {"tree"} and (tree["hits"] >= cfg["deep"]["trees"]["min_hits"]).all() and (tree["score"] > 0).all()
    assert set(tree["family"]) <= {"T6"}
    for r in tree.itertuples(index=False):
        ids = [int(x) for x in r.atoms]
        assert len(set(ids)) == len(ids) and rule_compatible([w["aset"].atoms[i] for i in ids])
    assert tinfo["by_direction"]["long"] == int((tree["direction"] == "long").sum()) and tinfo["n_rules"] == len(tree)
    tree2, tinfo2 = induce_rules(w["aset"], w["feats"], w["lab"], w["cfg"], w["tr"])          # deterministic function of config + data
    pd.testing.assert_frame_equal(tree.reset_index(drop=True), tree2.reset_index(drop=True))
    off = copy.deepcopy(w["cfg"])
    off["deep"]["trees"]["enabled"] = False
    empty, einfo = induce_rules(w["aset"], w["feats"], w["lab"], off, w["tr"])
    assert empty.empty and not einfo["enabled"] and einfo["n_rules"] == 0


def test_week_blocks_are_dense_and_split_table_is_the_ladder(m11):
    idx = pd.date_range("2024-01-01", periods=10, freq="h")                      # Mon 00-09 h: one calendar week
    idx2 = pd.DatetimeIndex(list(idx) + list(pd.date_range("2024-01-08", periods=5, freq="h")))
    b = week_blocks(idx2)
    assert b[:10].tolist() == [0] * 10 and b[10:].tolist() == [1] * 5            # dense ids, one per calendar week
    assert week_blocks(pd.RangeIndex(250)).tolist()[:5] == [0] * 5 and week_blocks(pd.RangeIndex(250))[150] == 1
    aset, ladder = m11["aset"], split_table(m11["aset"])
    assert ladder and len(ladder) < len({a.feature for a in aset.atoms})         # only quantile features have a ladder
    for f, rows in ladder.items():
        thr = [r[0] for r in rows]
        assert thr == sorted(thr) and len(set(thr)) == len(thr)
        for t, le, gt in rows:
            a_le, a_gt = aset.atoms[le], aset.atoms[gt]
            assert a_le.feature == f and a_le.op == "<=" and float(a_le.threshold) == t
            assert a_gt.feature == f and a_gt.op == ">" and float(a_gt.threshold) == t


def test_canonical_atoms_tightens_bounds_and_drops_illegal_paths(m11):
    aset, ladder = m11["aset"], split_table(m11["aset"])
    f = next(f for f, rows in ladder.items() if len(rows) >= 3)
    (t0, le0, gt0), (t1, le1, gt1), (t2, le2, gt2) = ladder[f][:3]
    lookup = {(a.feature, a.op, float(a.threshold)): int(a.id) for a in aset.atoms}
    # two "<=" on one feature tighten to the smaller threshold; one feature alone is not a rule
    assert canonical_atoms([(f, "<=", t2), (f, "<=", t1)], aset, lookup) is None
    # an interval (t0, t1] is the pair of atoms at its bounds
    got = canonical_atoms([(f, "<=", t1), (f, ">", t0)], aset, lookup)
    assert got == tuple(sorted([le1, gt0]))
    # the same atoms in any order and repetition give the same canonical rule
    assert canonical_atoms([(f, ">", t0), (f, "<=", t1), (f, "<=", t1)], aset, lookup) == got
    # an empty interval (nothing is <= t0 AND > t0) is dropped
    assert canonical_atoms([(f, "<=", t0), (f, ">", t0)], aset, lookup) is None
    # a threshold that is not on the ladder has no atom: not representable, dropped
    assert canonical_atoms([(f, "<=", (t0 + t1) / 2), (f, ">", t0)], aset, lookup) is None


def test_grow_splits_on_the_best_gain_and_respects_min_leaf():
    yv = np.array([5.0, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0])
    sq = yv * yv
    M = np.stack([np.array([1.0, 1, 1, 1, 0, 0, 0, 0]), np.array([1.0, 0, 1, 0, 1, 0, 1, 0])])       # split 0 is perfect, split 1 is noise
    from edge_research.discovery.trees import _grow
    leaves = []
    _grow(yv, sq, M, np.arange(8), 0, 1, 2, (), leaves)
    paths = {p: [int(x) for x in idx] for p, idx in leaves}
    assert set(paths) == {((0, 0),), ((0, 1),)}
    assert paths[((0, 0),)] == [0, 1, 2, 3] and paths[((0, 1),)] == [4, 5, 6, 7]
    leaves2 = []
    _grow(yv, sq, M, np.arange(8), 0, 3, 5, (), leaves2)                        # n < 2 * min_leaf: a single leaf, no split
    assert len(leaves2) == 1 and leaves2[0][0] == () and len(leaves2[0][1]) == 8


def test_lift_over_parents_covers_every_parent_in_stage6_order(m11):
    w = m11
    ap = w["ap"]
    r3 = next(r for r in ap.itertuples(index=False) if len(r.atoms) == 3) if len(ap) else None
    assert r3 is not None
    direction, ids = str(r3.direction), [int(x) for x in r3.atoms]
    y = np.array(net_R(w["lab"], direction, "T6").to_numpy(np.float64), copy=True)      # FULL length: lift_stats slices every input by rows
    hold = family_hold("T6", w["cfg"])
    out = lift_over_parents(w["aset"], ids, y, hold, w["tr"], int(w["cfg"]["candidates"]["lift_block_mult"]))
    assert out["n_parents"] == 3 and len(out["lifts"]) == 3
    child = rule_mask(_dense(w["aset"]), ids)
    blk = max(1, int(w["cfg"]["candidates"]["lift_block_mult"]) * int(hold))
    # stage 6 order: drop the LAST atom first (lift_a), then the FIRST (lift_b), then the middle ones
    exp_order = [ids[2], ids[0], ids[1]]
    for n, dropped in enumerate(exp_order):
        parent = rule_mask(_dense(w["aset"]), [i for i in ids if i != dropped])
        r = lift_stats(parent, child, y, hold, rows=w["tr"], block=blk)
        assert out["lifts"][n] == pytest.approx(float(r["lift"])) and out["ts"][n] == pytest.approx(float(r["t"]))
    assert out["lift_a_t"] == out["ts"][0] and out["lift_b_t"] == out["ts"][1]
    assert out["lift_min_t"] == pytest.approx(min(out["ts"])) and out["lift_min"] == pytest.approx(min(out["lifts"]))
    # a 2-atom rule keeps the M8 meaning: lift_a = over the parent WITHOUT the last atom, lift_b = without the first
    ids2 = ids[:2]
    out2 = lift_over_parents(w["aset"], ids2, y, hold, w["tr"], int(w["cfg"]["candidates"]["lift_block_mult"]))
    assert out2["n_parents"] == 2 and out2["lift_a_t"] == out2["ts"][0] and out2["lift_b_t"] == out2["ts"][1]
    assert out2["lift_min_t"] == pytest.approx(min(out2["ts"]))


def _passing_row(n_par=4):
    row = {"n_parents": n_par, "val_mean_R": 1.0, "val_t": 5.0, "t_crit_bonferroni": 3.0, "decay_ratio": 1.0, "val_trades": 50,
           "stress_mean_R": 0.5, "stress_net_profit": 10.0, "val_profit_factor": 2.0, "val_equity_dd_pct": 5.0,
           "g2_time_stability": True, "g3_exit_plateau": True}
    if n_par > 2:
        row.update(lift_min_t_train=3.0, lift_min_t_val=3.0)
    else:
        row.update(lift_a_t_train=3.0, lift_b_t_train=3.0, lift_a_t_val=3.0, lift_b_t_val=3.0)
    return row


V = {"lift_t_min": 2.0, "decay_min_ratio": 0.5, "min_trades_val": 20, "min_profit_factor": 1.2, "max_equity_drawdown_pct": 20.0, "require_g1": False}


def test_apply_gates_checks_every_parent_for_deep_rules():
    # a 4-atom rule passes only when the MINIMUM parent t clears the threshold on train AND validation
    ok = apply_gates(_passing_row(4), V, 2.5, True)
    assert ok["g4_lift"] and ok["accepted"] and ok["eligible_for_holdout"] and ok["fail_reasons"] == ""
    weak = apply_gates(_passing_row(4) | {"lift_min_t_val": 1.5}, V, 2.5, True)
    assert not weak["g4_lift"] and not weak["accepted"] and "g4_lift" in weak["fail_reasons"]
    weak_train = apply_gates(_passing_row(4) | {"lift_min_t_train": 2.4}, V, 2.5, True)
    assert not weak_train["g4_lift"]
    # a NaN never passes: one parent t is undefined, so the whole rule fails G4
    nan_row = apply_gates(_passing_row(4) | {"lift_min_t_train": float("nan")}, V, 2.5, True)
    assert not nan_row["g4_lift"]
    # a 2-atom rule keeps the a / b gate fields
    ok2 = apply_gates(_passing_row(2), V, 2.5, True)
    assert ok2["g4_lift"] and ok2["accepted"]
    bad2 = apply_gates(_passing_row(2) | {"lift_b_t_train": 1.0}, V, 2.5, True)
    assert not bad2["g4_lift"]


def test_deep_scope_note_names_what_the_null_repeats():
    base = {"deep": {"apriori": {"enabled": True, "include_in_null": False}, "trees": {"enabled": True, "include_in_null": False}}}
    assert "Apriori expansion" in deep_scope_note(base) and "tree engine" in deep_scope_note(base) and "depth-2" in deep_scope_note(base)
    both = {"deep": {"apriori": {"enabled": True, "include_in_null": True}, "trees": {"enabled": True, "include_in_null": True}}}
    assert "FULL deep stage" in deep_scope_note(both)
    only_a = {"deep": {"apriori": {"enabled": True, "include_in_null": True}, "trees": {"enabled": True, "include_in_null": False}}}
    assert "Apriori expansion inside every replicate" in deep_scope_note(only_a) and "NOT part" in deep_scope_note(only_a)
    only_t = {"deep": {"apriori": {"enabled": False, "include_in_null": True}, "trees": {"enabled": True, "include_in_null": True}}}
    assert "tree engine inside every replicate" in deep_scope_note(only_t)
    off = {"deep": {"apriori": {"enabled": False, "include_in_null": False}, "trees": {"enabled": False, "include_in_null": False}}}
    assert "depth-2 search only" in deep_scope_note(off)


def test_stage5b_pipeline_writes_deep_rules_and_stage6_consumes_both_sources(tmp_path):
    """End-to-end: stages 0-5b on the small world with reduced engine sizes, then stage 6 on pairs + deep rules."""
    import json
    from edge_research.pipeline import run_stage_6, run_stages_0_5b
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(6000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    # reduced engines and a lower episode bar: at 6000 bars a triple cannot reach the default 200 episodes, so the
    # exact stage would legitimately kill every deep rule; the point here is that stage 6 WALKS the deep rules.
    cfg = load_config(overrides={"deep": copy.deepcopy(DEEP), "discovery": {"max_survivors": 800}, "candidates": {"min_episodes_train": 80}})
    run = tmp_path / "run"
    out = run_stages_0_5b(csv, run, cfg, "SYNTH", tmp_path / "s_spread.csv")
    assert out["status"]["05b_deep"] == "ran" and out["n_deep"] > 0 and out["n_deep"] == out["n_apriori"] + out["n_trees"]
    d5b = run / "05b_deep"
    for f in ("survivors_deep.parquet", "deep_info.json", "summary.md"):
        assert (d5b / f).exists(), f
    deep = pd.read_parquet(d5b / "survivors_deep.parquet")
    assert set(deep["source"]) == {"apriori", "tree"} and (deep["n_atoms"] >= 3).all()
    info = json.loads((d5b / "deep_info.json").read_text(encoding="utf-8"))
    assert info["n_deep"] == len(deep) and info["null_scope"] == deep_scope_note(load_config(overrides={"deep": copy.deepcopy(DEEP)}))
    assert "Engine B (trees)" in (d5b / "summary.md").read_text(encoding="utf-8")
    s6 = run_stage_6(run, cfg)
    assert s6["status"] == "ran" and s6["n_candidates"] > 0
    assert set(int(k) for k in s6["by_depth"]) >= {2, 3, 4}                      # pairs, Apriori triples/quads and tree leaves all walked stage 6
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["06_candidates"]["upstream"].count("|") == 1        # stage 6 fingerprints BOTH upstream runs
    s6b = run_stage_6(run, cfg)
    assert s6b["status"] == "skipped" and s6b["n_candidates"] == s6["n_candidates"]


def test_trees_requested_without_a_feature_table_are_reported_not_silent(m11):
    """The real stage-9 path always passes the feature table; a caller that cannot supply one must see the shrink in the scope note (spec 13.3 forbids
    a null procedure narrower than the real search), never a silent skip."""
    from edge_research.nulltest.null import deep_rules_for_search, search_statistic
    aset, lab, cfg, tr, surv = m11["aset"], m11["lab"], m11["cfg"], m11["tr"], m11["surv"]
    cfg = copy.deepcopy(cfg)
    cfg["deep"]["trees"]["include_in_null"] = True
    deep, scope = deep_rules_for_search(aset, lab, cfg, tr, surv, None)
    assert len(deep) == 0 and scope["used"] == []
    assert "SKIPPED" in scope["note"] and "feature table" in scope["note"]          # the base note alone would imply the full deep stage
    real = search_statistic(aset, lab, cfg, tr, None)
    assert real["deep_used"] == "none" and "SKIPPED" in real["deep_note"]
    real_ft = search_statistic(aset, lab, cfg, tr, m11["feats"])
    assert "trees" in real_ft["deep_used"] and real_ft["n_deep"] > 0 and "SKIPPED" not in real_ft["deep_note"]


def test_trees_run_inside_the_null_replicates_when_included(m11, tmp_path):
    """Spec 13.3: the null procedure is IDENTICAL to the real one - when deep.trees.include_in_null is true the replicates run Engine B too
    (this is exactly the bug a verification pass found: replicates silently dropped the feature table and ran depth-2 only)."""
    import json
    from edge_research.nulltest.null import run_null, search_statistic
    aset, lab, feats, cfg, tr = m11["aset"], m11["lab"], m11["feats"], m11["cfg"], m11["tr"]
    cfg = copy.deepcopy(cfg)
    cfg["deep"]["trees"]["include_in_null"] = True
    cfg["nulltest"].update({"replicates": 1, "min_replicates": 1, "quantile": 0.8, "variants": ["random"]})
    real = search_statistic(aset, lab, cfg, tr, feats)
    assert "trees" in real["deep_used"] and real["n_deep"] > 0
    res = run_null(aset, lab, cfg, tr, tmp_path / "n", "fp", feature_table=feats)
    assert res["complete"] and res["real"]["deep_used"] == real["deep_used"] and res["real"]["n_deep"] == real["n_deep"]
    j = json.loads((tmp_path / "n" / "replicates" / "random_0000.json").read_text(encoding="utf-8"))
    assert j["deep_used"] == "trees" and "SKIPPED" not in j["deep_note"]            # Engine B RAN in the replicate even if the rotated labels yield 0 rules
    assert isinstance(j["n_deep"], int) and j["n_deep"] >= 0
