import numpy as np
import pandas as pd
import pytest

from edge_research.atoms.atoms import (QUANT_NAN, Atom, apply_atoms, build_atoms, compatible, family_map, load_atoms, popcount, round_sig,
                                       save_atoms, segment_packed)
from edge_research.config import load_config
from edge_research.features.cluster import cluster_features
from edge_research.features.table import build_feature_table
from edge_research.splits import make_splits
from edge_research.synth.generator import make_synthetic_bars

CFG = load_config()


def _synth_atoms(seed=1, n=12000):
    df, _ = make_synthetic_bars(n, seed=seed)
    f = build_feature_table(df, CFG)
    sp = make_splits(len(df), CFG)
    tr = slice(max(sp.train.start, f.first_valid_row), sp.train.stop)
    kept, cm = cluster_features(f.table.iloc[tr], f.specs, CFG["features"]["corr_cluster_threshold"])
    kinds = {s.name: s.kind for s in f.specs}
    return build_atoms(f.table, kinds, family_map(cm), kept, CFG, tr), f, sp, tr, kept, kinds, cm


def _eval(atom: Atom, col: np.ndarray) -> np.ndarray:
    """Reference implementation: direct comparison; NaN is False for every operator."""
    fin = np.isfinite(col)
    if atom.op == "<=":
        return fin & (col <= atom.threshold)
    if atom.op == ">":
        return fin & (col > atom.threshold)
    return fin & (col == atom.threshold)


def test_every_atom_mask_equals_brute_force_comparison_on_all_rows():
    a, f, *_ = _synth_atoms()
    assert len(a.atoms) > 200 and len(a.specs) < len(a.atoms)                 # complements are derived, not stored
    for atom in a.atoms:
        np.testing.assert_array_equal(a.dense(atom), _eval(atom, f.table[atom.feature].to_numpy(np.float64)), err_msg=atom.expr())


def test_complement_is_exact_and_nan_rows_belong_to_neither_side():
    a, f, *_ = _synth_atoms()
    by = {}
    for atom in a.atoms:
        if atom.kind == "quantile":
            by.setdefault((atom.feature, atom.threshold), {})[atom.op] = atom
    assert by
    for (feat, _), pair in by.items():
        lo, hi = a.dense(pair["<="]), a.dense(pair[">"])
        assert not (lo & hi).any()
        np.testing.assert_array_equal(lo | hi, np.isfinite(f.table[feat].to_numpy(np.float64)))
    assert np.isnan(f.table.to_numpy()).any()                                 # warm-up NaN rows exist, so the NaN path is exercised


def test_support_by_popcount_equals_boolean_sum_and_respects_limits():
    a, f, sp, tr, *_ = _synth_atoms()
    n_tr = tr.stop - tr.start
    for atom in a.atoms:
        d = a.dense(atom, tr)
        assert a.support(atom) == pytest.approx(d.sum() / n_tr)
        assert atom.train_support == pytest.approx(d.sum() / n_tr)
        assert CFG["atoms"]["min_support"] <= atom.train_support <= 1 - CFG["atoms"]["min_support"], atom.expr()
    m = a.packed_mask(0)
    assert popcount(m) == int(a.dense(0).sum()) and popcount(segment_packed(a.n_rows, tr)) == n_tr


def test_thresholds_are_rounded_train_quantiles_and_train_only():
    a, f, sp, tr, kept, kinds, cm = _synth_atoms()
    feat = next(x.feature for x in a.atoms if x.kind == "quantile")
    train_vals = f.table[feat].iloc[tr].to_numpy(np.float64)
    train_vals = train_vals[np.isfinite(train_vals)]
    got = sorted({x.threshold for x in a.by_feature(feat)})
    expected = sorted({round_sig(float(np.quantile(train_vals, p / 100)), 6) for p in CFG["atoms"]["quantile_pcts"]})
    assert set(got) <= set(expected) and len(got) >= 7
    # Destroy validation + holdout rows: atoms, thresholds and TRAIN masks must be identical.
    t2 = f.table.copy()
    t2.iloc[sp.train.stop:] = t2.iloc[sp.train.stop:].to_numpy() * 9.0 - 4.0
    b = build_atoms(t2, kinds, family_map(cm), kept, CFG, tr)
    assert [x.expr() for x in b.atoms] == [x.expr() for x in a.atoms]
    for i in (0, 5, len(a.atoms) - 1):
        np.testing.assert_array_equal(a.dense(i, tr), b.dense(i, tr))


def test_hand_example_ties_flags_constants_categories_and_nan():
    n = 1000
    x = np.r_[np.zeros(500), np.arange(1, 501)].astype(float)                  # train = first 800 rows: 500 zeros, then 1..300
    x[10:15] = np.nan
    flag = np.zeros(n); flag[[3, 700]] = 1                                     # 2 of 800 train rows = 0.25% -> dropped
    flag2 = np.zeros(n); flag2[::20] = 1                                       # 5% -> kept as "== 1" only
    cat = np.tile([0.0, 1.0, 2.0, 3.0], n // 4)
    t = pd.DataFrame({"x": x, "flag": flag, "flag2": flag2, "const": 7.0, "cat": cat}, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    kinds = {"x": "numeric", "flag": "event", "flag2": "event", "const": "numeric", "cat": "time"}
    a = build_atoms(t, kinds, {c: c for c in t.columns}, list(t.columns), CFG, slice(0, 800))
    xs = [at for at in a.by_feature("x")]
    thr = sorted({at.threshold for at in xs})
    assert thr[0] == 0.0 and thr.count(0.0) == 1 and len(thr) == len(set(thr))        # tied quantiles collapse to one threshold
    assert sum(d["reason"].startswith("duplicate") for d in a.dropped if d["feature"] == "x") >= 5
    assert [at.expr() for at in a.by_feature("flag2")] == ["flag2 == 1"]
    assert not a.by_feature("flag") and not a.by_feature("const")
    dropped = {(d["feature"], d["reason"]) for d in a.dropped}
    assert ("flag", "support outside limits") in dropped and ("const", "support outside limits") in dropped
    assert [at.threshold for at in a.by_feature("cat")] == [0.0, 1.0, 2.0, 3.0] and all(at.op == "==" for at in a.by_feature("cat"))
    first = xs[0]
    assert not a.dense(first)[10:15].any() and not a.dense(xs[1])[10:15].any()          # NaN rows are False on both sides


def test_compatibility_rules():
    mk = lambda i, f, op, thr, fam, kind="quantile": Atom(i, f, op, thr, fam, kind, 50, 0, op == ">", 0.5)
    a_le, a_gt = mk(0, "rsi_14", "<=", 70.0, "c1"), mk(1, "rsi_14", ">", 30.0, "c1")
    assert compatible(a_gt, a_le) and compatible(a_le, a_gt)                  # interval 30 < rsi <= 70
    assert not compatible(mk(2, "rsi_14", ">", 80.0, "c1"), a_le)             # empty interval
    assert not compatible(a_le, mk(3, "rsi_14", "<=", 60.0, "c1"))            # same side twice is redundant
    assert not compatible(a_le, a_le)
    assert not compatible(a_le, mk(4, "rsi_21", "<=", 60.0, "c1"))            # same family, different feature
    assert compatible(a_le, mk(5, "atr_ratio", ">", 1.0, "c2"))
    h1, h2 = mk(6, "hour", "==", 7.0, "hour", "category"), mk(7, "hour", "==", 8.0, "hour", "category")
    assert not compatible(h1, h2) and compatible(h1, a_le)


def test_quantized_bins_match_threshold_semantics():
    a, f, *_ = _synth_atoms()
    assert a.quantized is not None and a.quantized.dtype == np.uint8
    j = {name: i for i, name in enumerate(a.quantized_features)}
    for feat, cuts in list(a.cuts.items())[:6]:
        col, q = f.table[feat].to_numpy(np.float64), a.quantized[:, j[feat]]
        fin = np.isfinite(col)
        assert (q[~fin] == QUANT_NAN).all() and (q[fin] < QUANT_NAN).all()
        for k, c in enumerate(cuts):
            np.testing.assert_array_equal(q[fin] <= k, col[fin] <= c)


def test_save_load_roundtrip_and_frozen_apply(tmp_path):
    a, f, sp, tr, kept, kinds, cm = _synth_atoms()
    save_atoms(a, tmp_path / "atoms", {"data_hash": "abc"})
    b = load_atoms(tmp_path / "atoms")
    assert [x.expr() for x in b.atoms] == [x.expr() for x in a.atoms] and b.train == a.train and b.cuts == a.cuts
    np.testing.assert_array_equal(b.packed, a.packed)
    np.testing.assert_array_equal(b.quantized, a.quantized)
    for i in (0, 1, len(a.atoms) // 2, len(a.atoms) - 1):
        np.testing.assert_array_equal(a.dense(i), b.dense(i))
    # Frozen thresholds applied to a different table (e.g. validation rows only) must equal brute force with the SAME numbers.
    val = f.table.iloc[sp.val]
    c = apply_atoms(b, val)
    assert c.n_rows == len(val) and [x.expr() for x in c.atoms] == [x.expr() for x in a.atoms]
    for atom in c.atoms[:: max(1, len(c.atoms) // 40)]:
        np.testing.assert_array_equal(c.dense(atom), _eval(atom, val[atom.feature].to_numpy(np.float64)), err_msg=atom.expr())
    assert (tmp_path / "atoms" / "atoms.csv").exists() and (tmp_path / "atoms" / "dropped_atoms.csv").exists()


def test_quantization_is_exact_when_values_equal_the_cut():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 40, 3000).astype(float)                               # integer values: many rows sit exactly ON a threshold
    x[::97] = np.nan
    t = pd.DataFrame({"x": x}, index=pd.date_range("2024-01-01", periods=len(x), freq="1h"))
    a = build_atoms(t, {"x": "numeric"}, {"x": "x"}, ["x"], CFG, slice(0, 2400))
    cuts = a.cuts["x"]
    assert len(cuts) >= 5 and all(c == int(c) for c in cuts)
    q = a.quantized[:, 0]
    fin = np.isfinite(x)
    for k, c in enumerate(cuts):
        np.testing.assert_array_equal(q[fin] <= k, x[fin] <= c)
        assert (x[fin] == c).any()                                            # the boundary case is really exercised


def test_pipeline_stage4_end_to_end_resume_and_train_only(tmp_path):
    from edge_research.pipeline import run_stages_0_4
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(8000, seed=9)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    run = tmp_path / "run"
    out = run_stages_0_4(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")
    assert out["status"]["04_atoms"] == "ran" and out["n_atoms"] > 100
    b = load_atoms(run / "04_atoms")
    assert b.n_rows == 8000 and b.train[1] <= 8000 - 100 and (run / "04_atoms" / "summary.md").exists()
    assert b.meta["train_rows"] == b.train[1] - b.train[0]
    out2 = run_stages_0_4(csv, run, CFG, "SYNTH", tmp_path / "s_spread.csv")
    assert set(out2["status"].values()) == {"skipped"}
    cfg2 = load_config(overrides={"atoms": {"quantile_pcts": [25, 50, 75]}})                 # only atoms config changed -> only stage 4 reruns
    out3 = run_stages_0_4(csv, run, cfg2, "SYNTH", tmp_path / "s_spread.csv")
    assert out3["status"] == {"00_data_qa": "skipped", "01_features": "skipped", "02_labels": "skipped", "03_eda": "skipped", "04_atoms": "ran"}
    assert load_atoms(run / "04_atoms").meta["n_atoms"] < b.meta["n_atoms"]
