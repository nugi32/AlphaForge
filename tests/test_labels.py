import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.labels.builder import build_labels, family_list, intrabar_name, net_R
from edge_research.labels.costs import CostSpec, cumulative_rollovers, spread_array, swap_nights
from edge_research.mt5_backtest.engine import parse_family
from edge_research.synth.generator import make_synthetic_bars


def cfg_small(**over):
    base = {
        "costs": {"point": 0.01, "contract_size": 100, "fixed_spread_points": 10, "commission_per_lot_per_side": 1.0,
                  "swap_long_per_lot": -5, "swap_short_per_lot": 2, "swap_rollover_hour": 0, "swap_triple_weekday": 2},
        "labels": {"time_stops": [2], "barrier_grid": {"tp": [1], "sl": [1], "horizon": [4]}},
    }
    for k, v in over.items():
        tgt = base.setdefault(k, {})
        for kk, vv in v.items():
            if isinstance(vv, dict) and isinstance(tgt.get(kk), dict):     # nested sections merge (a bare update would drop the
                tgt[kk].update(vv)                                        # other barrier_grid keys and change the hand worlds)
            else:
                tgt[kk] = vv
    return load_config(overrides=base)


def make_df(opens, closes=None, start="2024-01-01 08:00", atr=1.0):
    idx = pd.date_range(start, periods=len(opens), freq="1h")
    closes = opens if closes is None else closes
    o = np.asarray(opens, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5,
                         "close": c, "atr_14": atr}, index=idx)


def make_ohl_df(opens, highs, lows, closes, start="2024-01-01 08:00", atr=1.0):
    idx = pd.date_range(start, periods=len(opens), freq="1h")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "atr_14": atr}, index=idx).astype(np.float64)


def test_time_stop_hand_example_with_per_bar_spread():
    df = make_df([10, 11, 12, 13, 14, 15, 16, 17])
    spread = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=df.index)
    t = build_labels(df, cfg_small(), spread).table
    # i=0 -> e=1 (open 11), x=3 (open 13). Commission = 2*1/100 = 0.02.
    assert t["long_T2_gross"].iloc[0] == pytest.approx(2.0)
    assert t["long_T2_cost"].iloc[0] == pytest.approx(0.20 + 0.02)          # long pays spread of ENTRY bar (bar 1 = 20 pts)
    assert t["short_T2_gross"].iloc[0] == pytest.approx(-2.0)
    assert t["short_T2_cost"].iloc[0] == pytest.approx(0.40 + 0.02)         # short pays spread of EXIT bar (bar 3 = 40 pts)
    assert net_R(t, "long", "T2").iloc[0] == pytest.approx(1.78)
    assert t["long_T2_hold"].iloc[0] == 2
    assert t["long_T2_gross"].iloc[5:].isna().all()                         # window leaves the data


def test_cost_symmetry_with_constant_spread():
    df = make_df(np.linspace(10, 20, 30))
    t = build_labels(df, cfg_small(costs={"swap_long_per_lot": 0, "swap_short_per_lot": 0}), None).table   # swap differs by side by design
    a, b = t["long_T2_cost"].dropna(), t["short_T2_cost"].dropna()
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-5)


def test_slippage_charged_on_both_fills():
    df = make_df(np.arange(10, 20, dtype=float))
    base = build_labels(df, cfg_small(), None).table
    slip = build_labels(df, cfg_small(costs={"slippage_points": 5}), None).table
    np.testing.assert_allclose((slip["long_T2_cost"] - base["long_T2_cost"]).dropna(), 0.10, rtol=1e-4)


def test_swap_charged_across_rollover_and_triple_wednesday():
    # Tuesday 22:00 start: bars 22,23,00,01,... i=0 -> e=1 (23:00), x=3 (01:00 Wed). Rollover 00:00 booked on Tuesday (x1).
    df = make_df(np.arange(10, 20, dtype=float), start="2024-01-02 22:00")
    t = build_labels(df, cfg_small(), None).table
    assert t["long_T2_cost"].iloc[0] == pytest.approx(0.10 + 0.02 + 0.05)     # swap_long -5/lot -> 0.05 cost
    assert t["short_T2_cost"].iloc[0] == pytest.approx(0.10 + 0.02 - 0.02)    # swap_short +2/lot -> 0.02 credit
    # Wednesday 22:00 start: the rollover is booked on Wednesday -> x3
    df = make_df(np.arange(10, 20, dtype=float), start="2024-01-03 22:00")
    t = build_labels(df, cfg_small(), None).table
    assert t["long_T2_cost"].iloc[0] == pytest.approx(0.10 + 0.02 + 3 * 0.05)


@pytest.mark.parametrize("hour,triple", [(0, 2), (17, 2), (22, 4), (3, 0)])
def test_vectorised_swap_equals_reference_loop(hour, triple):
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="1h")           # includes weekends on purpose
    cum = cumulative_rollovers(idx, hour, triple)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = int(rng.integers(0, len(idx) - 1))
        b = int(rng.integers(a, min(len(idx), a + 24 * 12)))
        assert cum[b] - cum[a] == swap_nights(idx[a], idx[b], hour, triple)


def test_barrier_family_hand_examples():
    opens = [10, 10, 10, 11, 12, 12, 12, 12, 12, 12]
    closes = [10, 10.2, 11.2, 11.5, 12, 12, 12, 12, 12, 12]
    df = make_df(opens, closes)
    t = build_labels(df, cfg_small(), None).table
    # long, i=0: entry = 10 + 0.1 ask; TP level 11.1; close[2] = 11.2 hits -> exit at open of bar 3
    assert t["long_B1_1_4_hold"].iloc[0] == 2
    assert t["long_B1_1_4_gross"].iloc[0] == pytest.approx(1.0)
    assert t["long_B1_1_4_cost"].iloc[0] == pytest.approx(0.10 + 0.02)
    # short, i=0: entry = 10 (bid); SL level 11; close[2] = 11.2 hits SL -> exit at open of bar 3
    assert t["short_B1_1_4_hold"].iloc[0] == 2
    assert t["short_B1_1_4_gross"].iloc[0] == pytest.approx(-1.0)
    # i=3 long: no level touched -> time exit at open of e + H = 4 + 4 = 8
    assert t["long_B1_1_4_hold"].iloc[3] == 4
    assert t["long_B1_1_4_gross"].iloc[3] == pytest.approx(0.0)
    assert np.isnan(t["long_B1_1_4_gross"].iloc[5])                          # i + 1 + 4 > n - 1


def test_barrier_never_exits_later_than_horizon_and_never_before_one_bar():
    df, _ = make_synthetic_bars(3000, seed=4)
    cfg = load_config()
    t = build_labels(df, cfg, None).table
    for col in [c for c in t.columns if c.startswith("long_B") and c.endswith("_hold")]:
        h = int(col.split("_")[-2])
        v = t[col].dropna()
        assert v.min() >= 1 and v.max() <= h


def test_gap_rule_and_invalid_atr():
    df, _ = make_synthetic_bars(1500, seed=2)
    on = build_labels(df, load_config(), None).table
    off = build_labels(df, load_config(overrides={"data": {"max_gap_hours": None}}), None).table
    assert on["long_T6_gross"].notna().sum() < off["long_T6_gross"].notna().sum()
    valid = on["long_T6_gross"].notna().to_numpy()
    pos = np.where(valid)[0]
    span_h = (df.index[pos + 1 + 6] - df.index[pos]).total_seconds() / 3600.0
    assert span_h.max() <= 7 + 1e-9                                           # no weekend inside any valid window
    bad = df.copy()
    bad.iloc[100, bad.columns.get_loc("atr_14")] = 0.0
    assert np.isnan(build_labels(bad, load_config(), None).table["long_T3_gross"].iloc[100])


def test_cost_stress_hook_and_spread_fallback_rule():
    df = make_df(np.arange(10, 20, dtype=float))
    t = build_labels(df, cfg_small(), None).table
    n1, n15 = net_R(t, "long", "T2"), net_R(t, "long", "T2", cost_mult=1.5)
    assert (n1 - n15).dropna().iloc[0] == pytest.approx(0.5 * t["long_T2_cost"].iloc[0])
    cost = CostSpec(point=0.01, fixed_spread_points=10)
    sp = pd.Series([0, 20, -5, 30], index=pd.date_range("2024-01-01", periods=4, freq="1h"))
    arr, fallback = spread_array(sp.index, sp, cost)
    np.testing.assert_allclose(arr, [0.10, 0.20, 0.20, 0.30])                 # <=0 -> last known; leading unknown -> fixed
    assert fallback == 1                                                      # only the leading unknown bar used the fixed spread


# ---------------------------------------------------------------- intrabar exits (spec 15.4 S{...}) and the catastrophic layer
ZERO = {"fixed_spread_points": 0, "commission_per_lot_per_side": 0}


def test_intrabar_family_hand_examples():
    """The same hand worlds as the engine intrabar tests (tests/test_engine_m9.py): entry bar e=1 at open 100, ATR 1, zero costs."""
    cfg = cfg_small(costs=ZERO, labels={"barrier_grid": {"tp": [2]}})         # grid tp 2, sl 1, horizon 4 -> I2_1_4
    o = [100, 100, 100.4, 100.8, 101, 101, 101, 101, 101]
    h = [100.1, 101.5, 102.4, 101, 101, 101, 101, 101, 101]
    l = [99.9, 99.5, 100.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    t = build_labels(make_ohl_df(o, h, l, o), cfg, None).table
    # long i=0: TP 102 touched by high[2]=102.4 -> exit INSIDE bar 2 at the level; short: SL 101 touched by ask high of bar 1 already
    assert t["long_I2_1_4_hold"].iloc[0] == 1 and t["long_I2_1_4_gross"].iloc[0] == pytest.approx(2.0)
    assert t["long_I2_1_4_cost"].iloc[0] == pytest.approx(0.0)
    assert t["short_I2_1_4_hold"].iloc[0] == 0 and t["short_I2_1_4_gross"].iloc[0] == pytest.approx(-1.0)
    # a gap through the SL after the entry bar fills at the OPEN of that bar
    o2 = [100, 100, 98.0, 100, 100, 100, 100, 100, 100]
    t2 = build_labels(make_ohl_df(o2, [100.5] * 9, [99.5, 99.5, 97.0] + [99.5] * 6, o2), cfg, None).table
    assert t2["long_I2_1_4_hold"].iloc[0] == 1 and t2["long_I2_1_4_gross"].iloc[0] == pytest.approx(-2.0)
    # both levels touched in one bar: the SL wins (pessimistic)
    o3 = [100] * 9
    t3 = build_labels(make_ohl_df(o3, [100.1, 100.5, 103.0] + [100.5] * 6, [99.9, 99.6, 98.0] + [99.5] * 6, o3), cfg, None).table
    assert t3["long_I2_1_4_hold"].iloc[0] == 1 and t3["long_I2_1_4_gross"].iloc[0] == pytest.approx(-1.0)
    # no touch anywhere: time exit at the open of the horizon bar
    t4 = build_labels(make_ohl_df(o3, [100.5] * 9, [99.5] * 9, o3), cfg, None).table
    assert t4["long_I2_1_4_hold"].iloc[0] == 4 and t4["long_I2_1_4_gross"].iloc[0] == pytest.approx(0.0)


def test_catastrophic_layer_cuts_before_the_base_exit_and_never_overrides_the_same_bar():
    o = [100] * 9
    cfg = cfg_small(costs=ZERO, labels={"cat_stop_atr": 3.0})
    h = [100.5] * 9
    l = [99.5, 99.5, 96.5] + [99.5] * 6                                       # bar 2 pierces the cat level 97 (3 ATR under the entry)
    t = build_labels(make_ohl_df(o, h, l, o), cfg, None).table
    assert t["long_T2_hold"].iloc[0] == 1 and t["long_T2_gross"].iloc[0] == pytest.approx(-3.0)     # strictly before the T2 exit at bar 3
    # B1_1_4 decides on the CLOSE (close[2] = 100, no exit) -> base exit at the open of bar 5; the intrabar cat touch inside bar 2 cuts it
    assert t["long_B1_1_4_hold"].iloc[0] == 1 and t["long_B1_1_4_gross"].iloc[0] == pytest.approx(-3.0)
    # the base intrabar SL (1 ATR, tighter) fires in the same bar as the cat pierce: the fill stays at the SL level, not at the cat level
    assert t["long_I1_1_4_hold"].iloc[0] == 1 and t["long_I1_1_4_gross"].iloc[0] == pytest.approx(-1.0)
    # the exit open pre-empts the bar it happens in: a short whose T2 exit is the open of bar 3 is NOT cut by a cat touch inside bar 3
    h2 = [100.5, 100.5, 100.5, 103.5] + [100.5] * 5
    t2 = build_labels(make_ohl_df(o, h2, [99.5] * 9, o), cfg, None).table
    assert t2["short_T2_hold"].iloc[0] == 2 and t2["short_T2_gross"].iloc[0] == pytest.approx(0.0)
    # ... but a cat touch inside bar 2 (strictly before) cuts it
    h3 = [100.5, 100.5, 103.5] + [100.5] * 6
    t3 = build_labels(make_ohl_df(o, h3, [99.5] * 9, o), cfg, None).table
    assert t3["short_T2_hold"].iloc[0] == 1 and t3["short_T2_gross"].iloc[0] == pytest.approx(-3.0)
    # the layer never overrides an intrabar exit in the SAME bar: bar 2 touches BOTH the TP (102.4 >= 102) and the SL (96 <= 99):
    # the pessimistic SL-first base exit fills at 99 (-1.0), never at the cat level 97 (-3.0)
    h4 = [100.1, 101.5, 102.4, 101, 101, 101, 101, 101, 101]
    l4 = [99.9, 99.5, 96.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    t4 = build_labels(make_ohl_df(o, h4, l4, o), cfg_small(costs=ZERO, labels={"barrier_grid": {"tp": [2]}, "cat_stop_atr": 3.0}), None).table
    assert t4["long_I2_1_4_hold"].iloc[0] == 1 and t4["long_I2_1_4_gross"].iloc[0] == pytest.approx(-1.0)


def test_intrabar_hold_bounds_family_names_and_graceful_skip_without_high_low():
    df, _ = make_synthetic_bars(3000, seed=4)
    cfg = load_config()
    t = build_labels(df, cfg, None).table
    names = [f for f, _ in family_list(cfg)]
    assert sum(1 for f in names if f.startswith("I")) == 18                   # the full B grid mirrored: 3 tp x 3 sl x 2 horizon
    for f, p in family_list(cfg):
        assert parse_family(f) == p
    for col in [c for c in t.columns if c.startswith("long_I") and c.endswith("_hold")]:
        h = parse_family(col.removeprefix("long_").removesuffix("_hold"))["h"]
        v = t[col].dropna()
        assert v.min() >= 0 and v.max() <= h                                  # an intrabar touch can happen on the entry bar itself
    # data without high/low: the I families are skipped with a spec note instead of crashing stage 2
    slim = df.drop(columns=["high", "low"])
    res = build_labels(slim, load_config(overrides={"labels": {"intrabar_families": True}}), None)
    assert all(not f.startswith("I") for f in res.families)
    assert res.spec["intrabar_families"] is False and res.spec["intrabar_dropped"]
    with pytest.raises(ValueError):                                           # an EXPLICIT cat stop without high/low is a config error
        build_labels(slim, load_config(overrides={"labels": {"cat_stop_atr": 3.0}}), None)


def _sig(n, bars):
    s = np.zeros(n, dtype=bool)
    s[list(bars)] = True
    return s


@pytest.mark.parametrize("seed", [1, 2])
def test_engine_matches_labels_with_intrabar_and_cat_layer(seed):
    """Label <-> engine equality (acceptance 19.4) with the intrabar families AND the catastrophic layer switched on."""
    from edge_research.mt5_backtest.engine import run_engine
    df, _ = make_synthetic_bars(4000, seed=seed)
    cfg = load_config(overrides={"labels": {"cat_stop_atr": 3.0}})
    labels = build_labels(df, cfg, None).table
    rng = np.random.default_rng(seed + 1)
    n = len(df)
    for fam, p in family_list(cfg):
        if p["kind"] not in ("T", "I"):
            continue
        spacing = (p["n"] if p["kind"] == "T" else p["h"]) + 3
        bars = np.arange(5, n - 1, spacing)
        for d in ("long", "short"):
            pick = bars[rng.random(len(bars)) < 0.5]
            res = run_engine(df, cfg, {d: _sig(n, pick)}, fam, cat_stop_atr=3.0)
            t = res.trades
            expected = (labels[f"{d}_{fam}_gross"] - labels[f"{d}_{fam}_cost"]).to_numpy()[pick]
            valid = np.isfinite(expected)
            assert res.stats["overlap_skipped"] == 0
            assert list(t["signal_bar"]) == list(pick[valid]), (fam, d)
            np.testing.assert_allclose(t["net_R"].to_numpy(), expected[valid], rtol=1e-5, atol=1e-5, err_msg=f"{fam} {d}")
            np.testing.assert_allclose(t["hold"].to_numpy(), labels[f"{d}_{fam}_hold"].to_numpy()[pick][valid], err_msg=f"{fam} {d}")
