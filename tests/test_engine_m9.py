"""M9 engine extensions: intrabar, catastrophic and condition exits, position limits, cost multiplier, end handling, plus a scalar reference simulation."""

import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.mt5_backtest.engine import ENGINE_DEFAULTS, engine_settings, parse_family, run_engine


def cfg0(**eng):
    return load_config(overrides={"costs": {"point": 0.01, "contract_size": 100, "fixed_spread_points": 0, "commission_per_lot_per_side": 0, "swap_long_per_lot": 0,
                                            "swap_short_per_lot": 0, "slippage_points": 0},
                                  "labels": {"time_stops": [3], "barrier_grid": {"tp": [1], "sl": [1], "horizon": [4]}}, "engine": eng})


def world(o, h=None, l=None, c=None, atr=1.0):
    n = len(o)
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1h")
    o = np.asarray(o, float)
    return pd.DataFrame({"open": o, "high": np.asarray(h, float) if h is not None else o + 0.1, "low": np.asarray(l, float) if l is not None else o - 0.1,
                         "close": np.asarray(c, float) if c is not None else o, "atr_14": atr}, index=idx)


def sig(n, bars):
    s = np.zeros(n, dtype=bool)
    s[list(bars)] = True
    return s


def one(df, fam, direction="long", spread=None, cfg=None, **kw):
    sp = None if spread is None else pd.Series(spread, index=df.index)
    r = run_engine(df, cfg or cfg0(), {direction: sig(len(df), [0])}, fam, sp, **kw)
    assert len(r.trades) == 1, r.stats
    return r.trades.iloc[0]


# ------------------------------------------------------------------------------------------------------------ settings and names
def test_engine_settings_defaults_unknown_keys_and_validation():
    assert engine_settings(load_config())["max_open_positions"] == 1 and set(ENGINE_DEFAULTS) >= {"cost_multiplier", "sizing_mode", "cat_stop_atr", "account_mode"}
    with pytest.raises(ValueError, match="unknown engine"):
        engine_settings({"engine": {"reenter_on_exit_bars": True}})
    for bad in ({"max_open_positions": 0}, {"sizing_mode": "kelly"}, {"account_mode": "x"}, {"cost_multiplier": -1}, {"conflict_policy": "long"}):
        with pytest.raises(ValueError):
            engine_settings({"engine": bad})
    assert parse_family("I2_1p5_12") == {"kind": "I", "tp": 2.0, "sl": 1.5, "h": 12} and parse_family("B2_1p5_12")["kind"] == "B"


# ------------------------------------------------------------------------------------------------------------ intrabar exits by hand
# entry bar e = 1 at open 100, ATR 1: TP = 100 + 2 = 102, SL = 100 - 1 = 99 (long)
def test_intrabar_long_tp_touched_in_a_later_bar_fills_at_the_level():
    o = [100, 100, 100.4, 100.8, 101, 101, 101, 101, 101]
    h = [100.1, 101.5, 102.4, 101, 101, 101, 101, 101, 101]
    l = [99.9, 99.5, 100.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    t = one(world(o, h, l), "I2_1_4")
    assert (t["entry_bar"], t["exit_bar"], t["exit_reason"], bool(t["exit_intrabar"])) == (1, 2, "tp", True)
    assert t["exit_price"] == pytest.approx(102.0) and t["gross_R"] == pytest.approx(2.0) and t["net_R"] == pytest.approx(2.0) and t["hold"] == 1
    assert t["mfe"] == pytest.approx(2.4) and t["mae"] == pytest.approx(0.5)                                      # the touching bar counts for both excursions
    assert t["stop_distance"] == pytest.approx(1.0)


def test_intrabar_both_levels_in_one_bar_stop_first():
    o = [100] * 9
    h = [100.1, 100.5, 103.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    l = [99.9, 99.6, 98.0, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5]
    t = one(world(o, h, l), "I2_1_4")
    assert t["exit_reason"] == "sl" and t["exit_price"] == pytest.approx(99.0) and t["gross_R"] == pytest.approx(-1.0) and t["exit_bar"] == 2


def test_intrabar_gap_through_a_level_fills_at_the_open_only_after_the_entry_bar():
    o = [100, 100, 98.0, 100, 100, 100, 100, 100]                                                                 # bar 2 opens below the stop 99
    l = [99.5, 99.5, 97, 99.5, 99.5, 99.5, 99.5, 99.5]
    t = one(world(o, [100.5] * 8, l), "I2_1_4")
    assert t["exit_reason"] == "sl" and t["exit_price"] == pytest.approx(98.0) and t["gross_R"] == pytest.approx(-2.0) and t["exit_bar"] == 2
    o = [100, 100, 103.0, 100, 100, 100, 100, 100]                                                                # favourable gap: fills at the open, better than the level
    t = one(world(o, [100.5, 100.5, 104, 100.5, 100.5, 100.5, 100.5, 100.5], [99.5] * 8), "I2_1_4")
    assert t["exit_reason"] == "tp" and t["exit_price"] == pytest.approx(103.0) and t["gross_R"] == pytest.approx(3.0)
    # on the ENTRY bar itself the open is the entry price: no gap test, the low touches the level
    t = one(world([100] * 8, [100.1] * 8, [98.9, 98.9, 100, 100, 100, 100, 100, 100]), "I2_1_4")
    assert t["exit_bar"] == 1 and t["exit_price"] == pytest.approx(99.0) and t["exit_reason"] == "sl"


def test_intrabar_without_touch_exits_at_the_horizon_open():
    t = one(world([100] * 8, [100.5] * 8, [99.5] * 8), "I2_1_4")
    assert t["exit_reason"] == "time" and not t["exit_intrabar"] and t["exit_bar"] == 5 and t["hold"] == 4 and t["gross_R"] == pytest.approx(0.0)
    assert t["mae"] == pytest.approx(0.5) and t["mfe"] == pytest.approx(0.5)                                     # bars e..x-1 only for an open exit


def test_intrabar_short_uses_ask_prices_and_pays_the_exit_spread():
    sp = [20] * 9                                                                                                # 20 points x 0.01 = 0.2
    o = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    # short: entry bid 100 -> TP ask level 99, SL ask level 101
    h, l = [100.1] * 9, [99.9] * 9
    l[2] = 98.5                                                                                                   # ask low 98.7 <= 99
    t = one(world(o, h, l), "I1_1_4", "short", spread=sp)
    assert t["exit_reason"] == "tp" and t["exit_bar"] == 2 and t["exit_price"] == pytest.approx(98.8)             # bid-equivalent = 99 - 0.2
    assert t["gross_R"] == pytest.approx(1.2) and t["cost_R"] == pytest.approx(0.2) and t["net_R"] == pytest.approx(1.0)
    h, l = [100.1] * 9, [99.9] * 9
    h[2] = 100.9                                                                                                  # ask high 101.1 >= 101
    t = one(world(o, h, l), "I1_1_4", "short", spread=sp)
    assert t["exit_reason"] == "sl" and t["net_R"] == pytest.approx(-1.0) and t["exit_price"] == pytest.approx(100.8)
    h, l = [100.1] * 9, [99.9] * 9
    h[2], l[2] = 100.85, 99.0                                                                                     # ask high 101.05 and ask low 99.2: only the stop is touched
    assert one(world(o, h, l), "I1_1_4", "short", spread=sp)["exit_reason"] == "sl"


def test_intrabar_long_pays_the_entry_spread_and_slippage_through_the_levels():
    cfg = load_config(overrides={"costs": {"point": 0.01, "contract_size": 100, "fixed_spread_points": 20, "slippage_points": 10, "commission_per_lot_per_side": 0},
                                 "labels": {"time_stops": [3], "barrier_grid": {"tp": [1], "sl": [1], "horizon": [4]}}})
    h, l = [100.1] * 9, [99.9] * 9
    h[2] = 101.6                                                                                                  # entry fill = 100 + 0.2 + 0.1 = 100.3 -> TP level 101.3
    t = one(world([100] * 9, h, l), "I1_1_4", cfg=cfg)
    assert t["exit_reason"] == "tp" and t["exit_price"] == pytest.approx(101.3)
    assert t["gross_R"] == pytest.approx(1.3) and t["cost_R"] == pytest.approx(0.2 + 0.2)                          # entry spread + 2 x slippage 0.1; net = TP - entry fill = 1.0 R


def test_intrabar_needs_high_and_low():
    df = world([100] * 8).drop(columns=["high", "low"])
    with pytest.raises(ValueError, match="high and low"):
        run_engine(df, cfg0(), {"long": sig(8, [0])}, "I2_1_4")
    with pytest.raises(ValueError, match="high and low"):
        run_engine(df, cfg0(), {"long": sig(8, [0])}, "T3", cat_stop_atr=2.0)
    assert len(run_engine(df, cfg0(), {"long": sig(8, [0])}, "T3").trades) == 1                                   # plain exits work without high / low (MAE / MFE are NaN)
    assert np.isnan(run_engine(df, cfg0(), {"long": sig(8, [0])}, "T3").trades["mae"].iloc[0])


# ------------------------------------------------------------------------------------------------------------ bar boundary reasons
def test_bar_boundary_reasons_and_the_old_ambiguity_is_gone():
    o = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    c = list(o)
    c[2] = 102.5                                                                                                  # close of bar 2 beyond TP 102 -> exit at the open of bar 3
    t = one(world(o, c=c), "B2_1_6")
    assert (t["exit_bar"], t["exit_reason"], bool(t["exit_intrabar"])) == (3, "tp", False)
    c = list(o)
    c[3] = 98.9
    assert one(world(o, c=c), "B2_1_6")["exit_reason"] == "sl"
    assert one(world(o), "B2_1_6")["exit_reason"] == "time"
    c = list(o)
    c[6] = 102.5                                                                                                  # the LAST bar of the horizon (e + h - 1 = 6) closes beyond TP
    t = one(world(o, c=c), "B2_1_6")
    assert t["exit_bar"] == 7 and t["exit_reason"] == "tp"
    assert one(world(o), "T3")["exit_reason"] == "time"


# ------------------------------------------------------------------------------------------------------------ catastrophic layer
def test_catastrophic_stop_fires_inside_a_bar_before_the_time_exit():
    o = [100] * 10
    l = [99.9] * 10
    l[3] = 97.5                                                                                                   # cat level = 100 - 2 = 98
    t = one(world(o, [100.5] * 10, l), "T6", cat_stop_atr=2.0)
    assert (t["exit_bar"], t["exit_reason"], bool(t["exit_intrabar"])) == (3, "cat", True) and t["exit_price"] == pytest.approx(98.0) and t["gross_R"] == pytest.approx(-2.0)
    assert t["stop_distance"] == pytest.approx(2.0)
    o2 = list(o)
    o2[3] = 97.0                                                                                                  # gap through the level: fills at the open
    l2 = [99.9] * 10
    l2[3] = 96.5
    t = one(world(o2, [100.5] * 10, l2), "T6", cat_stop_atr=2.0)
    assert t["exit_price"] == pytest.approx(97.0) and t["exit_reason"] == "cat"
    assert one(world(o, [100.5] * 10, [99.9] * 10), "T6", cat_stop_atr=2.0)["exit_reason"] == "time"             # never touched


def test_catastrophic_does_not_pre_empt_an_exit_at_the_same_open_and_comes_from_the_config():
    l = [99.9] * 10
    l[4] = 90.0                                                                                                   # T3: entry 1, exit at the open of bar 4; the crash happens INSIDE bar 4, after that open
    t = one(world([100] * 10, [100.5] * 10, l), "T3", cat_stop_atr=2.0)
    assert t["exit_reason"] == "time" and t["exit_bar"] == 4
    l[3] = 90.0
    assert one(world([100] * 10, [100.5] * 10, l), "T3", cfg=cfg0(cat_stop_atr=2.0))["exit_reason"] == "cat"
    with pytest.raises(ValueError, match="positive"):
        run_engine(world([100] * 10), cfg0(), {"long": sig(10, [0])}, "T3", cat_stop_atr=0)


def test_catastrophic_on_a_bar_boundary_exit_and_for_shorts():
    o = [100] * 10
    l = [99.9] * 10
    l[2] = 96.5                                                                                                   # bar-boundary SL 99 not hit by a CLOSE, but intrabar cat 97 is
    t = one(world(o, [100.5] * 10, l), "B2_1_6", cat_stop_atr=3.0)
    assert t["exit_reason"] == "cat" and t["exit_price"] == pytest.approx(97.0)
    h = [100.1] * 10
    h[2] = 103.5                                                                                                  # short: ask level 103 with zero spread
    t = one(world(o, h, [99.9] * 10), "T6", "short", cat_stop_atr=3.0)
    assert t["exit_reason"] == "cat" and t["exit_price"] == pytest.approx(103.0) and t["gross_R"] == pytest.approx(-3.0)


# ------------------------------------------------------------------------------------------------------------ condition exit
def test_condition_exit_leaves_at_the_next_open_and_loses_ties_to_the_base_exit():
    df = world([100, 100, 101, 102, 103, 104, 105, 106, 107, 108])
    cond = sig(10, [2])                                                                                           # true at the CLOSE of bar 2 -> exit at the open of bar 3
    t = one(df, "T6", exit_signals={"long": cond})
    assert (t["exit_bar"], t["exit_reason"], bool(t["exit_intrabar"])) == (3, "cond", False) and t["exit_price"] == pytest.approx(102.0)
    assert one(df, "T6", exit_signals={"long": sig(10, [6])})["exit_reason"] == "time"                           # T6 exits at the open of bar 7; a decision on bar 6 also means bar 7: base wins
    assert one(df, "T6", exit_signals={"long": sig(10, [5])})["exit_bar"] == 6                                   # one bar earlier it is the condition exit
    assert one(df, "T6", exit_signals={"long": sig(10, [0])})["exit_reason"] == "time"                           # before the entry bar: ignored
    assert one(df, "T6", exit_signals={"short": sig(10, [2])})["exit_reason"] == "time"                          # other direction
    assert one(df, "T6", exit_signals={"long": sig(10, [1, 2, 3])})["exit_bar"] == 2                             # the first true bar wins
    with pytest.raises(ValueError):
        run_engine(df, cfg0(), {"long": sig(10, [0])}, "T6", exit_signals={"up": cond})


def test_condition_exit_against_an_intrabar_touch_and_the_catastrophic_layer():
    o = [100] * 12
    h, l = [100.1] * 12, [99.9] * 12
    h[4] = 102.5                                                                                                  # I2_1_8: TP touched inside bar 4
    df = world(o, h, l)
    assert one(df, "I2_1_8", exit_signals={"long": sig(12, [3])})["exit_reason"] == "cond"                       # decided at the close of 3 -> open of 4, before the touch inside 4
    assert one(df, "I2_1_8", exit_signals={"long": sig(12, [4])})["exit_reason"] == "tp"                         # decided at the close of 4: too late
    l2 = [99.9] * 12
    l2[3] = 97.0
    df2 = world(o, [100.1] * 12, l2)
    t = one(df2, "T8", exit_signals={"long": sig(12, [4])}, cat_stop_atr=2.0)                                    # cat inside bar 3 happens before the open of 5
    assert (t["exit_reason"], t["exit_bar"]) == ("cat", 3)
    t = one(df2, "T8", exit_signals={"long": sig(12, [1])}, cat_stop_atr=2.0)                                    # cond exit at the open of bar 2 happens before bar 3's crash
    assert (t["exit_reason"], t["exit_bar"]) == ("cond", 2)


# ------------------------------------------------------------------------------------------------------------ positions
def test_max_open_positions_and_the_tick_that_closed_a_trade():
    df = world(np.arange(100, 112.0))
    s = sig(len(df), [0, 1, 2, 3, 4, 5])                                                                          # T3: signal k enters at k + 1 and exits at the open of k + 4
    n = lambda **eng: list(run_engine(df, cfg0(**eng), {"long": s}, "T3").trades["signal_bar"])
    assert n() == [0, 4]                                                                                          # signal 3 would enter on the exit bar 4 of trade 0 (blocked), signal 4 enters at 5
    assert n(reenter_on_exit_bar=True) == [0, 3]
    assert n(max_open_positions=2) == [0, 1, 5]                                                                   # 3 and 4 enter on bars that closed a trade (4: trade 0, 5: trade 1)
    assert n(max_open_positions=2, reenter_on_exit_bar=True) == [0, 1, 3, 4]
    assert n(max_open_positions=3) == [0, 1, 2]
    assert n(max_open_positions=9, reenter_on_exit_bar=True) == [0, 1, 2, 3, 4, 5]
    r = run_engine(df, cfg0(max_open_positions=2, reenter_on_exit_bar=True), {"long": s}, "T3")
    assert r.stats["overlap_skipped"] == 2 and r.stats["trades"] == 4


def test_an_intrabar_exit_frees_the_slot_for_the_next_bar_only():
    o = [100] * 12
    h, l = [100.1] * 12, [99.9] * 12
    h[2] = 102.5                                                                                                  # signal 0: TP inside bar 2
    r = run_engine(world(o, h, l), cfg0(reenter_on_exit_bar=True), {"long": sig(12, [0, 1, 2, 3])}, "I2_1_6")
    assert list(r.trades["signal_bar"]) == [0, 2]                                                                 # bar 2's signal enters at bar 3 (after the intrabar exit in bar 2); signal 1 (entry bar 2) is still blocked
    assert r.stats["overlap_skipped"] == 2


def test_close_open_at_end_and_out_of_data_counters():
    df = world(np.arange(100, 110.0))                                                                             # 10 bars; T6 needs the signal bar <= 2
    s = {"long": sig(10, [3, 7])}
    r = run_engine(df, cfg0(), s, "T6")
    assert r.trades.empty and r.stats["n_open_at_end"] == 2 and r.stats["out_of_data"] == 2                       # the window leaves the data, nothing closes
    r = run_engine(df, cfg0(close_open_at_end=True), s, "T6")
    t = r.trades.iloc[0]
    assert len(r.trades) == 1 and r.stats["closed_at_end"] == 2 and r.stats["overlap_skipped"] == 1             # the second one overlaps the first, as always
    assert t["exit_reason"] == "end" and t["entry_bar"] == 4 and t["exit_bar"] == 9 and t["exit_price"] == pytest.approx(109.0) and t["gross_R"] == pytest.approx(5.0)
    r = run_engine(df, cfg0(close_open_at_end=True), {"long": sig(10, [8, 9])}, "T6")                             # entry bar 9 is the last bar: nothing to close
    assert r.trades.empty and r.stats["out_of_data"] == 2 and r.stats["closed_at_end"] == 0


def test_cost_multiplier_scales_spread_and_slippage_only():
    cfg = lambda m: load_config(overrides={"costs": {"point": 0.01, "contract_size": 100, "fixed_spread_points": 20, "slippage_points": 5, "commission_per_lot_per_side": 2.0},
                                           "labels": {"time_stops": [3], "barrier_grid": {"tp": [1], "sl": [1], "horizon": [4]}}, "engine": {"cost_multiplier": m}})
    df = world(np.arange(100, 112.0))
    c1 = one(df, "T3", cfg=cfg(1.0))["cost_R"]
    c3 = one(df, "T3", cfg=cfg(3.0))["cost_R"]
    com = 2 * 2.0 / 100
    assert c1 == pytest.approx(0.2 + 0.1 + com) and c3 == pytest.approx(3 * (0.2 + 0.1) + com)                    # commission is not multiplied


# ------------------------------------------------------------------------------------------------------------ scalar reference simulation
def reference_trade(df, spread, atr, i, is_long, fam, cat_k, cond):
    """Bar-by-bar scalar simulation of one signal. Independent of the vectorised code: it walks time forward."""
    p = parse_family(fam)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    e = i + 1
    a = atr[i]
    sgn = 1 if is_long else -1
    entry = o[e] + spread[e] if is_long else o[e]
    tp_l = entry + sgn * p["tp"] * a if p["kind"] != "T" else None
    sl_l = entry - sgn * p["sl"] * a if p["kind"] != "T" else None
    cat_l = entry - sgn * cat_k * a if cat_k else None
    hor = p["n"] if p["kind"] == "T" else p["h"]
    ex = lambda b: o[b] + (0 if is_long else spread[b])                                                           # price the side exits at
    for b in range(e, e + hor + 1):
        # 1) exits decided at the previous close execute at this open
        if b > e and p["kind"] == "B":
            pc = c[b - 1]
            tp_h = pc >= tp_l if is_long else pc <= tp_l
            sl_h = pc <= sl_l if is_long else pc >= sl_l
            if sl_h or tp_h:                                                                                      # a hit on the last horizon bar is still a hit
                return b, o[b], "sl" if sl_h else "tp"
        if b == e + hor:
            return b, o[b], "time"
        if b > e and cond is not None and cond[b - 1]:
            return b, o[b], "cond"
        # 2) inside the bar: levels in the price they are quoted in (short: ask = price + spread)
        sp = 0 if is_long else spread[b]
        for kind, lvl in (("sl", sl_l), ("tp", tp_l), ("cat", cat_l)):
            if lvl is None or (kind != "cat" and p["kind"] != "I"):
                continue
            adverse = kind in ("sl", "cat")
            gap = b > e and ((o[b] + sp <= lvl) if (is_long == adverse) else (o[b] + sp >= lvl))
            if is_long:
                touched = (l[b] + sp <= lvl) if adverse else (h[b] + sp >= lvl)
            else:
                touched = (h[b] + sp >= lvl) if adverse else (l[b] + sp <= lvl)
            if gap or touched:
                fill = o[b] + sp if gap else lvl
                return b, fill - (0 if is_long else spread[b]), kind
    raise AssertionError("unreachable")


def random_world(seed, n=400):
    rng = np.random.default_rng(seed)
    o = 100 + np.cumsum(rng.normal(0, 0.5, n))
    gap = rng.normal(0, 0.4, n) * (rng.random(n) < 0.1)                                                            # occasional open gaps
    o = o + np.cumsum(gap)
    c = o + rng.normal(0, 0.4, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.4, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.4, n))
    df = world(o, h, l, c, atr=1.0)
    spread = rng.integers(2, 30, n) * 0.01
    return df, spread


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fam,cat_k,use_cond", [("T6", None, False), ("T6", 2.5, True), ("B2_1_8", None, False), ("B2_1_8", 3.5, True), ("I2_1_8", None, False),
                                                ("I1_1p5_6", 4.0, False), ("I2_1_8", 3.0, True), ("T3", 1.0, False)])
def test_engine_equals_a_scalar_reference_for_random_paths(seed, fam, cat_k, use_cond):
    df, spread = random_world(seed)
    n = len(df)
    rng = np.random.default_rng(100 + seed)
    pick = np.sort(rng.choice(np.arange(0, n - 20), 25, replace=False))
    cond = (rng.random(n) < 0.08) if use_cond else None
    cfg = cfg0(max_open_positions=100, reenter_on_exit_bar=True)
    sp_series = pd.Series(spread / 0.01, index=df.index)
    atr = df["atr_14"].to_numpy()
    for direction in ("long", "short"):
        r = run_engine(df, cfg, {direction: sig(n, pick)}, fam, sp_series, exit_signals={direction: cond} if use_cond else None, cat_stop_atr=cat_k)
        got = r.trades.set_index("signal_bar")
        assert len(got) == len(pick)
        for i in pick:
            xb, px, reason = reference_trade(df, spread, atr, i, direction == "long", fam, cat_k, cond)
            row = got.loc[i]
            assert (int(row["exit_bar"]), row["exit_reason"]) == (xb, reason), (i, direction, fam)
            assert row["exit_price"] == pytest.approx(px, abs=1e-9), (i, direction, fam, reason)                  # the reference already returns the bid-equivalent price
            e = i + 1
            fill = df["open"].iloc[e] + spread[e] if direction == "long" else df["open"].iloc[e]
            net_price = (row["exit_price"] - fill) if direction == "long" else (fill - row["exit_price"] - spread[xb])
            assert row["net_R"] == pytest.approx(net_price / atr[i], abs=1e-9)                                     # net = signed price move including the spread paid
