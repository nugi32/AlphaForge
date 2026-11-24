"""M9: accounting, metrics, compare_trades, canonical rule file and the multi-rule backtest."""

import json

import numpy as np
import pandas as pd
import pytest

from edge_research.atoms.atoms import build_atoms
from edge_research.config import load_config
from edge_research.mt5_backtest.accounting import account_trades, round_lots
from edge_research.mt5_backtest.backtest import run_backtest, write_backtest
from edge_research.mt5_backtest.compare import compare_trades
from edge_research.mt5_backtest.engine import engine_settings, run_engine
from edge_research.mt5_backtest.metrics import _max_run, drawdown, summarize, trimmed_mean
from edge_research.rules import FrozenRule, atom_mask, exit_condition_signal, load_rules, rule_signal, save_rules
from test_engine_m9 import random_world, sig, world


def cfgm(costs=None, **eng):
    c = {"point": 0.01, "contract_size": 100, "fixed_spread_points": 0, "commission_per_lot_per_side": 0, "swap_long_per_lot": 0, "swap_short_per_lot": 0, "slippage_points": 0}
    c.update(costs or {})
    return load_config(overrides={"costs": c, "labels": {"time_stops": [3], "barrier_grid": {"tp": [1], "sl": [1], "horizon": [4]}}, "engine": eng})


def acct(df, cfg, signals, fam, spread=None, **kw):
    r = run_engine(df, cfg, signals, fam, spread, **kw)
    return r, account_trades(r.trades, r.market, engine_settings(cfg), df.index)


# ------------------------------------------------------------------------------------------------------------ money by hand
RISING = [100, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


def test_long_and_short_money_by_hand_with_spread_slippage_and_commission():
    cfg = cfgm({"fixed_spread_points": 20, "slippage_points": 10, "commission_per_lot_per_side": 3.5}, fixed_lots=0.1)
    df = world(RISING, atr=1.0)
    r, a = acct(df, cfg, {"long": sig(12, [0])}, "T3")
    t = a.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(100.3) and t["exit_price"] == pytest.approx(102.9) and t["price_pnl"] == pytest.approx(2.6)
    assert t["gross_profit"] == pytest.approx(26.0) and t["commission"] == pytest.approx(-0.7) and t["swap"] == 0 and t["profit"] == pytest.approx(25.3)
    assert t["r_net"] == pytest.approx(2.53) == pytest.approx(r.trades["net_R"].iloc[0]) and t["lots"] == 0.1 and t["spread_entry"] == pytest.approx(0.2)
    r, a = acct(df, cfg, {"short": sig(12, [0])}, "T3")                                                         # short: bid - slip in, ask + slip out
    t = a.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(99.9) and t["exit_price"] == pytest.approx(103.3) and t["price_pnl"] == pytest.approx(-3.4)
    assert t["profit"] == pytest.approx(-3.4 * 10 - 0.7) and t["r_net"] == pytest.approx(r.trades["net_R"].iloc[0]) == pytest.approx(-3.47)


def test_swap_money_sign_and_nights():
    idx = pd.date_range("2024-01-01 20:00", periods=40, freq="1h")                                            # 20:00 -> crosses the 00:00 rollover each day
    df = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "atr_14": 1.0}, index=idx)
    cfg = cfgm({"swap_long_per_lot": -7.0, "swap_short_per_lot": 3.0, "swap_rollover_hour": 0, "swap_triple_weekday": 9}, fixed_lots=0.5)
    r, a = acct(df, cfg, {"long": sig(40, [0])}, "T24")
    t = a.trades.iloc[0]
    assert t["nights"] == 1 and t["swap"] == pytest.approx(-7.0 * 0.5)                                         # a negative long rate costs money
    r, a = acct(df, cfg, {"short": sig(40, [0])}, "T24")
    assert a.trades.iloc[0]["swap"] == pytest.approx(3.0 * 0.5)                                                # a positive short rate earns
    assert a.trades.iloc[0]["r_net"] == pytest.approx(r.trades["net_R"].iloc[0])


def test_r_net_equals_engine_net_r_for_any_sizing_on_random_paths():
    df, spread = random_world(3)
    n = len(df)
    sp = pd.Series(spread / 0.01, index=df.index)
    pick = np.arange(5, n - 30, 17)
    for sizing in ({"sizing_mode": "fixed_lots", "fixed_lots": 0.37}, {"sizing_mode": "risk_percent", "risk_percent": 2.0, "compounding": True}):
        cfg = cfgm({"fixed_spread_points": 5, "slippage_points": 3, "commission_per_lot_per_side": 2.0, "swap_long_per_lot": -4.0, "swap_short_per_lot": 1.0}, max_open_positions=3,
                   reenter_on_exit_bar=True, **sizing)
        for fam in ("B2_1_8", "I2_1_8"):
            for d in ("long", "short"):
                r, a = acct(df, cfg, {d: sig(n, pick)}, fam, sp)
                assert len(a.trades) == len(r.trades) > 3
                np.testing.assert_allclose(a.trades["r_net"].to_numpy(), r.trades["net_R"].to_numpy(), atol=1e-9)


# ------------------------------------------------------------------------------------------------------------ sizing
def test_risk_percent_sizing_by_hand_clipping_and_missing_stop():
    df = world([100] * 12, atr=2.0)
    base = dict(sizing_mode="risk_percent", risk_percent=1.0, initial_deposit=10000.0, lot_step=0.01, lot_min=0.01, lot_max=100.0)
    lot = lambda **o: acct(df, cfgm(**{**base, **o}), {"long": sig(12, [0])}, "I1_1_4")[1].trades["lots"].iloc[0]
    assert lot() == pytest.approx(0.5)                                                                         # 100 / (stop 2 x 100) = 0.5
    df3 = world([100] * 12, atr=3.0)
    a = acct(df3, cfgm(**base), {"long": sig(12, [0])}, "I1_1_4")[1]
    assert a.trades["lots"].iloc[0] == pytest.approx(0.33)                                                     # 0.3333 floored to the step, never rounded up
    a = acct(df3, cfgm(**{**base, "initial_deposit": 100.0}), {"long": sig(12, [0])}, "I1_1_4")[1]
    assert a.trades["lots"].iloc[0] == pytest.approx(0.01) and a.stats["n_lot_clipped_min"] == 1
    assert lot(lot_max=0.2) == pytest.approx(0.2)
    assert lot(risk_percent=2.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="needs a stop"):
        acct(df, cfgm(**base), {"long": sig(12, [0])}, "T3")
    r, a = acct(df, cfgm(**base), {"long": sig(12, [0])}, "T3", cat_stop_atr=3.0)                              # the catastrophic stop is the risk distance: 100 / (6 x 100)
    assert a.trades["lots"].iloc[0] == pytest.approx(0.16)


def test_round_lots_float_noise_and_bounds():
    assert round_lots(0.3, 0.01, 0.0, 100) == pytest.approx(0.30)                                              # 0.3 / 0.01 = 29.999999999999996
    assert round_lots(0.019999, 0.01, 0.0, 100) == pytest.approx(0.01)
    assert round_lots(250.0, 0.01, 0.0, 100) == 100.0 and round_lots(0.5, 0.1, 0.0, 100) == pytest.approx(0.5)


def test_compounding_uses_the_realised_balance_at_the_entry_open():
    # two trades in a row, each T3 (exit at the open 4 bars later); the second one is sized from the balance AFTER the first
    o = [100, 100, 101, 102, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103]
    df = world(o, h=np.array(o) + 0.1, l=np.array(o) - 0.1, atr=1.0)
    base = dict(sizing_mode="risk_percent", risk_percent=10.0, initial_deposit=1000.0, reenter_on_exit_bar=True, max_open_positions=1)
    s = {"long": sig(14, [0, 5])}
    flat = acct(df, cfgm(**base, compounding=False), s, "I2_1_6")[1].trades
    comp = acct(df, cfgm(**base, compounding=True), s, "I2_1_6")[1].trades
    assert flat["lots"].tolist() == [1.0, 1.0]                                                                  # 10 % of 1000 = 100 / (1 x 100) = 1 lot
    first = comp["profit"].iloc[0]
    assert first == pytest.approx(200.0)                                                                       # TP 2 R = 2 price units x 100 x 1 lot (touched inside bar 2)
    assert comp["lots"].iloc[1] == pytest.approx(np.floor((0.10 * (1000 + first) / 100) / 0.01) * 0.01)       # 1.2 lots


# ------------------------------------------------------------------------------------------------------------ balance and equity
def test_balance_and_equity_by_hand_for_open_and_intrabar_exits():
    df = world(RISING, atr=1.0)
    _, a = acct(df, cfgm(fixed_lots=1.0), {"long": sig(12, [0])}, "T3")                                         # long 1 lot, entry open 100 (bar 1), exit open 103 (bar 4)
    np.testing.assert_allclose(a.balance.to_numpy()[:6], [10000, 10000, 10000, 10000, 10300, 10300])
    np.testing.assert_allclose(a.equity.to_numpy()[:6], [10000, 10000, 10100, 10200, 10300, 10300])
    _, a = acct(df, cfgm(fixed_lots=1.0), {"short": sig(12, [0])}, "T3")
    np.testing.assert_allclose(a.equity.to_numpy()[:6], [10000, 10000, 9900, 9800, 9700, 9700])
    # intrabar TP in bar 2 (price 102): the position is still open at the OPEN of bar 2; the balance books it at the open of bar 3
    h = np.array(RISING, float) + 0.1
    h[2] = 102.5
    dfi = world(RISING, h=h, l=np.array(RISING, float) - 0.1, atr=1.0)
    r, a = acct(dfi, cfgm(fixed_lots=1.0), {"long": sig(12, [0])}, "I2_1_6")
    assert r.trades["exit_bar"].iloc[0] == 2 and r.trades["exit_intrabar"].iloc[0]
    np.testing.assert_allclose(a.balance.to_numpy()[:5], [10000, 10000, 10000, 10200, 10200])
    np.testing.assert_allclose(a.equity.to_numpy()[:5], [10000, 10000, 10100, 10200, 10200])
    # costs show up as a negative floating profit at the entry open (long marked at the bid)
    _, a = acct(df, cfgm({"fixed_spread_points": 20}, fixed_lots=1.0), {"long": sig(12, [0])}, "T3")
    assert a.equity.iloc[1] == pytest.approx(10000 - 20.0)


def test_empty_trade_list_gives_a_flat_account():
    df = world(RISING, atr=1.0)
    r, a = acct(df, cfgm(), {"long": sig(12, [])}, "T3")
    assert a.trades.empty and (a.balance == 10000).all() and (a.equity == 10000).all()
    s = summarize(a, df.index, engine_settings(cfgm()), 0.01, r.stats)
    assert s["total_trades"] == 0 and s["final_balance"] == 10000 and s["total_net_profit"] == 0


# ------------------------------------------------------------------------------------------------------------ metrics by hand
def test_drawdown_runs_and_trimmed_mean_by_hand():
    assert drawdown(np.array([10, 12, 9, 11, 7, 13, 12.0])) == (5.0, pytest.approx(5 / 12), 3)               # peak 12 -> 7; below the peak for bars 9, 11, 7 (3 bars); 13 is a new peak, then 12 is below for 1 bar
    assert drawdown(np.array([1, 2, 3.0])) == (0.0, 0.0, 0) and drawdown(np.array([])) == (0.0, 0.0, 0)
    assert _max_run(np.array([1, 1, 0, 1, 1, 1, 0], bool)) == 3
    r = np.arange(1, 101, dtype=float)                                                                         # 1..100
    assert trimmed_mean(r, 0.05) == pytest.approx(np.mean(np.arange(1, 96))) and trimmed_mean(r, 0.01) == pytest.approx(np.mean(np.arange(1, 100)))
    assert trimmed_mean(np.array([5.0]), 0.05) == 5.0 and np.isnan(trimmed_mean(np.array([]), 0.05))


def test_summary_metrics_on_a_known_trade_sequence():
    # five T1 trades, profits by construction: price changes +2, -1, +3, -1, -1 on one lot x 100
    o = [100, 100, 102, 102, 101, 101, 104, 104, 103, 103, 102, 102, 102, 102]
    df = world(o, atr=1.0)
    cfg = cfgm(fixed_lots=1.0, reenter_on_exit_bar=True)
    sg = {"long": sig(14, [0, 2, 4, 6, 8])}
    r, a = acct(df, cfg, sg, "T2")
    s = summarize(a, df.index, engine_settings(cfg), 0.01, r.stats)
    pr = a.trades["profit"].to_numpy()
    np.testing.assert_allclose(pr, [200, -100, 300, -100, -100])
    assert s["total_trades"] == len(pr) and s["long_trades"] == len(pr) and s["short_trades"] == 0
    assert s["total_net_profit"] == pytest.approx(pr.sum()) and s["gross_profit"] == pytest.approx(pr[pr > 0].sum()) and s["gross_loss"] == pytest.approx(-pr[pr < 0].sum())
    assert s["profit_factor"] == pytest.approx(pr[pr > 0].sum() / -pr[pr < 0].sum()) and s["expected_payoff"] == pytest.approx(pr.mean())
    assert s["max_consecutive_losses"] == _max_run(pr < 0) and s["profit_trades_pct"] == pytest.approx(100 * (pr > 0).mean())
    assert s["largest_profit_trade"] == pr.max() and s["largest_loss_trade"] == pr.min() and s["final_balance"] == pytest.approx(10000 + pr.sum())
    assert s["expectancy_R"] == pytest.approx(pr.mean() / 100) and s["exposure_pct"] <= 100 and s["max_concurrent_positions"] == 1
    assert s["recovery_factor"] == pytest.approx(pr.sum() / s["equity_drawdown_abs"]) and s["cost_share_of_gross"] == pytest.approx(0.0) and s["equity_drawdown_abs"] == pytest.approx(200.0)


def test_cost_share_and_exposure_by_hand():
    cfg = cfgm({"fixed_spread_points": 20}, fixed_lots=1.0)
    df = world(RISING, atr=1.0)
    r, a = acct(df, cfg, {"long": sig(12, [0])}, "T3")                                                         # move 3.0, spread 0.2 -> cost share 0.2 / 3.0
    s = summarize(a, df.index, engine_settings(cfg), 0.01, r.stats)
    assert s["cost_share_of_gross"] == pytest.approx(0.2 / 3.0) and s["avg_spread_entry_points"] == pytest.approx(20.0)
    assert s["exposure_pct"] == pytest.approx(100 * 3 / 12) and s["avg_duration_bars"] == 3.0                    # open at the opens of bars 1, 2, 3
    assert s["exit_reasons"] == {"time": 1}


def test_trade_tables_by_year_and_month():
    from edge_research.mt5_backtest.metrics import period_table
    t = pd.DataFrame({"exit_time": pd.to_datetime(["2023-12-30", "2024-01-02", "2024-01-20", "2024-02-01"]), "profit": [10.0, -5.0, 7.0, 3.0], "r_net": [1.0, -0.5, 0.7, 0.3]})
    y = period_table(t, "Y").set_index("period")
    assert y.loc["2024", "trades"] == 3 and y.loc["2024", "net_profit"] == 5.0 and y.loc["2024", "win_rate"] == pytest.approx(2 / 3)
    m = period_table(t, "M")
    assert list(m["period"]) == ["2023-12", "2024-01", "2024-02"] and m["expectancy_R"].iloc[1] == pytest.approx(0.1)
    assert period_table(t.iloc[0:0], "Y").empty


# ------------------------------------------------------------------------------------------------------------ compare_trades
def _tl(n=5):
    idx = pd.date_range("2024-01-01", periods=n * 10, freq="1h")[::10]
    return pd.DataFrame({"entry_time": idx, "exit_time": idx + pd.Timedelta(hours=3), "entry_price": 100.0 + np.arange(n), "exit_price": 101.0 + np.arange(n),
                         "profit": np.arange(n) * 1.5 + 1, "direction": "long"})


def test_compare_trades_perfect_match_and_name_aliases():
    e = _tl()
    mt5 = e.rename(columns={"entry_time": "Open Time".lower().replace(" ", "_"), "exit_time": "close_time", "direction": "type"})
    mt5["type"] = "buy"
    r = compare_trades(e, mt5)
    assert r["accepted"] and r["entry_match_pct"] == 100 and r["exit_match_pct"] == 100 and r["first_difference"] is None and r["n_matched"] == 5 and r["profit_rel_diff"] == 0


def test_compare_trades_reports_every_kind_of_difference():
    e = _tl()
    m = e.copy()
    m.loc[2, "exit_time"] += pd.Timedelta(hours=1)
    r = compare_trades(e, m)
    assert not r["accepted"] and r["exit_match_pct"] == pytest.approx(80.0) and r["first_difference"] == {"entry_time": str(e["entry_time"][2]), "reason": "exit time differs"}
    m = e.iloc[:4].copy()                                                                                      # MT5 has one trade less
    r = compare_trades(e, m)
    assert not r["accepted"] and r["entry_match_pct"] == pytest.approx(80.0) and r["only_engine"] == [str(e["entry_time"][4])] and r["first_difference"]["reason"] == "only in the engine"
    m = e.copy()
    m.loc[1, "entry_price"] += 0.01
    r = compare_trades(e, m, price_tol=1e-4)
    assert not r["accepted"] and r["n_price_mismatch"] == 1 and r["max_abs_entry_price_diff"] == pytest.approx(0.01)
    assert compare_trades(e, m, price_tol=0.05)["accepted"]                                                    # a tolerance for rounding is allowed
    m = e.copy()
    m["profit"] = m["profit"] * 1.05
    r = compare_trades(e, m)
    assert not r["accepted"] and r["profit_rel_diff"] == pytest.approx(0.05 / 1.05, rel=1e-6)
    assert compare_trades(e, m, pnl_rel_tol=0.1)["accepted"]
    m = e.copy()
    m["direction"] = "short"
    assert compare_trades(e, m)["direction_mismatches"] == 5
    d = pd.concat([e, e.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        compare_trades(d, e)
    with pytest.raises(ValueError, match="exit_time"):
        compare_trades(e, e.drop(columns=["exit_time"]))
    assert not compare_trades(e.iloc[0:0], e.iloc[0:0])["accepted"]


# ------------------------------------------------------------------------------------------------------------ rule file
def _rule(**kw):
    d = dict(rule_id="r1", direction="long", atoms=[{"feature": "x", "op": "<=", "threshold": 5.0}, {"feature": "y", "op": ">", "threshold": 0.5}], exit="T6")
    d.update(kw)
    return FrozenRule(**d)


def test_rule_validation_roundtrip_and_unknown_fields(tmp_path):
    r = _rule(cat_stop_atr=3.0, meta={"t": 4.2})
    save_rules([r, _rule(rule_id="r2", exit="B2_1p5_12")], tmp_path / "rules.json")
    back = load_rules(tmp_path / "rules.json")
    assert [x.rule_id for x in back] == ["r1", "r2"] and back[0].cat_stop_atr == 3.0 and back[0].meta == {"t": 4.2} and back[1].exit == "B2_1p5_12"
    assert not back[0].research_only and _rule(exit="I2_1_12").research_only and r.features() == ["x", "y"]
    for bad in (dict(direction="up"), dict(atoms=[]), dict(atoms=[{"feature": "x", "op": "<", "threshold": 1}]), dict(atoms=[{"feature": "x", "op": "<=", "threshold": float("nan")}]),
                dict(atoms=[{"feature": "x", "op": "<=", "threshold": 1, "extra": 1}]), dict(exit="Z9"), dict(cat_stop_atr=0)):
        with pytest.raises(ValueError):
            _rule(**bad).validate()
    d = r.to_dict()
    d["surprise"] = 1
    with pytest.raises(ValueError, match="unknown rule fields"):
        FrozenRule.from_dict(d)
    d = r.to_dict()
    d["version"] = 99
    with pytest.raises(ValueError, match="version"):
        FrozenRule.from_dict(d)
    with pytest.raises(ValueError, match="duplicate"):
        save_rules([r, r], tmp_path / "x.json")


def test_rule_signal_semantics_equal_the_atoms_masks():
    n = 1000
    x = np.r_[np.zeros(500), np.arange(1, 501)].astype(float)
    x[10:15] = np.nan
    cat = np.tile([0.0, 1.0, 2.0, 3.0], n // 4)
    t = pd.DataFrame({"x": x, "cat": cat}, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    a = build_atoms(t, {"x": "numeric", "cat": "time"}, {"x": "x", "cat": "cat"}, ["x", "cat"], load_config(), slice(0, 800))
    assert any(at.op == ">" for at in a.atoms) and any(at.op == "==" for at in a.atoms) and any(at.op == "<=" for at in a.atoms)
    for atom in a.atoms:
        np.testing.assert_array_equal(atom_mask(t[atom.feature].to_numpy(), atom.op, atom.threshold), a.dense(atom), err_msg=atom.expr())
    gt = next(at for at in a.atoms if at.op == ">" and at.feature == "x")
    assert not atom_mask(np.array([np.nan]), ">", 1.0)[0] and not atom_mask(np.array([np.nan]), "<=", 1.0)[0]    # NaN makes every atom False
    r = FrozenRule("a", "long", [{"feature": "x", "op": ">", "threshold": gt.threshold}, {"feature": "cat", "op": "==", "threshold": 2.0}], "T3")
    np.testing.assert_array_equal(rule_signal(r, t), a.dense(gt) & (cat == 2.0))
    with pytest.raises(KeyError, match="missing"):
        rule_signal(FrozenRule("a", "long", [{"feature": "nope", "op": "<=", "threshold": 1.0}], "T3"), t)


def test_exit_condition_signal_and_missing_feature():
    n = 60
    x = np.arange(n).astype(float)
    x[10:15] = np.nan
    t = pd.DataFrame({"x": x}, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    r = FrozenRule("a", "long", [{"feature": "x", "op": "<=", "threshold": 1e9}], "T3",
                   exit_condition=[{"feature": "x", "op": ">", "threshold": 20.0}])
    got = exit_condition_signal(r, t)
    assert got is not None and got.sum() == 39 and not got[10:15].any()        # x > 20: rows 21..59; NaN rows are False like every atom
    assert exit_condition_signal(FrozenRule("a", "long", [{"feature": "x", "op": "<=", "threshold": 1e9}], "T3"), t) is None
    with pytest.raises(KeyError, match="exit-condition"):
        exit_condition_signal(FrozenRule("a", "long", [{"feature": "x", "op": "<=", "threshold": 1e9}], "T3",
                                        exit_condition=[{"feature": "nope", "op": ">", "threshold": 1.0}]), t)


def test_condition_exit_carried_by_a_frozen_rule_end_to_end(tmp_path):
    """Design 15.4 condition exit through the WHOLE frozen-rule path: rule file -> run_backtest -> engine (exit at the open
    of the bar after the condition holds); the generator refuses the class (Q-008)."""
    df = world(RISING, atr=1.0)
    n = len(df)
    ft = pd.DataFrame({"x": df["close"].to_numpy()}, index=df.index)
    entry = [{"feature": "x", "op": "<=", "threshold": 1e9}]                  # always true: one trade, T8 time stop
    cond = [{"feature": "x", "op": ">", "threshold": 103.0}]
    with_cond = run_backtest(df, cfgm(), [_rule(atoms=entry, exit="T8", exit_condition=cond)], ft)
    without = run_backtest(df, cfgm(), [_rule(atoms=entry, exit="T8")], ft)
    t = with_cond.trades.iloc[0]
    assert t["exit_reason"] == "cond" and t["exit_bar"] == 6 and t["hold"] == 5           # close[5] = 104 > 103 -> exit at the open of bar 6
    assert without.trades.iloc[0]["exit_reason"] == "time" and without.trades.iloc[0]["exit_bar"] == 9
    save_rules([_rule(atoms=entry, exit="T8", exit_condition=cond)], tmp_path / "r.json")
    back = load_rules(tmp_path / "r.json")[0]
    assert back.exit_condition == cond                                        # the rule file keeps the condition
    with pytest.raises(ValueError):
        _rule(exit_condition=[{"feature": "x", "op": "<", "threshold": 1.0}]).validate()
    with pytest.raises(ValueError):
        _rule(exit_condition=[{"feature": "x", "op": ">", "threshold": float("nan")}]).validate()
    with pytest.raises(KeyError, match="exit-condition"):
        run_backtest(df, cfgm(), [_rule(atoms=entry, exit="T3", exit_condition=[{"feature": "zz", "op": ">", "threshold": 1.0}])], ft)


def test_engine_fidelity_key_and_per_symbol_tester_overrides(tmp_path):
    """engine.fidelity pins the Strategy Tester mode (open prices only); config/MT5_TESTER/<SYMBOL>/<TF>.yaml overrides
    costs + engine per market (design 18) and is deep-merged over the global config."""
    from edge_research.config import symbol_config, tester_overrides
    assert engine_settings(load_config())["fidelity"] == "open_prices_only"
    with pytest.raises(ValueError):
        engine_settings({"engine": {"fidelity": "every_tick"}})
    with pytest.raises(ValueError):
        load_config(overrides={"engine": {"fidelity": "1m_ohlc"}})
    d = tmp_path / "MT5_TESTER" / "EURUSD"
    d.mkdir(parents=True)
    (d / "H1.yaml").write_text("costs:\n  fixed_spread_points: 7\nengine:\n  max_open_positions: 2\n")
    base = load_config()
    assert tester_overrides("EURUSD", "H1", tmp_path)["costs"]["fixed_spread_points"] == 7
    assert tester_overrides("EURUSD", "M30", tmp_path) == {}                     # missing file: no override
    merged = symbol_config(base, "EURUSD", "H1", base_dir=tmp_path)
    assert merged["costs"]["fixed_spread_points"] == 7 and merged["engine"]["max_open_positions"] == 2
    assert merged["costs"]["point"] == base["costs"]["point"] and merged["engine"]["fidelity"] == "open_prices_only"
    assert symbol_config(base, "EURUSD", "M30", base_dir=tmp_path) is base
    (d / "H1.yaml").write_text("labels:\n  atr_column: other\n")
    with pytest.raises(ValueError, match="costs"):
        tester_overrides("EURUSD", "H1", tmp_path)                               # only costs / engine may be overridden


# ------------------------------------------------------------------------------------------------------------ run_backtest
def test_run_backtest_two_rules_share_one_account_and_write_files(tmp_path):
    df, spread = random_world(5, 500)
    n = len(df)
    ft = pd.DataFrame({"f": (np.arange(n) % 25).astype(float), "g": (np.arange(n) % 40).astype(float)}, index=df.index)
    cfg = cfgm({"fixed_spread_points": 3}, fixed_lots=0.1, reenter_on_exit_bar=True)
    r1 = FrozenRule("A", "long", [{"feature": "f", "op": "<=", "threshold": 1.0}], "B2_1_8")
    r2 = FrozenRule("B", "short", [{"feature": "g", "op": "<=", "threshold": 1.0}], "T6", cat_stop_atr=4.0)
    res = run_backtest(df, cfg, [r1, r2], ft)
    solo = {r.rule_id: run_backtest(df, cfg, [r], ft) for r in (r1, r2)}
    assert set(res.trades["rule_id"]) == {"A", "B"} and len(res.trades) == sum(len(s.trades) for s in solo.values())
    assert res.summary["total_net_profit"] == pytest.approx(sum(s.summary["total_net_profit"] for s in solo.values()))      # fixed lots: the account is additive
    assert res.per_rule["A"]["trades"] == len(solo["A"].trades) and res.summary["max_concurrent_positions"] >= 1
    assert res.portfolio["n_rules"] == 2 and "max_pairwise_correlation" in res.portfolio and res.portfolio["max_simultaneous_lots"] >= 0.1
    assert res.trades["entry_bar"].is_monotonic_increasing
    # restricting the rows removes every signal outside them
    half = run_backtest(df, cfg, [r1], ft, rows=slice(0, 250))
    assert half.trades["signal_bar" if "signal_bar" in half.trades else "entry_bar"].max() < 251 + 1 and 0 < len(half.trades) < len(solo["A"].trades)
    write_backtest(res, tmp_path / "bt", "T")
    cols = pd.read_csv(tmp_path / "bt" / "trades.csv").columns.tolist()
    assert cols[:3] == ["rule_id", "signal_time", "entry_time"] and not any(c.startswith("_") for c in cols) and "lots" in cols
    sm = json.loads((tmp_path / "bt" / "summary.json").read_text())
    assert sm["summary"]["total_trades"] == len(res.trades) and (tmp_path / "bt" / "report.md").read_text().startswith("# T")
    with pytest.raises(ValueError, match="aligned"):
        run_backtest(df, cfg, [r1], ft.iloc[:-1])
    with pytest.raises(ValueError, match="no rules"):
        run_backtest(df, cfg, [], ft)


def test_netting_mode_warns():
    df, _ = random_world(1, 200)
    ft = pd.DataFrame({"f": np.zeros(200)}, index=df.index)
    with pytest.warns(UserWarning, match="netting"):
        run_backtest(df, cfgm(account_mode="netting"), [FrozenRule("A", "long", [{"feature": "f", "op": "<=", "threshold": 1.0}], "T3")], ft)


def test_segment_backtest_statistics_refer_to_the_segment_window():
    df, _ = random_world(5, 600)
    n = len(df)
    ft = pd.DataFrame({"f": (np.arange(n) % 25).astype(float)}, index=df.index)
    cfg = cfgm(fixed_lots=0.1, reenter_on_exit_bar=True)
    rule = FrozenRule("A", "long", [{"feature": "f", "op": "<=", "threshold": 1.0}], "T6")
    seg = run_backtest(df, cfg, [rule], ft, rows=slice(200, 400))
    full = run_backtest(df, cfg, [rule], ft)
    t = seg.trades
    assert t["entry_bar"].min() >= 201 and t["entry_bar"].max() <= 401
    window = 200 + 6 + 2                                                                                      # the segment plus the exits that run past it
    assert seg.summary["longest_drawdown_bars"] <= window and full.summary["longest_drawdown_bars"] >= seg.summary["longest_drawdown_bars"]
    exposed_bars = seg.summary["exposure_pct"] / 100 * seg.summary["trades_per_year"] * 0 + seg.summary["exposure_pct"] / 100
    held = sum(int(h) for h in t["exit_bar"] - t["entry_bar"])                                                 # one position at a time, no overlap: exposed bars = total hold
    n_window = held / exposed_bars if exposed_bars else 0
    assert 200 <= round(n_window) <= window
    assert seg.summary["years"] == pytest.approx((df.index[min(n, 200 + window) - 1] - df.index[200]).total_seconds() / (365.25 * 86400), rel=0.05)


# ------------------------------------------------------------------------------------------------------------ cases added after the mutation check
def test_a_stop_tighter_than_the_spread_is_touched_inside_the_entry_bar_at_the_level():
    # long: entry fill = 100 + 0.5 spread = 100.5; stop = 0.3 below = 100.2 (ABOVE the bid open 100). The entry-bar low touches it: fill at the LEVEL, not at the open.
    from test_engine_m9 import one, cfg0
    h, l = [100.1] * 9, [99.9] * 9
    t = one(world([100] * 9, h, l), "I3_0p3_4", spread=[50] * 9)
    assert t["exit_reason"] == "sl" and t["exit_bar"] == 1 and t["exit_price"] == pytest.approx(100.2)


def test_short_equity_is_marked_at_the_ask_and_recovery_uses_the_equity_drawdown():
    _, a = acct(world([100] * 12, atr=1.0), cfgm({"fixed_spread_points": 20}, fixed_lots=1.0), {"short": sig(12, [0])}, "T3")
    assert a.equity.iloc[1] == pytest.approx(10000 - 20.0)                                                    # short opened at the bid 100, marked at the ask 100.2
    o = [100, 100, 98, 99, 103, 103, 103, 103, 103, 103, 103, 103]                                            # long dips to 98 inside the trade, balance never draws down
    df = world(o, atr=1.0)
    cfg = cfgm(fixed_lots=1.0)
    r, a = acct(df, cfg, {"long": sig(12, [0])}, "T3")
    s = summarize(a, df.index, engine_settings(cfg), 0.01, r.stats)
    assert s["balance_drawdown_abs"] == 0 and s["equity_drawdown_abs"] == pytest.approx(200.0) and s["recovery_factor"] == pytest.approx(300.0 / 200.0)


def test_trimmed_mean_drops_the_floor_of_the_share():
    assert trimmed_mean(np.arange(1, 31, dtype=float), 0.05) == pytest.approx(np.mean(np.arange(1, 30)))      # 1.5 trades -> drop 1, not 2


def test_compare_trades_when_mt5_has_extra_trades():
    e = _tl()
    r = compare_trades(e.iloc[:4], e)
    assert not r["accepted"] and r["n_matched"] == 4 and r["entry_match_pct"] == pytest.approx(80.0) and r["only_mt5"] == [str(e["entry_time"][4])] and r["only_engine"] == []
    assert r["first_difference"]["reason"] == "only in MT5" and r["exit_match_pct"] == 100.0
