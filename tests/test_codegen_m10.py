"""M10: MQL5 generator, EA algorithm model, parity harness (design chapter 16). No MetaTrader in the sandbox: the MQL5 text is checked statically (lint) and through the
recipe / plan strings that are also evaluated in Python; the EA algorithm is checked through ``ea_sim`` against the engine."""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_research.codegen import CodegenRefusal
from edge_research.codegen import ea_sim, parity as P
from edge_research.codegen.indicators import indicator_for
from edge_research.codegen.lint import lint_mql5
from edge_research.codegen.mql_helpers import ALL_HELPERS, helper_closure
from edge_research.codegen.plan import SpecLite, build_plan, magic_for, mql_double
from edge_research.codegen.pyeval import Namespace, eval_expr, plan_features, plan_signal
from edge_research.codegen.recipes import recipe_for
from edge_research.codegen.render import render_ea
from edge_research.codegen.stage import generate_eas
from edge_research.config import load_config
from edge_research.features.registry import build_registry
from edge_research.features.table import build_feature_table, write_features
from edge_research.mt5_backtest.backtest import run_backtest
from edge_research.rules import FrozenRule, rule_signal, save_rules
from edge_research.synth.generator import PlantedEdge, make_synthetic_bars, write_exporter_csv

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
STRONG = PlantedEdge(drop_sigmas=1.0, drift_sigmas=3.0, hours=tuple(range(6, 14)))


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def world(cfg):
    df, meta = make_synthetic_bars(6000, 11, plant_edge=STRONG)
    res = build_feature_table(df, cfg)
    return df, res, {s.name: s for s in res.specs}


def atoms(*a):
    return [{"feature": f, "op": o, "threshold": t} for f, o, t in a]


LONG = atoms(("cci_14", "<=", -48.2), ("candle_body", ">", 0.61))
SHORT = atoms(("cci_14", ">", 80.0), ("candle_body", "<=", -0.2))


def rule(rid="R1", d="long", a=LONG, ex="T6", **kw):
    return FrozenRule(rid, d, a, ex, **kw)


# ------------------------------------------------------------------------------------------------------------ indicators
@pytest.mark.parametrize("col,create,buf", [
    ("rsi_14", "iRSI(_Symbol,TF,14,PRICE_CLOSE)", 0),
    ("cci_20", "iCCI(_Symbol,TF,20,PRICE_TYPICAL)", 0),
    ("atr_14", "iATR(_Symbol,TF,14)", 0),
    ("ma_50_ema", "iMA(_Symbol,TF,50,0,MODE_EMA,PRICE_CLOSE)", 0),
    ("ma_5_sma", "iMA(_Symbol,TF,5,0,MODE_SMA,PRICE_CLOSE)", 0),
    ("stoch_14_3_3_k", "iStochastic(_Symbol,TF,14,3,3,MODE_SMA,STO_LOWHIGH)", 0),
    ("stoch_5_3_3_d", "iStochastic(_Symbol,TF,5,3,3,MODE_SMA,STO_LOWHIGH)", 1),
    ("bb_20_2_upper", "iBands(_Symbol,TF,20,0,2.0,PRICE_CLOSE)", 1),
    ("bb_20_2_middle", "iBands(_Symbol,TF,20,0,2.0,PRICE_CLOSE)", 0),
    ("bb_20_2p5_lower", "iBands(_Symbol,TF,20,0,2.5,PRICE_CLOSE)", 2),
    ("macd_12_26_9_main", "iMACD(_Symbol,TF,12,26,9,PRICE_CLOSE)", 0),
    ("macd_12_26_9_signal", "iMACD(_Symbol,TF,12,26,9,PRICE_CLOSE)", 1),
    ("ao", "iAO(_Symbol,TF)", 0), ("bears_13", "iBearsPower(_Symbol,TF,13)", 0),
    ("mfi_14", "iMFI(_Symbol,TF,14,VOLUME_TICK)", 0),
])
def test_indicator_defaults_by_hand(col, create, buf):
    s = indicator_for(col)
    assert (s.create, s.buffer, s.verified) == (create, buf, False)


@pytest.mark.parametrize("col", ["sar_002_02", "volregime_atr_pctl", "ps_hh", "frama_20_x", "nonsense", "rsi_", "ma_20_weird"])
def test_indicator_without_a_readable_recipe_is_refused(col):
    with pytest.raises(CodegenRefusal, match="no MQL5 recipe"):
        indicator_for(col)


def test_indicator_override_is_used_and_validated():
    s = indicator_for("sar_002_02", {"sar_002_02": {"create": "iSAR(_Symbol,TF,0.02,0.2)", "buffer": 0, "verified": True}})
    assert s.create == "iSAR(_Symbol,TF,0.02,0.2)" and s.verified and s.source == "config override"
    assert indicator_for("rsi_14", {"rsi_14": {"create": "iRSI(_Symbol,TF,14,PRICE_MEDIAN)"}}).create.endswith("PRICE_MEDIAN)")      # override wins over the name pattern
    for bad in ({"create": "Sleep(1)"}, {"create": "iRSI(1)", "bogus": 1}, {"buffer": 0}, "iRSI(1)", {"create": "iX(1); Alert(2)"}, {"create": "iX(1) // c"}, {"create": "iX(1) /* c */"},
                {"create": "iX(1)+iY(2)"}, {"create": 'iX("a")'}, {"create": "iX(1)\nAlert(1)"}, {"create": "iX(1"}, {"create": "iX(1)) ("}, {"create": "iX(1){}"},
                {"create": "iX(1 //)"}, {"create": "iX(1 /* c */)"}, {"create": "iX(1; Alert(2))"}):
        with pytest.raises(CodegenRefusal):
            indicator_for("x", {"x": bad})


# ------------------------------------------------------------------------------------------------------------ recipes
FAKE_COLUMNS = (["open", "high", "low", "close", "tick_volume", "volumes", "atr_14", "force_13", "obv", "ad", "chaikin_3_10", "mom_10", "mom_14", "ao", "ac", "bears_13", "bulls_13",
                 "osma_12_26_9", "macd_12_26_9_main", "macd_12_26_9_signal", "stddev_20", "bb_20_2_upper", "bb_20_2_lower", "bb_20_2_middle", "sar_002_02", "volregime_atr_pctl",
                 "volregime_atr_ratio", "volregime_regime", "ps_hh", "cal_next", "news_x", "dema_20_x", "frama_20_x"]
                + [f"{p}_{n}" for p in ("rsi", "stoch_5_3_3", "wpr", "adx", "adxwilder", "demarker", "mfi", "rvi", "cci", "trix") for n in (7, 14, 20)]
                + [f"ma_{n}_{t}" for n in (5, 10, 20, 50, 100, 200) for t in ("sma", "ema")] + ["stoch_5_3_3_k", "stoch_5_3_3_d", "stoch_14_3_3_k", "stoch_14_3_3_d"])


def test_every_registry_feature_is_either_translated_or_refused_with_a_reason(cfg):
    c = load_config(overrides={"features": {"include_calendar_news": True}})
    specs, _ = build_registry(FAKE_COLUMNS, c)
    assert len(specs) > 80
    supported = refused = 0
    for s in specs:
        try:
            recipe_for(s, c)
            supported += 1
        except CodegenRefusal as e:
            refused += 1
            assert "no MQL5 recipe in the generator" not in str(e), f"{s.name} has neither a recipe nor an explicit refusal"
    assert supported > 50 and refused > 8


def test_recipe_expressions_reproduce_the_python_feature_table(world, cfg):
    df, res, reg = world
    ns = Namespace(df.astype("float64"), cfg)
    n_checked = 0
    for s in res.specs:
        try:
            r = recipe_for(s, cfg)
        except CodegenRefusal:
            continue
        cols = list(r.sources) + [cfg["features"]["atr_column"]]
        v, t = eval_expr(ns, r.expr, cols), res.table[s.name].to_numpy(np.float64)
        assert (np.isnan(v) == np.isnan(t)).all(), f"{s.name}: NaN pattern differs"
        m = ~np.isnan(t)
        assert m.sum() > 1000, s.name
        np.testing.assert_allclose(v[m], t[m], rtol=2e-6, atol=1e-6, err_msg=s.name)
        n_checked += 1
    assert n_checked >= 55


def test_recipe_expressions_use_only_the_documented_vocabulary(world, cfg):
    _, res, _ = world
    allowed = {"O", "H", "L", "C", "ATRN", "DIV", "MAX2", "MIN2", "HHV", "LLV", "HOURV", "DOWV", "SESSIONV", "BRKUP", "BRKDN"}
    for s in res.specs:
        try:
            r = recipe_for(s, cfg)
        except CodegenRefusal:
            continue
        names = set(re.findall(r"([A-Za-z_]\w*)\(", r.expr))
        assert all(n in allowed or n.startswith("IND_") for n in names), (s.name, names)
        assert set(r.helpers) <= set(ALL_HELPERS)
        assert all(f"IND_{c}(" in r.expr for c in r.sources)


def tiny(open_, high, low, close, atr, start="2020-01-06 00:00", freq="h"):
    n = len(close)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "atr_14": atr}, index=pd.date_range(start, periods=n, freq=freq, name="time")).astype("float64")


def test_shift_semantics_and_hand_values(cfg):
    # decision row i reads shift 1 = row i; shift 2 = row i - 1
    df = tiny([1, 2, 3, 4], [2, 3, 4, 5], [0.5, 1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5], [0.5, 0.5, 0.5, 0.0])
    ns = Namespace(df, cfg)
    np.testing.assert_array_equal(np.isnan(eval_expr(ns, "C(2)", [])), [True, False, False, False])
    np.testing.assert_allclose(eval_expr(ns, "C(2)", [])[1:], [1.5, 2.5, 3.5])
    body = eval_expr(ns, "DIV(C(1) - O(1), ATRN(1))", ["atr_14"])
    np.testing.assert_allclose(body[:3], [1.0, 1.0, 1.0])
    assert np.isnan(body[3])                                         # ATR = 0 -> NaN, never a division by zero
    np.testing.assert_allclose(eval_expr(ns, "DIV(C(1) - C(3), ATRN(1))", ["atr_14"])[2], (3.5 - 1.5) / 0.5)
    np.testing.assert_allclose(eval_expr(ns, "HHV(2, 1)", [])[1:], [3, 4, 5])
    np.testing.assert_allclose(eval_expr(ns, "LLV(2, 2)", [])[2:], [0.5, 1.5])          # shifts 2..3 of the decision at row 3 = rows 2, 1... see below
    assert np.isnan(eval_expr(ns, "HHV(3, 1)", [])[1])                                   # history too short


def test_calendar_primitives_by_hand(cfg):
    idx = pd.to_datetime(["2020-01-06 06:00", "2020-01-06 07:00", "2020-01-06 13:00", "2020-01-06 22:00", "2020-01-12 12:00"])   # Mon x4, Sunday
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "atr_14": 1.0}, index=idx.rename("time"))
    ns = Namespace(df, cfg)
    np.testing.assert_array_equal(eval_expr(ns, "HOURV(1)", []), [6, 7, 13, 22, 12])
    np.testing.assert_array_equal(eval_expr(ns, "SESSIONV(1)", []), [0, 1, 2, 3, 1])      # asia 0-6, london 7-12, newyork 13-21, late 22-23
    np.testing.assert_array_equal(eval_expr(ns, "DOWV(1)", []), [0, 0, 0, 0, 6])          # Monday = 0, Sunday = 6
    assert np.isnan(eval_expr(ns, "HOURV(2)", [])[0])
    df0 = tiny([1, 2], [2, 3], [0.0, 1.5], [1.5, 2.5], [0.5, 0.5])
    assert np.isnan(eval_expr(Namespace(df0, cfg), "L(1)", [])[0])           # a price of 0 is missing data, like iLow() beyond the history


def test_breakout_flags_by_hand(cfg):
    high = [11, 12, 13, 12, 15, 11]                                   # prices must be > 0 (the EA treats iLow() == 0 beyond the history as missing)
    low = [10, 11, 12, 11, 14, 10]
    close = [10.5, 11.5, 12.5, 11.5, 16.0, 10.5]
    ns = Namespace(tiny(close, high, low, close, [1] * 6), cfg)
    up = eval_expr(ns, "BRKUP(2)", [])
    assert np.isnan(up[:2]).all() and list(up[2:]) == [1.0, 0.0, 1.0, 0.0]               # by hand: row 2 close 12.5 > max(11, 12); row 3 11.5 < 13; row 4 close 16 > max(13, 12); row 5 10.5 < 15
    dn = eval_expr(ns, "BRKDN(2)", [])
    assert list(dn[2:]) == [0.0, 0.0, 0.0, 1.0]                                          # by hand: only row 5, close 10.5 < min(low 11, 14)


def test_recipe_refusals_name_the_reason(cfg):
    for name, frag in [("atr_ratio", "rolling"), ("bars_since_rsi14_below30", "event"), ("volregime_atr_pctl", "custom"), ("ps_x", "custom"), ("tick_volume_rel", "rolling"), ("cal_x", "calendar")]:
        with pytest.raises(CodegenRefusal, match=frag):
            recipe_for(SpecLite(name, "x"), cfg)
    with pytest.raises(CodegenRefusal, match="live_only"):
        recipe_for(SpecLite("rsi_14", "osc", reproducible_in_tester=False), cfg)
    with pytest.raises(CodegenRefusal, match="no MQL5 recipe in the generator"):
        recipe_for(SpecLite("brand_new_feature", "x"), cfg)


def test_division_and_comparison_semantics_at_the_edges(cfg):
    ns = Namespace(tiny([1.0, 1.0], [1.0, 2.0], [1.0, 1.0], [1.0, 1.5], [1.0, 1.0]), cfg)
    flat = eval_expr(ns, "DIV(C(1) - L(1), H(1) - L(1))", [])                      # row 0 is a flat bar: high == low
    assert np.isnan(flat[0]) and flat[1] == pytest.approx(0.5)
    assert np.isnan(Namespace.DIV(1.0, 0.0)) and np.isnan(Namespace.DIV(np.nan, 2.0)) and np.isnan(Namespace.DIV(np.inf, 2.0)) and Namespace.DIV(6.0, 3.0) == 2.0
    v = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf])
    assert list(Namespace.LE(v, 2.0)) == [True, True, False, False, False, False]        # boundary: equal counts; NaN and +-inf never
    assert list(Namespace.GT(v, 2.0)) == [False, False, True, False, False, False]
    assert list(Namespace.EQ(v, 2.0)) == [False, True, False, False, False, False]
    assert list(Namespace.MAX2(np.array([1.0, np.nan]), np.array([2.0, 2.0]))[:1]) == [2.0] and np.isnan(Namespace.MAX2(np.nan, 2.0)) and np.isnan(Namespace.MIN2(1.0, np.nan))
    assert Namespace.MIN2(1.0, 2.0) == 1.0 and Namespace.MAX2(3.0, 2.0) == 3.0


def test_mql_helper_sources_state_their_semantics():
    """The MQL5 helper bodies cannot be executed here; these checks pin the lines that carry the semantics (shift windows, Monday = 0, NaN guards)."""
    s = {k: v[1] for k, v in ALL_HELPERS.items()}
    assert "if(Bars(_Symbol,TF) < s+n) return QNaN();" in s["HHV"] and "for(int k=s;k<s+n;k++)" in s["HHV"] and "double v=H(k);" in s["HHV"] and "if(v>r) r=v;" in s["HHV"]
    assert "if(Bars(_Symbol,TF) < s+n) return QNaN();" in s["LLV"] and "double v=L(k);" in s["LLV"] and "if(v<r) r=v;" in s["LLV"]
    assert "(d.day_of_week+6)%7" in s["DOWV"] and "return (double)d.hour;" in s["HOURV"] and "iTime(_Symbol,TF,s)" in s["HOURV"]
    assert "HHV(n,2)" in s["BRKUP"] and "c>hh ? 1.0 : 0.0" in s["BRKUP"] and "LLV(n,2)" in s["BRKDN"] and "c<ll ? 1.0 : 0.0" in s["BRKDN"]
    assert "b==0.0" in s["DIV"] and "return a/b;" in s["DIV"] and "MathIsValidNumber" in s["DIV"]
    assert "a>b ? a : b" in s["MAX2"] and "a<b ? a : b" in s["MIN2"]
    assert "v<=t" in s["LE"] and "v>t" in s["GT"] and "v==t" in s["EQ"] and all("MathIsValidNumber(v)" in s[k] for k in ("LE", "GT", "EQ"))
    for k in "OHLC":
        assert "v>0.0 ? v : QNaN()" in s[k]
    assert "MathArcsin(2.0)" in s["QNaN"]


# ------------------------------------------------------------------------------------------------------------ plan
def test_mql_double_round_trips():
    for x in (-48.2, 0.61, 1e-05, 3.0, 123456789.0, 1.5e-12, -2e20, 0.1 + 0.2):
        s = mql_double(x)
        assert float(s) == x and ("." in s)
    assert mql_double(1e-05) == "1.0e-5" and mql_double(3) == "3.0"


def test_plan_of_the_accepted_rule(world, cfg):
    _, _, reg = world
    p = build_plan(rule("R02441", ex="T24"), cfg, reg)
    assert p.feature_names == ["candle_body", "cci_14"]                       # sorted: f[0] candle_body, f[1] cci_14
    assert p.atom_exprs == ["LE(f[1], -48.2)", "GT(f[0], 0.61)"]
    assert [i.column for i in p.indicators] == ["atr_14", "cci_14"]
    assert p.exit == {"kind": "T", "n": 24}
    assert p.settings["magic"] == magic_for("R02441") and 7_000_000 <= p.settings["magic"] < 8_000_000
    assert p.settings["max_gap_hours"] == 6.0 and p.settings["reenter_on_exit_bar"] is False and p.settings["lots"] == 0.1
    assert any("ASSUMED" in w for w in p.warnings)
    assert p.helpers.index("QNaN") < p.helpers.index("O") and "DIV" in p.helpers


def test_plan_settings_follow_the_config(world):
    _, _, reg = world
    c = load_config(overrides={"engine": {"no_weekend_hold": False, "reenter_on_exit_bar": True, "fixed_lots": 0.5, "sizing_mode": "risk_percent"}, "data": {"max_gap_hours": None},
                               "costs": {"slippage_points": 3}})
    p = build_plan(rule(), c, reg)
    assert p.settings["max_gap_hours"] == 0.0 and p.settings["reenter_on_exit_bar"] is True and p.settings["lots"] == 0.5
    assert sum("risk_percent" in w for w in p.warnings) == 1 and sum("slippage" in w for w in p.warnings) == 1
    c2 = load_config(overrides={"data": {"max_gap_hours": 8.0}})
    assert build_plan(rule(), c2, reg).settings["max_gap_hours"] == 8.0
    c3 = load_config(overrides={"engine": {"no_weekend_hold": False}})
    assert build_plan(rule(), c3, reg).settings["max_gap_hours"] == 0.0


def test_plan_refuses_with_every_reason(world, cfg):
    _, _, reg = world
    bad = rule(a=atoms(("atr_ratio", ">", 1.0), ("tick_volume_rel", ">", 2.0), ("not_a_feature", ">", 0.0), ("cci_14", "<=", 0.0)), ex="I1_1p5_12")
    with pytest.raises(CodegenRefusal) as e:
        build_plan(bad, cfg, {**reg, "atr_ratio": SpecLite("atr_ratio"), "tick_volume_rel": SpecLite("tick_volume_rel")})
    txt = " | ".join(e.value.reasons)
    assert len(e.value.reasons) >= 4
    for frag in ("intrabar", "atr_ratio", "tick_volume_rel", "not in the feature registry"):
        assert frag in txt
    with pytest.raises(CodegenRefusal, match="live_only"):
        build_plan(rule(meta={"live_only": True}), cfg, reg)
    with pytest.raises(CodegenRefusal, match="condition exit"):               # Q-008: no EA generation for exit conditions yet
        build_plan(rule(exit_condition=atoms(("cci_14", ">", 0.0))), cfg, reg)
    with pytest.raises(CodegenRefusal, match="max_open_positions"):
        build_plan(rule(), load_config(overrides={"engine": {"max_open_positions": 2}}), reg)
    with pytest.raises(CodegenRefusal, match="timeframe"):
        build_plan(rule(), load_config(overrides={"data": {"timeframe": "H2", "timeframe_minutes": 120}}), reg)
    with pytest.raises(CodegenRefusal, match="disagree"):
        build_plan(rule(), load_config(overrides={"data": {"timeframe": "H4", "timeframe_minutes": 60}}), reg)


def test_plan_source_without_a_recipe_needs_an_override(world, cfg):
    df, res, reg = world
    reg2 = {**reg, "dist_sar_x": SpecLite("dist_sar_x", "price_dist_atr")}
    r = rule(a=atoms(("dist_sar_x", "<=", 0.0)))
    with pytest.raises(CodegenRefusal, match="sar_x"):
        build_plan(r, cfg, reg2)
    c = load_config(overrides={"codegen": {"indicators": {"sar_x": {"create": "iSAR(_Symbol,TF,0.02,0.2)", "buffer": 0}}}})
    p = build_plan(r, c, reg2)
    assert "sar_x" in [i.column for i in p.indicators] and "atr_14" in [i.column for i in p.indicators]


def test_plan_signal_equals_the_python_rule_signal_for_every_atom_type(world, cfg):
    df, res, reg = world
    d64 = df.astype("float64")
    cases = [LONG, SHORT, atoms(("hour", "==", 9.0), ("rsi_14", "<=", 40.0)), atoms(("rsi_14", ">", 30.0), ("rsi_14", "<=", 45.0)),          # a range on one feature
             atoms(("session", "==", 1.0), ("ret_3_atr", "<=", -1.0)), atoms(("breakout_up_20", "==", 1.0)), atoms(("dist_ll_20", "<=", 0.5), ("dayofweek", ">", 1.0))]
    for a in cases:
        r = rule(a=a)
        p = build_plan(r, cfg, reg)
        sig, ref = plan_signal(p, d64, cfg), rule_signal(r, res.table)
        assert ref.sum() > 5, a
        assert (sig != ref).sum() == 0, a


# ------------------------------------------------------------------------------------------------------------ render and lint
def test_rendered_ea_is_clean_ascii_and_contains_the_plan(world, cfg):
    _, _, reg = world
    p = build_plan(rule("R02441", ex="T24", meta={"val_mean_R": 1.73, "val_trades": 92}), cfg, reg, {"data_hash": "abc123"})
    src = render_ea(p)
    assert lint_mql5(src) == [] and src.isascii() and src == render_ea(p)
    for frag in ["const int    EXIT_KIND = 0;", "const int    EXIT_N = 24;", "const int    DIR = 1;", f"input long   InpMagic            = {p.settings['magic']};",
                 "const ENUM_TIMEFRAMES TF = PERIOD_H1;", "return (" + p.atom_exprs[0], "&& " + p.atom_exprs[1], "const int NF = 2;", "NOT FOR REAL MONEY",
                 "val_mean_R", "abc123", "ASSUMED", "iCCI(_Symbol,TF,14,PRICE_TYPICAL)", "IND_atr_14(s)", "REVIEW CHECKLIST"]:
        assert frag in src, frag
    for r in p.features:
        assert f"f[{p.feature_names.index(r.name)}] = {r.expr};" in src
    assert "time,candle_body,cci_14,signal" in src


def test_rendered_exit_parameters_and_direction(world, cfg):
    _, _, reg = world
    s = render_ea(build_plan(rule(d="short", a=SHORT, ex="B1_1p5_12", cat_stop_atr=3.0), cfg, reg))
    for frag in ["const int    DIR = -1;", "const int    EXIT_KIND = 1;", "const double EXIT_TP = 1.0;", "const double EXIT_SL = 1.5;", "const int    EXIT_H = 12;", "const double CAT_K = 3.0;", "catastrophic stop 3.0"]:
        assert frag in s, frag
    assert lint_mql5(s) == []


def test_every_supported_feature_renders_lint_clean(world, cfg):
    df, res, reg = world
    n = 0
    for s in res.specs:
        try:
            recipe_for(s, cfg)
        except CodegenRefusal:
            continue
        src = render_ea(build_plan(rule(a=atoms((s.name, ">", 0.0))), cfg, reg))
        assert lint_mql5(src) == [], s.name
        n += 1
    assert n >= 55


def test_template_exit_entry_and_logging_logic_text(world, cfg):
    """Golden lines of the EA template (the part of the algorithm that ``ea_sim`` models in Python): the model and this text must say the same thing."""
    _, _, reg = world
    src = render_ea(build_plan(rule(), cfg, reg))
    for line in ["if(EXIT_KIND==0) return held>=EXIT_N;", "if(held<1) return false;", "if(held>=EXIT_H) return true;", "double atr0=ATRN(held+1);", "double cl=C(1);",
                 "if(ptype==POSITION_TYPE_BUY) return (cl>=popen+EXIT_TP*atr0) || (cl<=popen-EXIT_SL*atr0);", "return (cl<=popen-EXIT_TP*atr0) || (cl>=popen+EXIT_SL*atr0);",
                 "int held=iBarShift(_Symbol,TF,ptime,false);", "if(held>=0 && ExitDue(held,popen,ptype)) closed_now=ClosePositionRetry(ticket);",
                 "if(bar0==0 || bar0==g_last_bar) return;", "g_last_bar=iTime(_Symbol,TF,0);", "if(closed_now && !InpReenterOnExitBar) return;",
                 "if(FindPosition(ticket,ptime,popen,ptype)) return;", "(double)(iTime(_Symbol,TF,0)-iTime(_Symbol,TF,1))>InpMaxGapHours*3600.0",
                 "if(CAT_K>0.0) sl=NormalizeDouble((DIR>0) ? price-CAT_K*atr : price+CAT_K*atr,_Digits);", "g_trade.Buy(InpLots,_Symbol,0.0,sl,0.0,RULE_ID)", "g_trade.Sell(InpLots,_Symbol,0.0,sl,0.0,RULE_ID)",
                 "double price=(DIR>0) ? SymbolInfoDouble(_Symbol,SYMBOL_ASK) : SymbolInfoDouble(_Symbol,SYMBOL_BID);", "if(trans.type!=TRADE_TRANSACTION_DEAL_ADD) return;",
                 "if(HistoryDealGetInteger(trans.deal,DEAL_ENTRY)!=DEAL_ENTRY_OUT) return;", "if(HistoryDealGetInteger(trans.deal,DEAL_MAGIC)!=InpMagic) return;",
                 "for(int s=s_out;s<=s_in;s++)", "TimeToString(iTime(_Symbol,TF,1),TIME_DATE|TIME_MINUTES)", "if(_Period!=TF)", "if(!MathIsValidNumber(ATRN(1)))",
                 "if(!EvalSignal(f)) return;", "ulong ticket=0; datetime ptime=0; double popen=0.0; long ptype=-1;", "return (MathIsValidNumber(a) && a>0.0) ? a : QNaN();",
                 "TimeToString(iTime(_Symbol,TF,s_in),TIME_DATE|TIME_MINUTES),TimeToString(iTime(_Symbol,TF,s_out),TIME_DATE|TIME_MINUTES),", "if(PositionGetInteger(POSITION_MAGIC)!=InpMagic) continue;", "if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;"]:
        assert line in src, line
    assert src.index("ExitDue(held,popen,ptype)) closed_now") < src.index("if(InpDumpFeatures) DumpRow();") < src.index("TryEnter();                                    // (c) entry")     # exit, dump, entry


def test_log_and_dump_formats_match_the_python_readers(world, cfg):
    _, _, reg = world
    src = render_ea(build_plan(rule(), cfg, reg))
    assert re.search(r'FileWriteString\(g_log,"' + re.escape(",".join(P.LOG_COLUMNS)) + r'\\r\\n"\)', src)
    assert "TIME_DATE|TIME_MINUTES" in src and P.TIME_FORMAT == "%Y.%m.%d %H:%M"
    fmt = re.search(r'StringFormat\("([^"]+)"', src).group(1)
    assert fmt.count("%") == len(P.LOG_COLUMNS)


def test_helper_closure_orders_dependencies():
    c = helper_closure(["BRKUP", "SESSIONV"])
    assert c.index("QNaN") < c.index("H") < c.index("HHV") < c.index("BRKUP") and c.index("HOURV") < c.index("SESSIONV")
    assert helper_closure(["O", "O"]).count("O") == 1
    with pytest.raises(KeyError):
        helper_closure(["nope"])


def test_session_helper_source_follows_the_registry_sessions():
    src = ALL_HELPERS["SESSIONV"][1]
    for i, (lo, hi) in enumerate([(0, 6), (7, 12), (13, 21), (22, 23)]):
        assert f"h>={lo}.0 && h<={hi}.0) return {i}.0;" in src


def test_lint_catches_what_it_is_for():
    good = "double QNaN(){ return 0.0; }\ndouble F(){ return QNaN(); }\nint OnInit(){ return 0; }\n"
    assert lint_mql5(good) == []
    assert any("unbalanced" in i for i in lint_mql5(good + "void g(){ if(true){ }\n"))
    assert any("before its definition" in i for i in lint_mql5("double A(){ return B(); }\ndouble B(){ return 1.0; }\n"))
    assert any("not defined" in i for i in lint_mql5("double A(){ return Missing(); }\n"))
    assert any("unfilled" in i for i in lint_mql5(good + "// @@NAME@@\n"))
    assert any("non-ASCII" in i for i in lint_mql5(good + "// caf\u00e9\n"))
    assert any("defined twice" in i for i in lint_mql5(good + "double QNaN(){ return 1.0; }\n"))
    assert lint_mql5("// Missing()\ndouble A(){ string s=\"Missing(\"; return 1.0; }\n") == []        # comments and strings are ignored


# ------------------------------------------------------------------------------------------------------------ EA model vs engine (trade parity)
def engine_trades(df, cfg, res, r, spread=None):
    return run_backtest(df, ea_sim.parity_cfg(cfg), [r], res.table, spread).trades


def compare(sim, eng):
    return P.compare_trades(eng.assign(profit=0.0), sim.assign(profit=0.0), price_tol=1e-9)


@pytest.mark.parametrize("direction,a", [("long", LONG), ("short", SHORT)])
@pytest.mark.parametrize("ex", ["T1", "T6", "T24", "B1_1p5_12", "B2_1_6", "B3_2_24"])
@pytest.mark.parametrize("reenter", [False, True])
def test_ea_model_matches_the_engine(world, cfg, direction, a, ex, reenter):
    df, res, reg = world
    c = load_config(overrides={"engine": {"reenter_on_exit_bar": reenter}})
    r = rule(d=direction, a=a, ex=ex)
    plan = build_plan(r, c, reg)
    sim = ea_sim.simulate_ea(df, ea_sim.parity_cfg(c), plan)
    eng = engine_trades(df, c, res, r)
    hmax = plan.exit["n"] if plan.exit["kind"] == "T" else plan.exit["h"]
    sim = P.tail_rule(sim, df.index, hmax)
    assert len(eng) >= 20, "test case has too few trades to mean anything"
    out = compare(sim, eng)
    assert out["accepted"] and out["n_matched"] == len(eng) == len(sim), out["first_difference"]


def test_ea_model_matches_the_engine_with_a_spread_series(world, cfg):
    df, res, reg = world
    rng = np.random.default_rng(5)
    spread = pd.Series(rng.integers(5, 60, len(df)).astype("float32"), index=df.index)
    for d, a in (("long", LONG), ("short", SHORT)):
        for ex in ("T6", "B1_1p5_12"):
            r = rule(d=d, a=a, ex=ex)
            plan = build_plan(r, cfg, reg)
            sim = ea_sim.simulate_ea(df, ea_sim.parity_cfg(cfg), plan, spread)
            eng = engine_trades(df, cfg, res, r, spread)
            out = compare(P.tail_rule(sim, df.index, 12 if ex[0] == "B" else 6), eng)
            assert out["accepted"] and out["n_matched"] == len(eng) > 20


def test_ea_model_the_engine_catches_a_wrong_exit_rule(world, cfg):
    """Sensitivity: an EA that exits one bar late is NOT accepted (the comparison can fail)."""
    df, res, reg = world
    r = rule(ex="T6")
    plan = build_plan(r, cfg, reg)
    plan7 = build_plan(rule(ex="T7"), cfg, reg)
    sim = P.tail_rule(ea_sim.simulate_ea(df, ea_sim.parity_cfg(cfg), plan7), df.index, 7)
    out = compare(sim, engine_trades(df, cfg, res, r))
    assert not out["accepted"] and out["exit_match_pct"] < 100.0


def test_ea_model_gap_handling(world, cfg):
    df, res, reg = world
    plan = build_plan(rule(ex="T24"), cfg, reg)
    pc = ea_sim.parity_cfg(cfg)
    gaps = (df.index.to_series().diff().dt.total_seconds() / 3600.0 > 6.0).to_numpy()
    sim = ea_sim.simulate_ea(df, pc, plan, max_gap_hours=6.0)
    assert not gaps[sim["entry_bar"].to_numpy()].any()                     # no entry on the bar right after a gap
    for _, t in sim.iterrows():
        assert t["gap_crossed"] == bool(gaps[t["entry_bar"]:t["exit_bar"] + 1].any())
    assert sim["gap_crossed"].any()                                         # T24 trades do hold over weekends
    forced = np.zeros(len(df), dtype=bool)                                  # a signal on every decision bar before a gap bar: the entry bar is the first bar after the gap
    forced[np.flatnonzero(gaps)[:20] - 1] = True
    with_check = ea_sim.simulate_ea(df, pc, plan, max_gap_hours=6.0, signal=forced)
    without = ea_sim.simulate_ea(df, pc, plan, max_gap_hours=0.0, signal=forced)
    assert len(with_check) == 0 and len(without) == 20 and gaps[without["entry_bar"].to_numpy()].all()
    assert not without["gap_crossed"].any()                                 # the flag needs the check on: with 0 the EA does not look at gaps


def test_parity_cfg_changes_only_what_it_should(cfg):
    pc = ea_sim.parity_cfg(load_config(overrides={"engine": {"cost_multiplier": 2.0}, "costs": {"slippage_points": 4}}))
    assert pc["engine"]["no_weekend_hold"] is False and pc["engine"]["cost_multiplier"] == 1.0 and pc["costs"]["slippage_points"] == 0
    assert cfg["engine"]["no_weekend_hold"] is True                         # the original is untouched


# ------------------------------------------------------------------------------------------------------------ parity checks
def write_dump(path, plan, df, cfg, rounding=10, shift=0, perturb=None):
    f = plan_features(plan, df.astype("float64"), cfg)
    sig = plan_signal(plan, df.astype("float64"), cfg)
    out = pd.DataFrame({k: np.round(v, rounding) for k, v in f.items()}, index=df.index)
    out["signal"] = sig.astype(int)
    if perturb:
        perturb(out)
    if shift:
        out = out.shift(shift).iloc[shift:]
    out.index = out.index.strftime(P.TIME_FORMAT)
    out.index.name = "time"
    out.to_csv(path, float_format="%.10g")
    return path


def test_feature_and_signal_parity_accept_a_faithful_dump(world, cfg, tmp_path):
    df, res, reg = world
    r = rule()
    plan = build_plan(r, cfg, reg)
    dump = P.read_feature_dump(write_dump(tmp_path / "d.csv", plan, df, cfg), plan.feature_names)
    fe = P.check_features(dump, res.table, plan.feature_names, skip_first=300)
    assert fe["accepted"] and fe["n_bars"] == len(df) - 300 and all(v["n_over_tol"] == 0 for v in fe["per_feature"].values())
    sg = P.check_signals(dump, r, res.table, skip_first=300)
    assert sg["accepted"] and sg["n_hard"] == 0 and sg["n_signals_python"] == sg["n_signals_ea"] > 20


def test_parity_detects_a_one_bar_shift(world, cfg, tmp_path):
    """The classic EA bug (values read at shift 0 or 2 instead of 1) must fail both checks."""
    df, res, reg = world
    r = rule()
    plan = build_plan(r, cfg, reg)
    dump = P.read_feature_dump(write_dump(tmp_path / "d.csv", plan, df, cfg, shift=1), plan.feature_names)
    assert not P.check_features(dump, res.table, plan.feature_names, skip_first=300)["accepted"]
    assert not P.check_signals(dump, r, res.table, skip_first=300)["accepted"]


def test_feature_parity_tolerance_and_nan_pattern(world, cfg, tmp_path):
    df, res, reg = world
    plan = build_plan(rule(), cfg, reg)
    rows = np.arange(400, 400 + 30)                                                                    # 30 of 5700 bars = 0.53 %
    def bump(o): o.iloc[rows, o.columns.get_loc("cci_14")] += 0.5
    dump = P.read_feature_dump(write_dump(tmp_path / "a.csv", plan, df, cfg, perturb=bump), plan.feature_names)
    fe = P.check_features(dump, res.table, plan.feature_names, skip_first=300)
    assert fe["per_feature"]["cci_14"]["n_over_tol"] == 30 and not fe["accepted"] and fe["per_feature"]["candle_body"]["ok"]
    assert P.check_features(dump, res.table, plan.feature_names, skip_first=300, max_deviating_pct=1.0)["accepted"]       # the share is a parameter
    def small(o): o.iloc[rows, o.columns.get_loc("cci_14")] += 1e-6                                         # below tol 1e-4 x max(1, |value|)
    d2 = P.read_feature_dump(write_dump(tmp_path / "b.csv", plan, df, cfg, perturb=small), plan.feature_names)
    assert P.check_features(d2, res.table, plan.feature_names, skip_first=300)["accepted"]
    def scaled(o): o.iloc[rows, o.columns.get_loc("cci_14")] += 4e-3                                        # |cci| ~ 50: tolerance is 1e-4 x |value| = ~5e-3, so this is inside
    d4 = P.read_feature_dump(write_dump(tmp_path / "s.csv", plan, df, cfg, perturb=scaled), plan.feature_names)
    f4 = P.check_features(d4, res.table, plan.feature_names, skip_first=300)
    assert f4["per_feature"]["cci_14"]["n_over_tol"] < 30 and f4["per_feature"]["cci_14"]["n_over_tol"] == int((np.abs(res.table["cci_14"].to_numpy()[rows]) * 1e-4 < 4e-3).sum())
    pct = 100.0 * 30 / (len(df) - 300)                                                                       # a share exactly at the limit is accepted
    assert P.check_features(dump, res.table, plan.feature_names, skip_first=300, max_deviating_pct=pct)["accepted"]
    assert not P.check_features(dump, res.table, plan.feature_names, skip_first=300, max_deviating_pct=pct * 0.999)["accepted"]
    def nan(o): o.iloc[rows, o.columns.get_loc("candle_body")] = np.nan
    d3 = P.read_feature_dump(write_dump(tmp_path / "c.csv", plan, df, cfg, perturb=nan), plan.feature_names)
    f3 = P.check_features(d3, res.table, plan.feature_names, skip_first=300)
    assert f3["per_feature"]["candle_body"]["nan_mismatch"] == 30 and not f3["accepted"]


def test_signal_parity_explains_only_near_threshold_differences(world, cfg, tmp_path):
    df, res, reg = world
    r = rule()
    plan = build_plan(r, cfg, reg)
    sig_rows = np.flatnonzero(rule_signal(r, res.table))
    far = sig_rows[sig_rows > 400][0]
    def flip_far(o): o.iloc[far, o.columns.get_loc("signal")] = 0
    d = P.read_feature_dump(write_dump(tmp_path / "f.csv", plan, df, cfg, perturb=flip_far), plan.feature_names)
    s = P.check_signals(d, r, res.table, skip_first=300)
    assert s["n_hard"] == 1 and not s["accepted"] and s["first_hard"]["time"] == str(df.index[far]) and s["first_hard"]["python"] is True
    def flip_near(o):                                       # the EA value sits 1e-6 beyond the threshold and the EA says "no signal": explained
        o.iloc[far, o.columns.get_loc("signal")] = 0
        o.iloc[far, o.columns.get_loc("cci_14")] = -48.2 + 1e-6
    d2 = P.read_feature_dump(write_dump(tmp_path / "n.csv", plan, df, cfg, perturb=flip_near), plan.feature_names)
    s2 = P.check_signals(d2, r, res.table, skip_first=300)
    assert s2["n_differ"] == 1 and s2["n_explained_near_threshold"] == 1 and s2["n_hard"] == 0 and s2["accepted"]


def test_signal_parity_near_threshold_on_the_python_side(world, cfg, tmp_path):
    df, res, reg = world
    r = rule()
    plan = build_plan(r, cfg, reg)
    far = np.flatnonzero(rule_signal(r, res.table))
    far = far[far > 400][0]
    table2 = res.table.copy()
    table2.iloc[far, table2.columns.get_loc("cci_14")] = np.float32(-48.2 + 1e-4)                          # the Python value sits just beyond the threshold: Python says no, the EA (value far below) says yes
    d = P.read_feature_dump(write_dump(tmp_path / "p.csv", plan, df, cfg), plan.feature_names)
    s = P.check_signals(d, r, table2, skip_first=300)
    assert s["n_differ"] == 1 and s["n_explained_near_threshold"] == 1 and s["n_hard"] == 0


def test_dump_reader_rejects_bad_files(world, cfg, tmp_path):
    df, res, reg = world
    plan = build_plan(rule(), cfg, reg)
    p = write_dump(tmp_path / "d.csv", plan, df.iloc[:50], cfg)
    with pytest.raises(ValueError, match="columns"):
        P.read_feature_dump(p, ["cci_14", "candle_body"])                                              # wrong order
    txt = Path(p).read_text().splitlines()
    (tmp_path / "dup.csv").write_text("\n".join(txt + [txt[1]]))
    with pytest.raises(ValueError, match="duplicate"):
        P.read_feature_dump(tmp_path / "dup.csv", plan.feature_names)
    other = res.table.copy()
    other.index = other.index + pd.Timedelta(days=4000)
    with pytest.raises(ValueError, match="share no bar times"):
        P.check_features(P.read_feature_dump(p, plan.feature_names), other, plan.feature_names)


def make_log(sim, volume=0.1):
    log = pd.DataFrame({"rule_id": "R1", "entry_time": sim["entry_time"].dt.strftime("%Y.%m.%d %H:%M:%S"), "exit_time": (sim["exit_time"] + pd.Timedelta(seconds=7)).dt.strftime("%Y.%m.%d %H:%M:%S"),
                        "direction": sim["direction"], "entry_price": sim["entry_price"], "exit_price": sim["exit_price"], "exit_reason": "DEAL_REASON_EXPERT", "volume": volume,
                        "profit": 0.0, "gap_crossed": sim["gap_crossed"].astype(int), "bars_held": sim["exit_bar"] - sim["entry_bar"]})
    return log


def test_trade_parity_end_to_end_and_its_sensitivity(world, cfg, tmp_path):
    df, res, reg = world
    r = rule(ex="B1_1p5_12")
    plan = build_plan(r, cfg, reg)
    sim = ea_sim.simulate_ea(df, ea_sim.parity_cfg(cfg), plan)
    eng = engine_trades(df, cfg, res, r)
    path = tmp_path / "log.csv"
    make_log(sim).to_csv(path, index=False)
    log = P.read_trade_log(path, 60)                                       # seconds are floored to the bar open
    ok = P.check_trades(log, eng, df.index, 12, use_profit=False)
    assert ok["accepted"] and ok["n_matched"] == len(eng) and ok["gap_crossed_in_ea_log"] == 0
    assert not P.check_trades(log.iloc[1:], eng, df.index, 12, use_profit=False)["accepted"]                   # a missing trade
    bad = log.copy(); bad.loc[3, "exit_time"] += pd.Timedelta(hours=1)
    o2 = P.check_trades(bad, eng, df.index, 12, use_profit=False)
    assert not o2["accepted"] and o2["first_difference"]["reason"] == "exit time differs" and o2["first_difference"]["entry_time"] == str(bad.loc[3, "entry_time"])
    bad = log.copy(); bad.loc[5, "entry_price"] += 1e-3
    assert P.check_trades(bad, eng, df.index, 12, use_profit=False)["n_price_mismatch"] == 1
    bad = log.copy(); bad.loc[2, "direction"] = "short"
    assert P.check_trades(bad, eng, df.index, 12, use_profit=False)["accepted"] is False
    with pytest.raises(ValueError, match="columns"):
        (tmp_path / "x.csv").write_text("a,b\n1,2\n"); P.read_trade_log(tmp_path / "x.csv", 60)


def test_trade_parity_period_tail_and_profit(world, cfg):
    df, res, reg = world
    r = rule(ex="T6")
    sim = ea_sim.simulate_ea(df, ea_sim.parity_cfg(cfg), build_plan(r, cfg, reg))
    eng = engine_trades(df, cfg, res, r)
    log = make_log(sim)
    log["entry_time"] = pd.to_datetime(log["entry_time"]); log["exit_time"] = pd.to_datetime(log["exit_time"]).dt.floor("60min")
    log["profit"] = eng["profit"].to_numpy() if len(eng) == len(log) else 0.0
    assert P.check_trades(log, eng, df.index, 6)["accepted"]                                              # profit identical
    off = log.copy(); off["profit"] = off["profit"] * 1.05
    o = P.check_trades(off, eng, df.index, 6)
    assert not o["accepted"] and o["profit_rel_diff"] == pytest.approx(0.05 / 1.05, rel=1e-6) or o["profit_rel_diff"] > 0.01
    assert P.check_trades(off, eng, df.index, 6, use_profit=False)["accepted"]
    t0, t1 = df.index[1000], df.index[3000]
    sub = P.check_trades(log, eng, df.index, 6, period=(t0, t1), use_profit=False)
    assert sub["accepted"] and 0 < sub["n_matched"] < len(eng)
    inside = eng[(eng["entry_time"] >= t0) & (eng["entry_time"] <= t1)]
    assert sub["n_matched"] == len(inside)
    e0, e1 = inside["entry_time"].iloc[0], inside["entry_time"].iloc[-1]                                   # both ends of the period are inclusive
    assert P.check_trades(log, eng, df.index, 6, period=(e0, e1), use_profit=False)["n_matched"] == len(inside)
    extra = pd.concat([log, log.iloc[[-1]].assign(entry_time=df.index[-2], exit_time=df.index[-1])], ignore_index=True)     # a position force-closed by the tester at the end
    assert P.check_trades(extra, eng, df.index, 6)["accepted"]
    assert not P.check_trades(extra, eng, None, None)["accepted"]


def test_tail_rule_boundary():
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    t = pd.DataFrame({"entry_time": idx[[0, 1, 11, 12, 13, 19]]})
    kept = P.tail_rule(t, idx, 6)["entry_time"].tolist()                  # signal_bar = entry - 1 must be <= n - 2 - 6 = 12 -> entry <= 13; an entry at bar 0 has no signal bar
    assert kept == [idx[1], idx[11], idx[12], idx[13]]
    assert P.tail_rule(t.iloc[0:0], idx, 6).empty


def test_g9_verdict_and_report():
    ok = {"accepted": True}
    no = {"accepted": False}
    assert P.g9_verdict(ok, ok, ok)["status"] == "PASS_AUTOMATED"
    assert P.g9_verdict(ok, no, ok) == {**P.g9_verdict(ok, no, ok), "status": "FAIL", "failed": ["signal_parity"]}
    v = P.g9_verdict(None, ok, ok)
    assert v["status"] == "NOT_RUN" and v["not_run"] == ["feature_parity"] and len(v["manual_remaining"]) == 3
    assert P.g9_verdict(no, None, ok)["status"] == "FAIL"                  # a failure beats a missing check
    md = P.parity_md("R1", None, None, None, v)
    assert "NOT_RUN" in md and "stress run" in md


# ------------------------------------------------------------------------------------------------------------ stage 12 and scripts
def make_run(tmp_path, cfg, rules, bars=2500):
    df, _ = make_synthetic_bars(bars, 11, plant_edge=STRONG)
    res = build_feature_table(df, cfg)
    run = tmp_path / "run"
    write_features(res, run / "01_features")
    (run / "10_validation").mkdir(parents=True)
    save_rules(rules, run / "10_validation" / "accepted_rules.json")
    (run / "manifest.json").write_text(json.dumps({"data_hash": "deadbeef", "stages": {"10_validation": {"fingerprint": "fp10"}}, "holdout_accesses": []}))
    return run, df, res


def test_generate_eas_writes_files_and_lists_refusals(tmp_path, cfg):
    rules = [rule("R_ok", ex="T6"), rule("R_ok2", "short", SHORT, "B1_1p5_12"), rule("R_ratio", a=atoms(("atr_ratio", ">", 1.0))), rule("R_intra", ex="I1_1p5_12")]
    run, _, _ = make_run(tmp_path, cfg, rules)
    info = generate_eas(run, cfg)
    out = run / "12_ea"
    assert info["generated"] == ["R_ok", "R_ok2"] and sorted(info["refused"]) == ["R_intra", "R_ratio"] and info["lint_issues"] == {}
    assert (out / "ER_R_ok.mq5").exists() and (out / "ER_R_ok2.plan.json").exists() and not (out / "ER_R_ratio.mq5").exists()
    assert "rolling" in json.loads((out / "refused.json").read_text())["R_ratio"][0]
    assert "Review checklist" in (out / "REVIEW_CHECKLIST.md").read_text()
    plan = json.loads((out / "ER_R_ok.plan.json").read_text())
    assert plan["provenance"]["data_hash"] == "deadbeef" and plan["provenance"]["run"] == "run" and plan["atoms"][0].startswith("LE(")
    assert "deadbeef" in (out / "ER_R_ok.mq5").read_text()
    only = generate_eas(run, cfg, only=["R_ok2"], out_dir=tmp_path / "o2")
    assert only["generated"] == ["R_ok2"] and only["n_rules"] == 1
    reg = json.loads((run / "01_features" / "feature_registry.json").read_text())                  # a registry row flagged not reproducible in the tester refuses the rule
    for f in reg["features"]:
        if f["name"] == "cci_14":
            f["reproducible_in_tester"] = False
    (run / "01_features" / "feature_registry.json").write_text(json.dumps(reg))
    again = generate_eas(run, cfg, out_dir=tmp_path / "o3")
    assert "R_ok" in again["refused"] and any("live_only" in x for x in again["refused"]["R_ok"]) and "R_ok2" in again["refused"]


def test_generate_eas_reports_lint_issues(tmp_path, cfg, monkeypatch):
    from edge_research.codegen import stage
    run, _, _ = make_run(tmp_path, cfg, [rule("R1")])
    monkeypatch.setattr(stage, "render_ea", lambda p: "double A(){ return Missing(); }\n")
    info = generate_eas(run, cfg)
    assert info["generated"] == ["R1"] and "Missing" in info["lint_issues"]["R1"][0]
    assert json.loads((run / "12_ea" / "ea_info.json").read_text())["lint_issues"] == info["lint_issues"]


def test_run_stage_12_is_fingerprinted_and_needs_stage_10(tmp_path, cfg):
    from edge_research import pipeline
    run, _, _ = make_run(tmp_path, cfg, [rule("R1")])
    r1 = pipeline.run_stage_12(run, cfg)
    assert r1["status"] == "ran" and r1["n_generated"] == 1 and r1["n_refused"] == 0
    again = pipeline.run_stage_12(run, cfg)
    assert again == {"status": "skipped", "n_generated": 1, "n_refused": 0}
    assert json.loads((run / "manifest.json").read_text())["stages"]["12_ea"]["n_generated"] == 1
    c2 = load_config(overrides={"codegen": {"indicators": {"x": {"create": "iAO(_Symbol,TF)"}}}})
    assert pipeline.run_stage_12(run, c2)["status"] == "ran"                                      # the codegen section is part of the fingerprint
    c3 = load_config(overrides={"engine": {"fixed_lots": 0.3}})
    assert pipeline.run_stage_12(run, c3)["status"] == "ran"
    m = json.loads((run / "manifest.json").read_text())
    m["stages"].pop("10_validation")
    (run / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(pipeline.MissingUpstream):
        pipeline.run_stage_12(run, c3)


def test_scripts_gen_ea_and_check_parity_end_to_end(tmp_path):
    """The whole G9 harness with the EA model standing in for MetaTrader: CSV -> stages 0-2 -> rules -> gen_ea -> fake EA files -> check_parity."""
    from edge_research.pipeline import run_stages_0_2
    cfg = load_config()
    df, meta = make_synthetic_bars(4000, 11, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "bars.csv")
    run = tmp_path / "run"
    run_stages_0_2(csv, run, cfg)
    (run / "10_validation").mkdir()
    r = rule("R02441", ex="T6")
    save_rules([r], run / "10_validation" / "accepted_rules.json")
    m = json.loads((run / "manifest.json").read_text())
    m["stages"]["10_validation"] = {"fingerprint": "fp"}
    (run / "manifest.json").write_text(json.dumps(m))
    g = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_ea.py"), "--run", str(run)], capture_output=True, text=True, cwd=ROOT)
    assert g.returncode == 0, g.stderr
    assert json.loads(g.stdout)["generated"] == ["R02441"] and (run / "12_ea" / "ER_R02441.mq5").exists()
    # fake MetaTrader files made from the plan evaluation and the EA model
    from edge_research.pipeline import load_run_inputs
    from edge_research.codegen.stage import load_registry
    df2, ft, _, _ = load_run_inputs(run, cfg)
    plan = build_plan(r, cfg, load_registry(run))
    dump = write_dump(tmp_path / "ER_R02441_features.csv", plan, df2, cfg)
    sim = ea_sim.simulate_ea(df2, ea_sim.parity_cfg(cfg), plan)
    make_log(sim).to_csv(tmp_path / "ER_R02441_trades.csv", index=False)
    c = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_parity.py"), "--run", str(run), "--rule", "R02441", "--features", str(dump), "--trades", str(tmp_path / "ER_R02441_trades.csv"),
                        "--no-profit", "--skip-first", "300"], capture_output=True, text=True, cwd=ROOT)
    assert c.returncode == 0, c.stderr
    assert json.loads(c.stdout)["status"] == "PASS_AUTOMATED"
    rep = json.loads((run / "12_ea" / "parity_R02441.json").read_text())
    assert rep["trades"]["n_matched"] > 10 and rep["features"]["accepted"] and rep["signals"]["accepted"]
    assert "PASS_AUTOMATED" in (run / "12_ea" / "parity_R02441.md").read_text()
    # a broken EA run is reported as a failure by the same script
    make_log(sim.iloc[1:]).to_csv(tmp_path / "bad.csv", index=False)
    c2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_parity.py"), "--run", str(run), "--rule", "R02441", "--trades", str(tmp_path / "bad.csv"), "--no-profit"],
                        capture_output=True, text=True, cwd=ROOT)
    assert c2.returncode == 0 and json.loads(c2.stdout)["status"] == "FAIL"
