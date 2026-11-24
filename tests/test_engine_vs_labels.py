import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.labels.builder import build_labels, family_list
from edge_research.mt5_backtest.engine import parse_family, run_engine
from edge_research.synth.generator import make_synthetic_bars


def cfg_small(**over):
    base = {
        "costs": {"point": 0.01, "contract_size": 100, "fixed_spread_points": 10, "commission_per_lot_per_side": 1.0,
                  "swap_long_per_lot": -5, "swap_short_per_lot": 2, "swap_rollover_hour": 0, "swap_triple_weekday": 2},
        "labels": {"time_stops": [3], "barrier_grid": {"tp": [1], "sl": [1], "horizon": [4]}},
    }
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return load_config(overrides=base)


def make_df(n=12, start="2024-01-01 08:00", atr=1.0):
    idx = pd.date_range(start, periods=n, freq="1h")
    o = np.arange(10.0, 10.0 + n)
    return pd.DataFrame({"open": o, "close": o, "atr_14": atr}, index=idx)


def sig(n, bars):
    s = np.zeros(n, dtype=bool)
    s[list(bars)] = True
    return s


def test_parse_family_roundtrips_all_config_families():
    cfg = load_config()
    for name, p in family_list(cfg):
        assert parse_family(name) == p
    with pytest.raises(ValueError):
        parse_family("X9")


def test_overlapping_signals_ignored_and_exit_bar_rule():
    df, cfg = make_df(), cfg_small()
    s = sig(len(df), [0, 1, 2, 3, 4])
    # T3: signal 0 -> enter bar 1, exit bar 4. Signals 1,2 (entry 2,3) overlap; signal 3 enters on the exit bar 4 -> skipped.
    r = run_engine(df, cfg, {"long": s}, "T3")
    assert list(r.trades["signal_bar"]) == [0, 4]
    assert list(r.trades["exit_bar"]) == [4, 8]
    assert r.stats["overlap_skipped"] == 3 and r.stats["trades"] == 2
    r2 = run_engine(df, load_config(overrides={**cfg, "engine": {"reenter_on_exit_bar": True}}), {"long": s}, "T3")
    assert list(r2.trades["signal_bar"]) == [0, 3]          # signal 3 enters at the open of bar 4 = previous exit bar


def test_trade_fields_and_cost_identity():
    df, cfg = make_df(), cfg_small()
    t = run_engine(df, cfg, {"long": sig(len(df), [0])}, "T3").trades.iloc[0]
    assert (t.entry_bar, t.exit_bar, t.hold) == (1, 4, 3)
    assert t.entry_price == 11 and t.exit_price == 14 and t.gross_price == pytest.approx(3.0)
    assert t.cost_price == pytest.approx(0.10 + 0.02)         # long: entry spread 10 pts + commission 2*1/100
    assert t.net_R == pytest.approx(3.0 - 0.12)
    assert t.net_R * t.atr == pytest.approx(t.gross_price - t.cost_price)


def test_long_and_short_conflict_is_skipped_and_other_bars_trade():
    df, cfg = make_df(), cfg_small()
    r = run_engine(df, cfg, {"long": sig(len(df), [0, 5]), "short": sig(len(df), [0])}, "T3")
    assert r.stats["conflict_skipped"] == 1
    assert list(r.trades["signal_bar"]) == [5] and list(r.trades["direction"]) == ["long"]


def test_signals_near_end_are_dropped_and_invalid_windows_do_not_block():
    df, cfg = make_df(), cfg_small()
    r = run_engine(df, cfg, {"long": sig(len(df), [8, 9, 10])}, "T3")       # last usable signal bar = 12 - 2 - 3 = 7
    assert r.trades.empty and r.stats["out_of_data"] == 3
    # A gap larger than max_gap_hours between bar 2 and 3 invalidates the signal at bar 1 (window 1 -> 5) but not the next one.
    df2 = make_df(14)
    df2.index = df2.index.where(np.arange(14) < 3, df2.index + pd.Timedelta(hours=60))
    r2 = run_engine(df2, cfg, {"long": sig(14, [1, 4])}, "T3")
    assert r2.stats["invalid_window"] == 1 and list(r2.trades["signal_bar"]) == [4]


def test_input_validation():
    df, cfg = make_df(), cfg_small()
    with pytest.raises(ValueError):
        run_engine(df, cfg, {"long": np.zeros(3, dtype=bool)}, "T3")
    with pytest.raises(ValueError):
        run_engine(df, cfg, {"up": sig(len(df), [0])}, "T3")
    with pytest.raises(ValueError):
        run_engine(df, cfg, {"long": np.full(len(df), 2.0)}, "T3")
    assert run_engine(df, cfg, {}, "T3").trades.empty


@pytest.mark.parametrize("seed", [1, 2])
def test_engine_matches_labels_trade_by_trade_for_every_family(seed):
    df, _ = make_synthetic_bars(4000, seed=seed)
    cfg = load_config()
    labels = build_labels(df, cfg, None).table
    rng = np.random.default_rng(seed)
    n = len(df)
    for fam, p in family_list(cfg):
        spacing = (p["n"] if p["kind"] == "T" else p["h"]) + 3          # guarantees non-overlapping trades
        bars = np.arange(5, n - 1, spacing)
        for d in ("long", "short"):
            pick = bars[rng.random(len(bars)) < 0.5]
            res = run_engine(df, cfg, {d: sig(n, pick)}, fam)
            t = res.trades
            expected = (labels[f"{d}_{fam}_gross"] - labels[f"{d}_{fam}_cost"]).to_numpy()[pick]
            valid = np.isfinite(expected)
            assert res.stats["overlap_skipped"] == 0
            assert list(t["signal_bar"]) == list(pick[valid]), (fam, d)
            np.testing.assert_allclose(t["net_R"].to_numpy(), expected[valid], rtol=1e-5, atol=1e-5, err_msg=f"{fam} {d}")
            np.testing.assert_allclose(t["hold"].to_numpy(), labels[f"{d}_{fam}_hold"].to_numpy()[pick][valid])
