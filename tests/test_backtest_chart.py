"""Equity + balance PNG written by every backtest, and the per-pair ``backtest:`` config section."""

import numpy as np
import pandas as pd
import pytest

from edge_research import config as C
from edge_research.config import load_config, symbol_config, validate_config
from edge_research.mt5_backtest.backtest import run_backtest, write_backtest
from edge_research.mt5_backtest.plots import active_window, chart_settings
from edge_research.rules import FrozenRule
from test_backtest_m9 import cfgm
from test_engine_m9 import random_world

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _result(cfg=None):
    df, _ = random_world(5, 500)
    ft = pd.DataFrame({"f": (np.arange(len(df)) % 25).astype(float)}, index=df.index)
    cfg = cfg or cfgm({"fixed_spread_points": 3}, fixed_lots=0.1)
    rule = FrozenRule("A", "long", [{"feature": "f", "op": "<=", "threshold": 1.0}], "B2_1_8")
    return run_backtest(df, cfg, [rule], ft, rows=slice(100, 300)), cfg


def test_write_backtest_writes_png_and_report_embeds_it(tmp_path):
    res, cfg = _result()
    files = write_backtest(res, tmp_path / "bt", "T - EURUSD H1", cfg)
    png = tmp_path / "bt" / "equity_balance.png"
    assert files["chart"] == png and png.read_bytes()[:8] == PNG_MAGIC and png.stat().st_size > 5_000
    md = (tmp_path / "bt" / "report.md").read_text()
    assert md.startswith("# T - EURUSD H1") and "![Equity and balance](equity_balance.png)" in md
    assert (tmp_path / "bt" / "trades.csv").exists() and (tmp_path / "bt" / "equity.csv").exists()


def test_chart_can_be_switched_off_and_customised_per_run(tmp_path):
    res, cfg = _result()
    off = load_config(overrides={**cfg, "backtest": {"chart": {"enabled": False}}})
    files = write_backtest(res, tmp_path / "off", "T", off)
    assert files["chart"] is None and not list((tmp_path / "off").glob("*.png")) and "![" not in (tmp_path / "off" / "report.md").read_text()
    cus = load_config(overrides={**cfg, "backtest": {"chart": {"filename": "eq.png", "drawdown_panel": False, "dpi": 60}, "account_currency": "USD"}})
    assert write_backtest(res, tmp_path / "c", "T", cus)["chart"].name == "eq.png"


def test_write_backtest_without_cfg_still_draws_the_default_chart(tmp_path):
    res, _ = _result()
    assert write_backtest(res, tmp_path / "d", "T")["chart"].read_bytes()[:8] == PNG_MAGIC          # old call signature keeps working


def test_crop_to_trades_cuts_the_flat_stretch():
    res, _ = _result()
    sl = active_window(res.trades, len(res.equity))
    assert sl.start >= 99 and sl.stop <= 330 and sl.stop - sl.start < len(res.equity)
    assert active_window(res.trades.iloc[0:0], 500) == slice(0, 500)


def test_backtest_section_validation_and_per_pair_override(tmp_path):
    assert chart_settings(load_config())["filename"] == "equity_balance.png"
    with pytest.raises(ValueError, match="unknown backtest.chart"):
        load_config(overrides={"backtest": {"chart": {"colour": "red"}}})
    with pytest.raises(ValueError, match="unknown backtest settings"):
        load_config(overrides={"backtest": {"png": True}})
    with pytest.raises(ValueError, match="holdout"):
        load_config(overrides={"backtest": {"segments": ["train", "holdout"]}})
    d = tmp_path / "MT5_TESTER" / "XAUUSD"
    d.mkdir(parents=True)
    (d / "H1.yaml").write_text("costs:\n  point: 0.01\nbacktest:\n  segments: [val]\n  chart:\n    height: 9.0\n")
    base = load_config()
    merged = symbol_config(base, "XAUUSD", "H1", base_dir=tmp_path)
    assert merged["backtest"]["segments"] == ["val"] and merged["backtest"]["chart"]["height"] == 9.0 and merged["backtest"]["chart"]["dpi"] == 150
    validate_config(merged)
    assert C.tester_overrides("EURUSD", "H1", tmp_path) == {}
    (d / "H1.yaml").write_text("labels:\n  atr_column: x\n")
    with pytest.raises(ValueError, match="backtest"):
        C.tester_overrides("XAUUSD", "H1", tmp_path)


def test_shipped_template_and_examples_are_valid_per_pair_files():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1] / "config" / "MT5_TESTER"
    for f in [root / "_TEMPLATE.yaml", *sorted((root / "_examples").glob("*.yaml"))]:
        over = yaml.safe_load(f.read_text())
        assert set(over) <= {"costs", "engine", "backtest"}, f
        cfg = load_config(overrides=over)                      # engine keys / fidelity / chart keys are validated here
        from edge_research.mt5_backtest.engine import engine_settings
        engine_settings(cfg)
