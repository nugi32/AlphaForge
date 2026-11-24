"""MQL5 loader mode: one exporter CSV -> bars + spread + features for EVERY exported column."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_research.codegen import CodegenRefusal
from edge_research.codegen.indicators import _is_single_call, indicator_for
from edge_research.codegen.pyeval import Namespace, eval_expr
from edge_research.codegen.recipes import recipe_for
from edge_research.config import _deep_merge, load_config
from edge_research.features.table import build_feature_table
from edge_research.mql5_loader.coverage import UnprocessedColumnsError, assert_all_processed, audit_header
from edge_research.mql5_loader.extension import codegen_config
from edge_research.mql5_loader.features import build_extended_registry
from edge_research.mql5_loader.loader import infer_digits, load_mql5_export, parse_export_filename, prepare_table, repair_envelope
from edge_research.mql5_loader.runner import build_config, run_pipeline, timeframe_settings
from edge_research.mql5_loader.sample import write_exporter_csv
from edge_research.mql5_loader.schema import ENVELOPE_COLUMNS, EXPORTER_BY_NAME, EXPORTER_COLUMNS, EXPORTER_HEADER, codegen_overrides, header_report

ROOT = Path(__file__).resolve().parents[1]
EXPORTER_SRC = ROOT / "mql5" / "dataExporter.mq5"


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    return write_exporter_csv(tmp_path_factory.mktemp("mql5") / "EURUSD_H1_export_20261101.csv", 1500, with_news=True)


@pytest.fixture(scope="module")
def table(csv_path):
    return load_mql5_export(csv_path)


def _cfg(table, csv_path, **kw):
    return build_config(table, csv_path, **kw)


# ------------------------------------------------------------------------------------------------------------ manifest vs the MQL5 source
def _source() -> str:
    return EXPORTER_SRC.read_text(encoding="utf-8")


def _norm(call: str) -> str:
    return re.sub(r"\s+", "", call.replace("_Symbol,TF", "g_symbol,g_timeframe"))


def test_manifest_has_the_108_exporter_columns():
    assert len(EXPORTER_COLUMNS) == 108 and len(set(EXPORTER_HEADER)) == 108
    assert EXPORTER_HEADER[:8] == ["time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread"]
    assert EXPORTER_HEADER[-7:][0] == "news_min_to_next" and EXPORTER_HEADER[-1] == "news_count_next_24h"


def test_manifest_handles_and_buffers_match_dataExporter_mq5():
    """Every indicator column: the creating call, its parameters and the buffer READ by the exporter, in exporter order."""
    src = _source()
    handles = dict(re.findall(r"g_handles\.(\w+)\s*=\s*(i[A-Za-z0-9]+\([^;]*\));", src))
    seq = []
    for fn in ("GetMAValues", "GetAdaptiveMAValues", "GetTrendValues", "GetOscillatorValues", "GetVolatilityValues", "GetVolumeBased"):
        body = src[src.index(f"string {fn}("): src.index("return result;", src.index(f"string {fn}("))]
        for m in re.finditer(r"GetIndicatorValue\(g_handles\.(\w+),\s*bar_index,\s*value(?:,\s*(\d+))?\)", body):
            seq.append((m[1], int(m[2] or 0)))
    cols = [c for c in EXPORTER_COLUMNS if c.create is not None]
    assert len(seq) == len(cols) == 81
    for (h, buf), c in zip(seq, cols):
        assert _norm(handles[h]) == _norm(c.create), f"{c.name}: exporter creates {handles[h]}, manifest says {c.create}"
        if c.name in ENVELOPE_COLUMNS:          # the exporter with the bug reads buffers 1 / 2, the exporter with mql5/dataExporter_envelope_fix.patch reads 0 / 1 (the real ones)
            assert buf in (c.exporter_buffer, c.buffer), f"{c.name}: exporter reads buffer {buf}"
        else:
            assert buf == c.exporter_buffer, f"{c.name}: exporter reads buffer {buf}, manifest says {c.exporter_buffer}"


def test_manifest_header_text_matches_the_exporter_header_literals():
    src = _source()
    head = src[src.index("string header = "): src.index("FileWrite(g_file_handle, header);")]
    for lit in ("time,open,high,low,close,tick_volume,real_volume,spread", ",ama_10_2_30", ",sar_002_02,sar_001_02,sar_003_02", ",stddev_20,envelope_20_01_upper,envelope_20_01_lower",
                ",bb_20_15_upper,bb_20_15_middle,bb_20_15_lower", ",obv,ad,mfi_14,force_13,chaikin_3_10,volumes", ",volregime_atr_pctl,volregime_atr_ratio,volregime_regime",
                ",news_min_to_next,news_next_importance"):
        assert lit in head
    assert [c.name for c in EXPORTER_COLUMNS if c.group == "custom_pricestructure"] == re.findall(r"ps_[a-z_]+", head)


def test_envelope_buffers_are_the_documented_mql5_ones():
    up, lo = EXPORTER_BY_NAME["envelope_20_01_upper"], EXPORTER_BY_NAME["envelope_20_01_lower"]
    assert (up.buffer, lo.buffer) == (0, 1)                    # iEnvelopes: UPPER_LINE = 0, LOWER_LINE = 1
    assert (up.exporter_buffer, lo.exporter_buffer) == (1, 2)  # what the exporter reads (the bug)


def test_header_report():
    ok = header_report(EXPORTER_HEADER)
    assert not ok["missing"] and not ok["unlisted"] and not ok["reordered"]
    old = header_report([c for c in EXPORTER_HEADER if c != "spread" and not c.startswith("news_")])
    assert not old["missing_required"] and "spread" in old["missing"]
    bad = header_report([c for c in EXPORTER_HEADER if c != "rsi_14"] + ["new_thing"])
    assert bad["missing_required"] == ["rsi_14"] and bad["unlisted"] == ["new_thing"]


# ------------------------------------------------------------------------------------------------------------ loader
def test_filename_symbol_timeframe_and_digits(csv_path):
    assert parse_export_filename("EURUSD_D1_export_20261101 3.csv") == ("EURUSD", "D1")
    assert parse_export_filename("XAUUSD.m_H4_export_20260101.csv") == ("XAUUSD.m", "H4")
    assert parse_export_filename("whatever.csv") == (None, None)
    assert infer_digits(csv_path) == 5


def test_single_csv_gives_bars_and_spread(table, csv_path):
    assert len(table.df) == 1500 and table.symbol == "EURUSD" and table.timeframe == "H1"        # the exporter's empty last line does not create a row
    assert table.spread is not None and table.spread.index.equals(table.df.index)
    assert np.array_equal(table.spread.to_numpy(), table.df["spread"].to_numpy())
    assert (table.spread.iloc[:40] == 0).all() and (table.spread.iloc[40:] > 0).all()
    assert not table.failed


def test_envelope_repair_exact_and_idempotent(csv_path):
    raw = pd.read_csv(csv_path)
    up, lo = ENVELOPE_COLUMNS
    assert raw[lo].isna().all() and np.allclose((raw[up] / raw["ma_20_sma"]).dropna(), 0.999, atol=2e-6)      # the bug as seen in the real export: "upper" is the lower band
    t = load_mql5_export(csv_path)
    ma = t.df["ma_20_sma"].to_numpy(np.float64)
    ok = np.isfinite(ma)
    assert np.allclose(t.df[up].to_numpy(np.float64)[ok] / ma[ok], 1.001, atol=2e-6) and np.allclose(t.df[lo].to_numpy(np.float64)[ok] / ma[ok], 0.999, atol=2e-6)
    assert len(t.repairs) == 1
    again, rep = repair_envelope(t.df)
    assert rep == [] and again[up].equals(t.df[up])                                              # idempotent: second pass changes nothing
    again2, _, rep2 = prepare_table(t.df)
    assert rep2 == [] and again2.equals(t.df)


def test_envelope_repair_leaves_a_fixed_exporter_alone(table):
    fixed = table.df.copy()                                   # an exporter that reads buffers 0 / 1 correctly: both bands present
    out, rep = repair_envelope(fixed)
    assert rep == [] and out.equals(fixed)


def test_integrity_findings_are_clean_on_a_faithful_export(table):
    bad = [f for f in table.findings if f.level in ("warn", "fail") and f.check != "news"]
    assert not bad, bad


def test_integrity_detects_wrong_band_parameters(table):
    df = table.df.copy()
    df["bb_20_15_upper"] = df["bb_20_2_upper"]                # a handle created with deviation 2.0 instead of 1.5
    _, findings, _ = prepare_table(df)
    assert any(f.check == "bb_20_15_deviation" and f.level == "warn" for f in findings)


# ------------------------------------------------------------------------------------------------------------ coverage
def test_stock_registry_leaves_columns_unread_and_loader_mode_reads_all(table):
    cfg = load_config()
    cfg["features"]["include_calendar_news"] = True
    from edge_research.features.registry import build_registry
    stock, _ = build_registry(list(table.df.columns), cfg)
    stock_read = {s for sp in stock for s in sp.sources}
    unread = {c for c in EXPORTER_HEADER if c != "time" and c not in stock_read}
    assert {"spread", "real_volume", "ama_10_2_30", "atr_20", "bb_20_2_middle", "bb_20_15_middle", "bb_20_25_middle"} <= unread      # the failure this module fixes
    ext, _ = build_extended_registry(list(table.df.columns), cfg)
    ext_read = {s for sp in ext for s in sp.sources}
    assert [c for c in EXPORTER_HEADER if c != "time" and c not in ext_read] == []


def test_audit_all_processed_and_guard(csv_path, table):
    cfg = _cfg(table, csv_path)
    audit = audit_header(["time"] + list(table.df.columns), cfg)
    assert_all_processed(audit)
    assert {r.status for r in audit.rows} == {"processed", "exempt"}
    with pytest.raises(UnprocessedColumnsError, match="brand_new_indicator"):
        assert_all_processed(audit_header(["time"] + list(table.df.columns) + ["brand_new_indicator"], cfg))


def test_gated_news_is_reported_as_gated_not_unprocessed(csv_path, table):
    cfg = _cfg(table, csv_path, gate_news=True)
    audit = audit_header(["time"] + list(table.df.columns), cfg)
    assert_all_processed(audit)
    assert {r.column for r in audit.rows if r.status == "gated"} == {c.name for c in EXPORTER_COLUMNS if c.group == "news"}


# ------------------------------------------------------------------------------------------------------------ features + recipes
@pytest.fixture(scope="module")
def feats(table, csv_path):
    cfg = _cfg(table, csv_path)
    df, _, _ = prepare_table(table.df)
    return cfg, df, build_feature_table(df, cfg)


def test_extension_features_are_built_and_finite(feats):
    cfg, df, res = feats
    names = {s.name for s in res.specs}
    assert {"spread_atr", "spread_rel", "real_volume_rel", "atr_ratio_14_20", "dist_ama_10_2_30", "bb_mid_dist_bb_20_2", "bb_mid_dist_bb_20_15", "bb_mid_dist_bb_20_25",
            "env_pctb_envelope_20_01", "env_width_envelope_20_01"} <= names
    t = res.table
    x = (df["close"].to_numpy(np.float64) - df["ama_10_2_30"].to_numpy(np.float64)) / df["atr_14"].to_numpy(np.float64)
    assert np.allclose(t["dist_ama_10_2_30"].to_numpy(np.float64), x, atol=1e-4, equal_nan=True)
    sp = df["spread"].to_numpy(np.float64)
    expect = np.where(sp > 0, sp * cfg["costs"]["point"] / df["atr_14"].to_numpy(np.float64), np.nan)
    assert np.allclose(t["spread_atr"].to_numpy(np.float64), expect, rtol=1e-4, equal_nan=True)
    assert t["spread_atr"].iloc[:40].isna().all()                                               # spread 0 = unknown, never "free"
    pct = t["env_pctb_envelope_20_01"].dropna()
    assert len(pct) > 1000 and pct.min() < 0 < 1 < pct.max() + 1                                 # the repaired band gives a meaningful %B (before: all NaN)
    assert (t["env_width_envelope_20_01"].dropna() > 0).all()


def test_every_loader_feature_has_a_recipe_or_an_explicit_refusal(feats):
    cfg, _, res = feats
    refused = {}
    for s in res.specs:
        try:
            recipe_for(s, cfg)
        except CodegenRefusal as e:
            refused[s.name] = str(e)
    assert all(len(v) > 20 for v in refused.values())
    for ours in ("atr_ratio_14_20", "dist_ama_10_2_30", "bb_mid_dist_bb_20_2", "env_pctb_envelope_20_01", "env_width_envelope_20_01"):
        assert ours not in refused
    for n in ("spread_atr", "spread_rel", "real_volume_rel"):
        assert n in refused                                                                      # honest: no approximate translation


@pytest.mark.parametrize("name", ["atr_ratio_14_20", "dist_ama_10_2_30", "bb_mid_dist_bb_20_2", "bb_mid_dist_bb_20_15", "bb_mid_dist_bb_20_25", "env_pctb_envelope_20_01", "env_width_envelope_20_01"])
def test_recipe_parity_with_the_feature_table(feats, name):
    """The MQL5 expression, evaluated by pyeval on the loader table, equals the Python feature (the parity the EA must reproduce)."""
    cfg, df, res = feats
    spec = next(s for s in res.specs if s.name == name)
    r = recipe_for(spec, cfg)
    got = eval_expr(Namespace(df, cfg), r.expr, list(r.sources))
    want = res.table[name].to_numpy(np.float64)
    ok = np.isfinite(want)
    assert ok.sum() > 1000 and np.array_equal(ok, np.isfinite(got))
    assert np.allclose(got[ok], want[ok], atol=2e-4, rtol=1e-4)


# ------------------------------------------------------------------------------------------------------------ code generator handles
def test_band_deviation_tokens_of_the_current_exporter():
    for col, dev in (("bb_20_2_upper", "2.0"), ("bb_20_15_upper", "1.5"), ("bb_20_25_lower", "2.5"), ("bb_20_2p5_lower", "2.5")):
        assert f",0,{dev},PRICE_CLOSE" in indicator_for(col).create


def test_codegen_overrides_cover_every_indicator_column_and_are_valid():
    ov = codegen_overrides()
    assert len(ov) == 81 and codegen_config()["codegen"]["indicators"] == ov
    for col, o in ov.items():
        assert _is_single_call(o["create"]), col
        s = indicator_for(col, ov)
        assert s.verified and s.buffer == o["buffer"]
    assert indicator_for("envelope_20_01_upper", ov).buffer == 0 and indicator_for("envelope_20_01_lower", ov).buffer == 1       # corrected buffers
    assert indicator_for("vidya_9", ov).create == "iVIDyA(_Symbol,TF,9,12,0,PRICE_CLOSE)"
    assert indicator_for("ama_10_2_30", ov).create == "iAMA(_Symbol,TF,10,2,30,0,PRICE_CLOSE)"
    assert indicator_for("bb_20_15_lower", ov).create == "iBands(_Symbol,TF,20,0,1.5,PRICE_CLOSE)"


# ------------------------------------------------------------------------------------------------------------ runner / settings
def test_timeframe_settings():
    assert timeframe_settings("H1")["qa"]["expected_bars_per_year"] == 6000 and timeframe_settings("H1")["data"]["max_gap_hours"] == 6.0
    assert timeframe_settings("D1")["data"] == {"timeframe": "D1", "timeframe_minutes": 1440, "max_gap_hours": 96.0}
    assert timeframe_settings("M15")["qa"]["expected_bars_per_year"] == 24000
    assert timeframe_settings("H4")["data"]["max_gap_hours"] == 6.0


def test_config_is_derived_from_the_file(table, csv_path):
    cfg = _cfg(table, csv_path)
    assert cfg["data"]["source"] == "mql5_loader" and cfg["data"]["timeframe"] == "H1" and cfg["costs"]["digits"] == 5 and cfg["costs"]["point"] == pytest.approx(1e-5)
    assert cfg["costs"]["fixed_spread_points"] == pytest.approx(float(np.median(table.spread[table.spread > 0])))
    assert len(cfg["codegen"]["indicators"]) == 81 and cfg["features"]["include_calendar_news"] is True
    assert _cfg(table, csv_path, timeframe="H4")["data"]["timeframe_minutes"] == 240
    assert _cfg(table, csv_path, overrides={"costs": {"digits": 3, "point": 0.001}})["costs"]["digits"] == 3                 # explicit override wins


def test_qa_news_nan_tolerance_is_per_prefix():
    from edge_research.qa.ingest_qa import QAReport, check_columns
    idx = pd.date_range("2020-01-01", periods=1000, freq="h")
    df = pd.DataFrame({"news_x": np.where(np.arange(1000) % 10 == 0, np.nan, 1.0) + np.arange(1000) * 1e-3, "rsi_14": np.where(np.arange(1000) % 10 == 0, np.nan, 1.0) + np.arange(1000) * 1e-3}, index=idx)
    rep = QAReport()
    check_columns(df, rep, 0.01, {"news_": 0.5})
    assert "rsi_14" in rep.excluded_columns and "news_x" not in rep.excluded_columns


# ------------------------------------------------------------------------------------------------------------ pipeline (one csv, no spread file)
def test_pipeline_stages_0_2_from_one_csv(tmp_path, csv_path):
    out = run_pipeline(csv_path, tmp_path / "run", through=2)
    assert out["pipeline"]["status"] == {"00_data_qa": "ran", "01_features": "ran", "02_labels": "ran"}
    run = tmp_path / "run"
    cov = pd.read_csv(run / "01_features" / "column_coverage.csv")
    assert "unprocessed" not in set(cov["verdict"])
    assert set(cov.loc[cov["verdict"] == "qa_excluded", "column"]) == {"real_volume"}
    assert set(cov.loc[cov["column"].isin(["spread", "ama_10_2_30", "atr_20"]), "verdict"]) == {"used"}
    import json
    spec = json.loads((run / "02_labels" / "label_spec.json").read_text())
    assert spec["spread_fallback_bars"] == 40                                                   # the 40 zero-spread rows only: every other bar used ITS spread from the data
    assert spec["costs"]["digits"] == 5
    assert (run / "00_data_qa" / "excluded_columns.json").exists() and (run / "_ingest" / "mql5_loader_report.json").exists()
    from edge_research.config import load_config as _lc
    from edge_research.pipeline import load_run_inputs
    df, ft, spread, splits = load_run_inputs(run, json.loads((run / "config_used.json").read_text()))
    assert spread is not None and (spread.iloc[:40] == 0).all() and df["envelope_20_01_lower"].notna().any()      # later stages see the repaired table and the same spread


def test_short_pipeline_warns_and_skips_split_dependent_feature_work(tmp_path, caplog):
    csv = write_exporter_csv(tmp_path / "EURUSD_H1_export_short.csv", 233, with_news=True)
    raw = pd.read_csv(csv)
    news = [c for c in raw.columns if c.startswith("news_")]
    raw[news] = np.nan
    raw.to_csv(csv, index=False)

    out = run_pipeline(csv, tmp_path / "short_run", through=2)
    run = tmp_path / "short_run"

    assert out["pipeline"]["status"] == {"00_data_qa": "ran", "01_features": "ran", "02_labels": "ran"}
    assert "Skipping feature leakcheck and correlation clustering" in caplog.text
    leakcheck = json.loads((run / "01_features" / "leakcheck.json").read_text())
    assert leakcheck["status"] == "skipped"
    coverage = pd.read_csv(run / "01_features" / "column_coverage.csv")
    assert set(coverage.loc[coverage["column"].isin(news), "verdict"]) == {"qa_excluded"}
    registry = json.loads((run / "01_features" / "feature_registry.json").read_text())
    assert not any(feature["name"].startswith("news_") for feature in registry["features"])


def test_pipeline_stops_on_an_unprocessed_column(tmp_path, csv_path):
    text = Path(csv_path).read_text().splitlines()
    mod = tmp_path / "EURUSD_H1_export_20261101.csv"
    mod.write_text("\n".join([text[0] + ",brand_new_indicator"] + [ln + ",1.5" for ln in text[1:] if ln]) + "\n")
    with pytest.raises(UnprocessedColumnsError, match="brand_new_indicator"):
        run_pipeline(mod, tmp_path / "run", through=2)
    out = run_pipeline(mod, tmp_path / "run2", through=2, allow_unprocessed=True)
    assert out["pipeline"]["n_features"] > 100


def test_stock_mode_is_unchanged(table, csv_path):
    cfg = load_config()                                                                          # no data.source: the stock registry, no extension features
    df, _, _ = prepare_table(table.df)
    res = build_feature_table(df, cfg)
    assert "spread_atr" not in {s.name for s in res.specs}
