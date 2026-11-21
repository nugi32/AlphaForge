"""Mutation checks for M10 (same harness as dev_mutate_m8.py). Each mutant edits ONE place of a scratch copy (/tmp/mut10); tests/test_codegen_m10.py must FAIL (caught).
Resumable: results in output/mut10_results.json. Run: python scripts/dev_mutate_m10.py [name-prefix]"""
import json, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
DST = Path("/tmp/mut10")
RES = SRC / "output" / "mut10_results.json"
C = "edge_research/codegen/"

M = [
    # name, file, old, new
    ("I01_cci_price", C + "indicators.py", "PRICE_TYPICAL", "PRICE_CLOSE"),
    ("I02_bb_buffers_swapped", C + "indicators.py", '{"middle": 0, "upper": 1, "lower": 2}', '{"middle": 0, "upper": 2, "lower": 1}'),
    ("I03_stoch_buffer", C + "indicators.py", '0 if m[4] == "k" else 1', '0 if m[4] == "d" else 1'),
    ("I04_override_ignored", C + "indicators.py", "if column in overrides:", "if False:"),
    ("I05_no_call_validation", C + "indicators.py", 'if not _is_single_call(str(o["create"]).strip()):', "if False:"),
    ("I06_comments_allowed", C + "indicators.py", ' or "//" in text or "/*" in text:', ":"),
    ("I07_semicolon_allowed", C + "indicators.py", "_FORBIDDEN_CHARS = set(';{}", "_FORBIDDEN_CHARS = set('{}"),
    ("I08_depth_check", C + "indicators.py", "if depth < 0 or (opened and depth == 0 and k < len(text) - 1):", "if depth < 0:"),
    ("I09_ma_mode", C + "indicators.py", '"ema": "MODE_EMA"', '"ema": "MODE_SMA"'),
    ("I10_extra_keys_allowed", C + "indicators.py", 'set(o) - {"create", "buffer", "verified"} or ', ""),
    ("R01_slope_shift", C + "recipes.py", "IND_{m[1]}({1 + k})", "IND_{m[1]}({k})"),
    ("R02_ret_shift", C + "recipes.py", "C({1 + int(m[1])})", "C({int(m[1])})"),
    ("R03_gap_shift", C + "recipes.py", '"gap_atr": f"DIV(O(1) - C(2), {A1})"', '"gap_atr": f"DIV(O(1) - C(1), {A1})"'),
    ("R04_wick_max_min", C + "recipes.py", "H(1) - MAX2(O(1), C(1))", "H(1) - MIN2(O(1), C(1))"),
    ("R05_hh_start", C + "recipes.py", "HHV({m[1]}, 1)", "HHV({m[1]}, 2)"),
    ("R06_pctb_swapped", C + "recipes.py", "DIV(C(1) - IND_{lo}(1), IND_{up}(1) - IND_{lo}(1))", "DIV(C(1) - IND_{up}(1), IND_{up}(1) - IND_{lo}(1))"),
    ("R07_macd_hist_sign", C + "recipes.py", "DIV(IND_macd_12_26_9_main(1) - IND_macd_12_26_9_signal(1)", "DIV(IND_macd_12_26_9_signal(1) - IND_macd_12_26_9_main(1)"),
    ("R08_live_only_check", C + "recipes.py", "if not spec.reproducible_in_tester:", "if False:"),
    ("R09_refusals_off", C + "recipes.py", "    if why:\n        raise", "    if False:\n        raise"),
    ("R10_mom_center", C + "recipes.py", "- {float(f['mom_center'])!r}", "- 0.0"),
    ("Y01_shift", C + "pyeval.py", "    k = s - 1\n", "    k = s\n"),
    ("Y02_div_zero", C + "pyeval.py", "& (b != 0.0)", ""),
    ("Y03_session_bounds", C + "pyeval.py", "(h >= lo) & (h <= hi)", "(h > lo) & (h <= hi)"),
    ("Y04_atr_positive", C + "pyeval.py", "return np.where(a > 0.0, a, np.nan)", "return np.where(a >= 0.0, a, np.nan)"),
    ("Y05_le_strict", C + "pyeval.py", "return np.isfinite(v) & (v <= t)", "return np.isfinite(v) & (v < t)"),
    ("Y06_gt_ge", C + "pyeval.py", "return np.isfinite(v) & (v > t)", "return np.isfinite(v) & (v >= t)"),
    ("Y07_hhv_min", C + "pyeval.py", "rolling(int(n)).max()", "rolling(int(n)).min()"),
    ("Y08_price_positive", C + "pyeval.py", "return np.where(v > 0.0, v, np.nan)", "return np.where(v >= 0.0, v, np.nan)"),
    ("Y09_brk_window", C + "pyeval.py", "hh, c = self.HHV(n, 2), self.C(1)", "hh, c = self.HHV(n, 1), self.C(1)"),
    ("Y10_dow", C + "pyeval.py", "self.df.index.dayofweek.to_numpy()", "((self.df.index.dayofweek.to_numpy() + 1) % 7)"),
    ("Y11_signal_or", C + "pyeval.py", "sig &= np.broadcast_to", "sig |= np.broadcast_to"),
    ("L01_atom_ops", C + "plan.py", '{"<=": "LE", ">": "GT", "==": "EQ"}', '{"<=": "GT", ">": "LE", "==": "EQ"}'),
    ("L02_gap_always", C + "plan.py", 'if (mg is not None and bool(eng["no_weekend_hold"])) else 0.0', "if mg is not None else 0.0"),
    ("L03_reenter", C + "plan.py", '"reenter_on_exit_bar": bool(eng["reenter_on_exit_bar"])', '"reenter_on_exit_bar": False'),
    ("L04_max_open", C + "plan.py", 'if int(eng["max_open_positions"]) != 1:', 'if int(eng["max_open_positions"]) > 5:'),
    ("L05_research_only", C + "plan.py", "    if rule.research_only:", "    if False:"),
    ("L06_live_only_meta", C + "plan.py", 'if rule.meta.get("live_only"):', "if False:"),
    ("L07_refusal_off", C + "plan.py", "    if reasons:\n        raise CodegenRefusal(reasons)", "    if False:\n        raise CodegenRefusal(reasons)"),
    ("L09_unverified_warning", C + "plan.py", "    if unverified:", "    if False:"),
    ("L10_tf_minutes", C + "plan.py", "!= TIMEFRAMES[tf]:", "!= 0:"),
    ("L12_feature_order", C + "plan.py", "names = rule.features()", "names = list(reversed(rule.features()))"),
    ("L13_lots", C + "plan.py", '"lots": float(eng["fixed_lots"])', '"lots": 0.1'),
    ("N01_direction", C + "render.py", '"1" if s["direction"] == "long" else "-1"', '"-1" if s["direction"] == "long" else "1"'),
    ("N02_exit_kind", C + "render.py", 'kind = 0 if ex["kind"] == "T" else 1', 'kind = 1 if ex["kind"] == "T" else 0'),
    ("N03_sl_sign", C + "render.py", "(cl<=popen-EXIT_SL*atr0);", "(cl<=popen+EXIT_SL*atr0);"),
    ("N04_atr_shift", C + "render.py", "ATRN(held+1)", "ATRN(held)"),
    ("N05_time_stop", C + "render.py", "return held>=EXIT_N;", "return held>EXIT_N;"),
    ("N06_horizon", C + "render.py", "if(held>=EXIT_H) return true;", "if(held>EXIT_H) return true;"),
    ("N07_reenter_gate", C + "render.py", "if(closed_now && !InpReenterOnExitBar) return;", "if(closed_now) return;"),
    ("N08_gap_compare", C + "render.py", "(double)(iTime(_Symbol,TF,0)-iTime(_Symbol,TF,1))>InpMaxGapHours", "(double)(iTime(_Symbol,TF,0)-iTime(_Symbol,TF,1))<InpMaxGapHours"),
    ("N09_dump_shift", C + "render.py", "TimeToString(iTime(_Symbol,TF,1),TIME_DATE|TIME_MINUTES);          // decision bar", "TimeToString(iTime(_Symbol,TF,0),TIME_DATE|TIME_MINUTES);          // decision bar"),
    ("N10_ask_bid", C + "render.py", "(DIR>0) ? SymbolInfoDouble(_Symbol,SYMBOL_ASK) : SymbolInfoDouble(_Symbol,SYMBOL_BID)", "(DIR>0) ? SymbolInfoDouble(_Symbol,SYMBOL_BID) : SymbolInfoDouble(_Symbol,SYMBOL_ASK)"),
    ("N11_signal_and", C + "render.py", '"\\n        && ".join(p.atom_exprs)', '"\\n        || ".join(p.atom_exprs)'),
    ("N12_new_bar_gate", C + "render.py", "if(bar0==0 || bar0==g_last_bar) return;", "if(bar0==0) return;"),
    ("N13_log_shift", C + "render.py", "TimeToString(iTime(_Symbol,TF,s_in),TIME_DATE|TIME_MINUTES)", "TimeToString(iTime(_Symbol,TF,s_out),TIME_DATE|TIME_MINUTES)"),
    ("N15_atrn_positive", C + "render.py", "return (MathIsValidNumber(a) && a>0.0) ? a : QNaN();", "return (MathIsValidNumber(a) && a>=0.0) ? a : QNaN();"),
    ("N14_atr_check", C + "render.py", "if(!MathIsValidNumber(ATRN(1))){ PrintFormat(\"%s: signal skipped (ATR invalid)", "if(false){ PrintFormat(\"%s: signal skipped (ATR invalid)"),
    ("E01_time_stop", C + "ea_sim.py", 'due = held >= ex["n"]', 'due = held > ex["n"]'),
    ("E02_horizon", C + "ea_sim.py", 'if held >= ex["h"]:', 'if held > ex["h"]:'),
    ("E03_atr_bar", C + "ea_sim.py", 'm.atr[pos["e"] - 1]', 'm.atr[pos["e"]]'),
    ("E04_close_bar", C + "ea_sim.py", "cl = m.close[b - 1]", "cl = m.close[b]"),
    ("E05_short_levels", C + "ea_sim.py", "if is_long else (cl <= lvl_tp or cl >= lvl_sl)", "if is_long else (cl >= lvl_tp or cl <= lvl_sl)"),
    ("E06_short_exit_ask", C + "ea_sim.py", "px = m.open[b] if is_long else m.open[b] + m.spread[b]", "px = m.open[b]"),
    ("E07_reenter", C + "ea_sim.py", "if closed_now and not reenter:", "if False:"),
    ("E08_entry_fill", C + "ea_sim.py", 'pos = {"e": b, "fill": m.open[b] + m.spread[b] if is_long else m.open[b]}', 'pos = {"e": b, "fill": m.open[b]}'),
    ("E09_gap_entry", C + "ea_sim.py", "if gap_s > 0 and gap_flag[b]:", "if False:"),
    ("E10_parity_cfg", C + "ea_sim.py", 'c["engine"]["no_weekend_hold"] = False', 'c["engine"]["no_weekend_hold"] = True'),
    ("E11_parity_slip", C + "ea_sim.py", 'c["costs"]["slippage_points"] = 0', "pass"),
    ("E12_decision_bar", C + "ea_sim.py", "        i = b - 1\n", "        i = b\n"),
    ("E13_long_levels", C + "ea_sim.py", "lvl_tp = pos[\"fill\"] + (ex[\"tp\"] if is_long else -ex[\"tp\"]) * atr0", "lvl_tp = pos[\"fill\"] + (ex[\"tp\"] if is_long else ex[\"tp\"]) * atr0"),
    ("A01_tol_scale", C + "parity.py", "tol * np.maximum(1.0, np.abs(b))", "tol * np.minimum(1.0, np.abs(b))"),
    ("A02_share_strict", C + "parity.py", "ok = pct <= max_deviating_pct", "ok = pct < max_deviating_pct"),
    ("A03_near_python_side", C + "parity.py", " | (np.abs(vb - thr) <= lim)", ""),
    ("A04_near_ea_side", C + "parity.py", "near |= (np.abs(va - thr) <= lim) | ", "near |= "),
    ("A05_tail_strict", C + "parity.py", "((pos - 1) <= len(index) - 2 - int(hmax))", "((pos - 1) < len(index) - 2 - int(hmax))"),
    ("A06_tail_zero", C + "parity.py", "keep = (pos >= 1) &", "keep = (pos >= 0) &"),
    ("A07_period_end", C + "parity.py", '(a["entry_time"] <= hi)', '(a["entry_time"] < hi)'),
    ("A08_period_start", C + "parity.py", '(b["entry_time"] >= lo)', '(b["entry_time"] > lo)'),
    ("A09_dup_times", C + "parity.py", 'if raw["time"].duplicated().any():', "if False:"),
    ("A10_signal_flag", C + "parity.py", '== 1)\n    diff = py != ea', '== 0)\n    diff = py != ea'),
    ("A11_skip_first", C + "parity.py", "table.index[skip_first:]", "table.index"),
    ("A12_verdict_order", C + "parity.py", 'status = "FAIL" if failed else ("NOT_RUN" if missing else "PASS_AUTOMATED")', 'status = "NOT_RUN" if missing else ("FAIL" if failed else "PASS_AUTOMATED")'),
    ("A13_hard_accept", C + "parity.py", '"accepted": bool(not hard.any() and len(idx) > 0)', '"accepted": bool(not diff.any() and len(idx) > 0) or True'),
    ("A14_nan_mismatch", C + "parity.py", "n_dev = int(over.sum()) + nan_mis", "n_dev = int(over.sum())"),
    ("A15_columns_check", C + "parity.py", "if list(raw.columns) != expect:", "if False:"),
    ("A16_log_columns", C + "parity.py", "if list(raw.columns) != LOG_COLUMNS:", "if False:"),
    ("A17_floor", C + "parity.py", '.dt.floor(f"{int(tf_minutes)}min")', ""),
    ("A18_seconds_fallback", C + "parity.py", "return pd.to_datetime(s, format=TIME_FORMAT + \":%S\")", "raise"),
    ("A19_profit_drop", C + "parity.py", "if not use_profit:", "if False:"),
    ("Z01_balance", C + "lint.py", "if code.count(o) != code.count(c):", "if False:"),
    ("Z02_order", C + "lint.py", "if defs[name] > pos:", "if defs[name] < pos:"),
    ("Z03_undefined", C + "lint.py", 'elif name not in ("MathMax", "MathMin"):', "elif False:"),
    ("Z04_ascii", C + "lint.py", "if not src.isascii():", "if False:"),
    ("Z05_placeholders", C + "lint.py", "    if left:\n        issues.append", "    if False:\n        issues.append"),
    ("Z06_comments_stripped", C + "lint.py", 'src = re.sub(r"//[^\\n]*", " ", src)', "pass"),
    ("T01_only_filter", C + "stage.py", "if r.rule_id in set(only)", "if r.rule_id not in set(only)"),
    ("T02_refusal_recorded", C + "stage.py", "refused[rule.rule_id] = e.reasons", "pass"),
    ("T03_data_hash", C + "stage.py", 'manifest.get("data_hash")', "None"),
    ("T04_lint_reported", C + "stage.py", "        if issues:\n            lint_issues", "        if False:\n            lint_issues"),
    ("T05_registry_reproducible", C + "stage.py", 'bool(f.get("reproducible_in_tester", True))', "True"),
    ("T06_default_rules", C + "stage.py", '"10_validation" / "accepted_rules.json"', '"10_validation" / "frozen_rules.json"'),
    ("P01_fp_codegen", "edge_research/pipeline.py", '"12_ea": ["codegen", "engine"', '"12_ea": ["engine"'),
    ("P02_no_upstream", "edge_research/pipeline.py", 'raise MissingUpstream("stage 10_validation not found; run it first")\n    fp = _fingerprint(stage, manifest["data_hash"], cfg)\n    entry = manifest["stages"].get(stage, {})\n    if entry.get("fingerprint") == fp and entry.get("upstream") == up and (run / stage / "ea_info.json")', 'raise MissingUpstream("x")\n    fp = _fingerprint(stage, manifest["data_hash"], cfg)\n    entry = manifest["stages"].get(stage, {})\n    if False and (run / stage / "ea_info.json")'),
    ("P03_stage_written", "edge_research/pipeline.py", '"n_generated": len(info["generated"]), "n_refused": len(info["refused"])}\n    mpath', '"n_generated": 0, "n_refused": 0}\n    mpath'),
]


def run(name, f, old, new):
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("output", "data", "__pycache__", "*.egg-info"))
    p = DST / f
    s = p.read_text()
    if s.count(old) != 1:
        return f"BAD-PATTERN({s.count(old)})"
    p.write_text(s.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_codegen_m10.py", "-x", "-q", "-p", "no:cacheprovider"], cwd=DST, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return ("CAUGHT" if r.returncode == 1 else "SURVIVED" if r.returncode == 0 else f"ERROR rc={r.returncode}") + " | " + tail[:90]


if __name__ == "__main__":
    RES.parent.mkdir(exist_ok=True)
    res = json.loads(RES.read_text()) if RES.exists() else {}
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, f, old, new in M:
        if not name.startswith(pref) or name in res:
            continue
        res[name] = run(name, f, old, new)
        RES.write_text(json.dumps(res, indent=1))
        print(name, res[name], flush=True)
