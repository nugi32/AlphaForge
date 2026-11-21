"""Mutation checks for M8. Each mutant edits ONE place of a scratch copy of the project (/tmp/mut); the selected tests must FAIL (mutant caught).
Resumable: finished mutants are kept in /home/claude/mut_results.json. Run: python scripts/dev_mutate_m8.py [name-prefix]  (scratch copy in /tmp/mut; results in output/mut_results.json; stages 0-8 worlds cached in output/m8cache)"""
import json, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
DST = Path("/tmp/mut")
RES = SRC / "output" / "mut_results.json"
V = "edge_research/validation/"

PURE_G = "t_critical or trade_clusters or trade_stats or apply_gates"
M = [
    # name, file, old, new, -k
    ("G01_alpha_not_divided_by_K", V + "gates.py", "float(alpha) / max(int(k), 1)", "float(alpha)", "t_critical"),
    ("G02_df_off_by_one", V + "gates.py", "int(n_clusters) - 1)", "int(n_clusters))", "t_critical"),
    ("G03_cluster_floor", V + "gates.py", "if int(n_clusters) < 2:", "if int(n_clusters) < 1:", "t_critical"),
    ("G04_gap_boundary", V + "gates.py", "np.diff(e[order]) > int(gap)", "np.diff(e[order]) >= int(gap)", "trade_clusters or trade_stats"),
    ("G05_cluster_order_lost", V + "gates.py", "    out = np.empty_like(new)\n    out[order] = new\n    return out", "    return new", "trade_clusters or trade_stats"),
    ("G06_ge_strict", V + "gates.py", "bool(x == x and x >= thr)", "bool(x == x and x > thr)", "apply_gates"),
    ("G07_g4_drop_b_val", V + "gates.py", ' and ge(row["lift_b_t_val"], lt_val))', ")", "apply_gates"),
    ("G08_g4_drop_train", V + "gates.py", 'bool(ge(row["lift_a_t_train"], lift_t_train) and ge(row["lift_b_t_train"], lift_t_train)\n                          and ', "bool(", "apply_gates"),
    ("G09_g5_mean_ge0", V + "gates.py", 'row["val_mean_R"] > 0 and ge(row["val_t"]', 'row["val_mean_R"] >= 0 and ge(row["val_t"]', "apply_gates"),
    ("G10_g5_no_decay", V + "gates.py", ' and ge(row["decay_ratio"], float(v["decay_min_ratio"])))', ")", "apply_gates"),
    ("G11_g5_no_tcrit", V + "gates.py", ' and ge(row["val_t"], row["t_crit_bonferroni"])', "", "apply_gates"),
    ("G12_g6_strict", V + "gates.py", 'row["val_trades"] >= int(v["min_trades_val"])', 'row["val_trades"] > int(v["min_trades_val"])', "apply_gates"),
    ("G13_g7_no_profit", V + "gates.py", 'row["stress_mean_R"] > 0 and row["stress_net_profit"] > 0)', 'row["stress_mean_R"] > 0)', "apply_gates"),
    ("G14_g8_dd_strict", V + "gates.py", 'row["val_equity_dd_pct"] <= float', 'row["val_equity_dd_pct"] < float', "apply_gates"),
    ("G15_g1_ignored", V + "gates.py", 'g1_ok = (row["g1_null"] is True) or (row["g1_null"] is None and not bool(v["require_g1"]))', "g1_ok = True", "apply_gates"),
    ("G16_accepted_any", V + "gates.py", 'row["accepted"] = bool(all(row[c] for c in GATE_COLS))', 'row["accepted"] = bool(any(row[c] for c in GATE_COLS))', "apply_gates"),
    ("G17_g2_true", V + "gates.py", 'row["g2_time_stability"] = bool(flt.at[cid, "pass_f2"])', 'row["g2_time_stability"] = True', "g2_g3"),
    ("G18_g3_or", V + "gates.py", 'bool(crow["pass_f3"]) and bool(flt.at[cid, "pass_f4"])', 'bool(crow["pass_f3"]) or bool(flt.at[cid, "pass_f4"])', "g2_g3"),
    ("G20_no_stress_in_table", V + "gates.py", 'float(v["cost_stress_mult"]))', "1.0)", "stage10_runs"),
    ("G21_stress_adds", V + "gates.py", 'eng.get("cost_multiplier", 1.0)) * float(stress_mult)', 'eng.get("cost_multiplier", 1.0)) + float(stress_mult)', "exactly_the_engine"),
    ("G22_lift_block_hold_only", V + "gates.py", "blk = max(1, int(block_mult) * int(hold))", "blk = max(1, int(hold))", "lift_over_parents"),
    ("G23_lift_parent_a_is_b", V + "gates.py", "la = lift_stats(a, child, y, int(hold), rows=rows, block=blk)", "la = lift_stats(b, child, y, int(hold), rows=rows, block=blk)", "lift_over_parents"),
    ("G24_val_lift_on_train", V + "gates.py", "lv = lift_over_parents(aset, rule.meta[\"atom_ids\"], y, hold, val, block_mult)", "lv = lift_over_parents(aset, rule.meta[\"atom_ids\"], y, hold, train_rows, block_mult)", "g2_g3"),
    ("G25_engine_on_train", V + "gates.py", "base = evaluate_rule(df, cfg, rule, ft, spread, val)", "base = evaluate_rule(df, cfg, rule, ft, spread, train_rows)", "never_the_holdout"),
    ("G26_recompute_nan", V + "gates.py", 'la_t, lb_t, src = lt["lift_a_t"], lt["lift_b_t"], "recomputed"', 'la_t, lb_t, src = float("nan"), float("nan"), "recomputed"', "recomputes_the_train"),
    ("H01_state_never_opened", V + "holdout.py", 'status = "opened" if done else "started"', 'status = "started"', "holdout_state"),
    ("H02_no_state_check", V + "holdout.py", 'if st["status"] != "untouched":', "if False:", "opens_once or crash_after_start"),
    ("H03_no_eligible_check", V + "holdout.py", "if not len(eligible):", "if False:", "refused_when_nothing"),
    ("H05_no_lock", V + "holdout.py", "os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)", "pass", "opens_once"),
    ("H06_inconclusive_boundary", V + "holdout.py", "if int(n_trades) < int(min_trades):", "if int(n_trades) <= int(min_trades):", "verdict_of"),
    ("H07_pass_ignores_profit", V + "holdout.py", "mean_R > 0 and net_profit > 0)", "mean_R > 0)", "verdict_of"),
    ("H08_verdict_ignores_sha", V + "holdout.py", 'elif st["rule_shas"].get(r.rule_id) != rule_sha(r):', "elif False:", "holdout_state"),
    ("H09_sha_includes_meta", V + "holdout.py", '"cat_stop_atr": rule.cat_stop_atr}', '"cat_stop_atr": rule.cat_stop_atr, "meta": rule.meta}', "rule_sha"),
    ("H10_empty_segment_allowed", V + "holdout.py", "if holdout_rows.stop - holdout_rows.start <= 0:", "if False:", "empty_segment"),
    ("H11_log_after_read", V + "holdout.py", "    _write_manifest(mpath, manifest)\n    # (2)", "    # (2)", "opens_once"),
    ("W01_window_fit", V + "walkforward.py", "while s + train + test <= n_rows:", "while s + train + test < n_rows:", "bar_windows"),
    ("W02_calendar_end", V + "walkforward.py", "if c > tlast + pd.Timedelta(seconds=1):", "if c > tlast - pd.Timedelta(days=60):", "calendar_windows"),
    ("W03_dd_row_order", V + "walkforward.py", 'p = trades.sort_values("exit_time")["profit"]', 'p = trades["profit"]', "exit_time_order"),
    ("W04_holdout_used", V + "walkforward.py", "else full.holdout.start - full.embargo", "else n_rows", "walkforward_end"),
    ("W05_pcts", V + "walkforward.py", "tp = train_n / float(train_n + test_n)", "tp = train_n / float(test_n + 1)", "bar_windows"),
    ("W06_ratio", V + "walkforward.py", "oos / is_mean if", "oos * is_mean if", "aggregate_hand"),
    ("W08_share_with_rules", V + "walkforward.py", '"share_windows_with_rules": len(with_rules) / n', '"share_windows_with_rules": len(windows) / n', "aggregate_hand"),
    ("W09_null_filter", V + "walkforward.py", 'chosen["best_t"].astype(float) > null_thr', 'chosen["best_t"].astype(float) > -null_thr', "walkforward_end"),
    ("W10_fp_no_threshold", V + "walkforward.py", '(f"|null{null_thr!r}" if null_thr is not None else "")', '("|null" if null_thr is not None else "")', "walkforward_end"),
    ("P01_fp_ignores_validation", "edge_research/pipeline.py", '"10_validation": ["validation", "engine"', '"10_validation": ["engine"', "stage10_runs"),
    ("P02_open_without_flag", "edge_research/pipeline.py", 'if bool(cfg["validation"]["open_holdout"]) and info["n_eligible"] > 0', 'if info["n_eligible"] > 0', "holdout_opens_once or stage10_runs"),
    ("P03_no_stale_check", "edge_research/pipeline.py", 'if entry.get("fingerprint") != _fingerprint(stage, manifest["data_hash"], cfg) or not (run / stage / "gate_table.csv").exists():', "if False:", "refused_when_nothing"),
    ("P04_stage11_all_frozen", "edge_research/pipeline.py", 'acc = load_rules(run / "10_validation" / "accepted_rules.json")', 'acc = load_rules(run / "10_validation" / "frozen_rules.json")', "stage11"),
    ("P05_accepted_is_frozen", "edge_research/pipeline.py", "save_rules([r for r in rules if r.rule_id in acc_ids], out_dir / \"accepted_rules.json\")", "save_rules(rules, out_dir / \"accepted_rules.json\")", "stage10_runs"),
    ("P06_verdicts_untouched", "edge_research/pipeline.py", 'if hold_mod.holdout_state(manifest)["status"] == "untouched":', "if False:", "apply_holdout_verdicts or stage10_runs"),
    ("P08_loader_no_feather", "edge_research/pipeline.py", '+ list((run / "_ingest").glob("*.feather"))', "", "load_run_inputs"),
    ("S01_zero_holdout", "edge_research/splits.py", 'if float(d["holdout_pct"]) == 0.0:', "if False:", "config_validation_section"),
    ("F01_freeze_order", V + "freeze.py", "ascending=[False, False, True]", "ascending=[True, False, True]", "freeze_orders"),
    ("F02_freeze_not_frozen_ok", V + "freeze.py", 'chosen["is_representative"].astype(bool) & chosen["frozen_ok"].astype(bool)', 'chosen["is_representative"].astype(bool)', "freeze_orders"),
]


def run(name, f, old, new, k):
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("output", "data", "__pycache__", "*.egg-info"))
    p = DST / f
    s = p.read_text()
    if s.count(old) != 1:
        return f"BAD-PATTERN({s.count(old)})"
    p.write_text(s.replace(old, new))
    env = dict(os.environ, M8_CACHE=str(SRC / "output" / "m8cache"))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_validation_m8.py", "-x", "-q", "-p", "no:cacheprovider", "-k", k], cwd=DST, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return ("CAUGHT" if r.returncode == 1 else "SURVIVED" if r.returncode == 0 else f"ERROR rc={r.returncode}") + " | " + tail[:90]


if __name__ == "__main__":
    res = json.loads(RES.read_text()) if RES.exists() else {}
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, f, old, new, k in M:
        if not name.startswith(pref) or name in res:
            continue
        res[name] = run(name, f, old, new, k)
        RES.write_text(json.dumps(res, indent=1))
        print(name, res[name], flush=True)
