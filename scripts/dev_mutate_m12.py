"""Mutation checks for M12 (stage 13, reporting layer). Each mutant edits ONE place of a scratch copy of the project (/tmp/mut); the selected
tests of tests/test_reporting_m12.py must FAIL (mutant caught). Resumable: finished mutants are kept in output/mut_results_m12.json.
Run: python scripts/dev_mutate_m12.py [name-prefix]  (the test world is a fabricated mini run folder, so this harness is CHEAP enough for a
slow machine; still, the scratch copy needs a full checkout in /tmp/mut)."""
import json, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
DST = Path("/tmp/mut")
RES = SRC / "output" / "mut_results_m12.json"
R = "edge_research/reporting/run_summary.py"
T = "tests/test_reporting_m12.py"

M = [
    # name, file, old, new, -k
    ("R01_gate_log_keeps_fail_reasons", R, '"stress_mean_R", "stress_net_profit", "lift_min_t_train", "lift_min_t_val", "fail_reasons"]', '"stress_mean_R", "stress_net_profit"]', "gate_log_keeps"),
    ("R02_gate_log_drops_gate_cols", R, '+ [c for c in GATE_COLS if c in t.columns]', "", "gate_log_keeps"),
    ("R03_gate_log_keeps_AND", R, '.str.replace(" AND ", " & ", regex=False)', "", "gate_log_keeps"),
    ("R04_gate_log_wrong_order", R, "out = t.reindex(columns=cols)", "out = t.reindex(columns=cols).iloc[::-1]", "gate_log_keeps"),
    ("R05_data_keeps_md_bold", R, 'f"- {crit_line.replace(\'**\', \'\')}."', 'f"- {crit_line}."', "run_summary_states"),
    ("R06_data_hides_bars", R, "n_bars if n_bars is not None else 'n/a'", "'n/a'", "run_summary_states"),
    ("R07_eda_no_verdict", R, "f\"- Verdict: **{v.get('verdict', 'n/a')}**; stop recommended: {v.get('stop_recommended', 'n/a')}.\",", "f\"- Verdict: {v.get('verdict', 'n/a')}; stop recommended: {v.get('stop_recommended', 'n/a')}.\",", "run_summary_states"),
    ("R08_funnel_wrong_deep", R, 'rows.append(("05b_deep", "deep rules (Apriori + trees)", s5b.get("n_deep") if s5b else "-"))', 'rows.append(("05b_deep", "deep rules (Apriori + trees)", "-"))', "run_summary_states"),
    ("R09_funnel_wrong_cand", R, 's6.get("n_candidates") if s6 else "-"', 's6.get("n_clusters") if s6 else "-"', "run_summary_states"),
    ("R10_null_pass_flipped", R, "f\"- **G1: {'PASS' if g1 else 'FAIL'}**", "f\"- **G1: {'FAIL' if g1 else 'PASS'}**", "run_summary_states"),
    ("R11_null_hides_complete", R, 'f"- Replicates: {run_i.get(\'n_done\', \'n/a\')} of {run_i.get(\'n_expected\', \'n/a\')} done; complete: {run_i.get(\'complete\', \'n/a\')}.",', "", "run_summary_states"),
    ("R12_gates_accepted_any", R, "f\"- Frozen rules tested: {frozen}; accepted by G2-G8: **{acc}**; eligible for the holdout: {info.get('n_eligible', 0)}.\",", "f\"- Frozen rules tested: {frozen}; accepted by G2-G8: **{frozen}**; eligible for the holdout: {info.get('n_eligible', 0)}.\",", "run_summary_states"),
    ("R13_gates_holdout_always_open", R, '"opened" if _manifest(run).get("holdout_accesses") else "untouched"', '"opened"', "run_summary_states"),
    ("R14_gates_no_counts", R, 'L += [f"| {k} | {v} / {frozen} |" for k, v in counts.items()]', "pass", "run_summary_states"),
    ("R15_backtest_segments_dropped", R, "for seg, s in sorted((info.get(\"summaries\") or {}).items()):", "for seg, s in []:", "run_summary_states"),
    ("R16_ea_count_wrong", R, "ng, nr = len(info.get(\"generated\", [])), len(info.get(\"refused\", []))", "ng, nr = 0, 0", "run_summary_states"),
    ("R17_open_questions_watered_down", R, '"- G9 (MQL5 parity): pending until the EA is compiled and the tester parity check has run on the user\'s machine.",', '"- G9 (MQL5 parity): see the stage logs.",', "run_summary_states"),
    ("R18_g1_verdict_flipped", R, "g1 = None if pd.isna(g1v) else bool(g1v)", "g1 = not bool(g1v)", "rule_report_only"),
    ("R19_rule_report_fail_reasons_hidden", R, 'L += [f"- Fail reasons (empty = none): `{r.get(\'fail_reasons\', \'\')}`.", ""]', "", "rule_report_only"),
    ("R20_rule_report_skips_val_bt", R, 'for seg, bt in bt_segments.items():', "for seg, bt in list(bt_segments.items())[:0]:", "rule_report_only"),
    ("R21_rule_report_off_by_one_trades", R, 'f"- {seg}: trades {per.get(\'trades\', 0)};', 'f"- {seg}: trades {per.get(\'trades\', 0) + 1};', "rule_report_only"),
    ("R22_rule_report_no_ea", R, "if gen is not None:", "if False:", "rule_report_only"),
    ("R23_ea_wrong_extension", R, "f\"- EA generated: `{gen.get('ea_name', 'n/a')}.mq5`", "f\"- EA generated: `{gen.get('ea_name', 'n/a')}.mqh`", "rule_report_only"),
    ("R24_accepted_from_frozen", R, 'p = run / "10_validation" / "accepted_rules.json"', 'p = run / "10_validation" / "frozen_rules.json"', "stage13_runs"),
    ("R25_cap_ignored", R, 'acc = acc[: int(rep["max_per_rule"])]', "pass", "reporting_config_switches"),
    ("R26_cap_worst_first", R, 'ascending=False, kind="stable")["rule_id"])', 'ascending=True, kind="stable")["rule_id"])', "reporting_config_switches"),
    ("R27_stale_run_summary_kept", R, 'elif (out / "RUN_SUMMARY.md").exists():\n        (out / "RUN_SUMMARY.md").unlink()', "elif False:\n        pass", "reporting_config_switches"),
    ("R28_stale_rule_pages_kept", R, 'for stale in (out / "rules").glob("*.md"):\n            stale.unlink()', "for stale in (out / \"rules\").glob(\"*.md\"):\n            pass", "reporting_config_switches"),
    ("R29_segments_ignore_config", R, '(rep.get("per_rule_segments") or ["val"])}', '["val"])}', "rule_report_only"),
    ("P01_stage13_upstream_sha_dropped", "edge_research/pipeline.py", 'gt_sha = hashlib.sha256(gt.read_bytes()).hexdigest()[:16] if gt.exists() else "none"', 'gt_sha = "none"', "stage13_runs"),
    ("P02_stage13_upstream_sha_unhashed", "edge_research/pipeline.py", "hashlib.sha256(gt.read_bytes()).hexdigest()[:16]", "gt.read_bytes().decode(errors='ignore')[:16]", "stage13_runs"),
    ("P03_stage13_skips_wrong_marker", "edge_research/pipeline.py", 'and (run / stage / "RUN_SUMMARY.md").exists()', "and True", "stage13_runs"),
    ("P04_stage13_no_manifest_write", "edge_research/pipeline.py", 'manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", **{k: info[k] for k in ("n_rules", "gates_logged")}}\n    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")', "pass", "stage13_runs"),
    ("P05_stage13_cfg_ignored", "edge_research/pipeline.py", "info = write_report(run, cfg)", "info = write_report(run, {})", "reporting_config_switches"),
    ("P06_no_missing_upstream", "edge_research/pipeline.py", 'if up10 is None:\n        raise MissingUpstream("stage 10_validation not found; run it first")', "if False:\n        pass", "refuses_a_run"),
]


def run(name, f, old, new, k):
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("output", "data", "__pycache__", "*.egg-info", ".pytest_cache"))
    p = DST / f
    s = p.read_text()
    if s.count(old) != 1:
        return f"BAD-PATTERN({s.count(old)})"
    p.write_text(s.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", T, "-x", "-q", "-p", "no:cacheprovider", "-k", k], cwd=DST, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return ("CAUGHT" if r.returncode == 1 else "SURVIVED" if r.returncode == 0 else f"ERROR rc={r.returncode}") + " | " + tail[:90]


if __name__ == "__main__":
    (SRC / "output").mkdir(exist_ok=True)
    res = json.loads(RES.read_text()) if RES.exists() else {}
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, f, old, new, k in M:
        if not name.startswith(pref) or name in res:
            continue
        res[name] = run(name, f, old, new, k)
        RES.write_text(json.dumps(res, indent=1))
        print(name, res[name], flush=True)
