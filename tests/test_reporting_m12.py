"""M12: stage 13 reporting layer (``reporting/run_summary.py``, ``pipeline.run_stage_13``).

The tests run on a fabricated mini run folder (the stage output files are written by hand): the reporting layer must be a pure view of
those files, so no pipeline world is needed and the module is cheap on a slow machine. The mutation harness for this module is
``scripts/dev_mutate_m12.py`` (run it on a stronger machine, like the other dev_mutate scripts).
"""

import copy
import json

import pandas as pd
import pytest

from edge_research.pipeline import MissingUpstream, run_stage_13
from edge_research.reporting import gate_log, write_report
from edge_research.rules import FrozenRule, save_rules
from test_discovery import CFG

GATES = ["g2_time_stability", "g3_exit_plateau", "g4_lift", "g5_oos", "g6_trades", "g7_cost_stress", "g8_quality"]


def _rule(rid: str) -> FrozenRule:
    return FrozenRule(rid, "long", [{"feature": "cci_14", "op": "<=", "threshold": -48.2}, {"feature": "candle_body", "op": ">", "threshold": 0.61}],
                      "T24", meta={"cand_id": int(rid[1:]) % 900, "exprs": ["cci_14 <= -48.2", "candle_body > 0.61"], "t": 2.9})


def _gate_row(rid: str, cand_id: int, accepted: bool) -> dict:
    """One gate-table row. The accepted rule passes every gate; the rejected one fails exactly G6 and G8 (its fail_reasons say so)."""
    row = {"rule_id": rid, "cand_id": cand_id, "direction": "long", "exit": "T24", "k_tested": 2, "expr": "cci_14 <= -48.2 AND candle_body > 0.61",
           "val_trades": 92, "val_mean_R": 1.73, "val_se_R": 0.5, "val_t": 3.46, "t_crit_bonferroni": 3.18, "decay_ratio": 1.1,
           "val_profit_factor": 3.55, "val_equity_dd_pct": 4.0, "stress_mean_R": 1.61, "stress_net_profit": 12.0,
           "lift_min_t_train": 4.0, "lift_min_t_val": 3.3, "fail_reasons": "" if accepted else "g6_trades;g8_quality"}
    for g in GATES:
        row[g] = accepted or g not in ("g6_trades", "g8_quality")
    row["g1_null"] = True
    row["accepted"] = accepted
    row["eligible_for_holdout"] = accepted
    row["g10_holdout"] = "NOT OPENED"
    return row


@pytest.fixture()
def run(tmp_path):
    """A mini run folder with one page of every stage output stage 13 reads: 2 frozen rules, 1 accepted, null PASS, holdout untouched."""
    r = tmp_path / "run1"
    for path, info in [("04_atoms/atoms.json", {"atoms": [{}, {}, {}]}),
                       ("05_discovery/screen_info.json", {"n_survivors": 10}),
                       ("05b_deep/deep_info.json", {"n_deep": 5, "apriori": {"truncated": False}}),
                       ("06_candidates/candidates_info.json", {"n_candidates": 7, "n_clusters": 3}),
                       ("07_filters/filters_info.json", {"n_passed": 4}),
                       ("08_exits/exits_info.json", {"n_frozen": 2}),
                       ("10_validation/validation_info.json", {"n_frozen": 2, "n_accepted": 1, "n_eligible": 1, "g1_pass": True,
                                                               "gate_counts": {"g6_trades": 1, "g8_quality": 1}}),
                       ("11_backtest/stage11_info.json", {"n_rules": 1, "segments": ["train", "val"],
                                                          "summaries": {"val": {"total_trades": 92, "total_net_profit": 55.0, "expectancy_R": 1.73,
                                                                                "profit_factor": 3.55, "equity_drawdown_pct": 4.0}}}),
                       ("11_backtest/train/summary.json", {"summary": {"total_trades": 236, "total_net_profit": 310.0, "equity_drawdown_pct": 6.0},
                                                           "per_rule": {"R001": {"trades": 236, "net_profit": 310.0, "expectancy_R": 2.31, "win_rate": 0.64}}}),
                       ("11_backtest/val/summary.json", {"summary": {"total_trades": 92, "total_net_profit": 55.0, "equity_drawdown_pct": 4.0},
                                                         "per_rule": {"R001": {"trades": 92, "net_profit": 55.0, "expectancy_R": 1.73, "win_rate": 0.6}}}),
                       ("12_ea/ea_info.json", {"generated": [{"rule_id": "R001", "ea_name": "ER_R001"}],
                                               "refused": [{"rule_id": "R999", "reason": "feature not reproducible in the tester"}], "lint_issues": []}),
                       ("09_null/null_info.json", {"summary": {"g1_pass": True, "g1_reason": "passes: p_global_S=0.010 <= 0.10, false discovery 0.100 < 0.5",
                                                               "real": {"n_pass_f2": 10},
                                                               "variants": {"random": {"p_global_S": 0.01, "false_discovery_estimate": 0.1}}},
                                                   "run": {"n_done": 100, "n_expected": 100, "complete": True}})]:
        p = r / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info), encoding="utf-8")
    (r / "_ingest").mkdir(parents=True)
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=1, freq="h"), "close": [1.0]}).to_parquet(r / "_ingest" / "bars.parquet")
    (r / "00_data_qa").mkdir(parents=True)
    (r / "00_data_qa" / "report.md").write_text("# Stage 0 data QA report\n\ndata_hash: `d0eadbeef`\n\n| check | status | detail |\n|---|---|---|\n"
                                                "| rows | PASS | 12000 bars |\n\ncritical_failed: **False**\n", encoding="utf-8")
    (r / "03_eda").mkdir()
    (r / "03_eda" / "verdict.json").write_text(json.dumps({"verdict": "GO", "stop_recommended": False}), encoding="utf-8")
    (r / "10_validation").mkdir(exist_ok=True)
    pd.DataFrame([_gate_row("R001", 41, True), _gate_row("R002", 42, False)]).to_csv(r / "10_validation" / "gate_table.csv", index=False)
    save_rules([_rule("R001"), _rule("R002")], r / "10_validation" / "frozen_rules.json")
    save_rules([_rule("R001")], r / "10_validation" / "accepted_rules.json")
    manifest = {"package_version": "test", "seed": 1, "data_hash": "d0eadbeef", "config_hash": "c0ffee",
                "stages": {"10_validation": {"fingerprint": "f10"}, "11_backtest": {"fingerprint": "f11"}}, "holdout_accesses": []}
    (r / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return r


# ---------------------------------------------------------------- pure view
def test_gate_log_keeps_every_verdict_and_shortens_expr(run):
    log = gate_log(run)
    assert list(log["rule_id"]) == ["R001", "R002"] and len(log) == 2
    for c in GATES + ["accepted", "eligible_for_holdout", "g10_holdout", "fail_reasons", "val_t", "t_crit_bonferroni"]:
        assert c in log.columns
    assert " & " in log.loc[0, "expr"] and " AND " not in log.loc[0, "expr"]          # readability: one line per rule
    assert bool(log.loc[0, "accepted"]) and not bool(log.loc[1, "accepted"])
    assert log.loc[1, "fail_reasons"] == "g6_trades;g8_quality"


def test_run_summary_states_the_facts_of_every_stage(run):
    info = write_report(run, copy.deepcopy(CFG))
    assert info["status"] == "ran" and info["n_rules"] == 1 and info["gates_logged"] == 2
    text = (run / "13_report" / "RUN_SUMMARY.md").read_text(encoding="utf-8")
    assert "d0eadbeef" in text and "critical_failed: False" in text and "Verdict: **GO**" in text
    assert "| 05_discovery | pair survivors | 10 |" in text and "| 06_candidates | qualified candidates | 7 |" in text
    assert "| 05b_deep | deep rules (Apriori + trees) | 5 |" in text and "| 08_exits | frozen representatives (with an exit) | 2 |" in text
    assert "**G1: PASS**" in text and "p_global_S = 0.010" in text and "complete: True" in text
    assert "accepted by G2-G8: **1**" in text and "Holdout: untouched." in text
    assert "expectancy 1.730 R" in text and "EAs generated: 1" in text and "G9 (MQL5 parity): pending" in text
    assert "Bars ingested: 1" in text                                                 # from the parquet metadata only, no data is read


def test_gate_log_is_none_without_stage10(tmp_path):
    assert gate_log(tmp_path) is None


def test_rule_report_only_for_accepted_rules(run):
    write_report(run, copy.deepcopy(CFG))
    good = (run / "13_report" / "rules" / "R001.md").read_text(encoding="utf-8")
    assert not (run / "13_report" / "rules" / "R002.md").exists()
    assert "`cci_14 <= -48.2 & candle_body > 0.61`" in good                            # the meta exprs, joined with & like the gate log
    assert "| accepted | YES |" in good and "| g6_trades | PASS |" in good and "| g1_null | PASS |" in good and "| g10_holdout | NOT OPENED |" in good
    assert "| val_t | 3.460 |" in good and "| t_crit_bonferroni | 3.180 |" in good and "Fail reasons (empty = none): ``" in good
    assert "Trades: 92" in good and "expectancy 1.730 R" in good and "11_backtest/val/report.md" in good
    assert "- train: trades 236" in good and "expectancy 2.310 R" in good               # both configured segments are quoted
    assert "ER_R001.mq5" in good and "check_parity.py" in good                          # the EA name comes from ea_info.json, not invented


def test_reporting_config_switches(run):
    cfg = copy.deepcopy(CFG)
    cfg["reporting"] = {"run_summary": True, "per_rule_reports": False, "per_rule_segments": ["val"], "max_per_rule": 25}
    info = write_report(run, cfg)
    assert info["n_rules"] == 0 and info["gates_logged"] == 2                           # gate log still written, no rule pages
    assert not (run / "13_report" / "rules" / "R001.md").exists() and (run / "13_report" / "RUN_SUMMARY.md").exists()
    cfg["reporting"]["run_summary"] = False
    info = write_report(run, cfg)
    assert info["summary_bytes"] == 0 and not (run / "13_report" / "RUN_SUMMARY.md").exists()
    cfg["reporting"].update({"per_rule_reports": True, "per_rule_segments": ["val"], "max_per_rule": 1})
    save_rules([_rule("R001"), _rule("R002")], run / "10_validation" / "accepted_rules.json")     # two accepted rules, cap 1
    t = pd.read_csv(run / "10_validation" / "gate_table.csv")
    t.loc[t["rule_id"] == "R001", "train_chosen_smooth_R"] = 2.0
    t.loc[t["rule_id"] == "R002", "train_chosen_smooth_R"] = 3.0                       # R002 has the better train plateau: the cap must keep IT
    t.to_csv(run / "10_validation" / "gate_table.csv", index=False)
    info = write_report(run, cfg)
    assert info["n_rules"] == 1 and (run / "13_report" / "rules" / "R002.md").exists() and not (run / "13_report" / "rules" / "R001.md").exists()


# ---------------------------------------------------------------- stage wiring
def test_stage13_runs_skips_and_refreshes_on_gate_table_change(run):
    cfg = copy.deepcopy(CFG)
    assert run_stage_13(run, cfg)["status"] == "ran"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["stages"]["13_report"]
    assert entry["status"] == "done" and entry["n_rules"] == 1 and entry["gates_logged"] == 2
    summary = run / "13_report" / "RUN_SUMMARY.md"
    mtime = summary.stat().st_mtime_ns
    assert run_stage_13(run, cfg)["status"] == "skipped"                                # fresh fingerprint: nothing rewritten
    assert summary.stat().st_mtime_ns == mtime
    # the holdout fills g10 WITHOUT a new stage 10 fingerprint: the gate-table hash in the upstream key must force a rewrite
    t = pd.read_csv(run / "10_validation" / "gate_table.csv")
    t.loc[t["rule_id"] == "R001", "g10_holdout"] = "PASS"
    t.to_csv(run / "10_validation" / "gate_table.csv", index=False)
    assert run_stage_13(run, cfg)["status"] == "ran"
    assert summary.stat().st_mtime_ns != mtime and "Holdout: untouched." not in summary.read_text(encoding="utf-8")


def test_stage13_refuses_a_run_without_stage10(tmp_path):
    with pytest.raises(MissingUpstream):
        run_stage_13(tmp_path, copy.deepcopy(CFG))
