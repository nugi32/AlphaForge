"""Stage 13 (M12): RUN_SUMMARY.md, per-rule reports and the gate log of a finished run.

Every file is a READ-ONLY view of the stage outputs (manifest, stage summaries, gate table, backtests): stage 13 computes no statistic of
its own and never changes a verdict. Numbers already reported by earlier stages are copied, not recomputed. A missing upstream stage is a
section that says so, not an error: the summary must be writable after any prefix of the pipeline (spec 19.3: the report shows what was run).

Outputs in ``<run>/13_report/``:
* ``RUN_SUMMARY.md``     - the whole run on one page: data, funnel, null verdict, gates, accepted rules, backtest, EA generation.
* ``gate_log.csv``       - one row per frozen rule and every gate column of stage 10 (the full gate table is linked, not duplicated).
* ``rules/<rule_id>.md`` - one page per rule accepted by stage 10: its definition, the gate row, the stage 11 backtest numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from edge_research.validation.gates import GATE_COLS


# ---------------------------------------------------------------------------------------------------------------- small readers (all optional)
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _fmt(x: Any, nd: int = 3) -> str:
    """One number cell; NaN / None become 'n/a' (never a bare empty cell, which reads as zero)."""
    if x is None:
        return "n/a"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{f:.{nd}f}" if f == f else "n/a"


def _link(rel: Path) -> str:
    return str(rel).replace("\\", "/")


def _manifest(run: Path) -> Dict[str, Any]:
    return _read_json(run / "manifest.json") or {"stages": {}}


# ---------------------------------------------------------------------------------------------------------------- gate log
def gate_log(run: Path) -> Optional[pd.DataFrame]:
    """The stage 10 gate table, reduced to identification + raw val numbers + the verdicts: the analyst-facing log of who passed what and why.

    None when stage 10 has not run; an EMPTY frame with the standard columns when it ran and froze nothing.
    """
    src = run / "10_validation" / "gate_table.csv"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    base = ["rule_id", "cand_id", "direction", "exit", "k_tested", "expr",
            "val_trades", "val_mean_R", "val_t", "t_crit_bonferroni", "decay_ratio", "val_profit_factor", "val_equity_dd_pct",
            "stress_mean_R", "stress_net_profit", "lift_min_t_train", "lift_min_t_val", "fail_reasons"]
    cols = [c for c in base if c in t.columns] + [c for c in GATE_COLS if c in t.columns] + [c for c in ("g1_null", "accepted", "eligible_for_holdout", "g10_holdout") if c in t.columns]
    out = t.reindex(columns=cols)
    if len(out) and "expr" in out.columns:                       # the expression is long; the log stays readable by keeping one line per rule
        out = out.copy()
        out["expr"] = out["expr"].astype(str).str.replace(" AND ", " & ", regex=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------- RUN_SUMMARY.md
def _data_section(run: Path, manifest: Dict[str, Any]) -> List[str]:
    """Stage 0 writes report.md (+ two CSVs) only: the few facts the summary needs are parsed from the report's own header lines."""
    report = run / "00_data_qa" / "report.md"
    L = ["## 1. Data and quality gate", ""]
    if not report.exists():
        L += ["Stage 0 has not run in this run folder."]
        return L
    text = report.read_text(encoding="utf-8")
    hash_line = next((ln for ln in text.splitlines() if ln.startswith("data_hash:")), None)
    crit_line = next((ln for ln in text.splitlines() if ln.startswith("critical_failed:")), "critical_failed: ?")
    n_bars = _n_ingest_rows(run)
    L += [f"- Bars ingested: {n_bars if n_bars is not None else 'n/a'}.",
          f"- {hash_line.replace('`', '')}." if hash_line else f"- Data hash: {manifest.get('data_hash', 'n/a')}.",
          f"- {crit_line.replace('**', '')}."]
    if "**True**" in crit_line:
        L += ["- **WARNING: the run continued past a critical QA failure; read 00_data_qa/report.md before anything else.**"]
    L += [f"- Full report: {_link(Path('00_data_qa') / 'report.md')}."]
    return L


def _n_ingest_rows(run: Path) -> Optional[int]:
    """Row count of the ingested bar table from the parquet metadata (no data is read)."""
    files = sorted((run / "_ingest").glob("*.parquet"))
    if not files:
        return None
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(files[0]).metadata.num_rows)
    except Exception:
        return None


def _eda_section(run: Path) -> List[str]:
    v = _read_json(run / "03_eda" / "verdict.json")
    L = ["## 2. Univariate verdict (stage 3)", ""]
    if v is None:
        L += ["Stage 3 has not run."]
        return L
    L += [f"- Verdict: **{v.get('verdict', 'n/a')}**; stop recommended: {v.get('stop_recommended', 'n/a')}.",
          f"- Summary: {_link(Path('03_eda') / 'summary.md')}."]
    return L


def _funnel_section(run: Path, manifest: Dict[str, Any]) -> List[str]:
    """The discovery funnel, one number per stage, read from the stage info files (the same numbers the stage summaries print)."""
    L = ["## 3. Discovery funnel (train rows only; selected numbers, not evidence)", ""]
    st = manifest.get("stages", {})
    rows: List[tuple] = []
    a = _read_json(run / "04_atoms" / "atoms.json")
    n_atoms = len(a.get("atoms", [])) if a else None
    rows.append(("04_atoms", "atoms built", n_atoms if n_atoms is not None else ("done" if st.get("04_atoms", {}).get("status") == "done" else "-")))
    s5 = _read_json(run / "05_discovery" / "screen_info.json")
    rows.append(("05_discovery", "pair survivors", s5.get("n_survivors") if s5 else "-"))
    s5b = _read_json(run / "05b_deep" / "deep_info.json")
    rows.append(("05b_deep", "deep rules (Apriori + trees)", s5b.get("n_deep") if s5b else "-"))
    s6 = _read_json(run / "06_candidates" / "candidates_info.json")
    rows.append(("06_candidates", "qualified candidates", s6.get("n_candidates") if s6 else "-"))
    if s6:
        rows.append(("06_candidates", "independent clusters", s6.get("n_clusters")))
    s7 = _read_json(run / "07_filters" / "filters_info.json")
    rows.append(("07_filters", "candidates passing F2/F4/F6/F7", s7.get("n_passed") if s7 else "-"))
    s8 = _read_json(run / "08_exits" / "exits_info.json")
    rows.append(("08_exits", "frozen representatives (with an exit)", s8.get("n_frozen") if s8 else "-"))
    L += ["| stage | quantity | value |", "|---|---|---|"]
    L += [f"| {stage} | {name} | {val} |" for stage, name, val in rows]
    if s5b and s5b.get("truncated") or (s5b and s5b.get("apriori", {}).get("truncated")):
        L += ["- Note: the deep stage was truncated by its survivor cap; the funnel numbers below the cap are conditional on it."]
    L += ["", "Every number above is a SELECTION over a wide search; nothing is evidence before the null test and validation."]
    return L


def _null_section(run: Path) -> List[str]:
    info = _read_json(run / "09_null" / "null_info.json")
    L = ["## 4. Null test (stage 9, G1)", ""]
    if info is None:
        L += ["Stage 9 has not run: G1 is undecided and no rule is eligible for the holdout (``require_g1``)."]
        return L
    s = info.get("summary", {})
    run_i = info.get("run", {})
    g1 = s.get("g1_pass")
    L += [f"- **G1: {'PASS' if g1 else 'FAIL'}** - {s.get('g1_reason', 'n/a')}",
          f"- Replicates: {run_i.get('n_done', 'n/a')} of {run_i.get('n_expected', 'n/a')} done; complete: {run_i.get('complete', 'n/a')}.",
          f"- Real rules passing F1-F2: {s.get('real', {}).get('n_pass_f2', 'n/a')}."]
    _du = s.get("real", {}).get("deep_used")
    if _du is not None:
        L += [f"- Deep engines priced by the null: {_du} - {s['real'].get('deep_note', '')}"]
    for v, it in sorted((s.get("variants") or {}).items()):
        L += [f"- Variant {v}: p_global_S = {_fmt(it.get('p_global_S'))}, false-discovery estimate = {_fmt(it.get('false_discovery_estimate'))}."]
    L += [f"- Full report: {_link(Path('09_null') / 'summary.md')}."]
    return L


def _gates_section(run: Path, manifest: Dict[str, Any]) -> List[str]:
    info = _read_json(run / "10_validation" / "validation_info.json")
    L = ["## 5. Validation gates (stage 10)", ""]
    if info is None:
        L += ["Stage 10 has not run."]
        return L
    acc, frozen = info.get("n_accepted", 0), info.get("n_frozen", 0)
    L += [f"- Frozen rules tested: {frozen}; accepted by G2-G8: **{acc}**; eligible for the holdout: {info.get('n_eligible', 0)}.",
          f"- G1 at validation time: {'PASS' if info.get('g1_pass') else ('FAIL' if info.get('g1_pass') is not None else 'not run')}."]
    counts = info.get("gate_counts") or {}
    if counts:
        L += ["", "| gate | passed |", "|---|---|"]
        L += [f"| {k} | {v} / {frozen} |" for k, v in counts.items()]
    # the holdout stage fills the g10_holdout column WITHOUT a new stage-10 fingerprint, so the gate table is evidence of an
    # open holdout even when the manifest hash was not re-read: the report must reflect the table, not only the manifest.
    verd = _read_csv(run / "10_validation" / "gate_table.csv")
    verdicts = []
    if verd is not None and len(verd) and "g10_holdout" in verd.columns:
        verdicts = [v for v in verd["g10_holdout"].dropna().astype(str).str.strip().tolist()
                    if v and v.lower() != "nan" and v != "NOT OPENED"]
    state = "opened" if (manifest.get("holdout_accesses") or verdicts) else "untouched"
    L += ["", f"- Holdout: {state}."]
    if state == "opened" and verdicts:
        vc = pd.Series(verdicts).value_counts().to_dict()
        L += [f"- Holdout verdicts: {', '.join(f'{k} {v}' for k, v in sorted(vc.items()))}."]
    L += [f"- Full gate table: {_link(Path('10_validation') / 'gate_table.csv')}; report: {_link(Path('10_validation') / 'gates.md')}.",
          f"- Regime diagnostic (spec 14.6): {_link(Path('10_validation') / 'regime_diag.csv')}."]
    return L


def _backtest_section(run: Path) -> List[str]:
    L = ["## 6. Engine backtest of the accepted rules (stage 11)", ""]
    info = _read_json(run / "11_backtest" / "stage11_info.json")
    if info is None:
        L += ["Stage 11 has not run (or nothing was accepted)."]
        return L
    L += [f"- Rules backtested: {info.get('n_rules', 0)} on segments {', '.join(info.get('segments', []))}."]
    for seg, s in sorted((info.get("summaries") or {}).items()):
        if not s:
            continue
        L += [f"- {seg}: trades {_fmt(s.get('total_trades'), 0)}, net {_fmt(s.get('total_net_profit'), 2)}, expectancy {_fmt(s.get('expectancy_R'))} R, "
              f"PF {_fmt(s.get('profit_factor'), 2)}, equity DD {_fmt(s.get('equity_drawdown_pct'), 1)} %."]
    L += [f"- Reports: {_link(Path('11_backtest'))}/<segment>/report.md."]
    for seg in sorted((info.get("summaries") or {})):
        png = (info["summaries"][seg] or {}).get("chart")
        if png:
            L += ["", f"**{seg}** - equity and balance", "", f"![{seg} equity and balance](../11_backtest/{png})"]
    return L


def _ea_section(run: Path) -> List[str]:
    info = _read_json(run / "12_ea" / "ea_info.json")
    L = ["## 7. MQL5 generation (stage 12, M10)", ""]
    if info is None:
        L += ["Stage 12 has not run."]
        return L
    ng, nr = len(info.get("generated", [])), len(info.get("refused", []))
    L += [f"- EAs generated: {ng}; rules refused (no exact translation): {nr}.",
          f"- G9 (parity with the Strategy Tester) is NOT decided here; run ``scripts/check_parity.py`` after the tester runs (see 12_ea/REVIEW_CHECKLIST.md)."]
    if nr:
        L += [f"- Refused rules: {_link(Path('12_ea') / 'refused.json')}."]
    return L


def run_summary_md(run: Path) -> str:
    """The whole run on one page, from the stage outputs only (spec 19.3)."""
    run = Path(run)
    manifest = _manifest(run)
    L = ["# RUN_SUMMARY", "",
         f"- Run folder: {run.name}; package version: {manifest.get('package_version', 'n/a')}; seed: {manifest.get('seed', 'n/a')}.",
         f"- Data hash: {manifest.get('data_hash', 'n/a')}; config hash: {manifest.get('config_hash', 'n/a')}.",
         "- This page is produced by stage 13 from the stage outputs; it computes nothing new. Selection bias disclaimers of the stages apply here unchanged.", ""]
    L += _data_section(run, manifest)
    L += [""] + _eda_section(run)
    L += [""] + _funnel_section(run, manifest)
    L += [""] + _null_section(run)
    L += [""] + _gates_section(run, manifest)
    L += [""] + _backtest_section(run)
    L += [""] + _ea_section(run)
    L += ["", "## Open questions and pending gates",
          "- G9 (MQL5 parity): pending until the EA is compiled and the tester parity check has run on the user's machine.",
          "- G10 (holdout): see section 5; the holdout must stay closed until the parity work is final.",
          "- G11 (demo forward): pending; it needs a live demo account and is not part of this pipeline."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------- per-rule report
def rule_report_md(rule_id: str, table: pd.DataFrame, bt_segments: Dict[str, Dict[str, Any]], ea_info: Optional[Dict[str, Any]] = None) -> str:
    """One accepted rule on one page: definition, every gate verdict with its raw value, the stage 11 backtest numbers per segment."""
    rows = table[table["rule_id"] == rule_id]
    L = [f"# Rule {rule_id}", ""]
    if rows.empty:
        L += ["This rule id is not in the stage 10 gate table (it was not frozen in this run)."]
        return "\n".join(L) + "\n"
    r = rows.iloc[0]
    L += ["## Definition", "",
          f"- Expression (train): `{str(r.get('expr', 'n/a')).replace(' AND ', ' & ')}`",
          f"- Direction: {r.get('direction', 'n/a')}; exit: {r.get('exit', 'n/a')}; cand_id {r.get('cand_id', 'n/a')}.",
          f"- Rule id: {rule_id} (canonical rule JSON: 10_validation/frozen_rules.json).", ""]
    L += ["## Gates (stage 10, validation segment)", "", "| gate | verdict |", "|---|---|"]
    for c in GATE_COLS:
        if c in rows.columns:
            L.append(f"| {c} | {'PASS' if bool(r[c]) else 'FAIL'} |")
    g1v = r.get("g1_null")
    g1 = None if pd.isna(g1v) else bool(g1v)                                       # a CSV bool comes back as numpy.bool_ / NaN, never compare with `is`
    L += [f"| g1_null | {'PASS' if g1 else ('FAIL' if g1 is False else 'not priced')} |",
          f"| accepted | {'YES' if bool(r['accepted']) else 'no'} |"]
    if "g10_holdout" in rows.columns:
        L += [f"| g10_holdout | {r['g10_holdout']} |"]
    L += ["", "| raw value | number |", "|---|---|"]
    for c in ("val_trades", "val_mean_R", "val_se_R", "val_t", "t_crit_bonferroni", "decay_ratio", "val_profit_factor", "val_equity_dd_pct",
              "stress_mean_R", "stress_net_profit", "lift_min_t_train", "lift_min_t_val"):
        if c in rows.columns:
            L.append(f"| {c} | {_fmt(r[c])} |")
    fr = r.get("fail_reasons", "")                                   # an empty CSV cell comes back as NaN, not ""
    L += [f"- Fail reasons (empty = none): `{'' if pd.isna(fr) else fr}`.", ""]
    L += ["## Engine backtest (stage 11, one shared account for all accepted rules)", ""]
    if not bt_segments:
        L += ["Stage 11 has not run for this run."]
    val_bt = bt_segments.get("val") or bt_segments.get(next(iter(bt_segments), "")) or {}   # headline from the val segment: the decision segment
    head = (val_bt.get("per_rule") or {}).get(rule_id)
    if head is not None:                                               # the headline verdict of the rule at a glance
        L += [f"- Trades: {head.get('trades', 0)}, expectancy {_fmt(head.get('expectancy_R'))} R."]
    for seg, bt in bt_segments.items():
        per = (bt.get("per_rule") or {}).get(rule_id)
        if per is None:
            L += [f"- {seg}: no trades of this rule (it shares the account with the other accepted rules but never triggers)."]
            continue
        L += [f"- {seg}: trades {per.get('trades', 0)}; net profit {_fmt(per.get('net_profit'), 2)}; expectancy {_fmt(per.get('expectancy_R'))} R; win rate {_fmt(per.get('win_rate'))}.",
              f"  Portfolio (all rules on one account): trades {bt.get('summary', {}).get('total_trades', 'n/a')}, "
              f"net {bt.get('summary', {}).get('total_net_profit', 'n/a')}, DD {bt.get('summary', {}).get('equity_drawdown_pct', 'n/a')} %.",
              f"  Full report: {_link(Path('11_backtest') / seg / 'report.md')}."]
    L += [""]
    L += ["## MQL5 EA (stage 12)", ""]
    gen = next((g for g in (ea_info or {}).get("generated", []) if g.get("rule_id") == rule_id), None)
    ref = next((g for g in (ea_info or {}).get("refused", []) if g.get("rule_id") == rule_id), None)
    if gen is not None:
        L += [f"- EA generated: `{gen.get('ea_name', 'n/a')}.mq5` in 12_ea/ (compile in MetaEditor, then ``scripts/check_parity.py``)."]
    elif ref is not None:
        L += [f"- The generator REFUSED this rule: {ref.get('reason', 'see 12_ea/refused.json')}. The rule stays research-only."]
    else:
        L += ["Stage 12 has not run: no EA exists yet."]
    L += ["", "## Reminders", "",
          "- This rule was SELECTED from a wide search; its validation numbers are optimistic by construction. The holdout (G10), the MQL5 parity check (G9)",
          "  and the demo forward (G11) are the remaining tests.",
          "- Do not retune the rule after looking at the holdout: an opened holdout is burned for this run (manifest)."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------- writer
def write_report(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 13 entry point: write ``13_report/RUN_SUMMARY.md``, ``gate_log.csv`` and one ``rules/<rule_id>.md`` per accepted rule.

    Honours the ``reporting:`` config section: ``run_summary`` / ``per_rule_reports`` switch the outputs off, ``per_rule_segments`` selects
    the stage 11 segments quoted in the rule pages, ``max_per_rule`` caps their number (best train plateau first).
    """
    run = Path(run_dir)
    rep = cfg.get("reporting", {}) or {}
    out = run / "13_report"
    (out / "rules").mkdir(parents=True, exist_ok=True)
    n_rules = 0
    log = gate_log(run)
    if log is not None:
        log.to_csv(out / "gate_log.csv", index=False)
    if rep.get("per_rule_reports", True) and log is not None:
        table = _read_csv(run / "10_validation" / "gate_table.csv")      # a DataFrame is never truthy/falsy: test for None explicitly
        if table is None:
            table = pd.DataFrame()
        bt = {seg: _read_json(run / "11_backtest" / seg / "summary.json") or {} for seg in (rep.get("per_rule_segments") or ["val"])}
        bt = {seg: b for seg, b in bt.items() if b}
        ea = _read_json(run / "12_ea" / "ea_info.json")
        acc = load_accepted(run)
        if rep.get("max_per_rule") is not None and len(acc) > int(rep["max_per_rule"]):
            order = {rid: i for i, rid in enumerate(acc)}
            if "train_chosen_smooth_R" in table.columns:
                acc = list(table.loc[table["rule_id"].isin(acc)].sort_values("train_chosen_smooth_R", ascending=False, kind="stable")["rule_id"])
            else:
                acc = sorted(acc, key=lambda r: -order[r])                       # no plateau column: keep the accepted_rules.json order
            acc = acc[: int(rep["max_per_rule"])]
        for rid in acc:
            (out / "rules" / f"{rid}.md").write_text(rule_report_md(rid, table, bt, ea), encoding="utf-8")
        n_rules = len(acc)
    else:
        for stale in (out / "rules").glob("*.md"):                               # the stage owns 13_report: an output switched off leaves no stale file
            stale.unlink()
    summary = run_summary_md(run) if rep.get("run_summary", True) else ""
    if summary:
        (out / "RUN_SUMMARY.md").write_text(summary, encoding="utf-8")
    elif (out / "RUN_SUMMARY.md").exists():
        (out / "RUN_SUMMARY.md").unlink()
    return {"status": "ran", "n_rules": n_rules, "gates_logged": int(len(log)) if log is not None else 0, "summary_bytes": len(summary)}


def load_accepted(run: Path) -> List[str]:
    """Rule ids accepted by stage 10, from the canonical accepted_rules.json (the same list stage 11 backtests)."""
    p = run / "10_validation" / "accepted_rules.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))["rules"]
    except (json.JSONDecodeError, KeyError):
        return []
    return [str(d.get("rule_id")) for d in data if isinstance(d, dict) and d.get("rule_id")]
