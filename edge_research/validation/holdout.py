"""Holdout protocol (design 14.4).

* The holdout is read ONCE, for a rule set fixed before it is read (the rules eligible after G1-G8).
* Order of operations: (1) a ``start`` record is written to ``manifest.json`` BEFORE any holdout row is touched; (2) the engine runs on the holdout rows;
  (3) the results file is created exclusively (``x`` mode, a second write fails) and made read-only; (4) a ``complete`` record is appended.
* A second opening is refused, also after a crash between (1) and (4): the holdout may have been seen, so it counts as used (``started`` state).
* Rules cannot be changed after the opening: every verdict is bound to the hash of the rule it was computed for; a rule whose hash differs, or that was not opened, is ``BURNED``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from edge_research.mt5_backtest.backtest import run_backtest
from edge_research.rules import FrozenRule
from edge_research.validation.gates import trade_stats
from edge_research.discovery.pairs import family_hold

RESULTS_NAME = "holdout_results.json"
HOLDOUT_DIR = "holdout"


class HoldoutError(RuntimeError):
    pass


class HoldoutNotAllowed(HoldoutError):
    """Opening refused: stale stage 10, nothing eligible, empty holdout, or a holdout that was already opened (or started) once."""


def rule_sha(rule: FrozenRule) -> str:
    """Hash of what the engine reads from a rule (never of ``meta``)."""
    d = {"direction": rule.direction, "atoms": [[a["feature"], a["op"], float(a["threshold"])] for a in rule.atoms], "exit": rule.exit, "cat_stop_atr": rule.cat_stop_atr}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}, "holdout_accesses": []}


def _write_manifest(path: Path, m: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def holdout_state(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """``untouched`` (no record), ``started`` (a start record without its complete record) or ``opened`` (complete)."""
    acc = list(manifest.get("holdout_accesses", []))
    starts = [a for a in acc if a.get("event") == "start"]
    done = [a for a in acc if a.get("event") == "complete"]
    if not starts:
        return {"status": "untouched", "n_accesses": 0, "rule_shas": {}}
    last = starts[-1]
    status = "opened" if done else "started"
    return {"status": status, "n_accesses": len(starts), "rule_shas": dict(last.get("rule_shas", {})), "gate_table_sha": last.get("gate_table_sha"), "time": last.get("time")}


def results_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "10_validation" / HOLDOUT_DIR / RESULTS_NAME


def read_results(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    p = results_path(run_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def verdict_of(n_trades: int, mean_R: float, net_profit: float, min_trades: int) -> str:
    """INCONCLUSIVE below ``min_trades`` (never a pass); PASS for a positive mean R and positive net profit; FAIL otherwise (a NaN mean is a FAIL)."""
    if int(n_trades) < int(min_trades):
        return "INCONCLUSIVE"
    return "PASS" if (mean_R == mean_R and mean_R > 0 and net_profit > 0) else "FAIL"


def holdout_verdict_for(manifest: Dict[str, Any], eligible: Sequence[FrozenRule], results: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Verdict per rule id for the rules that are eligible NOW. A rule carries a verdict only if it was opened with the SAME hash; otherwise it is BURNED
    (the holdout cannot be opened again for it). Before the opening, or while it is incomplete, nothing is decided."""
    st = holdout_state(manifest)
    out: Dict[str, str] = {}
    for r in eligible:
        if st["status"] == "untouched":
            out[r.rule_id] = "NOT OPENED"
        elif st["status"] == "started" or results is None:
            out[r.rule_id] = "BURNED (access incomplete)"
        elif st["rule_shas"].get(r.rule_id) != rule_sha(r):
            out[r.rule_id] = "BURNED"
        else:
            out[r.rule_id] = results["rules"][r.rule_id]["verdict"]
    return out


def open_holdout(run_dir: str | Path, cfg: Dict[str, Any], eligible: Sequence[FrozenRule], df: pd.DataFrame, ft: pd.DataFrame, spread: Optional[pd.Series],
                 holdout_rows: slice, gate_table_sha: str) -> Dict[str, Any]:
    """Open the holdout once. Returns the stored results. Raises ``HoldoutNotAllowed`` BEFORE reading any holdout row when it must be refused."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _read_manifest(mpath)
    st = holdout_state(manifest)
    if st["status"] != "untouched":
        raise HoldoutNotAllowed(f"the holdout was already accessed ({st['status']}, {st['n_accesses']} time(s), first at {st.get('time')}); it is used up. Collect new data for a new holdout")
    if not len(eligible):
        raise HoldoutNotAllowed("no rule is eligible for the holdout (G1-G8 not all passed); opening it would burn it for nothing")
    if holdout_rows.stop - holdout_rows.start <= 0:
        raise HoldoutNotAllowed("the holdout segment is empty (data.holdout_pct = 0)")
    ids = [r.rule_id for r in eligible]
    if len(set(ids)) != len(ids):
        raise HoldoutNotAllowed("duplicate rule ids")
    out_dir = run / "10_validation" / HOLDOUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # (1) log BEFORE reading
    manifest.setdefault("holdout_accesses", []).append({"event": "start", "time": _now(), "rule_ids": ids, "rule_shas": {r.rule_id: rule_sha(r) for r in eligible},
                                                        "gate_table_sha": gate_table_sha, "rows": [holdout_rows.start, holdout_rows.stop]})
    _write_manifest(mpath, manifest)
    # (2) evaluate: each rule alone (as in validation), plus the shared account of all of them
    mt = int(cfg["validation"]["min_holdout_trades"])
    per: Dict[str, Any] = {}
    for r in eligible:
        res = run_backtest(df, cfg, [r], ft, spread, holdout_rows)
        s = trade_stats(res.trades, family_hold(r.exit, cfg))
        net = float(res.summary.get("total_net_profit", 0.0))
        per[r.rule_id] = {"rule_sha": rule_sha(r), "n_trades": s["n_trades"], "mean_R": s["mean_R"], "t": s["t"], "net_profit": net,
                          "profit_factor": float(res.summary.get("profit_factor", float("nan"))), "equity_dd_pct": float(res.summary.get("equity_drawdown_pct", float("nan"))),
                          "verdict": verdict_of(s["n_trades"], s["mean_R"], net, mt)}
    results = {"opened_at": manifest["holdout_accesses"][-1]["time"], "rows": [holdout_rows.start, holdout_rows.stop], "min_holdout_trades": mt, "rules": per}
    # (3) write once, then lock
    blob = json.dumps(results, indent=2, default=float)
    path = out_dir / RESULTS_NAME
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(blob)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    # (4) complete record
    manifest = _read_manifest(mpath)
    manifest["holdout_accesses"].append({"event": "complete", "time": _now(), "results_sha": hashlib.sha256(blob.encode()).hexdigest()[:16], "verdicts": {k: v["verdict"] for k, v in per.items()}})
    _write_manifest(mpath, manifest)
    return results
