"""G11 (demo forward): the OFFLINE half of the gate. The demo run itself needs a live demo account by definition and can
never be part of the pipeline; everything around it can:

1. Attach the generated EAs (``12_ea``) of the ACCEPTED rules to demo charts and run the agreed window (no parameter changes).
2. Export the account history of the window per EA (MT5: History tab -> right click -> "Report", save the trade rows as one
   CSV per rule: ``demo_<RULEID>.csv``) with one row per closed position and the columns ``open_time, close_time, type, price_in,
   price_out, profit`` (the aliases of ``compare_trades`` are accepted: time / type / open_price / ... are understood).
3. ``python scripts/g11_demo.py --run output/run1 --data data/export.csv --statements demo/`` re-runs the engine over the same
   window, compares trade by trade (``compare_trades``) and updates the ``g11_demo`` column of the gate table.

The gate is PASS only when every exported rule matches the engine 100 % on entry and exit time (the same acceptance criterion
as G9 parity) with the profit within tolerance and at least ``min_trades`` demo trades. FAIL / NO_TRADES are verdicts too —
a demo run that disagrees with the engine is evidence, not an inconvenience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from edge_research.mt5_backtest.compare import compare_trades

DEMO_ALIASES = {"open_time": ["open_time", "time_in", "time", "entry_time"], "close_time": ["close_time", "time_out", "exit_time"],
                "price_in": ["price_in", "open_price", "price"], "price_out": ["price_out", "close_price"],
                "type": ["type", "direction", "side"], "profit": ["profit", "net_profit"], "volume": ["volume", "lots"]}


def load_demo_statement(path: str | Path) -> pd.DataFrame:
    """One row per closed demo position; the columns of ``DEMO_ALIASES`` are matched case-insensitively (MT5 report exports use
    several spellings). ``open_time`` / ``close_time`` must exist — the window and the matching are built on them."""
    df = pd.read_csv(path)
    low = {c.strip().lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for key, names in DEMO_ALIASES.items():
        for nm in names:
            if nm in low:
                out[key] = df[low[nm]]
                break
    missing = [k for k in ("open_time", "close_time") if k not in out.columns]
    if missing:
        raise ValueError(f"{path}: demo statement has no {missing} column (found: {list(df.columns)}); save the MT5 history export as CSV with the trade times")
    out["open_time"] = pd.to_datetime(out["open_time"])
    out["close_time"] = pd.to_datetime(out["close_time"])
    return out.sort_values("open_time").reset_index(drop=True)


def evaluate_demo_gate(engine_trades: pd.DataFrame, statement: str | Path, min_trades: int = 10,
                       price_tol: float = 1e-4, pnl_rel_tol: float = 0.01) -> Dict[str, Any]:
    """Compare ONE rule's demo statement with its engine trades on the same window (compare one rule at a time: several EAs may
    enter on the same bar). Returns the comparison plus the verdict and the demo window."""
    demo = load_demo_statement(statement)
    cmp = compare_trades(engine_trades, demo, price_tol=price_tol, pnl_rel_tol=pnl_rel_tol)
    window = {"start": str(demo["open_time"].min()), "end": str(demo["close_time"].max())}
    n_demo = int(cmp["n_mt5"])
    if n_demo == 0:
        verdict = "NO_TRADES"
    elif not cmp["accepted"]:
        verdict = "FAIL"
    elif n_demo < int(min_trades):
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "n_demo": n_demo, "min_trades": int(min_trades), "window": window, **{k: v for k, v in cmp.items() if k != "only_engine" and k != "only_mt5"},
            "only_engine": cmp.get("only_engine", [])[:5], "only_mt5": cmp.get("only_mt5", [])[:5],
            "first_difference": cmp.get("first_difference")}


def update_gate_table(run_dir: str | Path, results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Write ``10_validation/g11_demo.json`` and put the verdict into the ``g11_demo`` column of ``10_validation/gate_table.csv``.
    Rules without a statement keep ``PENDING``. A result for a rule id that is not in the table is APPENDED as its own row
    (a demo verdict is never silently dropped). Returns the updated rule ids."""
    run = Path(run_dir)
    gpath = run / "10_validation" / "gate_table.csv"
    if not gpath.exists():
        raise FileNotFoundError(f"{gpath}: run stage 10 first")
    table = pd.read_csv(gpath)
    updated: List[str] = []
    appended: List[Dict[str, Any]] = []
    for rid, r in results.items():
        pos = table.index[table["rule_id"] == rid]
        txt = r["verdict"]
        if r["verdict"] == "PASS":
            txt = f"PASS (n={r['n_demo']}, {r['window']['start']}..{r['window']['end']})"
        elif r["verdict"] == "WARN":
            txt = f"WARN (n={r['n_demo']} < min {r['min_trades']}, {r['window']['start']}..{r['window']['end']})"
        elif r["verdict"] == "FAIL":
            fd = r.get("first_difference") or {}
            txt = f"FAIL (n={r['n_demo']}; first difference: {fd.get('reason', 'see g11_demo.json')})"
        if len(pos):
            table.loc[pos, "g11_demo"] = txt
        else:                                             # unknown rule id: append it so the verdict stays visible in the table
            appended.append({"rule_id": rid, "g11_demo": txt})
        updated.append(rid)
    if appended:
        table = pd.concat([table, pd.DataFrame(appended)], ignore_index=True)
    table.to_csv(gpath, index=False)
    (run / "10_validation" / "g11_demo.json").write_text(json.dumps({"results": results, "updated": updated}, indent=2, default=float), encoding="utf-8")
    return updated
