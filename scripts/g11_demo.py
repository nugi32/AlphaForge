"""G11 (demo forward) offline part: compare the demo-account statement of each ACCEPTED rule with the engine on the same window
and update the ``g11_demo`` column of the gate table. The demo run itself happens in MetaTrader (see edge_research/validation/demo.py).

Example:
  python scripts/g11_demo.py --run output/run1 --statements demo/ --symbol EURUSD --timeframe H1
where demo/ holds one CSV per rule: demo_R02441.csv (open_time, close_time, type, price_in, price_out, profit).
The bar table is read from the run (10_validation/_ingest); the engine is re-run over the demo window.
Writes: 10_validation/g11_demo.json and updates 10_validation/gate_table.csv (g11_demo column).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_research.config import load_config, symbol_config
from edge_research.mt5_backtest.backtest import run_backtest
from edge_research.pipeline import load_run_inputs
from edge_research.rules import load_rules
from edge_research.validation.demo import evaluate_demo_gate, update_gate_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--statements", required=True, help="directory with demo_<RULEID>.csv files (or a single CSV)")
    ap.add_argument("--rules", default=None, help="default: <run>/10_validation/accepted_rules.json")
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--config", default=None)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--symbol", default=None, help="merge config/MT5_TESTER/<SYMBOL>/<TF>.yaml (costs / engine)")
    ap.add_argument("--timeframe", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.symbol or a.timeframe:
        if not (a.symbol and a.timeframe):
            ap.error("--symbol and --timeframe must be given together")
        cfg = symbol_config(cfg, a.symbol, a.timeframe)
    run = Path(a.run)
    rules = load_rules(a.rules or run / "10_validation" / "accepted_rules.json")
    st_dir = Path(a.statements)
    files = ([st_dir] if st_dir.is_file() else sorted(st_dir.glob("demo_*.csv")))
    if not files:
        sys.exit(f"no demo statement files found in {st_dir}")
    df, ft, spread, _splits = load_run_inputs(run, cfg, a.spread_csv)
    results: dict = {}
    for f in files:
        rid = f.stem.replace("demo_", "").replace(".csv", "")
        rule = next((r for r in rules if r.rule_id == rid), None)
        if rule is None:
            results[rid] = {"verdict": "FAIL", "n_demo": 0, "min_trades": a.min_trades, "window": {},
                            "first_difference": {"reason": f"{rid} is not in the accepted rules"}}
            continue
        times = pd.to_datetime(pd.read_csv(f).iloc[:, 0])                 # first column of the statement holds the open times
        lo_pos = int(df.index.searchsorted(times.min()))
        hi_pos = int(df.index.searchsorted(times.max(), side="right"))
        res = run_backtest(df, cfg, [rule], ft, spread, rows=slice(lo_pos, min(hi_pos + 1, len(df))))
        results[rid] = evaluate_demo_gate(res.trades, f, min_trades=a.min_trades)
    updated = update_gate_table(run, results)
    print(json.dumps({k: {"verdict": v["verdict"], "n_demo": v["n_demo"]} for k, v in results.items()}, indent=2))
    print(f"gate table updated for: {updated or 'no rule'}")


if __name__ == "__main__":
    main()
