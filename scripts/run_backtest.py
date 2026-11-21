"""Backtest the frozen rules of a canonical rule file on a finished run (train / val / all; the holdout is opened only by the validation stage). Example:
python scripts/run_backtest.py --run output/run1 --rules output/run1/rules.json --segment val --spread-csv data/synth_edge_spread.csv
With per-market overrides: --symbol EURUSD --timeframe H1 (merges config/MT5_TESTER/EURUSD/H1.yaml, costs + engine only).
Outputs: <run>/11_backtest/<segment>/trades.csv, equity.csv, by_year.csv, by_month.csv, summary.json, report.md
"""

import argparse
import json

from edge_research.config import load_config, symbol_config
from edge_research.pipeline import run_backtest_stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--segment", default="val", choices=["train", "val", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--symbol", default=None, help="merge config/MT5_TESTER/<SYMBOL>/<TF>.yaml (costs / engine) before running")
    ap.add_argument("--timeframe", default=None, help="e.g. H1; required together with --symbol")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.symbol or a.timeframe:
        if not (a.symbol and a.timeframe):
            ap.error("--symbol and --timeframe must be given together")
        cfg = symbol_config(cfg, a.symbol, a.timeframe)
    res = run_backtest_stage(a.run, cfg, a.rules, a.segment, a.spread_csv, a.out)
    print(json.dumps({k: v for k, v in res.summary.items() if not isinstance(v, dict)}, indent=2, default=float))


if __name__ == "__main__":
    main()
