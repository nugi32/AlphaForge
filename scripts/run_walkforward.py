"""Outer walk-forward (design 14.3): the whole procedure (stages 0-8) on rolling windows of the data, evaluated on each test window with the engine.
Never uses the holdout rows of the full run. Resumable (finished windows are kept). Windows are configured in validation.walkforward. Example:
python scripts/run_walkforward.py --csv data/synth_strong.csv --spread-csv data/synth_strong_spread.csv --run output/run1
"""

import argparse
import json

from edge_research.config import load_config
from edge_research.validation.walkforward import run_walkforward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", required=True, help="output folder (a finished stage 9 here is only needed for require_null_approx)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--symbol", default="SYMBOL")
    ap.add_argument("--max-windows", type=int, default=None, help="stop after this many NEW windows in this call; run again to continue")
    a = ap.parse_args()
    print(json.dumps(run_walkforward(a.csv, a.run, load_config(a.config), a.spread_csv, a.symbol, None, a.max_windows), indent=2, default=str))


if __name__ == "__main__":
    main()
