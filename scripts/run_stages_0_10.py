"""Run stages 0-10 (everything up to the frozen-rule validation on the validation segment; the holdout stays closed unless validation.open_holdout is true in the config).
Resumable. Example:
python scripts/run_stages_0_10.py --csv data/synth_strong.csv --spread-csv data/synth_strong_spread.csv --out output/run1
"""

import argparse
import json

from edge_research.config import load_config
from edge_research.pipeline import run_stages_0_10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="YAML overriding config/default.yaml")
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--symbol", default="SYMBOL")
    ap.add_argument("--max-replicates", type=int, default=None, help="stop after this many NEW null replicates in this call; run again to continue")
    a = ap.parse_args()
    print(json.dumps(run_stages_0_10(a.csv, a.out, load_config(a.config), a.symbol, a.spread_csv, a.max_replicates), indent=2, default=str))


if __name__ == "__main__":
    main()
