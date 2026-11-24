"""Run stages 0-2 (QA, features, labels) on an exporter CSV. Example:
python scripts/run_stages_0_2.py --csv data/synth_edge.csv --spread-csv data/synth_edge_spread.csv --out output/run1
"""

import argparse
import json

from edge_research.config import load_config
from edge_research.pipeline import run_stages_0_2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="YAML overriding config/default.yaml")
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--symbol", default="SYMBOL")
    a = ap.parse_args()
    print(json.dumps(run_stages_0_2(a.csv, a.out, load_config(a.config), a.symbol, a.spread_csv), indent=2))


if __name__ == "__main__":
    main()
