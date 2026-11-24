"""Run stages 0-13, the full pipeline including the MQL5 EAs (stage 12) and the reporting layer (stage 13, M12). Example:
python scripts/run_stages_0_13.py --csv data/synth_strong.csv --spread-csv data/synth_strong_spread.csv --out output/run1
"""

import argparse
import json

from edge_research.config import load_config
from edge_research.pipeline import run_stages_0_13
from edge_research.utils.memory import MemoryGuard, set_guard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--symbol", default="SYMBOL")
    ap.add_argument("--max-replicates", type=int, default=None)
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="stop cleanly when system-available memory drops below this (spec 17)")
    ap.add_argument("--max-rss-gb", type=float, default=None, help="stop cleanly when the process RSS exceeds this")
    a = ap.parse_args()
    set_guard(MemoryGuard(max_rss_gb=a.max_rss_gb, min_free_gb=a.min_free_gb))
    print(json.dumps(run_stages_0_13(a.csv, a.out, load_config(a.config), a.symbol, a.spread_csv, a.max_replicates), indent=2, default=str))


if __name__ == "__main__":
    main()
