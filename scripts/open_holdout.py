"""Open the holdout ONCE for the rules that passed stage 10 (G1-G8). The access is logged in manifest.json; a second call is refused (design 14.4).
Run this only when you will not change any rule afterwards. Example:
python scripts/open_holdout.py --run output/run1 --yes-i-will-not-tune-after-this
"""

import argparse
import json

from edge_research.config import load_config
from edge_research.pipeline import run_holdout_stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run folder with a finished stage 10")
    ap.add_argument("--config", default=None)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--yes-i-will-not-tune-after-this", action="store_true", dest="confirmed")
    a = ap.parse_args()
    if not a.confirmed:
        raise SystemExit("refusing: pass --yes-i-will-not-tune-after-this (the holdout can be opened only once)")
    print(json.dumps(run_holdout_stage(a.run, load_config(a.config), a.spread_csv), indent=2, default=str))


if __name__ == "__main__":
    main()
