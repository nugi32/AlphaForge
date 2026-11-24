"""Create a synthetic exporter-style CSV (and spread CSV) for pipeline tests. Example:
python scripts/make_synth.py --out data/synth_edge.csv --bars 30000 --edge
"""

import argparse
from pathlib import Path

from edge_research.synth.generator import PlantedEdge, make_synthetic_bars, write_exporter_csv


STRONG = PlantedEdge(drop_sigmas=1.0, drift_sigmas=3.0, hours=tuple(range(6, 14)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bars", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--edge", action="store_true", help="plant a known mean-reversion edge (default PlantedEdge: weak, usually NOT found, NOTES Q-005)")
    ap.add_argument("--strong", action="store_true", help="plant the strong edge used by the tests (drop 1.0, drift 3.0, hours 6-13)")
    args = ap.parse_args()
    df, meta = make_synthetic_bars(args.bars, args.seed, plant_edge=(STRONG if args.strong else PlantedEdge()) if (args.edge or args.strong) else None)
    out = write_exporter_csv(df, args.out)
    spread = meta["spread_points"].to_frame()
    spread.insert(0, "time", spread.index.strftime("%Y%m%d %H:%M"))
    spread.to_csv(Path(out).with_name(Path(out).stem + "_spread.csv"), index=False)
    print(f"wrote {out} ({len(df)} bars, planted triggers: {int(meta['trigger'].sum())})")


if __name__ == "__main__":
    main()
