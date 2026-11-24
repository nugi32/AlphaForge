"""Run the pipeline on ONE CSV written by mql5/dataExporter.mq5 (no separate spread file). Examples:

    python scripts/run_mql5_pipeline.py audit --csv data/EURUSD_H1_export_20261101.csv
    python scripts/run_mql5_pipeline.py run   --csv data/EURUSD_H1_export_20261101.csv --out output/eurusd_h1 --through 2
    python scripts/run_mql5_pipeline.py sample --out data/EURUSD_H1_export_sample.csv --bars 3000      # synthetic file in the exporter format (no MT5 needed)

``audit`` lists, for every column of the file, the features that read it and stops with an error code when a column is read by no feature.
``run`` executes stages 0..THROUGH (2 = QA + features + labels, 3 = + EDA, ... 10 = + validation, 13 = everything incl. the report).
Symbol and timeframe come from the exporter file name (<SYMBOL>_<TF>_export_<date>.csv); price digits / point, bars per year and the weekend-gap threshold are
derived from the file. Override anything with --config (YAML) or config/MT5_TESTER/<SYMBOL>/<TF>.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # run from a checkout without `pip install -e .`

from edge_research.mql5_loader.coverage import UnprocessedColumnsError
from edge_research.mql5_loader.runner import audit_file, run_pipeline
from edge_research.mql5_loader.sample import write_exporter_csv


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML merged over config/default.yaml + config/mql5_loader.yaml")
    p.add_argument("--symbol", default=None, help="default: from the file name")
    p.add_argument("--timeframe", default=None, help="M1 M5 M15 M30 H1 H4 D1 W1 MN1; default: from the file name / bar spacing")
    p.add_argument("--digits", type=int, default=None, help="price digits (default: read from the close column)")
    p.add_argument("--gate-news", action="store_true", help="exclude the news_* columns from the features (they are processed by default)")
    p.add_argument("--allow-unprocessed", action="store_true", help="do not stop when an exporter column is read by no feature")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="column coverage + integrity findings, no pipeline run")
    a.add_argument("--csv", required=True)
    _common(a)
    r = sub.add_parser("run", help="stages 0..THROUGH on one CSV")
    r.add_argument("--csv", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--through", type=int, default=2, help="last stage function: 2..11 or 13 (run_stages_0_N)")
    r.add_argument("--max-replicates", type=int, default=None, help="stage 9+: bound the null replicates (slow machines)")
    _common(r)
    s = sub.add_parser("sample", help="write a synthetic CSV in the exporter format")
    s.add_argument("--out", required=True)
    s.add_argument("--bars", type=int, default=3000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--timeframe-minutes", type=int, default=60)
    s.add_argument("--with-news", action="store_true")
    args = ap.parse_args()

    if args.cmd == "sample":
        print(write_exporter_csv(args.out, args.bars, args.seed, args.timeframe_minutes, with_news=args.with_news))
        return 0
    kw = dict(user_config=args.config, symbol=args.symbol, digits=args.digits, timeframe=args.timeframe, gate_news=args.gate_news, allow_unprocessed=args.allow_unprocessed)
    try:
        if args.cmd == "audit":
            res = audit_file(args.csv, **kw)
            print(res["markdown"])
            for f in res["table"].findings:
                print(f"[{f.level}] {f.check}: {f.message}")
            for note in res["table"].repairs:
                print(f"[repair] {note}")
            return 1 if res["audit"].unprocessed else 0
        print(json.dumps(run_pipeline(args.csv, args.out, args.through, max_replicates=args.max_replicates, **kw), indent=2, default=str))
        return 0
    except UnprocessedColumnsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
