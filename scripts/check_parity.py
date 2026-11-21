"""Parity checks of an EA run against the Python side (design 16.6, gate G9 automated part). Example:
python scripts/check_parity.py --run output/run1 --rule R02441 --features ER_R02441_features.csv --trades ER_R02441_trades.csv --period 2015-03-01 2016-01-01

Get the files like this: copy the EA's Common\\Files\\ER_<rule>_features.csv (run once with InpDumpFeatures = true) and ER_<rule>_trades.csv (Strategy Tester, Open prices only,
InpMaxGapHours = 0) next to this repository. Either file may be omitted; a missing check is reported as NOT_RUN. The engine side of the trade check runs with
``ea_sim.parity_cfg`` (no weekend exclusion, no slippage). Outputs: <run>/12_ea/parity_<rule>.md and .json.
"""

import argparse
import json
from pathlib import Path

from edge_research.codegen import parity as P
from edge_research.codegen.ea_sim import parity_cfg
from edge_research.codegen.plan import build_plan
from edge_research.codegen.stage import load_registry
from edge_research.config import load_config
from edge_research.mt5_backtest.backtest import run_backtest
from edge_research.pipeline import load_run_inputs
from edge_research.rules import load_rules


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--features", default=None, help="the EA's feature dump")
    ap.add_argument("--trades", default=None, help="the EA's trade log")
    ap.add_argument("--config", default=None)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--period", nargs=2, default=None, metavar=("START", "END"), help="restrict the trade comparison to entries in this period")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--skip-first", type=int, default=500, help="warm-up bars skipped in the feature / signal comparison")
    ap.add_argument("--no-profit", action="store_true", help="do not compare profit (tester account settings differ from the config costs)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = Path(a.run)
    rules = load_rules(a.rules or run / "10_validation" / "accepted_rules.json")
    rule = next((r for r in rules if r.rule_id == a.rule), None)
    if rule is None:
        raise SystemExit(f"rule {a.rule} not found")
    plan = build_plan(rule, cfg, load_registry(run))
    df, ft, spread, _ = load_run_inputs(run, cfg, a.spread_csv)
    feats = sigs = trades = None
    if a.features:
        dump = P.read_feature_dump(a.features, plan.feature_names)
        feats = P.check_features(dump, ft, plan.feature_names, a.tol, a.skip_first)
        sigs = P.check_signals(dump, rule, ft, a.tol, a.skip_first)
    if a.trades:
        pc = parity_cfg(cfg)
        eng = run_backtest(df, pc, [rule], ft, spread).trades
        log = P.read_trade_log(a.trades, int(cfg["data"]["timeframe_minutes"]))
        hmax = plan.exit["n"] if plan.exit["kind"] == "T" else plan.exit["h"]
        trades = P.check_trades(log, eng, df.index, hmax, tuple(a.period) if a.period else None, not a.no_profit)
    verdict = P.g9_verdict(feats, sigs, trades)
    out = run / "12_ea"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"parity_{rule.rule_id}.md").write_text(P.parity_md(rule.rule_id, feats, sigs, trades, verdict), encoding="utf-8")
    (out / f"parity_{rule.rule_id}.json").write_text(json.dumps({"features": feats, "signals": sigs, "trades": trades, "verdict": verdict}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
