"""Run frozen rules through the engine and the accounting (stage 11 building block). One engine run per rule (hedging: several rules may hold positions at once, each rule
respects its own ``max_open_positions``); sizing, balance and equity are computed on the combined trade list so the rules share one account."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from edge_research.mt5_backtest.accounting import AccountResult, account_trades
from edge_research.mt5_backtest.engine import engine_settings, run_engine
from edge_research.mt5_backtest.metrics import period_table, report_md, summarize
from edge_research.rules import FrozenRule, exit_condition_signal, rule_signal


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    balance: pd.Series
    equity: pd.Series
    summary: Dict[str, Any]
    by_year: pd.DataFrame
    by_month: pd.DataFrame
    per_rule: Dict[str, Dict[str, Any]]
    portfolio: Dict[str, Any] = field(default_factory=dict)
    engine_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, cfg: Dict[str, Any], rules: Sequence[FrozenRule], feature_table: pd.DataFrame, spread_points: Optional[pd.Series] = None,
                 rows: Optional[slice] = None) -> BacktestResult:
    """``rows``: only signals whose decision bar lies in this row range are used (the exits may run past it; the embargo of the splits covers that)."""
    eng = engine_settings(cfg)
    if eng["account_mode"] == "netting":
        warnings.warn("account_mode=netting: opposite positions of one symbol would net in MT5, the engine keeps them separate (hedging semantics)")
    if len(feature_table) != len(df) or not feature_table.index.equals(df.index):
        raise ValueError("feature_table must be aligned with the bar table")
    n = len(df)
    lo, hi = (0, n) if rows is None else (rows.start or 0, rows.stop if rows.stop is not None else n)
    pieces, stats, last_market = [], {}, None
    for rule in rules:
        rule.validate()
        sig = rule_signal(rule, feature_table)
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        sig = sig & mask
        exc = exit_condition_signal(rule, feature_table)             # design 15.4: an optional condition exit closes the position early
        exit_signals = {rule.direction: exc} if exc is not None else None
        r = run_engine(df, cfg, {rule.direction: sig}, rule.exit, spread_points, exit_signals=exit_signals, cat_stop_atr=rule.cat_stop_atr)
        tr = r.trades.copy()
        tr["rule_id"] = rule.rule_id
        pieces.append(tr)
        stats[rule.rule_id] = r.stats
        last_market = r.market
    if last_market is None:
        raise ValueError("no rules given")
    allt = pd.concat([p for p in pieces if len(p)] or [pieces[0]], ignore_index=True)
    acc = account_trades(allt, last_market, eng, df.index, [r.rule_id for r in rules])
    point = last_market.cost.point
    agg_stats = {k: int(sum(s.get(k, 0) for s in stats.values())) for k in next(iter(stats.values()))}
    last_bar = int(acc.trades["exit_bar"].max()) + 2 if len(acc.trades) else hi
    window = slice(lo, min(n, max(hi, last_bar)))                                  # flat outside the segment: do not count it as drawdown time or idle exposure
    summary = summarize(acc, df.index, eng, point, agg_stats, window if rows is not None else None)
    per_rule: Dict[str, Dict[str, Any]] = {}
    for rid, g in acc.trades.groupby("rule_id"):
        per_rule[rid] = {"trades": int(len(g)), "net_profit": float(g["profit"].sum()), "expectancy_R": float(g["r_net"].mean()), "win_rate": float((g["profit"] > 0).mean())}
    return BacktestResult(acc.trades, acc.balance, acc.equity, summary, period_table(acc.trades, "Y"), period_table(acc.trades, "M"), per_rule,
                          portfolio_diagnostics(acc, df.index, [r.rule_id for r in rules]), stats)


def portfolio_diagnostics(acc: AccountResult, index: pd.DatetimeIndex, rule_ids: List[str]) -> Dict[str, Any]:
    """Correlation of the rules' daily realised P&L, maximum simultaneous exposure (lots), and the combined drawdown."""
    t = acc.trades
    out: Dict[str, Any] = {"n_rules": len(rule_ids)}
    if t.empty:
        return out
    day = pd.to_datetime(t["exit_time"]).dt.normalize()
    daily = t.assign(day=day).pivot_table(index="day", columns="rule_id", values="profit", aggfunc="sum").reindex(columns=rule_ids).fillna(0.0)
    if len(rule_ids) > 1 and len(daily) > 2:
        corr = daily.corr().fillna(0.0)
        out["daily_pnl_correlation"] = {a: {b: float(corr.loc[a, b]) for b in rule_ids} for a in rule_ids}
        off = corr.to_numpy()[~np.eye(len(rule_ids), dtype=bool)]
        out["max_pairwise_correlation"] = float(off.max())
    n = len(index)
    diff_lots = np.zeros(n + 1)
    last = np.where(t["_ib"].to_numpy(dtype=bool), t["exit_bar"].to_numpy(), t["exit_bar"].to_numpy() - 1)
    for a, b, lots in zip(t["entry_bar"].to_numpy(), last, t["lots"].to_numpy()):
        if b >= a:
            diff_lots[a] += lots
            diff_lots[b + 1] -= lots
    out["max_simultaneous_lots"] = float(np.cumsum(diff_lots)[:n].max())
    return out


def write_backtest(res: BacktestResult, out_dir: str | Path, title: str = "Backtest report") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.drop(columns=[c for c in res.trades.columns if c.startswith("_")]).to_csv(out / "trades.csv", index=False)
    pd.DataFrame({"balance": res.balance, "equity": res.equity}).rename_axis("time").to_csv(out / "equity.csv")
    res.by_year.to_csv(out / "by_year.csv", index=False)
    res.by_month.to_csv(out / "by_month.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"summary": res.summary, "per_rule": res.per_rule, "portfolio": res.portfolio, "engine_stats": res.engine_stats}, indent=2, default=float))
    (out / "report.md").write_text(report_md(res.summary, res.by_year, res.by_month, title))
