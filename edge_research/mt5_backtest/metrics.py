"""Summary metrics (design 15.8): Strategy-Tester style, R based, structure, time tables, costs, path risk, diagnostics. Pure functions of the accounted trades."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from edge_research.mt5_backtest.accounting import AccountResult


def _max_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return int(best)


def drawdown(series: np.ndarray) -> Tuple[float, float, int]:
    """(max absolute drawdown, max drawdown as a fraction of the running peak, longest stretch in bars below the running peak)."""
    s = np.asarray(series, dtype=np.float64)
    if s.size == 0:
        return 0.0, 0.0, 0
    peak = np.maximum.accumulate(s)
    dd = peak - s
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(peak > 0, dd / peak, 0.0)
    return float(dd.max()), float(pct.max()), _max_run(dd > 1e-12)


def trimmed_mean(r: np.ndarray, frac: float) -> float:
    """Mean of R after dropping the ``frac`` share of the LARGEST trades (at least one trade is kept)."""
    r = np.sort(np.asarray(r, dtype=np.float64))
    k = int(np.floor(frac * len(r)))
    return float(r[: max(1, len(r) - k)].mean()) if len(r) else float("nan")


def period_table(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "trades", "net_profit", "expectancy_R", "win_rate"])
    key = pd.to_datetime(trades["exit_time"]).dt.to_period(freq).astype(str)
    g = trades.groupby(key)
    return pd.DataFrame({"trades": g.size(), "net_profit": g["profit"].sum(), "expectancy_R": g["r_net"].mean(), "win_rate": g["profit"].apply(lambda v: float((v > 0).mean()))}
                        ).rename_axis("period").reset_index()


def summarize(res: AccountResult, index: pd.DatetimeIndex, settings: Dict[str, Any], point: float, engine_stats: Optional[Dict[str, Any]] = None,
              window: Optional[slice] = None) -> Dict[str, Any]:
    """``window``: bar range the drawdown, exposure and trades-per-year statistics refer to (a segment backtest: the account is flat outside it). Default: all bars."""
    t = res.trades
    init = float(settings["initial_deposit"])
    n_all = len(index)
    sl = window if window is not None else slice(0, n_all)
    idx_w = index[sl]
    n_bars = len(idx_w)
    years = max((idx_w[-1] - idx_w[0]).total_seconds() / (365.25 * 86400), 1e-9) if n_bars > 1 else float("nan")
    out: Dict[str, Any] = {"initial_deposit": init, "total_trades": int(len(t)), "years": float(years)}
    eq_dd, eq_dd_pct, eq_dd_len = drawdown(res.equity.to_numpy()[sl])
    bal_dd, bal_dd_pct, _ = drawdown(res.balance.to_numpy()[sl])
    out.update(final_balance=float(res.balance.iloc[-1]), balance_drawdown_abs=bal_dd, balance_drawdown_pct=bal_dd_pct * 100.0,
               equity_drawdown_abs=eq_dd, equity_drawdown_pct=eq_dd_pct * 100.0, longest_drawdown_bars=eq_dd_len)
    if t.empty:
        out.update(total_net_profit=0.0, gross_profit=0.0, gross_loss=0.0, profit_factor=float("nan"), expected_payoff=float("nan"), recovery_factor=float("nan"),
                   long_trades=0, short_trades=0, profit_trades_pct=float("nan"), expectancy_R=float("nan"))
    else:
        p = t["profit"].to_numpy()
        r = t["r_net"].to_numpy()
        gp, gl = float(p[p > 0].sum()), float(-p[p < 0].sum())
        wins, losses = r > 0, r < 0
        net = float(p.sum())
        out.update(
            total_net_profit=net, gross_profit=gp, gross_loss=gl, profit_factor=gp / gl if gl > 0 else float("inf"), expected_payoff=float(p.mean()),
            recovery_factor=net / eq_dd if eq_dd > 0 else float("nan"),
            long_trades=int((t["direction"] == "long").sum()), short_trades=int((t["direction"] == "short").sum()),
            profit_trades_pct=float((p > 0).mean() * 100.0), largest_profit_trade=float(p.max()), largest_loss_trade=float(p.min()),
            average_profit_trade=float(p[p > 0].mean()) if (p > 0).any() else float("nan"), average_loss_trade=float(p[p < 0].mean()) if (p < 0).any() else float("nan"),
            max_consecutive_wins=_max_run(p > 0), max_consecutive_losses=_max_run(p < 0),
            # R based
            expectancy_R=float(r.mean()), median_R=float(np.median(r)), win_rate=float((r > 0).mean()),
            avg_win_R=float(r[wins].mean()) if wins.any() else float("nan"), avg_loss_R=float(r[losses].mean()) if losses.any() else float("nan"),
            expectancy_R_trim1=trimmed_mean(r, 0.01), expectancy_R_trim5=trimmed_mean(r, 0.05),
            drawdown_R=drawdown(np.cumsum(r))[0],
            # structure
            trades_per_year=float(len(t) / years), avg_duration_bars=float((t["exit_bar"] - t["entry_bar"]).mean()),
            # costs
            cost_share_of_gross=float(res.stats["cost_money"] / res.stats["gross_move_money"]) if res.stats.get("gross_move_money", 0) > 0 else float("nan"),
            avg_spread_entry_points=float(t["spread_entry"].mean() / point), avg_spread_exit_points=float(t["spread_exit"].mean() / point),
            exit_reasons={k: int(v) for k, v in t["exit_reason"].value_counts().items()},
        )
        # exposure: share of bars with at least one position
        diff = np.zeros(n_all + 1)
        last = np.where(t["_ib"].to_numpy(dtype=bool), t["exit_bar"].to_numpy(), t["exit_bar"].to_numpy() - 1)
        for a, b in zip(t["entry_bar"].to_numpy(), last):
            if b >= a:
                diff[a] += 1
                diff[b + 1] -= 1
        conc = np.cumsum(diff)[:n_all][sl]
        out["exposure_pct"] = float((conc > 0).mean() * 100.0)
        out["max_concurrent_positions"] = int(conc.max())
        for name in ("mae", "mfe"):
            v = t[name].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            out[f"{name}_quantiles_R"] = {str(q): float(np.quantile(v, q / 100.0)) for q in (50, 75, 90, 95)} if len(v) else {}
    if engine_stats:
        for k in ("signals_long", "signals_short", "overlap_skipped", "n_open_at_end", "n_spread_fallback", "invalid_window", "conflict_skipped", "closed_at_end"):
            if k in engine_stats:
                out[k] = engine_stats[k]
        out["n_signals"] = int(engine_stats.get("signals_long", 0) + engine_stats.get("signals_short", 0))
        out["n_skipped_busy"] = int(engine_stats.get("overlap_skipped", 0))
    return out


def report_md(summary: Dict[str, Any], by_year: pd.DataFrame, by_month: pd.DataFrame, title: str = "Backtest report") -> str:
    s = summary
    f = lambda k, nd=2: ("n/a" if s.get(k) is None or (isinstance(s.get(k), float) and not np.isfinite(s.get(k))) else f"{s[k]:.{nd}f}")
    L = [f"# {title}", "",
         f"- Trades: {s['total_trades']} (long {s.get('long_trades', 0)}, short {s.get('short_trades', 0)}); initial deposit {s['initial_deposit']:.2f}; final balance {s['final_balance']:.2f}",
         f"- Net profit {f('total_net_profit')}; profit factor {f('profit_factor')}; expected payoff {f('expected_payoff')}; recovery factor {f('recovery_factor')}",
         f"- Equity drawdown {f('equity_drawdown_abs')} ({f('equity_drawdown_pct')} %); balance drawdown {f('balance_drawdown_abs')} ({f('balance_drawdown_pct')} %); longest drawdown {s['longest_drawdown_bars']} bars",
         f"- R: expectancy {f('expectancy_R', 3)}, median {f('median_R', 3)}, win rate {f('win_rate', 3)}, trimmed (1 % / 5 % largest removed) {f('expectancy_R_trim1', 3)} / {f('expectancy_R_trim5', 3)}, drawdown {f('drawdown_R')} R",
         f"- Structure: {f('trades_per_year', 1)} trades/year, exposure {f('exposure_pct', 1)} %, mean duration {f('avg_duration_bars', 1)} bars, max concurrent positions {s.get('max_concurrent_positions', 0)}",
         f"- Costs: {f('cost_share_of_gross', 3)} of the gross profit; mean spread entry / exit {f('avg_spread_entry_points', 1)} / {f('avg_spread_exit_points', 1)} points",
         f"- Diagnostics: signals {s.get('n_signals', 'n/a')}, skipped (busy) {s.get('n_skipped_busy', 'n/a')}, open at end {s.get('n_open_at_end', 'n/a')}, spread fallback bars {s.get('n_spread_fallback', 'n/a')}",
         f"- Exit reasons: {s.get('exit_reasons', {})}", ""]
    for name, tab in (("By year", by_year), ("By month (first 24)", by_month.head(24))):
        if len(tab):
            L += [f"## {name}", "", "| period | trades | net profit | expectancy R | win rate |", "|---|---|---|---|---|"]
            L += [f"| {r.period} | {r.trades} | {r.net_profit:.2f} | {r.expectancy_R:.3f} | {r.win_rate:.2f} |" for r in tab.itertuples()]
            L.append("")
    return "\n".join(L)
