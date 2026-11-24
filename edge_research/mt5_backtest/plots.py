"""Equity / balance chart (PNG) of a finished backtest.

Written next to ``trades.csv`` / ``equity.csv`` / ``report.md`` by :func:`edge_research.mt5_backtest.backtest.write_backtest`, so EVERY backtest run (stage 11,
``scripts/run_backtest.py``, the MQL5 loader runner) ends with a picture. Settings come from the ``backtest.chart`` config section, which can differ per
symbol / timeframe through ``config/MT5_TESTER/<SYMBOL>/<TF>.yaml``.

The figure uses the object-oriented matplotlib API with the Agg canvas only (no pyplot, no global backend switch), so it is safe on a headless machine.
A missing matplotlib never fails a run: a warning is issued and no PNG is written.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

CHART_DEFAULTS: Dict[str, Any] = {
    "enabled": True,            # false = no PNG at all
    "filename": "equity_balance.png",
    "dpi": 150,
    "width": 14.0,              # inches
    "height": 8.0,
    "drawdown_panel": True,     # lower panel: equity drawdown in % of the running peak
    "crop_to_trades": True,     # cut the flat stretch before the first entry and after the last exit (a train / val segment is flat outside its range)
    "stats_box": True,          # key numbers in a box on the equity panel
}


def chart_settings(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Defaults + ``cfg["backtest"]["chart"]``. An unknown key is an ERROR (it catches typos, same rule as the engine section)."""
    over = dict(((cfg or {}).get("backtest", {}) or {}).get("chart", {}) or {})
    unknown = sorted(set(over) - set(CHART_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown backtest.chart settings: {unknown}")
    out = {**CHART_DEFAULTS, **over}
    if int(out["dpi"]) < 30 or float(out["width"]) <= 0 or float(out["height"]) <= 0:
        raise ValueError("backtest.chart: dpi must be >= 30 and width / height > 0")
    return out


def _num(x: Any, nd: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(v):
        return "n/a"
    if np.isinf(v):
        return "inf"
    return f"{v:,.{nd}f}"


def active_window(trades: Optional[pd.DataFrame], n: int) -> slice:
    """Bar range from one bar before the first entry to two bars after the last exit (the whole range when there is no trade)."""
    if trades is None or len(trades) == 0 or "entry_bar" not in trades.columns:
        return slice(0, n)
    lo = max(int(trades["entry_bar"].min()) - 1, 0)
    hi = min(int(trades["exit_bar"].max()) + 2, n)
    return slice(lo, max(hi, lo + 2))


def plot_equity_balance(balance: pd.Series, equity: pd.Series, out_path: str | Path, title: str = "Backtest", summary: Optional[Dict[str, Any]] = None,
                        trades: Optional[pd.DataFrame] = None, settings: Optional[Dict[str, Any]] = None, currency: str = "") -> Optional[Path]:
    """Draw balance (step line) and equity (line) of one backtest into ``out_path`` (PNG). Returns the path, or None when the chart is switched off / matplotlib is missing."""
    s = {**CHART_DEFAULTS, **(settings or {})}
    if not s["enabled"]:
        return None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
        from matplotlib.figure import Figure
    except ImportError:
        warnings.warn("matplotlib is not installed: the equity / balance PNG was not written (pip install matplotlib)")
        return None
    if len(equity) != len(balance) or not equity.index.equals(balance.index):
        raise ValueError("balance and equity must share one index")

    sl = active_window(trades, len(equity)) if s["crop_to_trades"] else slice(0, len(equity))
    bal, eq = balance.iloc[sl], equity.iloc[sl]
    x = pd.DatetimeIndex(eq.index).to_pydatetime()
    unit = f" ({currency})" if currency else ""
    two = bool(s["drawdown_panel"])

    fig = Figure(figsize=(float(s["width"]), float(s["height"])), dpi=int(s["dpi"]))
    FigureCanvasAgg(fig)
    if two:
        ax, ax_dd = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    else:
        ax, ax_dd = fig.subplots(1, 1), None

    ax.plot(x, eq.to_numpy(), color="#1f77b4", lw=1.1, label="Equity")
    ax.step(x, bal.to_numpy(), where="post", color="#d62728", lw=1.3, label="Balance")
    ax.axhline(float((summary or {}).get("initial_deposit", bal.iloc[0])), color="#888888", lw=0.8, ls="--", label="Initial deposit")
    ax.set_ylabel(f"Account{unit}")
    ax.grid(True, alpha=0.3)
    fig.text(0.07, 0.975, title, fontsize=13, fontweight="bold", ha="left", va="top")                 # title, key numbers and legend sit ABOVE the axes: nothing covers the curves
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=3, fontsize=9, frameon=False)
    if s["stats_box"] and summary:
        line = (f"Trades {summary.get('total_trades', 'n/a')}   Net {_num(summary.get('total_net_profit'))}   Final balance {_num(summary.get('final_balance'))}   "
                f"PF {_num(summary.get('profit_factor'))}   Max equity DD {_num(summary.get('equity_drawdown_pct'), 1)} %   Expectancy {_num(summary.get('expectancy_R'), 3)} R")
        ax.text(0.0, 1.015, line, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, family="monospace", color="#333333")
    if ax_dd is not None:
        e = eq.to_numpy(dtype=np.float64)
        peak = np.maximum.accumulate(e)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (peak - e) / peak * 100.0, 0.0)
        ax_dd.fill_between(x, -dd, 0.0, color="#d62728", alpha=0.35, lw=0)
        ax_dd.plot(x, -dd, color="#d62728", lw=0.7)
        ax_dd.set_ylabel("Equity DD (%)")
        ax_dd.grid(True, alpha=0.3)
        last = ax_dd
    else:
        last = ax
    loc = AutoDateLocator()
    last.xaxis.set_major_locator(loc)
    last.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.885, bottom=0.07)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png")
    return out
