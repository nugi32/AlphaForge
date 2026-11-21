"""Parity checks of an EA run against the Python side (design 16.6, gate G9). The EA writes two files (Common\\Files):

``ER_<rule>_features.csv``  per closed bar: ``time`` (open time of the decision bar = the Python table row), the rule's feature values, ``signal`` (0/1). Written when ``InpDumpFeatures``.
``ER_<rule>_trades.csv``    per closed trade: rule_id, entry_time, exit_time (bar open times), direction, entry_price, exit_price, exit_reason, volume, profit, gap_crossed, bars_held.

* feature parity   Python table vs dump, tolerance ``tol`` (the exporter writes 6 decimals); the share of deviating bars is reported.
* signal parity    boolean signal per bar; a mismatch is EXPLAINED only when a feature of one of the rule's atoms lies within ``tol`` of its threshold (reported), otherwise it is HARD.
* trade parity     ``compare_trades`` (engine list vs EA log). Run it with ``ea_sim.parity_cfg`` (no weekend exclusion) and ``InpMaxGapHours`` = 0, see ``ea_sim``.

Nothing here claims G9 as a whole: stress runs (tick / 1-minute OHLC mode) and the review checklist are manual; ``g9_verdict`` says so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from edge_research.mt5_backtest.compare import compare_trades
from edge_research.rules import FrozenRule, atom_mask, rule_signal

TIME_FORMAT = "%Y.%m.%d %H:%M"
LOG_COLUMNS = ["rule_id", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "exit_reason", "volume", "profit", "gap_crossed", "bars_held"]


def _parse_time(s: pd.Series) -> pd.Series:
    """EA files use minutes; a Strategy Tester export may carry seconds (floored to the bar open by the callers that need it)."""
    try:
        return pd.to_datetime(s, format=TIME_FORMAT)
    except ValueError:
        return pd.to_datetime(s, format=TIME_FORMAT + ":%S")


def read_feature_dump(path: str, names: Sequence[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    expect = ["time", *names, "signal"]
    if list(raw.columns) != expect:
        raise ValueError(f"feature dump columns {list(raw.columns)} differ from the expected {expect}")
    raw["time"] = _parse_time(raw["time"])
    if raw["time"].duplicated().any():
        raise ValueError("feature dump has duplicate times (a tester run appended to an old file?)")
    return raw.set_index("time").sort_index()


def read_trade_log(path: str, tf_minutes: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if list(raw.columns) != LOG_COLUMNS:
        raise ValueError(f"trade log columns {list(raw.columns)} differ from the expected {LOG_COLUMNS}")
    for c in ("entry_time", "exit_time"):
        raw[c] = _parse_time(raw[c]).dt.floor(f"{int(tf_minutes)}min")
    return raw


def _align(dump: pd.DataFrame, table: pd.DataFrame, skip_first: int) -> pd.DatetimeIndex:
    idx = dump.index.intersection(table.index[skip_first:])
    if len(idx) == 0:
        raise ValueError("the dump and the feature table share no bar times (different symbol, period or time zone?)")
    return idx


def check_features(dump: pd.DataFrame, table: pd.DataFrame, names: Sequence[str], tol: float = 1e-4, skip_first: int = 0, max_deviating_pct: float = 0.1) -> Dict[str, Any]:
    """Per feature: bars compared, NaN-pattern mismatches, bars over tolerance (|a - b| <= tol x max(1, |b|)), the worst difference. ``accepted`` needs every
    feature's deviating share (NaN mismatches included) <= ``max_deviating_pct`` percent."""
    idx = _align(dump, table, skip_first)
    per: Dict[str, Any] = {}
    ok_all = True
    for nme in names:
        a = dump.loc[idx, nme].to_numpy(np.float64)
        b = table.loc[idx, nme].to_numpy(np.float64)
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        nan_mis = int((fa != fb).sum())
        diff = np.abs(a - b)
        over = both & (diff > tol * np.maximum(1.0, np.abs(b)))
        n_dev = int(over.sum()) + nan_mis
        pct = 100.0 * n_dev / len(idx)
        ok = pct <= max_deviating_pct
        ok_all &= ok
        per[nme] = {"n_compared": int(len(idx)), "n_both_finite": int(both.sum()), "nan_mismatch": nan_mis, "n_over_tol": int(over.sum()), "deviating_pct": pct,
                    "max_abs_diff": float(diff[both].max()) if both.any() else 0.0, "ok": bool(ok)}
    return {"n_bars": int(len(idx)), "tol": tol, "max_deviating_pct": max_deviating_pct, "per_feature": per, "accepted": bool(ok_all)}


def check_signals(dump: pd.DataFrame, rule: FrozenRule, table: pd.DataFrame, tol: float = 1e-4, skip_first: int = 0) -> Dict[str, Any]:
    """Signal parity. A differing bar is ``explained`` when some atom feature (EA value or Python value) lies within ``tol`` of the threshold, else ``hard``."""
    idx = _align(dump, table, skip_first)
    py = pd.Series(rule_signal(rule, table), index=table.index).loc[idx].to_numpy()
    ea = (dump.loc[idx, "signal"].to_numpy() == 1)
    diff = py != ea
    near = np.zeros(len(idx), dtype=bool)
    for a in rule.atoms:
        thr = float(a["threshold"])
        va, vb = dump.loc[idx, a["feature"]].to_numpy(np.float64), table.loc[idx, a["feature"]].to_numpy(np.float64)
        lim = tol * max(1.0, abs(thr))
        near |= (np.abs(va - thr) <= lim) | (np.abs(vb - thr) <= lim)
    hard = diff & ~near
    first = None
    if hard.any():
        k = int(np.flatnonzero(hard)[0])
        first = {"time": str(idx[k]), "python": bool(py[k]), "ea": bool(ea[k])}
    return {"n_bars": int(len(idx)), "n_signals_python": int(py.sum()), "n_signals_ea": int(ea.sum()), "n_differ": int(diff.sum()), "n_explained_near_threshold": int((diff & near).sum()),
            "n_hard": int(hard.sum()), "first_hard": first, "accepted": bool(not hard.any() and len(idx) > 0)}


def tail_rule(trades: pd.DataFrame, index: pd.DatetimeIndex, hmax: int) -> pd.DataFrame:
    """The engine skips signals that need bars beyond the data (``signal_bar > n - 2 - hmax``); apply the same cut to an EA log (a position the tester force-closes at the end)."""
    if trades.empty:
        return trades
    pos = index.get_indexer(pd.DatetimeIndex(trades["entry_time"]))
    keep = (pos >= 1) & ((pos - 1) <= len(index) - 2 - int(hmax))
    return trades[keep].reset_index(drop=True)


def check_trades(ea_log: pd.DataFrame, engine_trades: pd.DataFrame, index: Optional[pd.DatetimeIndex] = None, hmax: Optional[int] = None,
                 period: Optional[tuple] = None, use_profit: bool = True, price_tol: float = 1e-4, pnl_rel_tol: float = 0.01) -> Dict[str, Any]:
    """EA trade log vs the engine's accounted trade list (``run_backtest(...).trades`` with ``parity_cfg``)."""
    a, b = engine_trades.copy(), ea_log.copy()
    if period is not None:
        lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1])
        a, b = a[(a["entry_time"] >= lo) & (a["entry_time"] <= hi)], b[(b["entry_time"] >= lo) & (b["entry_time"] <= hi)]
    if index is not None and hmax is not None:
        a, b = tail_rule(a.reset_index(drop=True), index, hmax), tail_rule(b.reset_index(drop=True), index, hmax)
    if not use_profit:
        a, b = a.drop(columns=[c for c in ("profit",) if c in a.columns]), b.drop(columns=[c for c in ("profit",) if c in b.columns])
    res = compare_trades(a, b, price_tol=price_tol, pnl_rel_tol=pnl_rel_tol)
    res["gap_crossed_in_ea_log"] = int(ea_log["gap_crossed"].sum()) if "gap_crossed" in ea_log.columns else None
    return res


def g9_verdict(features: Optional[Dict[str, Any]], signals: Optional[Dict[str, Any]], trades: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """FAIL when a check that ran failed; NOT_RUN when one of the three automated checks is missing; PASS_AUTOMATED otherwise. The stress run and the review checklist stay manual."""
    parts = {"feature_parity": features, "signal_parity": signals, "trade_parity": trades}
    missing = [k for k, v in parts.items() if v is None]
    failed = [k for k, v in parts.items() if v is not None and not v["accepted"]]
    status = "FAIL" if failed else ("NOT_RUN" if missing else "PASS_AUTOMATED")
    return {"status": status, "failed": failed, "not_run": missing,
            "manual_remaining": ["stress run in tick / 1-minute OHLC mode (design 16.6)", "review checklist in the EA header", "custom-indicator (iCustom) timing check when ps_* / volregime_* are used"]}


def parity_md(rule_id: str, feats: Optional[Dict[str, Any]], sigs: Optional[Dict[str, Any]], trades: Optional[Dict[str, Any]], verdict: Dict[str, Any]) -> str:
    L: List[str] = [f"# Parity report {rule_id}", "", f"**G9 automated verdict: {verdict['status']}**" + (f" (failed: {', '.join(verdict['failed'])})" if verdict["failed"] else "") +
                    (f" (not run: {', '.join(verdict['not_run'])})" if verdict["not_run"] else ""), ""]
    if feats:
        L += ["## Feature parity", f"{feats['n_bars']} bars, tolerance {feats['tol']}, accepted: {feats['accepted']}", "", "| feature | deviating % | NaN mismatch | over tol | max abs diff |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['deviating_pct']:.4f} | {v['nan_mismatch']} | {v['n_over_tol']} | {v['max_abs_diff']:.3g} |" for k, v in feats["per_feature"].items()]
        L.append("")
    if sigs:
        L += ["## Signal parity", f"python {sigs['n_signals_python']} / EA {sigs['n_signals_ea']} signals, differing bars {sigs['n_differ']} (near a threshold: {sigs['n_explained_near_threshold']}, hard: {sigs['n_hard']}); accepted: {sigs['accepted']}"]
        if sigs["first_hard"]:
            L.append(f"First hard difference: {sigs['first_hard']}")
        L.append("")
    if trades:
        L += ["## Trade parity", f"engine {trades['n_engine']} / EA {trades['n_mt5']} trades, matched {trades['n_matched']}, entry match {trades['entry_match_pct']:.2f} %, exit match {trades['exit_match_pct']:.2f} %, "
              f"price mismatches {trades['n_price_mismatch']}, accepted: {trades['accepted']}"]
        if "profit_rel_diff" in trades:
            L.append(f"Profit: engine {trades['profit_engine']:.2f}, EA {trades['profit_mt5']:.2f}, relative difference {trades['profit_rel_diff']:.4f}")
        if trades.get("first_difference"):
            L.append(f"First difference (fix this before trusting any number): {trades['first_difference']}")
        L.append("")
    L += ["## Still manual (not covered by this report)"] + [f"- {x}" for x in verdict["manual_remaining"]]
    return "\n".join(L) + "\n"
