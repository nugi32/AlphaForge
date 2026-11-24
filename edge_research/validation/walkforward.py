"""Outer walk-forward (design 14.3): the WHOLE procedure (stages 0-8) on rolling windows, evaluated on the next test window with the engine.

Per window: the window's bars are cut from the raw export (the file keeps its format), stages 0-8 run on the TRAIN part only (``train_pct`` of the window; the test part is the
"validation" segment of that mini run, never read by stages 1-8), the best ``max_rules_per_window`` representatives (train-only ``plateau_score``) are frozen, and the engine
trades them on the test part. A window without rules is a valid result (no trades). All OOS trades of all windows form one series.

The rows after the FULL run's holdout start (minus the embargo) are never used by any window, so the walk-forward does not read the holdout (design 14.4).
The null test per window is not run (cost): ``require_null_approx`` filters by the null threshold of the FULL train run instead, an approximation that the report flags.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_research.config import config_hash
from edge_research.mt5_backtest.backtest import run_backtest
from edge_research.mt5_backtest.metrics import drawdown
from edge_research.atoms.atoms import load_atoms
from edge_research.splits import make_splits
from edge_research.validation.freeze import freeze_rules

WF_SECTIONS = ["data", "features", "labels", "costs", "qa", "eda", "atoms", "discovery", "candidates", "filters", "exits", "engine"]


# ---------------------------------------------------------------------------------------------------------------- geometry (pure)
def bar_windows(n_rows: int, train: int, test: int, step: int) -> List[Tuple[int, int, int]]:
    """(start, train_end, test_end) in rows for rolling windows [start, test_end); the last window must fit completely."""
    if min(train, test, step) < 1:
        raise ValueError("train, test and step must be >= 1")
    out, s = [], 0
    while s + train + test <= n_rows:
        out.append((s, s + train, s + train + test))
        s += step
    return out


def calendar_windows(times: pd.DatetimeIndex, train_years: float, test_years: float, step_years: float) -> List[Tuple[int, int, int]]:
    """Same, with window edges at calendar offsets from the first timestamp (days = years x 365.25). A window needs data up to its test end."""
    if min(train_years, test_years, step_years) <= 0:
        raise ValueError("train_years, test_years and step_years must be > 0")
    t0, tlast = times[0], times[-1]
    out, k = [], 0
    while True:
        a = t0 + pd.Timedelta(days=step_years * 365.25 * k)
        b = a + pd.Timedelta(days=train_years * 365.25)
        c = b + pd.Timedelta(days=test_years * 365.25)
        if c > tlast + pd.Timedelta(seconds=1):
            break
        i, j, l = (int(times.searchsorted(x, side="left")) for x in (a, b, c))
        out.append((i, j, l))
        k += 1
    return out


def window_pcts(train_n: int, test_n: int) -> Tuple[float, float, float]:
    tp = train_n / float(train_n + test_n)
    return tp, 1.0 - tp, 0.0


# ---------------------------------------------------------------------------------------------------------------- aggregation (pure)
def aggregate_windows(windows: List[Dict[str, Any]], trades: pd.DataFrame, initial_deposit: float) -> Dict[str, Any]:
    """Procedure metrics (design 14.3): share of windows with at least one rule, pooled OOS expectancy, OOS/IS ratio, stability across windows, pooled OOS drawdown."""
    n = len(windows)
    with_rules = [w for w in windows if w["n_rules"] > 0]
    out: Dict[str, Any] = {"n_windows": n, "n_windows_with_rules": len(with_rules), "share_windows_with_rules": len(with_rules) / n if n else float("nan"), "n_trades": int(len(trades))}
    if trades.empty:
        out.update(oos_mean_R=float("nan"), oos_net_profit=0.0, is_mean_R=float("nan"), oos_is_ratio=float("nan"), share_windows_positive=float("nan"), window_mean_R_std=float("nan"),
                   oos_equity_dd_pct=0.0, oos_equity_dd_abs=0.0, positive_oos=False)
        return out
    r = trades["r_net"].to_numpy(np.float64)
    p = trades.sort_values("exit_time")["profit"].to_numpy(np.float64)
    eq = initial_deposit + np.cumsum(p)
    dd_abs, dd_pct, _ = drawdown(np.concatenate([[initial_deposit], eq]))
    means = np.array([w["oos_mean_R"] for w in windows if w["n_trades"] > 0], dtype=np.float64)
    is_vals = np.array([w["is_mean_R"] for w in with_rules if np.isfinite(w["is_mean_R"])], dtype=np.float64)
    is_mean = float(is_vals.mean()) if is_vals.size else float("nan")
    oos = float(r.mean())
    out.update(oos_mean_R=oos, oos_net_profit=float(p.sum()), is_mean_R=is_mean, oos_is_ratio=oos / is_mean if is_mean and np.isfinite(is_mean) and is_mean > 0 else float("nan"),
               share_windows_positive=float((means > 0).mean()), window_mean_R_std=float(means.std(ddof=1)) if means.size > 1 else float("nan"),
               oos_equity_dd_pct=float(dd_pct * 100.0), oos_equity_dd_abs=float(dd_abs), positive_oos=bool(oos > 0 and p.sum() > 0))
    return out


# ---------------------------------------------------------------------------------------------------------------- one window
def _window_fp(csv_sha: str, bounds: Tuple[int, int, int], cfg: Dict[str, Any]) -> str:
    parts = [csv_sha, ",".join(map(str, bounds))] + [config_hash(cfg, s) for s in WF_SECTIONS] + [config_hash({"wf": cfg["validation"]["walkforward"], "n": cfg["validation"]["max_rules"]})]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _null_threshold(run: Path, cfg: Dict[str, Any]) -> Optional[float]:
    p = run / "09_null" / "null_info.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text(encoding="utf-8"))["summary"]["variants"].get("random")
    return float(v["S_threshold"]) if v else None


def _cut_csv(lines: List[str], a: int, b: int, out: Path) -> None:
    out.write_text("".join([lines[0]] + lines[1 + a: 1 + b]), encoding="utf-8")


def _run_window(k: int, bounds: Tuple[int, int, int], lines: List[str], times: pd.Series, spread: Optional[pd.DataFrame], cfg: Dict[str, Any], wf_dir: Path, symbol: str,
                null_thr: Optional[float]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    from edge_research.pipeline import NoGo, load_run_inputs, run_stages_0_8

    a, tr_end, b = bounds
    wd = wf_dir / f"w{k:03d}"
    wd.mkdir(parents=True, exist_ok=True)
    csv_w = wd / "window.csv"
    _cut_csv(lines, a, b, csv_w)
    sp_path = None
    if spread is not None:
        t0, t1 = times.iloc[a], times.iloc[b - 1]
        sp = spread[(spread["time"] >= t0) & (spread["time"] <= t1)]
        sp_path = wd / "window_spread.csv"
        sp.assign(time=sp["time"].dt.strftime("%Y%m%d %H:%M")).to_csv(sp_path, index=False)
    tp, vp, hp = window_pcts(tr_end - a, b - tr_end)
    cw = json.loads(json.dumps(cfg))
    cw["data"].update(train_pct=tp, val_pct=vp, holdout_pct=hp)
    info: Dict[str, Any] = {"window": k, "start_row": a, "train_end_row": tr_end, "end_row": b, "start_time": str(times.iloc[a]), "test_start_time": str(times.iloc[tr_end]), "end_time": str(times.iloc[b - 1]),
                            "n_rules": 0, "n_trades": 0, "oos_mean_R": float("nan"), "oos_net_profit": 0.0, "is_mean_R": float("nan"), "status": "ok", "rules": []}
    empty = pd.DataFrame()
    try:
        run_stages_0_8(csv_w, wd / "run", cw, symbol, sp_path)
    except NoGo:
        info["status"] = "no_go"
        return info, empty
    run = wd / "run"
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    if null_thr is not None and len(chosen):
        chosen = chosen[chosen["best_t"].astype(float) > null_thr]
    aset = load_atoms(run / "04_atoms")
    rules = freeze_rules(chosen, aset, cw, max_rules=int(cfg["validation"]["walkforward"]["max_rules_per_window"]))
    if not rules:
        info["status"] = "no_rules"
        return info, empty
    df, ft, spread_s, splits = load_run_inputs(run, cw, sp_path)
    res = run_backtest(df, cw, rules, ft, spread_s, splits.val)
    tr = res.trades.copy()
    tr["window"] = k
    s = res.summary
    info.update(n_rules=len(rules), n_trades=int(len(tr)), oos_mean_R=float(tr["r_net"].mean()) if len(tr) else float("nan"), oos_net_profit=float(s.get("total_net_profit", 0.0)),
                is_mean_R=float(np.mean([r.meta["chosen_smooth"] for r in rules])), test_rows=[splits.val.start, splits.val.stop],
                rules=[{"rule_id": r.rule_id, "exit": r.exit, "direction": r.direction, "expr": " AND ".join(r.meta.get("exprs", [])), "train_smooth_R": r.meta["chosen_smooth"],
                        "oos_trades": int((tr["rule_id"] == r.rule_id).sum()), "oos_mean_R": float(tr.loc[tr["rule_id"] == r.rule_id, "r_net"].mean()) if (tr["rule_id"] == r.rule_id).any() else float("nan")}
                       for r in rules])
    return info, tr.drop(columns=[c for c in tr.columns if c.startswith("_")])


# ---------------------------------------------------------------------------------------------------------------- driver
def run_walkforward(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], spread_csv: Optional[str | Path] = None, symbol: str = "SYMBOL",
                    progress=None, max_windows: Optional[int] = None) -> Dict[str, Any]:
    """Run (or resume) the outer walk-forward. Finished windows are kept (fingerprint: data file, window bounds, config); ``max_windows`` bounds the NEW windows of one call."""
    from edge_research.pipeline import load_spread_csv

    wf = cfg["validation"]["walkforward"]
    run = Path(run_dir)
    wf_dir = run / "walkforward"
    wf_dir.mkdir(parents=True, exist_ok=True)
    text = Path(csv_path).read_text(encoding="utf-8").splitlines(keepends=True)
    n_rows = len(text) - 1
    times = pd.to_datetime(pd.read_csv(csv_path, usecols=["time"])["time"], format="%Y%m%d %H:%M")
    sp = None
    if spread_csv:
        raw = pd.read_csv(spread_csv)
        sp = pd.DataFrame({"time": pd.to_datetime(raw["time"], format="%Y%m%d %H:%M"), "spread": raw["spread"]})
    full = make_splits(n_rows, cfg)
    usable_end = n_rows if full.holdout.stop - full.holdout.start <= 0 else full.holdout.start - full.embargo      # never reach into the holdout of the full run
    idx = pd.DatetimeIndex(times.iloc[:usable_end])
    mode = str(wf["mode"])
    if mode == "bars":
        wins = bar_windows(usable_end, int(wf["train_bars"]), int(wf["test_bars"]), int(wf["step_bars"]))
    elif mode == "calendar":
        wins = calendar_windows(idx, float(wf["train_years"]), float(wf["test_years"]), float(wf["step_years"]))
    else:
        raise ValueError(f"unknown walkforward.mode {mode!r}")
    wins = [w for w in wins if w[1] - w[0] >= int(wf["min_window_bars"]) and w[2] > w[1]]
    csv_sha = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]
    null_thr = None
    if bool(wf["require_null_approx"]):
        null_thr = _null_threshold(run, cfg)
        if null_thr is None:
            raise FileNotFoundError("walkforward.require_null_approx needs a finished stage 9 in the run folder (09_null/null_info.json)")
    infos: List[Dict[str, Any]] = []
    trades: List[pd.DataFrame] = []
    n_new = 0
    for k, b in enumerate(wins):
        fp = _window_fp(csv_sha, b, cfg) + (f"|null{null_thr!r}" if null_thr is not None else "")      # the threshold value is part of the fingerprint
        rp, tp_ = wf_dir / f"w{k:03d}" / "window_result.json", wf_dir / f"w{k:03d}" / "oos_trades.parquet"
        if rp.exists():
            stored = json.loads(rp.read_text(encoding="utf-8"))
            if stored.get("fingerprint") == fp:
                infos.append(stored["info"])
                if tp_.exists():
                    trades.append(pd.read_parquet(tp_))
                continue
        if max_windows is not None and n_new >= int(max_windows):
            break
        if (wf_dir / f"w{k:03d}").exists() and rp.exists():
            rp.unlink()
        info, tr = _run_window(k, b, text, times, sp, cfg, wf_dir, symbol, null_thr)
        if len(tr):
            tr.to_parquet(tp_)
        elif tp_.exists():
            tp_.unlink()
        rp.write_text(json.dumps({"fingerprint": fp, "info": info}, indent=2, default=float), encoding="utf-8")
        infos.append(info)
        trades.append(tr)
        n_new += 1
        if progress:
            progress(k, len(wins), info)
    all_tr = pd.concat([t for t in trades if len(t)], ignore_index=True) if any(len(t) for t in trades) else pd.DataFrame()
    complete = len(infos) == len(wins)
    agg = aggregate_windows(infos, all_tr, float(cfg["engine"].get("initial_deposit", 10000.0)))
    agg.update(complete=complete, n_windows_planned=len(wins), mode=mode, usable_rows=usable_end, null_approximation=null_thr is not None, null_threshold=null_thr)
    pd.DataFrame([{k: v for k, v in i.items() if k != "rules"} for i in infos]).to_csv(wf_dir / "windows.csv", index=False)
    if len(all_tr):
        all_tr.to_csv(wf_dir / "oos_trades.csv", index=False)
    (wf_dir / "summary.json").write_text(json.dumps({"summary": agg, "windows": infos}, indent=2, default=float), encoding="utf-8")
    (wf_dir / "summary.md").write_text(walkforward_md(agg, infos, cfg), encoding="utf-8")
    agg["new_windows"] = n_new
    return agg


def walkforward_md(agg: Dict[str, Any], infos: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    f = lambda x, d=3: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"
    L = ["# Outer walk-forward (design 14.3)", "",
         f"Mode {agg['mode']}; windows {agg['n_windows']} of {agg['n_windows_planned']} planned{'' if agg['complete'] else ' (INCOMPLETE: run again to continue)'}; rows used: first {agg['usable_rows']} (the holdout of the full run is never used).",
         "Null test per window: " + (f"NOT run; approximation with the full-run threshold {f(agg['null_threshold'], 2)} (best-family t) - flagged." if agg["null_approximation"] else "not run (require_null_approx false)."), "",
         "## Procedure metrics", "",
         f"- Windows with at least one rule: {agg['n_windows_with_rules']} / {agg['n_windows']} ({f(100 * agg['share_windows_with_rules'], 0)} %)",
         f"- Pooled OOS trades: {agg['n_trades']}; pooled OOS mean net R: {f(agg['oos_mean_R'])}; net profit {f(agg['oos_net_profit'], 2)}",
         f"- In-sample (train smoothed) mean R of the selected rules: {f(agg['is_mean_R'])}; OOS / IS ratio: {f(agg['oos_is_ratio'], 2)}",
         f"- Windows with positive OOS mean R: {f(100 * agg['share_windows_positive'], 0) if np.isfinite(agg['share_windows_positive']) else 'n/a'} %; std of window means: {f(agg['window_mean_R_std'])}",
         f"- Pooled OOS equity drawdown: {f(agg['oos_equity_dd_pct'], 1)} %", "",
         f"**Verdict: {'positive pooled OOS expectancy' if agg['positive_oos'] else 'NO positive pooled OOS expectancy - the procedure is not to be trusted, however good a single rule looks (design 14.3)'}.**", "",
         "## Windows", "", "| window | test from | status | rules | OOS trades | OOS mean R | IS mean R |", "|---|---|---|---|---|---|---|"]
    for i in infos:
        L.append(f"| {i['window']} | {i['test_start_time']} | {i['status']} | {i['n_rules']} | {i['n_trades']} | {f(i['oos_mean_R'])} | {f(i['is_mean_R'])} |")
    return "\n".join(L) + "\n"
