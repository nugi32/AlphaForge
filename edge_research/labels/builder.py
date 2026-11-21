"""
Stage 2: label table. For EVERY bar i: net result (in ATR_i units) of entering at the open of bar e=i+1, long and short
separately, for each exit family. Three columns per (direction, family):

  {dir}_{family}_gross  result before costs, R
  {dir}_{family}_cost   costs (spread, slippage, commission, swap), R      -> net = gross - cost
  {dir}_{family}_hold   bars held (exit bar - entry bar)

Families: T{N} time-stop; B{tp}_{sl}_{H} bar-boundary SL/TP (decision on the CLOSED bar, fill at next open);
I{tp}_{sl}_{H} intrabar (spec 15.4 ``S{...}``, RESEARCH ONLY): the high/low of a bar touches the level, fill AT the level
(pessimistic: SL first when one bar touches both; a bar opening beyond a level fills at the open).
The optional layer ``labels.cat_stop_atr`` (wide intrabar stop) is applied on top of EVERY family when set, exactly like the
engine's catastrophic layer. All exit mechanics live HERE and the engine imports them — one implementation, not a copy.
A label is NaN when the window leaves the data, prices/spread are not finite, ATR is invalid, or the window crosses a gap
larger than max_gap_hours (decisions D-003, D-004, D-008 in docs/NOTES.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_research.labels.costs import CostSpec, cumulative_rollovers, spread_array

DIRECTIONS = ("long", "short")


def _num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def time_stop_name(n: int) -> str:
    return f"T{n}"


def barrier_name(tp: float, sl: float, h: int) -> str:
    return f"B{_num(tp)}_{_num(sl)}_{h}"


def intrabar_name(tp: float, sl: float, h: int) -> str:
    return f"I{_num(tp)}_{_num(sl)}_{h}"


@dataclass
class LabelResult:
    table: pd.DataFrame
    families: List[str]
    spec: Dict[str, Any]


def family_list(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    lab = cfg["labels"]
    fams: List[Tuple[str, Dict[str, Any]]] = [(time_stop_name(n), {"kind": "T", "n": int(n)}) for n in lab["time_stops"]]
    g = lab["barrier_grid"]
    for h in g["horizon"]:
        for tp in g["tp"]:
            for sl in g["sl"]:
                fams.append((barrier_name(tp, sl, h), {"kind": "B", "tp": float(tp), "sl": float(sl), "h": int(h)}))
                if bool(lab.get("intrabar_families", True)):
                    fams.append((intrabar_name(tp, sl, h), {"kind": "I", "tp": float(tp), "sl": float(sl), "h": int(h)}))
    return fams


class _Market:
    """Arrays shared by all families."""

    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any], spread_points: Optional[pd.Series], cost_mult: float = 1.0):
        self.cost = CostSpec.from_config(cfg)
        self.cost_mult = float(cost_mult)                        # engine cost stress: multiplies spread and slippage (design 15.5); labels always use 1.0
        self.slip = self.cost.slippage_price * self.cost_mult
        self.n = len(df)
        self.open = df["open"].to_numpy(np.float64)
        self.close = df["close"].to_numpy(np.float64)
        atr = df[cfg["labels"]["atr_column"]].to_numpy(np.float64)
        self.atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
        self.spread, self.spread_fallback = spread_array(df.index, spread_points, self.cost)
        self.spread = self.spread * self.cost_mult
        self.high = df["high"].to_numpy(np.float64) if "high" in df.columns else None
        self.low = df["low"].to_numpy(np.float64) if "low" in df.columns else None
        mg = cfg["data"]["max_gap_hours"]
        if mg is None:
            self.gapcum = np.zeros(self.n, dtype=np.int64)
        else:
            dt = np.diff(df.index.to_numpy()).astype("timedelta64[s]").astype(np.float64) / 3600.0
            self.gapcum = np.concatenate([[0], np.cumsum(dt > float(mg))]).astype(np.int64)
        self.roll = cumulative_rollovers(df.index, self.cost.swap_rollover_hour, self.cost.swap_triple_weekday)

    def window_ok(self, i: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Window from signal bar i to exit bar x must not cross a gap and prices must be finite."""
        ok = (self.gapcum[x] - self.gapcum[i]) == 0
        e = i + 1
        ok &= np.isfinite(self.open[e]) & np.isfinite(self.open[x]) & np.isfinite(self.spread[e]) & np.isfinite(self.spread[x])
        ok &= np.isfinite(self.atr[i])
        return ok

    def result(self, i: np.ndarray, x: np.ndarray, is_long: bool) -> Tuple[np.ndarray, np.ndarray]:
        """(gross_R, cost_R) for trades entered at i+1 and exited at the open of bar x."""
        return self.result_at(i, x, self.open[x], is_long)

    def result_at(self, i: np.ndarray, x: np.ndarray, px: np.ndarray, is_long: bool) -> Tuple[np.ndarray, np.ndarray]:
        """(gross_R, cost_R) for trades entered at the open of i+1 and exited in bar x at the BID-equivalent price ``px`` (the open of x for the label families).
        A short that is closed at an ask level L passes px = L - spread[x]; its cost is the spread of the exit bar, so net = open[e] - L - fees either way."""
        e = i + 1
        c = self.cost
        gross = (px - self.open[e]) if is_long else (self.open[e] - px)
        spread = self.spread[e] if is_long else self.spread[x]
        nights = self.roll[x] - self.roll[e]
        cost = spread + 2.0 * self.slip + c.commission_price + c.swap_cost_price(nights, is_long)
        return gross / self.atr[i], cost / self.atr[i]


def barrier_exit_bars(m: _Market, i: np.ndarray, tp: float, sl: float, h: int, is_long: bool) -> np.ndarray:
    """Exit bar x for each signal bar in ``i`` (all with i + 1 + h <= n - 1).

    After bar b = e + k closes, if close_b is beyond the TP or SL level the exit fills at the open of b + 1; otherwise the trade
    is closed at the open of e + h. Levels are measured from the entry fill price (long: ask, short: bid).
    """
    e = i + 1
    slip = m.slip
    atr = m.atr[i]
    if is_long:
        entry = m.open[e] + m.spread[e] + slip
        tp_lvl, sl_lvl = entry + tp * atr, entry - sl * atr
    else:
        entry = m.open[e] - slip
        tp_lvl, sl_lvl = entry - tp * atr, entry + sl * atr
    x = e + h
    done = np.zeros(len(i), dtype=bool)
    for k in range(h):
        b = e + k
        cb = m.close[b]
        hit = ((cb >= tp_lvl) | (cb <= sl_lvl)) if is_long else ((cb <= tp_lvl) | (cb >= sl_lvl))
        new = hit & ~done
        x = np.where(new, b + 1, x)
        done |= new
    return x


# ------------------------------------------------- shared exit mechanics (the engine imports these, one implementation)
def entry_and_levels(m: "_Market", i: np.ndarray, is_long: bool, tp: Optional[float], sl: Optional[float]):
    """Entry fill price (long: ask, short: bid, both with adverse slippage) and TP / SL levels, from the entry fill and the ATR of the signal bar."""
    e = i + 1
    atr = m.atr[i]
    entry = (m.open[e] + m.spread[e] + m.slip) if is_long else (m.open[e] - m.slip)
    sgn = 1.0 if is_long else -1.0
    tp_lvl = entry + sgn * tp * atr if tp is not None else None
    sl_lvl = entry - sgn * sl * atr if sl is not None else None
    return entry, tp_lvl, sl_lvl


def _touch(m: "_Market", b: np.ndarray, is_long: bool, lvl: np.ndarray, kind: str, gap_ok: np.ndarray):
    """Does bar ``b`` touch ``lvl`` ('sl' = adverse level, 'tp' = favourable)? Returns (hit, fill price in the PRICE the level is quoted in).
    Long exits at Bid (= open, low, high of the bar); short exits at Ask (= those + spread). A bar whose open is already beyond the level fills at the open when ``gap_ok``."""
    sp = 0.0 if is_long else m.spread[b]
    op = m.open[b] + sp
    if is_long:
        beyond_open = (op <= lvl) if kind == "sl" else (op >= lvl)
        touched = (m.low[b] + sp <= lvl) if kind == "sl" else (m.high[b] + sp >= lvl)
    else:
        beyond_open = (op >= lvl) if kind == "sl" else (op <= lvl)
        touched = (m.high[b] + sp >= lvl) if kind == "sl" else (m.low[b] + sp <= lvl)
    gap = beyond_open & gap_ok
    hit = gap | touched
    fill = np.where(gap, op, lvl)
    return hit, fill


def _to_bid(m: "_Market", x: np.ndarray, fill: np.ndarray, is_long: bool) -> np.ndarray:
    """Convert an ask-quoted fill (short side) to the bid-equivalent exit price the result is measured at."""
    return fill if is_long else fill - m.spread[x]


def intrabar_exit(m: "_Market", i: np.ndarray, is_long: bool, tp: float, sl: float, h: int) -> Dict[str, np.ndarray]:
    """Intrabar research exits (label family ``I{tp}_{sl}_{H}``, spec 15.4 ``S{...}``): the high/low of a bar touches a level, the fill IS the
    level (or the open when the bar opens beyond it, only after the entry bar). SL is assumed first when one bar touches both (pessimistic).
    Without any touch the trade exits at the open of the horizon bar. Returns x / px (bid-equivalent) / reason / intrabar flag."""
    e = i + 1
    n_sig = len(i)
    if m.high is None or m.low is None:
        raise ValueError("intrabar exits need high and low columns")
    x = e + h
    px = m.open[x].copy()
    reason = np.full(n_sig, "time", dtype=object)
    intrabar = np.zeros(n_sig, dtype=bool)
    _, tp_lvl, sl_lvl = entry_and_levels(m, i, is_long, tp, sl)
    done = np.zeros(n_sig, dtype=bool)
    for k in range(h):
        b = e + k
        gap_ok = np.full(n_sig, k > 0)                                       # bar e opens at the entry price: no gap test there
        sl_h, sl_f = _touch(m, b, is_long, sl_lvl, "sl", gap_ok)
        tp_h, tp_f = _touch(m, b, is_long, tp_lvl, "tp", gap_ok)
        hit_sl = sl_h & ~done
        hit_tp = tp_h & ~sl_h & ~done
        new = hit_sl | hit_tp
        if new.any():
            fill = np.where(hit_sl, sl_f, tp_f)
            x = np.where(new, b, x)
            px = np.where(new, _to_bid(m, b, fill, is_long), px)
            reason = np.where(hit_sl, "sl", np.where(hit_tp, "tp", reason)).astype(object)
            intrabar = intrabar | new
            done |= new
    return {"x": x, "px": px, "reason": reason, "intrabar": intrabar}


def apply_cat_layer(m: "_Market", i: np.ndarray, is_long: bool, cat_k: float, hmax: int,
                    x: np.ndarray, px: np.ndarray, reason: np.ndarray, intrabar: np.ndarray):
    """Catastrophic stop (design 15.4 layer): a WIDE intrabar SL at ``cat_k`` ATR that cuts a trade strictly BEFORE the exit decided so far
    (an exit at the open of bar b pre-empts the cat touch inside b; the layer never overrides an intrabar exit in the same bar).
    Returns (x, px, reason, intrabar) with cat hits marked ``reason="cat"``."""
    e = i + 1
    n_sig = len(i)
    if m.high is None or m.low is None:
        raise ValueError("cat_stop_atr needs high and low columns")
    _, _, cat_lvl = entry_and_levels(m, i, is_long, None, float(cat_k))
    x0 = x.copy()
    done = np.zeros(n_sig, dtype=bool)
    for k in range(hmax):
        b = e + k
        active = (b < x0) & ~done
        if not active.any():
            break
        bb = np.minimum(b, m.n - 1)
        hit, fill = _touch(m, bb, is_long, cat_lvl, "sl", np.full(n_sig, k > 0))
        new = hit & active
        if new.any():
            x = np.where(new, b, x)
            px = np.where(new, _to_bid(m, bb, fill, is_long), px)
            reason = np.where(new, "cat", reason).astype(object)
            intrabar = intrabar | new
            done |= new
    return x, px, reason, intrabar


def build_labels(df: pd.DataFrame, cfg: Dict[str, Any], spread_points: Optional[pd.Series] = None) -> LabelResult:
    m = _Market(df, cfg, spread_points)
    fams = family_list(cfg)
    cat_k = cfg["labels"].get("cat_stop_atr")
    if cat_k is not None and float(cat_k) <= 0:
        raise ValueError("labels.cat_stop_atr must be positive")
    if cat_k is not None and (m.high is None or m.low is None):
        raise ValueError("labels.cat_stop_atr needs high and low columns")
    if bool(cfg["labels"].get("intrabar_families", True)) and (m.high is None or m.low is None):
        fams = [(f, p) for f, p in fams if p["kind"] != "I"]            # no intrabar data: the I families are skipped, not an error
        intrabar_dropped = "high/low columns missing from the data: intrabar (I) label families were NOT built"
    else:
        intrabar_dropped = None
    n = m.n
    cols: Dict[str, np.ndarray] = {}
    for fam, p in fams:
        hmax = p["n"] if p["kind"] == "T" else p["h"]
        i_all = np.arange(0, max(0, n - 1 - hmax))               # need e + hmax = i + 1 + hmax <= n - 1
        for direction in DIRECTIONS:
            is_long = direction == "long"
            gross = np.full(n, np.nan, dtype=np.float32)
            cost = np.full(n, np.nan, dtype=np.float32)
            hold = np.full(n, np.nan, dtype=np.float32)
            if len(i_all):
                if p["kind"] == "T":
                    x, px = i_all + 1 + p["n"], m.open[i_all + 1 + p["n"]].copy()
                    reason = intrabar = None
                elif p["kind"] == "B":
                    x = barrier_exit_bars(m, i_all, p["tp"], p["sl"], p["h"], is_long)
                    px, reason, intrabar = m.open[x].copy(), None, None
                else:
                    ex = intrabar_exit(m, i_all, is_long, p["tp"], p["sl"], p["h"])
                    x, px, reason, intrabar = ex["x"], ex["px"], ex["reason"], ex["intrabar"]
                if cat_k is not None:                            # catastrophic layer on EVERY family, before the window filter (engine order)
                    if reason is None:
                        reason = np.full(len(i_all), "time", dtype=object)
                        intrabar = np.zeros(len(i_all), dtype=bool)
                    x, px, reason, intrabar = apply_cat_layer(m, i_all, is_long, cat_k, hmax, x, px, reason, intrabar)
                ok = m.window_ok(i_all, x)
                i_ok, x_ok, px_ok = i_all[ok], x[ok], px[ok]
                g, c = m.result_at(i_ok, x_ok, px_ok, is_long)
                gross[i_ok], cost[i_ok], hold[i_ok] = g, c, x_ok - (i_ok + 1)
            cols[f"{direction}_{fam}_gross"], cols[f"{direction}_{fam}_cost"], cols[f"{direction}_{fam}_hold"] = gross, cost, hold
    table = pd.DataFrame(cols, index=df.index)
    spec = {"atr_column": cfg["labels"]["atr_column"], "families": [f for f, _ in fams], "max_gap_hours": cfg["data"]["max_gap_hours"],
            "costs": cfg["costs"], "spread_fallback_bars": m.spread_fallback,
            "intrabar_families": bool(cfg["labels"].get("intrabar_families", True)) and intrabar_dropped is None,
            "intrabar_dropped": intrabar_dropped, "cat_stop_atr": cat_k,
            "valid_rows": {c: int(table[c].notna().sum()) for c in table.columns if c.endswith("_gross")}}
    return LabelResult(table, [f for f, _ in fams], spec)


def net_R(table: pd.DataFrame, direction: str, family: str, cost_mult: float = 1.0) -> pd.Series:
    """Net result in R. ``cost_mult`` > 1 is the cost-stress hook (no relabelling needed)."""
    return table[f"{direction}_{family}_gross"] - cost_mult * table[f"{direction}_{family}_cost"]


def write_labels(res: LabelResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_parquet(out / "label_table.parquet", compression="zstd")
    (out / "label_spec.json").write_text(json.dumps(res.spec, indent=2, default=str), encoding="utf-8")
