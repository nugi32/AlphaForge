"""Stage 10 gates G1-G8 on the validation segment (design 14.1, 14.2, 14.5).

* The frozen rule is evaluated with the ENGINE (one position per rule, full costs, MT5 semantics), not with labels (``evaluate_rule``).
* G4 (lift over both parents) is the one gate that is computed on LABELS, on the validation rows: it asks whether the second atom still adds
  something, which the engine cannot isolate (``lift_over_parents``).
* Selection correction: the one-sided t threshold is Bonferroni over the K frozen rules that are tested (alpha / K), with a Student-t
  quantile at (number of trade clusters - 1) degrees of freedom (``t_critical``).
* ``apply_gates`` is a PURE function of one row of raw values: every threshold of the stage is applied there and nowhere else.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from edge_research.atoms.atoms import AtomSet
from edge_research.discovery.episodes import lift_stats
from edge_research.discovery.pairs import family_hold
from edge_research.eda.univariate import cluster_mean_se
from edge_research.labels.builder import net_R
from edge_research.mt5_backtest.backtest import run_backtest
from edge_research.rules import FrozenRule

GATE_COLS = ["g2_time_stability", "g3_exit_plateau", "g4_lift", "g5_oos", "g6_trades", "g7_cost_stress", "g8_quality"]


# ---------------------------------------------------------------------------------------------------------------- statistics
def t_critical(alpha: float, k: int, n_clusters: int) -> float:
    """One-sided Bonferroni critical t: ``t.isf(alpha / K, df = n_clusters - 1)``. Fewer than 2 clusters: no inference possible, the threshold is +inf (nothing passes)."""
    if int(n_clusters) < 2:
        return float("inf")
    return float(student_t.isf(float(alpha) / max(int(k), 1), int(n_clusters) - 1))


def trade_clusters(entry_bar: np.ndarray, gap: int) -> np.ndarray:
    """Cluster id per trade: trades whose entry bars are at most ``gap`` bars apart (their price paths overlap) share a cluster. Ids are 0..G-1 in the ORIGINAL trade order."""
    e = np.asarray(entry_bar, dtype=np.int64)
    if e.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(e, kind="stable")
    new = np.concatenate([[0], np.cumsum(np.diff(e[order]) > int(gap))]).astype(np.int64)
    out = np.empty_like(new)
    out[order] = new
    return out


def trade_stats(trades: pd.DataFrame, gap: int) -> Dict[str, Any]:
    """Mean net R of the engine trades with a cluster-robust SE and t (clusters = ``trade_clusters``)."""
    out = {"n_trades": int(len(trades)), "mean_R": float("nan"), "se_R": float("nan"), "t": float("nan"), "n_clusters": 0}
    if trades.empty:
        return out
    y = trades["r_net"].to_numpy(np.float64)
    g = trade_clusters(trades["entry_bar"].to_numpy(), gap)
    n_g = int(g.max()) + 1
    mean, se, _ = cluster_mean_se(y, g, n_g)
    out.update(mean_R=mean, se_R=se, n_clusters=int(len(np.unique(g))))
    out["t"] = float(mean / se) if np.isfinite(se) and se > 0 else float("nan")
    return out


def lift_over_parents(aset: AtomSet, atom_ids: Sequence[int], y: np.ndarray, hold: int, rows: slice, block_mult: int) -> Dict[str, Any]:
    """F5 on ``rows``: lift of the rule over each of its LEAVE-ONE-OUT parents, with the Stage 6 estimator (time blocks of ``block_mult`` x hold rows).

    Parents are ordered as in Stage 6 (``candidates.parent_order``): first the rule without its LAST atom (``lift_a``), then without its FIRST
    (``lift_b``), then the remaining ones (3- and 4-atom rules). ``lift_min_t`` is the smallest parent t: a rule must add value over EVERY parent.
    """
    ids = [int(i) for i in atom_ids]
    masks = [aset.dense(i) for i in ids]                           # FULL-length masks; lift_stats slices all inputs by ``rows``
    child = masks[0].copy()
    for m in masks[1:]:
        child &= m
    blk = max(1, int(block_mult) * int(hold))
    order = [len(ids) - 1, 0] + list(range(1, len(ids) - 1))
    out: Dict[str, Any] = {"n_child": 0, "n_parents": len(ids), "lifts": [], "ts": []}
    for n, i in enumerate(order):
        parent = None
        for j, m in enumerate(masks):
            if j == i:
                continue
            parent = m if parent is None else (parent & m)
        r = lift_stats(parent, child, y, int(hold), rows=rows, block=blk)
        out["lifts"].append(float(r["lift"]))
        out["ts"].append(float(r["t"]))
        if n == 0:
            out.update(lift_a=r["lift"], lift_a_t=r["t"], n_rest_a=r["n_rest"], n_child=r["n_child"])
        elif n == 1:
            out.update(lift_b=r["lift"], lift_b_t=r["t"], n_rest_b=r["n_rest"])
    finite = all(np.isfinite(t) for t in out["ts"])
    out["lift_min"] = float(np.min(out["lifts"])) if finite else float("nan")
    out["lift_min_t"] = float(np.min(out["ts"])) if finite else float("nan")
    return out


def evaluate_rule(df: pd.DataFrame, cfg: Dict[str, Any], rule: FrozenRule, ft: pd.DataFrame, spread: Optional[pd.Series], rows: slice, stress_mult: float = 1.0) -> Dict[str, Any]:
    """Engine backtest of ONE frozen rule on ``rows`` (signals decided in the range; exits may run past it, the embargo covers that). ``stress_mult`` multiplies the engine's
    spread and slippage on top of the configured ``engine.cost_multiplier``."""
    c = cfg
    if float(stress_mult) != 1.0:
        c = copy.deepcopy(cfg)
        eng = dict(c.get("engine", {}) or {})
        eng["cost_multiplier"] = float(eng.get("cost_multiplier", 1.0)) * float(stress_mult)
        c["engine"] = eng
    res = run_backtest(df, c, [rule], ft, spread, rows)
    gap = family_hold(rule.exit, cfg)
    st = trade_stats(res.trades, gap)
    s = res.summary
    st.update(net_profit=float(s.get("total_net_profit", 0.0)), profit_factor=float(s.get("profit_factor", float("nan"))), equity_dd_pct=float(s.get("equity_drawdown_pct", float("nan"))))
    return st


# ---------------------------------------------------------------------------------------------------------------- pure gate logic
def apply_gates(row: Dict[str, Any], v: Dict[str, Any], lift_t_train: float, g1_pass: Optional[bool]) -> Dict[str, Any]:
    """All gate verdicts G4-G8, ``accepted`` (G2-G8) and ``eligible_for_holdout`` (also G1) from the raw values of one row. Pure: every threshold of the stage lives here.

    Boundaries: G4 lift t >= threshold (train: ``lift_t_train``, validation: ``v['lift_t_min']``); G5 mean R > 0, t >= the Bonferroni critical value, decay ratio >= ``decay_min_ratio``;
    G6 trades >= ``min_trades_val``; G7 stressed mean R > 0 and stressed net profit > 0; G8 profit factor >= ``min_profit_factor`` and equity drawdown <= ``max_equity_drawdown_pct``.
    A NaN value never passes; an infinite profit factor (no losing trade) does.
    """
    ge = lambda x, thr: bool(x == x and x >= thr)
    lt_val = float(v["lift_t_min"])
    n_par = int(row.get("n_parents", 2) or 2)
    if n_par > 2:                                                    # 3- and 4-atom rules: every leave-one-out parent must add value
        row["g4_lift"] = bool(ge(row.get("lift_min_t_train"), lift_t_train) and ge(row.get("lift_min_t_val"), lt_val))
    else:
        row["g4_lift"] = bool(ge(row["lift_a_t_train"], lift_t_train) and ge(row["lift_b_t_train"], lift_t_train)
                              and ge(row["lift_a_t_val"], lt_val) and ge(row["lift_b_t_val"], lt_val))
    row["g5_oos"] = bool(row["val_mean_R"] == row["val_mean_R"] and row["val_mean_R"] > 0 and ge(row["val_t"], row["t_crit_bonferroni"]) and ge(row["decay_ratio"], float(v["decay_min_ratio"])))
    row["g6_trades"] = bool(row["val_trades"] >= int(v["min_trades_val"]))
    row["g7_cost_stress"] = bool(row["stress_mean_R"] == row["stress_mean_R"] and row["stress_mean_R"] > 0 and row["stress_net_profit"] > 0)
    row["g8_quality"] = bool(ge(row["val_profit_factor"], float(v["min_profit_factor"])) and np.isfinite(row["val_equity_dd_pct"]) and row["val_equity_dd_pct"] <= float(v["max_equity_drawdown_pct"]))
    row["g1_null"] = None if g1_pass is None else bool(g1_pass)
    row["accepted"] = bool(all(row[c] for c in GATE_COLS))
    g1_ok = (row["g1_null"] is True) or (row["g1_null"] is None and not bool(v["require_g1"]))
    row["eligible_for_holdout"] = bool(row["accepted"] and g1_ok)
    fails = [c for c in GATE_COLS if not row[c]] + ([] if g1_ok else ["g1_null"])
    row["fail_reasons"] = ";".join(fails)
    return row


# ---------------------------------------------------------------------------------------------------------------- the gate table
def build_gate_table(rules: Sequence[FrozenRule], chosen: pd.DataFrame, filtered: pd.DataFrame, fam: pd.DataFrame, aset: AtomSet, label_table: pd.DataFrame, df: pd.DataFrame,
                     ft: pd.DataFrame, spread: Optional[pd.Series], val: slice, cfg: Dict[str, Any], g1_pass: Optional[bool]) -> pd.DataFrame:
    """One row per frozen rule with the raw values and the verdict of every gate. Reads only validation rows (plus the train statistics stored by the earlier stages)."""
    v = cfg["validation"]
    k = len(rules)
    ch = chosen.set_index("cand_id")
    flt = filtered.set_index("cand_id")
    for c in ("pass_f2", "pass_f4"):
        if c not in flt.columns:
            raise KeyError(f"the stage 7 table has no column {c!r}: G2 / G3 cannot be read (never defaulted to True)")
    fam_i = fam.set_index(["cand_id", "family"])
    train_rows = slice(*aset.train)
    block_mult = int(cfg["candidates"]["lift_block_mult"])
    rows_out: List[Dict[str, Any]] = []
    for rule in rules:
        cid = int(rule.meta["cand_id"])
        crow = ch.loc[cid]
        family, direction = str(rule.exit), rule.direction
        hold = family_hold(family, cfg)
        row: Dict[str, Any] = {"rule_id": rule.rule_id, "cand_id": cid, "direction": direction, "exit": family, "k_tested": k, "expr": " AND ".join(rule.meta.get("exprs", [])),
                               "train_chosen_mean_R": float(crow["chosen_mean"]), "train_chosen_smooth_R": float(crow["chosen_smooth"]), "train_chosen_t": float(crow["chosen_t"])}
        # G2 (F2 time stability) and G3 (F3 exit neighbourhood + F4 threshold plateau): evaluated on train by the earlier stages
        row["g2_time_stability"] = bool(flt.at[cid, "pass_f2"])
        row["g3_exit_plateau"] = bool(crow["pass_f3"]) and bool(flt.at[cid, "pass_f4"])
        # G4: F5 on train AND on validation. Stage 6 stores the lift only for families that already passed episodes and t; for the others it is recomputed on train
        y = net_R(label_table, direction, family).to_numpy(np.float64)
        atom_ids = [int(i) for i in rule.meta["atom_ids"]]
        n_par = len(atom_ids)
        stored = fam_i.loc[(cid, family)] if (cid, family) in fam_i.index else None
        la_t, lb_t = (float(stored["lift_a_t"]), float(stored["lift_b_t"])) if stored is not None else (float("nan"), float("nan"))
        src = "stored"
        lmin_t = float(min(la_t, lb_t)) if (np.isfinite(la_t) and np.isfinite(lb_t)) else float("nan")
        if n_par > 2 or not (np.isfinite(la_t) and np.isfinite(lb_t)) or bool(rule.meta.get("f8_simplified")):
            # an F8-simplified rule keeps the cand_id of the ORIGINAL candidate: its stored lift columns belong to the pre-elimination atoms
            lt = lift_over_parents(aset, atom_ids, y, hold, train_rows, block_mult)
            la_t, lb_t, lmin_t, src = lt["lift_a_t"], lt["lift_b_t"], lt["lift_min_t"], "recomputed"
        lv = lift_over_parents(aset, atom_ids, y, hold, val, block_mult)
        row.update(lift_a_t_train=la_t, lift_b_t_train=lb_t, lift_train_source=src, lift_min_t_train=lmin_t, n_parents=n_par,
                   lift_a_val=lv["lift_a"], lift_a_t_val=lv["lift_a_t"], lift_b_val=lv["lift_b"], lift_b_t_val=lv["lift_b_t"],
                   lift_min_t_val=lv["lift_min_t"], lift_n_child_val=lv["n_child"])
        # engine on the validation segment: normal and stressed costs
        base = evaluate_rule(df, cfg, rule, ft, spread, val)
        stress = evaluate_rule(df, cfg, rule, ft, spread, val, float(v["cost_stress_mult"]))
        tcrit = t_critical(float(v["alpha"]), k, base["n_clusters"])
        smooth = row["train_chosen_smooth_R"]
        ratio = base["mean_R"] / smooth if smooth > 0 and np.isfinite(base["mean_R"]) else float("nan")
        row.update(val_trades=base["n_trades"], val_mean_R=base["mean_R"], val_se_R=base["se_R"], val_t=base["t"], val_clusters=base["n_clusters"], t_crit_bonferroni=tcrit,
                   decay_ratio=ratio, val_net_profit=base["net_profit"], val_profit_factor=base["profit_factor"], val_equity_dd_pct=base["equity_dd_pct"],
                   stress_mean_R=stress["mean_R"], stress_net_profit=stress["net_profit"])
        apply_gates(row, v, float(cfg["candidates"]["lift_t_min"]), g1_pass)
        row["g9_parity"] = "PENDING (M10)"
        row["g10_holdout"] = "NOT OPENED"
        row["g11_demo"] = "PENDING"
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def regime_diag(rules: Sequence[FrozenRule], ft: pd.DataFrame, label_table: pd.DataFrame, val: slice, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Post-selection diagnostic (spec 14.6): net R of each rule on the validation rows, per volatility regime.

    The regime is the first ``volregime_*`` column of the feature table (a small integer, one atom per value in the registry). A rule whose
    validation profit comes from ONE regime only is a regime bet, not an edge; the table makes that visible. Computed from LABELS (the same
    what-if table as the search), not from the engine: it is a diagnostic, not a gate.
    """
    col = next((c for c in ft.columns if c.startswith("volregime")), None)
    if col is None:
        return pd.DataFrame(columns=["rule_id", "regime", "n", "mean_R", "share"])
    reg = ft[col].to_numpy()[val]
    rows: List[Dict[str, Any]] = []
    for rule in rules:
        y = net_R(label_table, rule.direction, rule.exit).to_numpy(np.float64)[val]
        ok = np.isfinite(y) & np.isfinite(reg)
        yy, rr = y[ok], reg[ok]
        for v in sorted(set(rr.tolist())):
            m = rr == v
            rows.append({"rule_id": rule.rule_id, "regime": float(v), "n": int(m.sum()), "mean_R": float(yy[m].mean()) if m.any() else float("nan"),
                         "share": float(yy[m].sum() / yy.sum()) if yy.sum() != 0 else float("nan")})
    return pd.DataFrame(rows)


def gates_md(table: pd.DataFrame, cfg: Dict[str, Any], g1: Optional[Dict[str, Any]], holdout_note: str) -> str:
    v = cfg["validation"]
    k = len(table)
    g1_txt = "not run" if g1 is None else ("PASS" if g1.get("g1_pass") else "FAIL")
    L = ["# Stage 10 - validation gates", "",
         f"Rules tested on validation: {k} (K of the Bonferroni correction; alpha {v['alpha']} one-sided, per rule {float(v['alpha']) / max(k, 1):.4g}).",
         f"G1 (null test, stage 9): {g1_txt}. G9 (MQL5 parity) and G11 (demo forward) are outside this stage. {holdout_note}", ""]
    if table.empty:
        return "\n".join(L + ["No frozen rules (no representative passed F3 in Stage 8)."]) + "\n"
    L += [f"Accepted by G2-G8: {int(table['accepted'].sum())}. Eligible for the holdout (also G1): {int(table['eligible_for_holdout'].sum())}.", "",
          "| gate | passed |", "|---|---|"]
    L += [f"| {c} | {int(table[c].sum())} / {k} |" for c in GATE_COLS]
    L += ["", "## Rules (best train plateau first)", "",
          "| rule | dir | exit | train smooth R | val trades | val mean R | val t | t crit | decay | PF | DD % | stress R | lift t (a, b) | accepted | holdout | fails |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        L.append(f"| {r['rule_id']} | {r['direction']} | {r['exit']} | {r['train_chosen_smooth_R']:.3f} | {int(r['val_trades'])} | {r['val_mean_R']:.3f} | {r['val_t']:.2f} | {r['t_crit_bonferroni']:.2f} | "
                 f"{r['decay_ratio']:.2f} | {r['val_profit_factor']:.2f} | {r['val_equity_dd_pct']:.1f} | {r['stress_mean_R']:.3f} | {r['lift_a_t_val']:.1f}, {r['lift_b_t_val']:.1f} | "
                 f"{'yes' if r['accepted'] else 'no'} | {r.get('g10_holdout', 'NOT OPENED')} | {r['fail_reasons']} |")
    return "\n".join(L) + "\n"
