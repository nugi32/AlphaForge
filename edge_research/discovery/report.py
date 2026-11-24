"""Writers for the coarse-screen output (Stage 5, step 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from edge_research.atoms.atoms import AtomSet


def summary_md(surv: pd.DataFrame, info: Dict[str, Any], aset: AtomSet) -> str:
    d = info["survivors_by_direction"]
    lines = ["# Stage 5 - rule discovery, coarse screen of atom pairs (train rows only)", "",
             f"- Atoms: {info['n_atoms']}; legal pairs: {info['n_legal_pairs']}; train rows: {info['n_train_rows']}; tiles processed: {info['n_tiles']}; "
             f"time: {info['seconds']} s.",
             f"- Screening labels: {', '.join(info['labels'])}.",
             f"- Survivors (pair, direction): {info['n_survivors']} (long {d['long']}, short {d['short']})"
             + (f"; TRUNCATED from {info['n_survivors_before_cap']} by discovery.max_survivors." if info["truncated"] else "."),
             "", "Survivors are SCREENING candidates, not evidence: the filters are coarse (hold-deflated t, fold signs, lift over parents) and the search",
             "is wide. Nothing here is a finding until the exact statistics (M5), the exit selection (M6) and the null test over the whole search (M7)."]
    if info["n_survivors"] == 0:
        lines += ["", "No pair passed the screen. On real data this is a valid result for depth 2; deeper rules or finer atoms are not tried automatically."]
    else:
        both = surv.groupby(["atom_a", "atom_b"])["direction"].nunique()
        n_both = int((both > 1).sum())
        if n_both:
            lines += ["", f"WARNING: {n_both} pair(s) survive in BOTH directions; such rules are suspected artefacts (a volatility marker, spec 10.6)."]
        lines += ["", "## Top 15 by score (sum of coarse t over the passing labels)", "", "| rule A | rule B | direction | labels passed | score |", "|---|---|---|---|---|"]
        for r in surv.head(15).itertuples():
            lines.append(f"| {r.expr_a} | {r.expr_b} | {r.direction} | {r.n_labels_pass} | {r.score:.2f} |")
    return "\n".join(lines) + "\n"


def write_screen(surv: pd.DataFrame, info: Dict[str, Any], aset: AtomSet, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    surv.to_parquet(out / "survivors_pairs.parquet", compression="zstd")
    (out / "screen_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summary_md(surv, info, aset), encoding="utf-8")


def candidates_summary_md(cand: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    c = cfg["candidates"]
    f = info["funnel"]
    L = ["# Stage 6 - candidate filters and dedup (exact statistics on train rows)", "",
         f"- Filters: episodes >= {c['min_episodes_train']} (gap = hold of the label family), cluster t >= {c['t_exact']}, lift over BOTH parents t >= {c['lift_t_min']}, "
         f"each on >= {c['min_families_pass']} label families of the direction; dedup = Jaccard on trigger rows >= {c['dedup_jaccard']} (same direction).",
         f"- Funnel (candidates with >= {c['min_families_pass']} passing families): survivors {f['survivors']} -> episodes {f['episodes']} -> + t {f['t']} -> + lift {f['lift']}.",
         f"- Qualified candidates: {info['n_candidates']}; independent clusters (the number of distinct IDEAS): {info['n_clusters']}; time {info['seconds']} s.", "",
         "These are SELECTED numbers from a wide search: they are not evidence. Still missing: time stability beyond folds (F2), exit neighbourhood (F3), threshold plateau (F4),",
         "cost stress (F7), exit selection (M6), null test over the whole search (M7), validation (M8)."]
    if info["n_candidates"] == 0:
        L += ["", "No candidate passed. On real data this is a valid result for depth 2."]
        return "\n".join(L) + "\n"
    bd, cd = info["by_direction"], info["clusters_by_direction"]
    L += ["", f"- By direction: long {bd['long']} ({cd['long']} clusters), short {bd['short']} ({cd['short']} clusters)."]
    if info["n_both_directions"]:
        L += [f"- WARNING: {info['n_both_directions']} candidate row(s) belong to pairs that qualify in BOTH directions; suspected volatility markers (spec 10.6)."]
    reps = cand[cand["is_representative"]].head(15)
    L += ["", "## Top 15 cluster representatives (by median t over all label families of the direction)", "",
          "| rule A | rule B | dir | families passed | best family | t | net R | gross R | episodes | min lift | cluster size |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps.itertuples():
        L.append(f"| {r.expr_a} | {r.expr_b} | {r.direction} | {r.n_families_pass} | {r.best_family} | {r.best_t:.2f} | {r.best_mean:.3f} | {r.best_mean_gross:.3f} | "
                 f"{r.best_n_episodes} | {r.best_lift_min:.3f} | {r.cluster_size} |")
    return "\n".join(L) + "\n"


def write_candidates(cand: pd.DataFrame, fam: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(out / "candidates.parquet", compression="zstd")
    fam.to_parquet(out / "candidate_families.parquet", compression="zstd")
    if len(cand):
        reps = cand[cand["is_representative"]].drop(columns=["is_representative"])
        reps.to_csv(out / "clusters.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "clusters.csv", index=False)
    (out / "candidates_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(candidates_summary_md(cand, info, cfg), encoding="utf-8")


def filters_summary_md(log: pd.DataFrame, passed: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    c = cfg["filters"]
    f = info["funnel"]
    L = ["# Stage 7 - candidate filters F2, F7, F4 (+ F6 soft) and dedup (best family of each candidate, train rows)", "",
         f"- F2: >= {c['min_pos_folds']} positive folds, >= {c['min_pos_period_share']:.0%} positive {info.get('period_kind')}s (with >= {c['min_episodes_period']} episodes), leave-one-out >= {c['min_loyo_share']:.0%}, "
         f"no fold/period over {c['max_period_profit_share']:.0%} of the profit, 5 largest episodes <= {c['max_top5_episode_share']:.0%}.",
         f"- F7: mean > 0 with all costs x {c['cost_stress_mult']}.  F4: neighbours +-{c['plateau_steps']} on the quantile ladder, median >= {c['plateau_min_ratio']:.0%} of the original and >= {c['plateau_min_pos_share']:.0%} positive.",
         f"- Funnel: candidates {f['candidates']} -> F2 {f['f2']} -> + F7 {f['f7']} -> + F4 {f['f4']}.",
         f"- Passed: {info['n_passed']}; independent clusters after the second dedup: {info['n_clusters']}; time {info['seconds']} s.", "",
         "Still selected numbers: not evidence before the null test (M7) and validation (M8). F3 (exit neighbourhood) is checked in Stage 8."]
    if not len(passed):
        return "\n".join(L + ["", "No candidate passed."]) + "\n"
    reps = passed[passed["f_is_representative"]].head(15)
    L += ["", "## Top 15 cluster representatives (by plateau score = smoothed R)", "",
          "| rule A | rule B | dir | best family | t | net R | plateau R | plateau ratio | mono | cluster size |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps.itertuples():
        L.append(f"| {r.expr_a} | {r.expr_b} | {r.direction} | {r.best_family} | {r.best_t:.2f} | {r.best_mean:.3f} | {r.plateau_score:.3f} | {r.plateau_ratio:.2f} | {r.mono_score:.2f} | {r.f_cluster_size} |")
    return "\n".join(L) + "\n"


def write_filters(log: pd.DataFrame, passed: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.to_parquet(out / "filter_log.parquet", compression="zstd")
    passed.to_parquet(out / "candidates_filtered.parquet", compression="zstd")
    (passed[passed["f_is_representative"]] if len(passed) else passed).to_csv(out / "clusters.csv", index=False)
    (out / "filters_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(filters_summary_md(log, passed, info, cfg), encoding="utf-8")


def deep_summary_md(frame: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any], exprs: Dict[str, List[str]]) -> str:
    """Stage 5b (M11) summary: the Apriori depth 3-4 survivors and the Engine B leaves, side by side."""
    ap, tr = cfg["deep"]["apriori"], cfg["deep"]["trees"]
    a, t = info["apriori"], info["trees"]
    L = ["# Stage 5b (M11) - deepening the search: Apriori depth 3-4 (Engine A) and shallow trees (Engine B)", ""]
    L += [f"- Engine A (Apriori): {'on' if ap['enabled'] else 'off'}; pairs in: {a['n_pairs_in']}; deep survivors: {a['n_survivors']}; "
          f"cells screened: {a['n_cells_screened']}; time {a['seconds']} s."
          + (" TRUNCATED by deep.apriori.max_survivors." if a["truncated"] else ""),
          f"  Beams: {ap['beam_triple']} triples / {ap['beam_quad']} quads; max atoms {ap['max_atoms']}; the same coarse filters as depth 2 "
          f"(rows >= {cfg['discovery']['min_rows_screen']}, coarse t >= {cfg['discovery']['t_screen']}, >= {cfg['discovery']['min_pos_folds']} positive folds, lift over the parent > 0)."]
    if a.get("by_depth"):
        L += ["  Per depth / direction: " + ", ".join(f"{k} {v}" for k, v in sorted(a["by_depth"].items()))]
    L += [f"- Engine B (trees): {'on' if tr['enabled'] else 'off'}; trees {t['n_trees']} x depth {tr['max_depth']}; features used {t['n_features']}; splits {t['n_splits']}; "
          f"leaf rules {t['n_leaf_rules']}; rules kept {t['n_rules']}; time {t['seconds']} s.",
          f"  Target families {', '.join(tr['families'])} per direction, winsorized at +-{tr['winsor_R']} R; a leaf must hold >= {tr['min_leaf_rows']} rows, "
          f"appear in >= {tr['min_hits']} trees and have a positive TRAIN mean.", ""]
    L += ["Both lists are CANDIDATES for stage 6 only: the exact episode / lift statistics, the filters, the exit choice, the null test and validation decide",
          "what survives. Engine B is a comparison engine: rules found by both engines are more credible than rules found by one (spec 10.3)."]
    if info.get("null_scope"):
        L += ["", f"Null-test scope: {info['null_scope']}"]
    for title, src in (("Engine A (Apriori) survivors", "apriori"), ("Engine B (trees) rules", "tree")):
        sub = frame[frame["source"] == src] if len(frame) and "source" in frame.columns else frame.head(0)
        if sub.empty:
            L += ["", f"## {title}", "", "None."]
            continue
        if src == "tree":
            cols = ["n_atoms", "direction", "score", "hits", "family"]
        else:
            cols = ["n_atoms", "direction", "score", "n_labels_pass"]
        L += ["", f"## {title} (top 15 by score)", "", "| rule | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for r in sub.head(15).itertuples():
            cells = [(f"{getattr(r, c):.3f}" if isinstance(getattr(r, c), float) else str(getattr(r, c))) for c in cols]
            rule_txt = " & ".join(exprs.get(str(x), str(x)) for x in r.atoms)
            L.append("| " + rule_txt + " | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def write_deep(frame: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any], exprs: Dict[str, List[str]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "survivors_deep.parquet", compression="zstd")
    (out / "deep_info.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    (out / "summary.md").write_text(deep_summary_md(frame, info, cfg, exprs), encoding="utf-8")


def exits_summary_md(chosen: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    e = cfg["exits"]
    L = ["# Stage 8 - exit selection on the smoothed surface and F3 (train rows)", "",
         f"- Smoothed surface (3-point on time stops, 3 x 3 on barriers); a barrier must beat the best time stop by {e['min_improvement_R']} R; F3: >= {e['f3_min_positive']} positive neighbouring exits.",
         f"- Candidates in: {info['n_in']}; with an eligible exit: {info['n_with_exit']}; pass F3: {info['n_pass_f3']}; frozen cluster representatives: {info['n_frozen']}; "
         f"chosen exit differs from the raw best cell: {info['n_changed_from_raw_best']}.", "",
         "The chosen exit is a SELECTED parameter: validation must use it frozen (M8). Mean R per trade favours long holds; `r_per_bar` is reported for comparison only."]
    if not info["n_frozen"]:
        return "\n".join(L + ["", "No candidate could be frozen."]) + "\n"
    reps = chosen[chosen["is_representative"]].head(15)
    L += ["", "## Frozen representatives (top 15 by plateau score)", "", "| rule A | rule B | dir | exit | smooth R | raw R | vs best T | F3 | R/bar |", "|---|---|---|---|---|---|---|---|---|"]
    for r in reps.itertuples():
        L.append(f"| {r.expr_a} | {r.expr_b} | {r.direction} | {r.chosen_family} | {r.chosen_smooth:.3f} | {r.chosen_mean:.3f} | {r.improvement_over_T:+.3f} | {r.f3_positive}/{r.f3_cells} | {r.r_per_bar:.3f} |")
    return "\n".join(L) + "\n"


def write_exits(surface: pd.DataFrame, chosen: pd.DataFrame, info: Dict[str, Any], cfg: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    surface.to_csv(out / "exit_surface.csv", index=False)
    chosen.to_parquet(out / "chosen_exits.parquet", compression="zstd")
    chosen.to_csv(out / "chosen_exits.csv", index=False)
    (out / "exits_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(exits_summary_md(chosen, info, cfg), encoding="utf-8")
