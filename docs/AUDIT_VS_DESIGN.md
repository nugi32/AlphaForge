# Implementation audit vs EDGE_RESEARCH_V2_DESIGN.md (2026-11-18)

Per-chapter status of the design doc (uploaded to a session) against the code. Method: three parallel audits (chapters 4-9, 10-13, 14-19),
key claims re-verified directly by grep. Statuses: IMPLEMENTED / PARTIAL / MISSING / DEVIATES (content matches, naming or mechanism differs).

## RESOLUTION STATUS (updated 2026-11-18, after the gap-fix sessions — details NOTES D-054..D-058, G-019)

ALL ten confirmed gaps below are closed in the code. Nothing here is EXECUTION evidence: `pytest -q` and the mutation
harnesses have still never been run (NOTES G-018) — that is the first job on a real machine.

1. F8 — IMPLEMENTED (`filters._f8_backward_eliminate`, wired; hardened + typed flag columns, D-055).
2. Null covers the deep search by default — `include_in_null: true` for both engines (spec 13.3).
3. Intrabar `I{...}` (spec `S{...}`) exits + catastrophic layer — built by `build_labels`; ONE shared exit implementation
   in `labels/builder.py` that the engine imports; `labels.cat_stop_atr` applies to every family (D-054).
4. F9 `dedup_corr` — implemented (P&L-series correlation, simplest-first representatives, D-055).
5. G11 — offline harness done (`validation/demo.py`, `scripts/g11_demo.py`, gate-table update, D-056); the demo run itself
   needs a live demo account and stays manual by definition.
6. Condition exits — carried by `FrozenRule.exit_condition`, passed to the engine by `run_backtest`, refused by codegen
   (D-057); no discovery stage searches for them yet (Q-008/Q-010).
7. Per-symbol tester config + `engine.fidelity` — `config/MT5_TESTER/<SYMBOL>/<TF>.yaml`, `symbol_config`, D-058.
8. QA/feature remainder — `envelope_*` registry rules, scale summary (`column_stats.csv`), `compute_warmup` used,
   `qa.min_time_parse_fraction` wired, leak check automatic in stage 1 (G-019). `eda.stop_on_no_go: false` stays a
   documented decision (Q-003).
9. Engine B — `max_depth: [3, 4]`; tree `hits`/`family` survive into the candidate outputs (`tree_hits`/`tree_family`).
10. Statistics extras — all implemented (block-bootstrap CI, `cluster_extra`, per-rule PF/W-L/net-per-year, F6 soft
    threshold at |rho| >= 0.6 + sign check, `regime_diag.csv`, `--min-free-gb/--max-rss-gb` on the 0_11/0_13 scripts).

## Confirmed gaps worth acting on (ordered, original audit text)

1. **F8 (backward elimination on validation) is missing entirely** — one of the spec's nine filters has no counterpart anywhere (ch. 11).
2. **Null test does not price the deep search by default** — `deep.apriori.include_in_null` / `deep.trees.include_in_null` are false; G1
   covers depth-2 only. Disclosed in every report (`deep_scope_note`), but spec 13.3 wants the identical procedure. Cost: the deep engines
   must run inside every replicate. Decision needed (config exists, default is the deviation).
3. **Intrabar `S{tp}_{sl}_{H}` exits and the catastrophic layer exist only in the engine, not in the label table** — `labels/builder.py`
   has no S/C families (grep confirms: no `cat_stop` reference in builder.py), so stages 3-8 cannot select intrabar/cat exits, and
   `labels.cat_stop_atr` is a dead config key for labels. Engine-only by design decision (M2b/M9); NOTES Q-011 covers the parity side.
4. **F9 dedup lacks the P&L-correlation criterion (`dedup_corr`)** — only Jaccard on trigger rows is implemented; representatives are
   ranked by score, not "simplest first" (ch. 11 / ch. 18).
5. **G11 (demo forward) has no implementation** — only the `"PENDING"` string in gates; needs a live demo account by definition.
6. **Condition-exit family is a dead path** — the engine supports `exit_signals`, but `run_backtest` never passes them and no stage
   produces a condition exit; nothing frozen can carry one (ch. 15/16).
7. **No per-symbol/TF tester config (`config/MT5_TESTER/<SYMBOL>/<TF>.yaml`)** — one global `costs:` block; `engine.fidelity` key missing.
8. **Small QA/feature gaps**: `envelope_*` features never generated (no rule in `features/registry.py`); `qa.compute_warmup` is dead code
   (warm-up handled implicitly by `first_valid_row` later); per-column min/median/max scale summary not produced;
   `qa.min_time_parse_fraction` config key unread (loader hardcodes 0.99); leak check is a library+test, not a pipeline stage;
   stage 3 NO_GO does not stop the run (`eda.stop_on_no_go: false`, deliberate — NOTES Q-003).
9. **Engine B details**: `max_depth: 3` single value instead of the 3-4 range; tree `hits` frequency dropped by `normalise_rules` and
   never reaches the candidate outputs (ch. 10).
10. **Statistics niceties missing**: block-bootstrap CI for survivors; optional calendar-week cross-clustering in the cluster SE;
    per-rule PF / W-L / net-per-year; F6 monotonicity score stored but never thresholded at |rho| >= 0.6 nor sign-checked; regime
    diagnostic per vol/trend regime (14.6) absent; memory-guard flags `--min-free-gb/--max-rss-gb` exist only as a class in
    `utils/memory.py`, not exposed on any CLI.

## Accepted, documented deviations (no action)

- Output numbering: `10_validation` / `11_backtest/<segment>` / `12_ea` / `13_report` instead of the spec's 08/09/10 + RUN_SUMMARY inside
  `13_report/`; parquet instead of csv for the big tables; `nulltest` section name (YAML `null:` parses as None, NOTES G-006).
- Lift clusters are fixed time blocks of `5 x hold`, not parent episodes (D-027); episode gap = the hold of the evaluated family, not the
  family-maximum (tighter, deliberate).
- Bonferroni over the K frozen rules, not over independent clusters (df uses trade clusters).
- Smoothing on the exit surface = grid-neighbour mean (triple on T axis, 5-cell neighbourhood on B), not a literal 3x3 kernel.
- `parity_tolerance` hardcoded 1e-4 in `codegen/parity.py` instead of a config key.
- G9 (real Strategy Tester parity) pending by environment: automated parity (feature/signal/trade via `ea_sim`) is done; the tester runs,
  stress run and review checklist need the user's MetaTrader (NOTES G-015, CHECKPOINT "Next step A").
- Intrabar exits selected by stage 8 are refused by the MQL5 generator (research-only label, NOTES Q-008/M10b): a rule whose chosen exit
  is an `I...` family cannot become an EA until the generator supports it.

## Chapter 19 acceptance tests: 11 of 12 satisfied, 1 partial

Satisfied: no look-ahead; labels match hand computation; cost symmetry; label<->engine trade-for-trade; pair stats = brute force;
episodes/effective n; recall on planted edge; null false-pass rate; frozen rule unchanged; second holdout refused; codegen refusal reasons.
Partial: parity within tolerance — complete in Python (ea_sim), pending against the real tester (G9). The label<->engine acceptance now
also covers the intrabar families and the catastrophic layer (tests in `test_labels.py` / `test_engine_vs_labels.py`).

## Post-checkpoint-12 verification (this session)

A dedicated read-only verification pass (3 agents over the design chapters + direct line checks) found one violation of
spec 13.3 and two minor deviations, all fixed (NOTES D-059, D-060, G-020):
- Spec 13.3 ("prosedur null identik dengan prosedur nyata"): null replicates silently dropped the feature table, so
  Engine B ran on the real data but in NO replicate. Fixed + test-pinned; a SEARCH_VERSION now binds stored replicates
  to the procedure that computed them.
- Spec 13.4 (null over the whole search): `rule_flags` turned a missing calendar variant into `calendar_driven = beats_random`
  on partial runs; now UNKNOWN (NaN/None/"?"), never a silent True.
- Spec 13.5 (report contents): the deep scope note was computed but never written to any stage-9 output; now it reaches
  `summary.md` ("Deep scope" line) and `null_info.json`, with an explicit SKIPPED warning if trees are requested without
  a feature table.
The audit table above is unchanged — no new gap rows; these were verification findings against already-implemented behaviour.
