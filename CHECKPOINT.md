# CHECKPOINT (read this first in every new session)

Last updated: checkpoint 13 (post-checkpoint-12 verification findings FIXED, NOTES D-059/D-060/G-020: null replicates now run the FULL search procedure including Engine B — the feature table reaches every replicate — with a SEARCH_VERSION folded into the null base fingerprint so old replicates are never reused, the deep scope note reaches summary.md / null_info.json, and rule_flags treats a missing calendar variant as UNKNOWN instead of flagging every rule calendar_driven. NOTHING has been executed: pytest + mutation harnesses still pending, NOTES G-018; checkpoint 12 closed the audit-vs-design gaps D-054..D-058, G-019).
Tests: as checkpoint 12 (359 + M11/M12 + checkpoint-12 additions, none executed), PLUS the checkpoint-13 additions in `tests/test_null.py` (deep scope note propagation, rule_flags NaN semantics, `null_base_fingerprint` version binding, stage-9 report scope) and `tests/test_deep_m11.py` (trees requested without a feature table are REPORTED not silent; trees actually run inside a stored null replicate). First job on your machine remains `pip install -e .[dev]` + `pytest -q`.
**IMPORTANT for anyone with an EXISTING stage-9 run from checkpoint 12 or earlier: the stored `real.json` and replicate files are from the OLD procedure (deep engines priced on the real data only). The new base fingerprint invalidates them automatically — stage 9 recomputes everything; expect a full null run, not a resume.**

## Status board
| Item | Status |
|---|---|
| M0 synthetic data with planted edge; M1 QA + features; M2 labels + engine core; M3 univariate; M4 atoms + pair screen | DONE |
| M5 candidates (F1, F5, F9, partial F3), stage `06_candidates` | DONE |
| M5b filters F2, F4, F6 (soft), F7, F8 (backward elimination) + F9 dedup (jaccard + dedup_corr), stage `07_filters` | DONE in code; pytest pending (G-018) |
| M6 exit selection on the smoothed surface + F3 (now incl. intrabar I exits), stage `08_exits` | DONE in code; pytest pending (G-018) |
| M7 null test over the whole search (random + calendar variants, resumable, DEEP INCLUDED by default) | DONE in code; pytest pending (G-018) |
| M9 engine extensions (all exit kinds incl. intrabar/cat/condition, position limits, risk % sizing, money + equity) | DONE; exit mechanics now SHARED with the label builder |
| Label table: T / B / intrabar I families + optional catastrophic layer on every family | DONE in code (D-054); pytest pending |
| M8 validation (stage 10, gates G1-G8, G10 machinery), holdout protocol, outer walk-forward, stage 11 wiring | DONE |
| M10 MQL5 codegen + parity harness (gate G9) | DONE IN THE SANDBOX. **The generated EA has never been compiled or run in a tester** (NOTES G-015): G9 still open; intrabar-exit and condition-exit rules are REFUSED by the generator (Q-008) |
| M11 Engine B (trees) + depth 3-4 (`05b_deep`, `discovery/{deep,trees}.py`) | WRITTEN; pytest + mutation harness pending (G-018) |
| M12 RUN_SUMMARY and reporting (`13_report`, `reporting/run_summary.py`) | WRITTEN; pytest + mutation harness pending (G-018) |
| Audit gaps 1-10 (F8, null-deep default, label intrabar/cat, dedup_corr, G11 offline, condition-exit carry, MT5_TESTER config, QA remainder, tree hits, statistics extras) | CLOSED (NOTES D-054..D-058, G-019; `docs/AUDIT_VS_DESIGN.md` has the resolution status) |
| G11 demo forward — the demo run itself | needs a live demo account: `scripts/g11_demo.py` + `validation/demo.py` are the offline half |
| **First execution on the user's machine: `pip install -e .[dev]` + `pytest -q`, then `dev_mutate_m11.py` / `dev_mutate_m12.py`** | **NEXT (G-018)** |
| M10 acceptance on the user's machine (compile, feature dump, tester run, `check_parity.py`) | after the tests are green (needs MetaEditor + Strategy Tester) |

Project stage numbers (NOTES D-032): 00 qa, 01 features, 02 labels, 03 eda, 04 atoms, 05 discovery, 06 candidates, 07 filters, 08 exits, 09 null, planned 10 validation, 11 backtest, 12 ea.

## Next step
**0. FIRST, on your machine (nothing of M11/M12 has ever been executed):**
1. `pip install -e .[dev]`, `pytest -q` — fix anything the new files broke (they are reviewed but unrun; see NOTES G-018).
2. `python scripts/dev_mutate_m11.py` and `python scripts/dev_mutate_m12.py` (cheap; each -k run uses the small worlds). Close every SURVIVED mutant with a new test assertion before continuing.

**A. On your machine (needs MetaTrader 5; nothing else can be verified from the sandbox):**
1. `pip install -e .[dev]`, `pytest -q`; run the pipeline on REAL data (`scripts/run_stages_0_11.py`), read `00_data_qa/report.md`, `10_validation/gates.md`, `walkforward/summary.md`.
2. `python scripts/gen_ea.py --run output/run1` -> `12_ea/ER_<rule>.mq5`. Read `refused.json` (rules the generator will not approximate) and the `ASSUMED` indicator lines in each EA header: compare every line with your `dataExporter.mq5` (period, method, applied price, buffer); fix wrong ones through `codegen.indicators` in the config and regenerate.
3. Compile in MetaEditor. **Send the next session the exact compiler messages** (NOTES G-015: the file was never compiled).
4. Feature dump (`InpDumpFeatures = true`, tester over the research period) and trade run (Open prices only, `InpMaxGapHours = 0`), then `scripts/check_parity.py` (see `12_ea/REVIEW_CHECKLIST.md`). The first difference it reports is the clue.
5. Decide when to open the holdout (once, `scripts/open_holdout.py --yes-i-will-not-tune-after-this`); it has NOT been opened on any run. Do it only after the parity work is final.

**B. Next milestone in a session without MetaTrader:** M11 Engine B (shallow trees) + depth 3-4, or M12 reporting (RUN_SUMMARY), or M10b (generator support for the refused feature classes, only if real accepted rules need them: NOTES Q-008). Open decisions: Q-009 (weekend holds), Q-010 (risk-percent sizing not generated), Q-011 (catastrophic stop needs 1-minute OHLC / tick mode).

## Checkpoint 14 facts (first pytest execution round; fixes applied 2026-11-19, re-run pending)
- First `pytest -q` on the user's machine (M11/M12 code never executed before, NOTES G-018) exposed the expected breakage; fixed in two rounds across tests and sources. Status: fixes applied, **a fresh `pytest -q` run is still pending** (the sandbox cannot execute; the user verifies).
- Source fixes: `pipeline.py` imports `regime_diag`; `filters.py` `_ensure_f8_cols` before the F8 `.at` writes (pandas >= 2 refuses True into float64) + `_ladder_means` accepts a single Atom; `trees.py` harvests leaves per tree (cross-tree `hits` counting now works, min_hits reachable); `exits.py` `exit_grid` keeps B and I grids separate (a dict keyed (tp, sl) let I cells overwrite B cells); `engine.py`/`accounting.py` carry a `hold` column; `config.tester_overrides` takes `base_dir`; `validation/demo.py` `update_gate_table` APPENDS a result for a rule id missing from the gate table (a demo verdict is never silently dropped).
- Test fixes: family grid widened to 42 (18 I families) in candidates/exits tests; reporting helper `mkdir(exist_ok=True)`; `test_validation_m8._rule` passes `meta` as a keyword (positional slot now belongs to `exit_condition`); labels intrabar tests use `parse_family`; dedup-corr test reads the first value per cluster (`f_cluster_size` is broadcast over the cluster); deep tests give the mixed pairs row a distinct atom pair so keep-first dedup cannot drop it; stage-6 resume test pins `DEEP_OFF` so only stage 6 reruns.
- Pandas4Warning deprecations (`weekday`/`dayofweek`) are cosmetic and left for a later pass.
- Second pytest round (6 failures left, all fixed): `run_summary.py` line with `_read_csv(...) or pd.DataFrame()` tested a DataFrame for truthiness — explicit None check; `symbol_config` gained `base_dir` and the tester-override test passes it (it wrote the override into tmp_path but the code read the default config dir); the directions-test `evaluate_rule` monkeypatch stub lacked the M11 `lift_min` / `lift_min_t` keys that `build_candidates` stores as `best_lift_min`.
- Third pytest round (2 failures left, both M12 reporting, fixed): the RUN_SUMMARY holdout line read the manifest only — the holdout stage fills `g10_holdout` in the gate table WITHOUT a new stage-10 fingerprint, so the summary now reports "opened" when EITHER the manifest logs an access OR any g10 verdict is present; the per-rule page's `Fail reasons` line rendered `nan` for an empty CSV cell (NaN, not "") — now ``.
- Fourth round (last failure, fixed): the per-rule page lacked the one-line headline `Trades: N, expectancy X R.` — added from the first stage-11 segment that has per-rule data. Windows pytest run: 374 tests, only this one failed.

## Checkpoint 13 facts (verification fixes; details NOTES D-059, D-060, G-020)
- Found by the post-checkpoint-12 verification (3 read-only agents over the design chapters + direct line checks): the null test violated spec 13.3 ("the null procedure is IDENTICAL to the real one") — `run_null` passed the feature table for the real statistic but NOT for the replicates, so with `deep.trees.include_in_null: true` Engine B was silently skipped in every replicate. Fixed in `null.py` (one line) and pinned by tests.
- Staleness: `base_fp` (the replicate reuse key) hashed only `07_filters` + seed, so a change to the SEARCH PROCEDURE could never invalidate stored replicates. `null_base_fingerprint` (pipeline) now hashes `07_filters | seed | SEARCH_VERSION`; `SEARCH_VERSION = "2"` in `null.py` — bump it on any change to `search_statistic` / `deep_rules_for_search`.
- Deep scope note: `deep_rules_for_search` built a scope note that reached NOTHING; `search_statistic` now carries `deep_used` + `deep_note`, `null_summary` forwards them, and `summary.md` has a "Deep scope" line + `null_info.json` stores it. Trees requested without a feature table append an explicit SKIPPED warning to the note.
- `rule_flags` (spec 13.4): without calendar replicates the old code used an infinite calendar threshold, so `calendar_driven` degenerated to `beats_random` for EVERY representative and flowed into `freeze_rules` meta / the gate table on partial runs. Now NaN = UNKNOWN in `rule_flags.csv`, `None` in frozen-rule meta, "?" in `summary.md`.

## Checkpoint 12 facts (audit-gap fixes; details NOTES D-054..D-058, G-019)
- Label table: `I{tp}_{sl}_{H}` intrabar families (spec `S{...}`) on the same barrier grid (`labels.intrabar_families`, default true) + the catastrophic layer `labels.cat_stop_atr` applied to EVERY family; the exit mechanics (`entry_and_levels`, `_touch`, `_to_bid`, `intrabar_exit`, `apply_cat_layer`) now live in `labels/builder.py` and the ENGINE imports them (behaviour unchanged — the 48-case scalar-reference engine test must stay green). No high/low in the data: I families are skipped with a spec note (an explicit cat stop is an error). `exits.exit_grid` treats I like B.
- F8/F9: two latent bugs fixed (`.at` column enlargement, unsorted simplified atoms) and the f8 flag columns are always present with clean dtypes; `dedup_corr` (P&L correlation over the union of trigger bars) merges on top of Jaccard; representative = simplest first. New tests in `tests/test_filters.py`.
- Condition exits carried: `FrozenRule.exit_condition` + `exit_condition_signal`; `run_backtest` passes them to the engine; codegen refuses (Q-008). Tests in `test_backtest_m9.py` / `test_codegen_m10.py`.
- Per-symbol tester config: `config/MT5_TESTER/<SYMBOL>/<TF>.yaml` (costs + engine only), `symbol_config`, `scripts/run_backtest.py --symbol --timeframe`; new key `engine.fidelity: open_prices_only` (the only valid value, checked in two places). Tests + `config/MT5_TESTER/README.md`.
- G11 offline harness: `validation/demo.py` + `scripts/g11_demo.py` (per-rule demo statement CSV -> `compare_trades` -> PASS/WARN/FAIL/NO_TRADES -> `g11_demo.json` + gate-table column update). Tests in `test_validation_m8.py`.

## M11 / M12 facts (WRITTEN AND REVIEWED, NEVER EXECUTED; details NOTES D-051..D-053, G-018)
- Code: `edge_research/discovery/{deep,trees}.py` (stage 5b), `edge_research/reporting/run_summary.py` (stage 13), wiring in `pipeline.py` (`run_stage_5b`, stage-6 two-source input, `run_stage_13`, `run_stages_0_12/0_13`), config sections `deep:` and `reporting:` in `config/default.yaml`, scripts `run_stages_0_13.py`, `dev_mutate_m11.py`, `dev_mutate_m12.py`.
- Engine A (Apriori): beam expansion of the depth-2 survivors to triples/quads with the SAME coarse filters as depth 2, merged by canonical rule key; Engine B (trees): shallow trees split only at atom quantile thresholds, week-block bootstrap, leaf -> candidate rule with `min_hits`; both feed stage 6 through `normalise_rules` (one input shape; dedup on rule_key + direction via a `_key` string column — pandas cannot hash the atoms lists).
- Stage 6 walks pairs AND deep rules (fingerprint has both upstreams); `lift_over_parents` generalised to every leave-one-out parent for 3-4 atom rules (stage 6 order: drop last, drop first, then the middle), `apply_gates` uses `lift_min_t` for rules with > 2 atoms.
- Stage 13: RUN_SUMMARY.md + gate log + per-rule pages for ACCEPTED rules; a pure view of the stage outputs, honours `reporting:` config; re-reports when the gate table changes without a new stage 10 fingerprint (holdout fills g10).
- Bug fixed this checkpoint: `normalise_rules` dedup originally passed the unhashable atoms lists to drop_duplicates — replaced by the `_key` string column (test: duplicate rows, same rule in both directions, repeated atom, same-feature pair).

## M10 facts (synthetic strong edge, 6000-bar test world; details NOTES D-046..D-050, G-015..G-017)
- Code: `edge_research/codegen/{indicators,recipes,mql_helpers,plan,render,pyeval,ea_sim,parity,lint,stage}.py`, `pipeline.run_stage_12`, config section `codegen:`, scripts `gen_ea.py`, `check_parity.py`, dev script `dev_mutate_m10.py`.
- Recipes: 58 of the 64 registry features on the synthetic columns reproduce the Python feature table (values and NaN pattern); the others are refused with a reason (rolling state, bars-since events, custom indicators, calendar).
- EA algorithm vs engine: `ea_sim` (bar-by-bar model of the EA's `OnTick`) gives 100 % entry / exit time and price match with `run_backtest` for long and short, exits T1 / T6 / T24 / B1_1p5_12 / B2_1_6 / B3_2_24, `reenter_on_exit_bar` true and false, and with a random spread series. Sensitivity: an exit one bar late is rejected, a one-bar shifted feature dump is rejected.
- Weekend parity mode (NOTES D-048): `InpMaxGapHours = 0` in the tester and `ea_sim.parity_cfg` on the engine side.
- Findings from writing the tests (NOTES G-016): the indicator-override validation allowed statement injection (fixed); prices <= 0 are missing data.
- Mutation checks: 104 mutants, 103 caught, 1 equivalent; the MQL5 template logic that Python cannot execute is pinned by golden-text tests (weak by nature: only the tester proves it).
- NOT done: compile, tester, G9 as a whole (stress run, review checklist), risk-percent sizing, condition exits, features needing rolling state.

## M8 facts (synthetic strong edge, 30k bars, seed 11, default config; details NOTES D-042..D-045, G-010..G-014)
- Stage 10 (`validation/{freeze,gates,holdout,walkforward}.py`, `pipeline.run_stage_10`, `run_holdout_stage`, `run_stage_11`, `run_stages_0_10`, `run_stages_0_11`; scripts `run_stages_0_10.py`, `run_stages_0_11.py`, `open_holdout.py`, `run_walkforward.py`, `make_synth.py --strong`). Config section `validation:`. `data.holdout_pct: 0` = no holdout (walk-forward windows).
- Real run: G1 PASS (100 replicates, p_global 0.0099 = 1/101, false discovery 0.008 / 0.011). 50 frozen rules; G2 50, G3 50, G4 4, G5 9, G6 50, G7 50, G8 50; accepted 1: R02441 (`cci_14 <= -48.2 AND candle_body > 0.61`, T24 long): validation 92 trades, mean R 1.73, t 3.46 vs critical 3.18, PF 3.55, stress R 1.61. Holdout untouched.
- Outer walk-forward (train 12000 / test 3000 / step 3000 bars, 10 rules per window, no per-window null test): 4 of 4 windows with rules; pooled OOS mean R 0.96 vs IS 1.20 (ratio 0.80); every window positive (0.69 to 1.46); pooled OOS drawdown 8.7 %. The 3352 pooled trades come from only 1852 distinct entry bars (rules share signals, G-014). A planted edge this strong says nothing about real data.
- Mutation checks: 54 mutants of gates, holdout, walk-forward, pipeline wiring, splits, freeze: all caught (two first survivors, a calendar-window boundary and the pre-opening holdout label, were closed with new tests). Bugs found by writing tests: the walk-forward fingerprint ignored the null threshold; `load_run_inputs` ignored parquet.
- Not covered: G9 (needs M10), G11 (demo), a portfolio-level holdout verdict (Q-007), per-window null test (Q-006).

## M9 facts
- Engine: `mt5_backtest/engine.py` (exits T / B / I, catastrophic layer, condition exits, `max_open_positions`, `close_open_at_end`, `cost_multiplier`, strict `engine:` keys), `accounting.py` (lots, money, balance / equity per bar open), `metrics.py`, `compare.py`, `backtest.py` (multi-rule, one account), `rules.py` (canonical rule JSON). Details and conventions: NOTES D-037..D-041.
- `labels/builder._Market` gained `cost_mult`, `slip`, `high`, `low` and `result_at`; label results are unchanged (the label <-> engine tests pass).
- Tests: scalar bar-by-bar reference simulation vs the vectorised engine on random paths (48 parametrised cases incl. gaps, spreads, cat and condition layers), hand-computed money / equity / sizing / compare cases. Mutation check: 37 mutants, 35 caught, 2 equivalent (`op < lvl` vs `<=` on a gap exactly at the level fills at the same price; the B reason order cannot matter because one close cannot hit both levels).
- Smoke test on the synthetic strong-edge run (30k bars, rule 710 `T24`): train 236 trades, mean net R 2.31; val 77 trades, mean net R 1.42, PF 1.99 (synthetic edge is very strong).

## M5b / M6 / M7 facts (synthetic, H1, default config, strong planted edge hours 6-13)
- 30k bars, seed 11: 643 atoms -> 9077 survivors -> 5850 candidates (806 clusters) -> 5598 pass F2/F7/F4 (763 clusters after the second dedup, 13 s) -> 759 frozen representatives (5572 pass F3). Mean net R per trade of the top rules about 2 R at T24 (the synthetic edge is very strong; do not expect this on real data).
- Null test, 10 replicates per variant: real S = 10.0; random null threshold (q 0.9) 8.2, calendar 8.6, p_global 0.091 (= 1/11, the smallest possible with R = 10), false discovery 0.001 / 0.007. G1 PASS. With R = 10 the representatives' best-family t (6-7) is below the null MAXIMUM, so no individual rule is flagged as beating it; only the best rule overall (t = 10) does. Individual rules need much more t or a larger R; this is the honest reading of the null test (NOTES G-007).
- A rotated-label null on the planted world still yields about 8 (random) to 40 (calendar) rules passing F1-F2 per replicate, because hour/session structure and heavy-tailed labels survive the rotation. The calendar variant is the stricter one.
- Mutation checks: M5b 14 mutants, M6 13 mutants, M7 21 mutants: all caught (5 M7 survivors were closed with new tests: G1 alpha equality incl. the float `1 - 0.9`, replicate minimum, variant seeds, week-bars fallback and tie).
- Bug found by tests: the YAML key `null:` parses as None; the config section is `nulltest` (G-006).

## M5 facts (synthetic, H1, default config)
- Strong planted edge (drop 1.0, drift 3.0, hours 6-13), 20k bars, seed 1: 3587 survivors -> 3069 pass episodes -> 2571 + t -> 2093 + lift -> 349 clusters, 18 s. 40k bars, seed 7: 3446 -> 2530 candidates, 474 clusters, 47 s. All long, 0 short. The top representatives are `session == 1` paired with price-drop atoms (`ret_3_atr`, `ret_6_atr`, `candle_body`, `cci_14`), passing 22-23 of 24 families.
- Null data, 4 seeds at 20k bars: 0 survivors with the default screen. With the screen deliberately relaxed (979 / 182 / 469 / 1693 survivors) the exact filters still gave 0 candidates: most died at the episode count, none passed t >= 2.5 on 2 families.
- Cost of Stage 6: about 5 ms per survivor (24 families x episodes, lift only where episodes and t pass). The cap `discovery.max_survivors` (100k) would mean about 8 min; the evaluation loop is serial.
- Cluster count (474 of 2530) is an upper bound on independent ideas: one planted structure still yields hundreds of Jaccard < 0.6 variants (different thresholds, different companion atoms).
- A lift test with parent EPISODES as clusters is degenerate for dense parents (D-027); lift uses time blocks of 5 x hold bars.
- First ranking (mean t over passing families) put the planted rule 12th (D-028); rank is now the median t over all 24 families.
- Mutation checks: 21 deliberate breaks of `candidates.py`, `episodes.py` and `pipeline.py` (lift parent ignored, remainder = whole parent, SE factor, block width, fold edge, gross mean, min-episodes boundary, family-count off by one, Jaccard boundary, worst-ranked representative, directions merged, best family = min t, cost sign, long labels for short rules, rank = max t, dead both-direction flag, stage 6 ignoring the upstream fingerprint, ...) were all caught by `tests/test_candidates.py`.

## M4b facts (synthetic, H1, default config)
- 20k bars: about 620 atoms, 180k-200k legal pairs, screen takes 0.7-1.1 s; 60k bars: 2.7 s; 100k bars: 3.9 s.
- Null data (no planted edge), 8 seeds at 20k bars: 0 survivors every time. This may also mean the default screen is conservative: check on real data.
- Strong planted edge (drop 1.0, drift 3.0, hours 6-13): 3587 and 1113 survivors on two seeds, all long, 0 short; the top ones pair `session == 1` with a price-drop atom (`ret_6_atr`, `ret_3_atr`, `cci_14`, `rsi_14`), passing all 3 long labels.
- DEFAULT `PlantedEdge()` (hours 7-11): not found at 30k-100k bars (see Q-005).
- Mutation checks: 8 deliberate breaks of `pairs.py` (ignore label validity, NaN not zeroed, wrong lift combination, no hold deflation, lower triangle kept, fold positivity ignoring min rows, wrong parent remainder, chunk losing the last row) are all caught by `tests/test_discovery.py`.

## M3 calibration facts (synthetic data, 20-30k H1 bars)
- Null data (no planted edge): 0 of 16 seeds gave GO with `t_min: auto` (Bonferroni, alpha 0.05). With a fixed t_min 4.0, 2 of 8 seeds gave false GO.
- Strong planted edge (drop 1.0 sigma, drift 3.0 sigma, hours 6-13): GO; long edge gives long hits only, short edge short hits only.
- DEFAULT `PlantedEdge()` (drop 1.5 sigma, drift 0.8 sigma): univariate verdict NO_GO (conjunction edge, too diluted in single-feature bins). See Q-003.
- Stage 3 time on 20-30k bars: about 2-3 s.

## M4a facts (synthetic, 12-20k H1 bars)
- 607 atoms from 38 features (572 quantile, 33 category, 2 event), 321 stored masks (about 0.8 MB packed for 20k rows); build takes < 0.1 s.
- The 4 mutation checks done on the atom tests (complement without valid mask, thresholds from all rows, `<` instead of `<=`, quantization side) are all caught.

## NOT verified yet (no real data was available in sessions 1-2)
- Real exporter column names vs the registry (see NOTES G-002). Run Stage 0 on a real CSV first and read `00_data_qa/report.md`.
- The alignment check on real data, the `mom_*` centre, and `ps_*` timing (spec 19.2 "first checks").
- Whether the real spread CSV from `spreadExporter.mq5` has columns `time,spread` (pipeline.load_spread_csv assumes so).

## Free-tier working rules
- One milestone (or half) per session. Stop after updating this file; do not push to the limit.
- Write code and tests in the same step. Keep tool output short (`pytest -q`, `head`, `tail`).
- Never print large files; patch with small edits.
