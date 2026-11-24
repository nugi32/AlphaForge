# Roadmap

Status legend: TODO / IN PROGRESS / DONE. Update after every step.
Spec reference: `docs/SPEC_V2_CONDENSED.md` (section numbers in brackets).

| Milestone | Scope | Acceptance test | Status |
|---|---|---|---|
| M0 | Synthetic data generator with optional planted edge | Reproducible by seed; CSV passes the unchanged loader | DONE |
| M1 | Stage 0 QA + Stage 1 feature table, registry, correlation clusters, look-ahead test [5,6] | Look-ahead test green for every feature; alignment check works | DONE |
| M2a | Stage 2 label table (T and B families), shared cost model [7] | Hand-computed examples; long/short cost symmetry; vectorized swap == old swap_nights | DONE |
| M2b | Backtest engine core (time-stop + bar-boundary) [15] | Label <-> engine trade-by-trade match | DONE |
| M3 | Univariate analysis, go/no-go [8] | Planted edge visible on synthetic data; none on null data | DONE |
| M4a | Atoms: train quantile thresholds, bit-packed masks, derived complements, support filter, pairing rules [9] | Mask == brute force; popcount support == boolean sum; train-only thresholds | DONE |
| M4b | Engine A depth 2: exhaustive atom-pair statistics via matrix products, screening labels, coarse filters, episodes + effective n [10] | Pair stats == brute force on small data | DONE |
| M5 | Candidate filters F1 (exact stats, min episodes, t), F5 (lift over both parents), F9 (Jaccard dedup), partial F3 (>= 2 passing exit families) [11] | Planted edge survives, duplicates clustered, null gives none | DONE |
| M5b | Remaining candidate filters [11]: F2 time stability (years, profit shares, leave-one-year-out), F4 threshold plateau, F6 monotonicity (soft score), F7 cost stress x1.5. F3 neighbourhood and F8 simplification go with M6 / M8 | Plateau rule rejects a one-threshold spike; cost stress flips a marginal rule | DONE |
| M6 | Exit selection [12] | Plateau chosen, not peak | DONE |
| M7 | Null test over the whole search [13] | False pass rate near nominal on null data | DONE |
| M8 | Validation + outer walk-forward + holdout protocol [14] | Holdout access logged once | DONE (G1-G8 + G10 machinery; G9 needs M10, G11 needs a demo; the real holdout has not been opened on any run) |
| M9 | Engine extensions (intrabar, catastrophic, risk %, portfolio) [15] | Matches a MT5 trade list for a simple EA | DONE in code (the MT5 match itself needs a real tester list: `compare_trades` is ready, NOTES G-009) |
| M10 | Codegen + parity tests [16] | All parity tests green for one real rule | DONE in code and in the sandbox (generator, EA model vs engine, parity harness, stage 12). NOT compiled and NOT run in a tester: the real-rule acceptance needs MetaEditor / Strategy Tester on the user's machine (NOTES G-015) |
| M11 | Engine B (shallow trees) + triple/quad rules [10] | Two engines compared | TODO |
| M12 | Reporting [17] | One end-to-end run writes every artifact | TODO |
