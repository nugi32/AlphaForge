# Edge Research V2 - condensed English spec

Condensed from the original (Indonesian) design document supplied by the project owner. It keeps the contracts
that code must obey. Section numbers refer to the original document, so they can be cross-checked.

## 1. Goal and principles [1]
Turn per-bar CSV exports (OHLCV + MT5 indicators) into complete trading rules (entry condition, direction, exit)
whose NET expectancy is positive, stable over time, based on enough independent trades, and exactly reproducible
in an MT5 EA. "No viable edge" is a valid result.

P1 metric used at every stage = net trade outcome in ATR units with the real exit rule.
P2 compute outcomes once (label table), evaluate any condition by aggregating on its trigger rows.
P3 no early pruning on a narrow proxy. P4 filter by evidence (effective n, stability, search-wide null test,
unseen data), not by quotas. P5 direction is a label (separate long/short columns), never inferred.
P6 every feature/rule must be exactly rewritable in MQL5 (feature registry has an MT5 recipe).
P7 label table and backtest engine must match trade by trade. P8 validation/holdout use is logged.
P9 reproducible: manifest with data hash, config hash, version, seed.

## 2. What was wrong in the old pipeline [2]
Ranking used a 1-bar proxy; binary outcomes ignored magnitude; z-tests assumed independent samples; exits were
time-stop only with in-sample hold selection; atom generator ignored most exported indicators (adx, macd, bb, sar...);
codegen translated crosses to plain comparisons and the EA had no SL/TP. Suspected bugs to verify on old code:
signed effect-size sort, `iloc[0]` direction, `mom_*` centered near 100 (not 0), price-scale columns used with fixed thresholds.

## 3. Pipeline [3]
T0 ingest+QA -> T1 features -> T2 labels -> T3 univariate go/no-go -> T4 atoms -> T5 rule discovery
(Engine A pairs->triples, Engine B shallow trees) -> T6 filters -> T7 exit selection -> T8 null test over the whole
search -> T9 OOS validation -> T10 backtest engine (MT5 emulation) -> T11 MQL5 codegen + parity.
Package layout: ingestion, qa, features, labels, eda, atoms, discovery, filters, exits, nulltest, validation,
mt5_backtest, codegen, reporting, checkpoint, utils.

## 4. Time and split contract [4]
1. One row = one CLOSED bar; features at row i use data up to close of bar i only.
2. Decide at close of bar i, execute at OPEN of bar i+1 (entry bar e = i+1).
3. Server time everywhere, no timezone conversion.
4. Holds are counted in bars (weekend gaps are inside the window; see label validity).
5. No future info in features, thresholds, normalization (quantile thresholds from TRAIN only).
Split chronologically: train ~60%, validation ~22%, holdout ~18%; embargo >= max_hold + margin between segments
(drop embargo bars on both sides). Holdout is used once; access is logged in the manifest.
Unit of outcome: R = net price result / ATR_i (atr column at signal bar i, default atr_14).

## 5. Stage 0 QA checks [5]
monotonic unique time (stop) | time parse >= 99% (stop) | bar count vs expectation (warn) | gaps > 2x interval excluding
weekends (warn, report) | warm-up NaN rows dropped | NaN% per column, >1% column flagged | constant columns dropped |
scale summary, `mom_*` centre check | ALIGNMENT: recompute SMA from close, compare with exported column at shifts -1/0/+1;
must match at shift 0 else STOP | `ps_*` flagged "needs verification" | spread coverage and fallback count.
Manifest: hashes of data and config, package version, seed.

## 6. Stage 1 features [6]
Stationary, MT5 recipe recorded, float32 table + 8-bit bins later. Registry fields: name, kind, sources, formula,
mt5_recipe, warmup_bars, reproducible_in_tester, family, notes.
Groups: bounded oscillators as is; unbounded oscillators as is; momentum (mom-100 if centre ~100); (close-MA)/ATR;
MA pair relations (fixed pairs 10/20, 20/50, 50/200)/ATR; slope (MA_t-MA_{t-k})/(k*ATR), k in {3,10}; Bollinger %B and
width/ATR; (close-SAR)/ATR; MACD family / ATR (force: rolling z-score); volatility ratios (atr/rolling mean atr, 100);
volume / rolling median; candle shape in ATR; return_k in ATR, k in {1,3,6,12,24}; distance to N-bar high/low, N in {20,50};
time (hour, dayofweek, session); events (bars since oscillator threshold cross / MA cross); `ps_*` as is (flag);
`cal_*`/`news_*` EXCLUDED by default (not reproducible in tester).
De-duplication: Spearman on train, hierarchical clusters at |rho| >= 0.9, keep one representative per cluster
(simplest in MQL5 first), save cluster map. Target 40-60 features. Leak test: perturb all rows after bar i, rebuild,
features at rows <= i must not change.

## 7. Stage 2 labels [7]
For EVERY bar i: hypothetical net result of entering at open of i+1, long and short separately, per exit family.
Costs (single source of truth, same as the old engine): Bid = open, Ask = open + spread. Long buys at Ask on entry bar and
sells at Bid on exit bar -> pays spread of ENTRY bar. Short sells at Bid, buys at Ask on exit bar -> pays spread of EXIT bar.
Slippage per fill (default 0). Commission = 2*commission_per_lot_per_side/contract_size (price units).
Swap = nights*swap_per_lot/contract_size using rollover hour / triple weekday.
Time-stop result: long = open[x]-slip-(open[e]+spread[e]+slip)-comm-swap ; short = (open[e]-slip)-(open[x]+spread[x]+slip)-comm-swap.
Store two components per column: gross_R and cost_R (net = gross - cost), so cost stress = extra subtraction.
Families: T{N} time-stop N in {1,2,3,6,12,24}; B{tp}_{sl}_{H} bar-boundary: after bar b closes, if close_b crosses SL/TP level,
exit at open of b+1, else continue to H (exact in Open-prices-only and live); S{...} intrabar (research only, pessimistic
SL-first); C{...} catastrophic stop layer (optional). Use T and B as primary families.
Validity: NaN if window passes last bar, non-finite prices/spread, window crosses a gap > max_gap_hours, or ATR invalid/<=0.
Label vs engine must agree trade by trade on simple rules.

## 8-14 Analysis, discovery, validation (summary) [8-14]
T3 univariate: per-feature decile profile of net R, effective-n aware; go/no-go on whether any signal exists.
T4 atoms: quantile thresholds (10..90 step 10) on TRAIN, support 1%..99%, stored as bit-packed masks.
T5 Engine A: exhaustive pairs per tile, then beam extension (beam_triple 5000, beam_quad 2000, max_atoms 4);
Engine B: shallow trees (200 trees, depth 3-4, feature_frac 0.3-0.5, winsor_R 3). Episodes merge consecutive triggers
(episode_gap = family max hold); effective n = episodes; lift over parent (t >= 1.5).
T6 filters: 5 folds, >= 4/5 positive folds and >= 70% positive years, max single-period profit share 50%, top5 share 30%,
plateau (+-1, +-2 steps), monotonicity, cost stress 1.5x, dedup (jaccard 0.6 / corr 0.7), simplification.
T7 exits: smooth exit surface (3x3 kernel), choose plateau not peak.
T8 null: rerun the IDENTICAL search on label series rotated in time (>= 50, default 100 replicates), compare best statistic
to null quantile 0.95; also calendar-week-aligned shift variant.
T9 validation: frozen rules on validation (decay_min_ratio 0.5, min 50 trades, PF >= 1.15, Bonferroni over clusters),
outer walk-forward re-running the WHOLE procedure per window, holdout once.

## 15. Backtest engine [15]
Emulates MT5 Open-prices-only: one position, decisions on closed bar, fills at next open, full costs, same exit semantics
as labels. Outputs trades, equity, balance, summary, report, comparison with MT5 Strategy Tester. Later: intrabar,
catastrophic stop, risk-percent sizing, multi-rule portfolio. Config = old `MT5TesterConfig` keys plus new ones.

## 16. MQL5 codegen [16]
Registry-driven (not string replace). EA computes features from the registry recipes, implements the same exits, one position.
Parity tests are mandatory: features, signals, trades Python == MQL5 within tolerance (default 1e-4). Reject EA generation if an
atom cannot be translated exactly. `live_only` rules (calendar/news) cannot pass tester parity.

## 17-18 Artifacts and config [17,18]
Run folder: manifest.json, 00_data_qa, 01_features, 02_labels, 03_eda, 04_discovery, 05_filters, 06_exits, 07_null,
08_validation, 09_backtest/<rule>, 10_ea/<rule>, RUN_SUMMARY.md. Resume per stage by fingerprint (data hash + stage config
hash + stage code version). Null replicates checkpointed one by one. Memory guard (min free GB, max RSS GB).
Memory rule of thumb: never store combination masks; process in row chunks/atom tiles; float32 labels; 8-bit bins.
Defaults are in `config/default.yaml`.

## 19. Test list [19.1]
no look-ahead | label hand examples | cost symmetry | label<->engine | pair stats == brute force | episodes/effective n |
recall on planted edge | null false-pass rate | frozen rules unchanged | holdout second access flagged | codegen rejects
untranslatable atoms | Python==MQL5 parity.
