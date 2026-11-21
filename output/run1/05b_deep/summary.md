# Stage 5b (M11) - deepening the search: Apriori depth 3-4 (Engine A) and shallow trees (Engine B)

- Engine A (Apriori): on; pairs in: 0; deep survivors: 0; cells screened: 0; time 0.0 s.
  Beams: 5000 triples / 2000 quads; max atoms 4; the same coarse filters as depth 2 (rows >= 500, coarse t >= 2.0, >= 4 positive folds, lift over the parent > 0).
- Engine B (trees): on; trees 200 x depth [3, 4]; features used 35; splits 304; leaf rules 4525; rules kept 59; time 116.74 s.
  Target families T6, B2_1_12 per direction, winsorized at +-3.0 R; a leaf must hold >= 200 rows, appear in >= 2 trees and have a positive TRAIN mean.

Both lists are CANDIDATES for stage 6 only: the exact episode / lift statistics, the filters, the exit choice, the null test and validation decide
what survives. Engine B is a comparison engine: rules found by both engines are more credible than rules found by one (spec 10.3).

Null-test scope: the null test repeats the FULL deep stage (Apriori expansion AND tree engine) inside every replicate.

## Engine A (Apriori) survivors

None.

## Engine B (trees) rules (top 15 by score)

| rule | n_atoms | direction | score | hits | family |
|---|---|---|---|---|---|
| adx_14 > 19.4399 & ret_6_atr > 2.28326 & bars_since_ma10_20_cross_down > 36 | 3 | short | 0.716 | 2 | T6 |
| adx_14 <= 14.7481 & dist_ll_50 <= 7.47065 & bars_since_rsi14_above70 <= 46 | 3 | long | 0.692 | 2 | T6 |
| ret_6_atr > 2.28326 & bars_since_ma10_20_cross_down > 36 | 2 | short | 0.664 | 3 | T6 |
| rel_ma_20_sma_ma_50_sma > 1.07024 & slope_ma_20_sma_10 <= -0.029755 | 2 | short | 0.565 | 3 | B2_1_12 |
| stoch_14_3_3_k > 88.7666 & rel_ma_20_sma_ma_50_sma > 2.50781 | 2 | short | 0.555 | 2 | T6 |
| stoch_14_3_3_k <= 88.7666 & rel_ma_20_sma_ma_50_sma > 1.07024 & slope_ma_20_sma_10 <= -0.029755 | 3 | short | 0.540 | 2 | B2_1_12 |
| stoch_14_3_3_k > 88.7666 & rel_ma_50_ema_ma_200_ema > 2.62962 & bars_since_ma10_20_cross_up > 19 | 3 | short | 0.537 | 2 | T6 |
| macd_12_26_9_signal_atr > 1.08932 & dist_hh_20 <= 0.349656 | 2 | short | 0.533 | 2 | T6 |
| dist_hh_20 <= 0.349656 & dist_ll_50 > 10.0873 | 2 | short | 0.533 | 2 | T6 |
| dist_hh_50 <= 0.531305 & dist_ll_50 > 10.0873 | 2 | short | 0.506 | 2 | T6 |
| stoch_14_3_3_k > 88.7666 & bars_since_ma10_20_cross_down > 36 | 2 | short | 0.502 | 2 | T6 |
| dist_hh_20 <= 0.349656 & bars_since_ma10_20_cross_down > 36 | 2 | short | 0.493 | 5 | T6 |
| dist_ma_200_ema > 8.70357 & dist_ll_50 <= 7.47065 & bars_since_rsi14_above70 <= 46 | 3 | long | 0.469 | 3 | B2_1_12 |
| bbwidth_bb_20_2 > 7.7348 & dist_hh_20 <= 0.349656 | 2 | short | 0.465 | 2 | T6 |
| bars_since_rsi14_below30 > 22 & bars_since_rsi14_below30 <= 39 & bars_since_ma10_20_cross_down > 45 | 3 | short | 0.449 | 2 | T6 |
