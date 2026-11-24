# Stage 5b (M11) - deepening the search: Apriori depth 3-4 (Engine A) and shallow trees (Engine B)

- Engine A (Apriori): on; pairs in: 0; deep survivors: 0; cells screened: 0; time 0.0 s.
  Beams: 5000 triples / 2000 quads; max atoms 4; the same coarse filters as depth 2 (rows >= 500, coarse t >= 2.0, >= 4 positive folds, lift over the parent > 0).
- Engine B (trees): on; trees 200 x depth [3, 4]; features used 67; splits 581; leaf rules 3584; rules kept 94; time 40.62 s.
  Target families T6, B2_1_12 per direction, winsorized at +-3.0 R; a leaf must hold >= 200 rows, appear in >= 2 trees and have a positive TRAIN mean.

Both lists are CANDIDATES for stage 6 only: the exact episode / lift statistics, the filters, the exit choice, the null test and validation decide
what survives. Engine B is a comparison engine: rules found by both engines are more credible than rules found by one (spec 10.3).

Null-test scope: the null test repeats the FULL deep stage (Apriori expansion AND tree engine) inside every replicate.

## Engine A (Apriori) survivors

None.

## Engine B (trees) rules (top 15 by score)

| rule | n_atoms | direction | score | hits | family |
|---|---|---|---|---|---|
| spread_atr > 0.609942 & spread_rel > 1.5 | 2 | short | 1.680 | 2 | T6 |
| env_width_envelope_20_01 > 1.68366 & ret_6_atr <= -0.094375 | 2 | short | 1.330 | 2 | T6 |
| stoch_5_3_3_k <= 46.8123 & env_width_envelope_20_01 > 1.68366 | 2 | short | 1.325 | 7 | T6 |
| env_width_envelope_20_01 > 1.68366 & ps_gap_atr <= -0.0265671 | 2 | short | 1.297 | 2 | T6 |
| env_width_envelope_20_01 > 1.68366 & tick_volume_rel > 1.00196 | 2 | short | 1.268 | 2 | T6 |
| adx_14 <= 24.4054 & env_width_envelope_20_01 > 1.10798 & spread_atr <= 0.363795 | 3 | long | 0.855 | 2 | T6 |
| ao_atr <= -0.751835 & spread_atr > 0.609942 | 2 | short | 0.821 | 2 | B2_1_12 |
| rel_ma_50_ema_ma_200_ema <= -5.08332 & atr_ratio_14_20 <= 0.81857 | 2 | short | 0.790 | 3 | T6 |
| adx_14 <= 24.4054 & atr_ratio <= 0.956965 & spread_atr <= 0.260607 | 3 | long | 0.756 | 2 | B2_1_12 |
| adx_14 <= 19.8541 & spread_atr <= 0.220734 | 2 | long | 0.737 | 3 | T6 |
| tick_volume_rel > 0.821767 & atr_ratio_14_20 <= 0.742255 | 2 | short | 0.708 | 2 | T6 |
| chaikin_z > 0.256561 & spread_rel <= 0.625 | 2 | long | 0.675 | 2 | T6 |
| dist_dema_50 > 0.909205 & env_width_envelope_20_01 > 1.68366 | 2 | short | 0.666 | 2 | B2_1_12 |
| env_width_envelope_20_01 > 1.10798 & spread_atr <= 0.260607 | 2 | long | 0.655 | 3 | T6 |
| dist_tema_50 > 0.0563166 & env_width_envelope_20_01 > 1.68366 | 2 | short | 0.647 | 2 | B2_1_12 |
