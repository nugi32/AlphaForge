# Stage 0 data QA report

data_hash: `ab13b6db19a196149de38494c63dfbd9`

| check | status | detail |
|---|---|---|
| time_index | pass | monotonic increasing, no duplicates |
| bar_count | pass | 261 bars/year vs expected ~260 |
| gaps | pass | no non-weekend gaps; 46 weekend gaps |
| columns | warn | 11 columns excluded: ['news_count_next_24h', 'news_in_window', 'news_last_importance', 'news_last_surprise', 'news_min_since_last', 'news_min_to_next', 'news_next_importance', 'ps_dist_sess_h_atr'] |
| alignment (critical) | pass | exported SMAs match close-based recomputation at shift 0 |
| spread | warn | coverage 1.7%; median 0.0 pts, p95 0.0 pts |
| warmup | warn | 231 leading rows are not fully finite; stage 1 starts the feature table at the first valid row |

critical_failed: **False**
warmup_rows: **231**

## Excluded columns
- `news_count_next_24h`: constant or all-NaN
- `news_in_window`: constant or all-NaN
- `news_last_importance`: constant or all-NaN
- `news_last_surprise`: constant or all-NaN
- `news_min_since_last`: constant or all-NaN
- `news_min_to_next`: constant or all-NaN
- `news_next_importance`: constant or all-NaN
- `ps_dist_sess_h_atr`: 19.9% NaN after warm-up
- `ps_dist_sess_l_atr`: 19.9% NaN after warm-up
- `ps_gap_fill_pct`: constant or all-NaN
- `real_volume`: constant or all-NaN

## Flagged columns
- `mom_10`: centre ~100 (median 99.7); transform to mom-100
- `mom_14`: centre ~100 (median 99.6); transform to mom-100
- `news_count_next_24h`: live_only: not reproducible in Strategy Tester
- `news_in_window`: live_only: not reproducible in Strategy Tester
- `news_last_importance`: live_only: not reproducible in Strategy Tester
- `news_last_surprise`: live_only: not reproducible in Strategy Tester
- `news_min_since_last`: live_only: not reproducible in Strategy Tester
- `news_min_to_next`: live_only: not reproducible in Strategy Tester
- `news_next_importance`: live_only: not reproducible in Strategy Tester
- `ps_dist_pdh_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_dist_pdl_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_dist_round_dn_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_dist_round_up_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_dist_sess_h_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_dist_sess_l_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_gap_atr`: needs_verification: custom indicator may expose a forming bar (parity test required)
- `ps_gap_fill_pct`: needs_verification: custom indicator may expose a forming bar (parity test required)
