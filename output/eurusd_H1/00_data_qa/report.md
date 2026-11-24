# Stage 0 data QA report

data_hash: `617695beee2533846280854daf621d9e`

| check | status | detail |
|---|---|---|
| time_index | pass | monotonic increasing, no duplicates |
| bar_count | pass | 6208 bars/year vs expected ~6000 |
| gaps | warn | 48 non-weekend gaps (max 79.0h); 1349 weekend gaps |
| columns | warn | 9 columns excluded: ['news_count_next_24h', 'news_in_window', 'news_last_importance', 'news_last_surprise', 'news_min_since_last', 'news_min_to_next', 'news_next_importance', 'ps_dist_sess_h_atr'] |
| alignment (critical) | pass | exported SMAs match close-based recomputation at shift 0 |
| spread | warn | coverage 85.3%; median 8.0 pts, p95 50.0 pts |
| warmup | warn | 161423 leading rows are not fully finite; stage 1 starts the feature table at the first valid row |

critical_failed: **False**
warmup_rows: **161423**

## Excluded columns
- `news_count_next_24h`: constant or all-NaN
- `news_in_window`: constant or all-NaN
- `news_last_importance`: constant or all-NaN
- `news_last_surprise`: constant or all-NaN
- `news_min_since_last`: constant or all-NaN
- `news_min_to_next`: constant or all-NaN
- `news_next_importance`: constant or all-NaN
- `ps_dist_sess_h_atr`: 6.7% NaN after warm-up
- `ps_dist_sess_l_atr`: 6.7% NaN after warm-up

## Flagged columns
- `mom_10`: centre ~100 (median 100.0); transform to mom-100
- `mom_14`: centre ~100 (median 100.0); transform to mom-100
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
