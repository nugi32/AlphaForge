# Stage 0 data QA report

data_hash: `fa9347a7d7632967ec43ff70b8bfe9e9`

| check | status | detail |
|---|---|---|
| time_index | pass | monotonic increasing, no duplicates |
| bar_count | pass | 6269 bars/year vs expected ~6000 |
| gaps | pass | no non-weekend gaps; 249 weekend gaps |
| columns | pass | 0 columns excluded: [] |
| alignment (critical) | pass | exported SMAs match close-based recomputation at shift 0 |
| spread | pass | coverage 100.0%; median 12.0 pts, p95 31.0 pts |
| warmup | pass | 199 leading rows are not fully finite; stage 1 starts the feature table at the first valid row |

critical_failed: **False**
warmup_rows: **199**

## Flagged columns
- `mom_10`: centre ~100 (median 100.0); transform to mom-100
