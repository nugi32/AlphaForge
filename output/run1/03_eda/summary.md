# Stage 3 univariate summary

**Verdict: GO**

- Train rows: 17771; weekly clusters: 149; consistency unit: year (3 periods)
- Cells tested (n >= 200): 32590; hits (gross lift > 0, t >= 4.67): 8; consistent hits: 8
- Expected false hits if cells were independent: 0.05 (t_min is a Bonferroni bound; cells are correlated, so it is conservative)

## Baseline (random entry, train)

| label | n | gross R | cost R | net R | t net |
|---|---|---|---|---|---|
| long_B1_1_12 | 16747 | 0.0108 | 0.2302 | -0.2195 | -9.00 |
| long_B1_1_24 | 16731 | 0.0146 | 0.2303 | -0.2157 | -7.92 |
| long_B1_1p5_12 | 16589 | 0.0379 | 0.2308 | -0.1929 | -5.99 |
| long_B1_1p5_24 | 16548 | 0.0453 | 0.2309 | -0.1856 | -4.85 |
| long_B1_2_12 | 16488 | 0.0539 | 0.2309 | -0.1770 | -4.72 |
| long_B1_2_24 | 16399 | 0.0745 | 0.2312 | -0.1567 | -3.38 |
| long_B2_1_12 | 16586 | -0.0031 | 0.2306 | -0.2337 | -7.24 |
| long_B2_1_24 | 16509 | 0.0006 | 0.2308 | -0.2302 | -5.78 |
| long_B2_1p5_12 | 16407 | 0.0286 | 0.2311 | -0.2025 | -4.89 |
| long_B2_1p5_24 | 16271 | 0.0370 | 0.2314 | -0.1944 | -3.58 |
| long_B2_2_12 | 16298 | 0.0503 | 0.2313 | -0.1809 | -3.77 |
| long_B2_2_24 | 16080 | 0.0787 | 0.2318 | -0.1531 | -2.32 |
| ... 72 more labels in baseline.csv | | | | | |

Baseline check passed: no significant gross drift and no positive random-entry net R.

## Features with hits (best cell per feature)

| feature | hit cells | consistent | best label | bin | gross lift R | net mean R | t | positive periods |
|---|---|---|---|---|---|---|---|---|
| dist_hh_20 | 3 | 3 | short_T6 | 0 (0..0.3497) | 0.348 | 0.066 | 5.5 | 1.00 |
| stoch_14_3_3_k | 3 | 3 | short_T3 | 9 (88.78..98.57) | 0.255 | 0.003 | 5.5 | 1.00 |
| dist_hh_50 | 1 | 1 | short_T6 | 0 (0..0.5313) | 0.327 | 0.055 | 4.8 | 1.00 |
| rsi_14 | 1 | 1 | short_T3 | 9 (66.07..86.56) | 0.204 | -0.038 | 4.7 | 1.00 |

## Notes

- Hits use the GROSS lift over the baseline of the same label; the cell may still have negative net R after costs (see net mean R). CI columns in univariate.csv refer to net R.
- This stage selects nothing: thresholds and atoms are built in Stage 4. Use this output only to see where information may sit.
