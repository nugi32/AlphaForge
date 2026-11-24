# Stage 3 univariate summary

**Verdict: NO_GO**

- Train rows: 4034; weekly clusters: 36; consistency unit: quarter (4 periods)
- Cells tested (n >= 200): 57282; hits (gross lift > 0, t >= 4.78): 0; consistent hits: 0
- Expected false hits if cells were independent: 0.05 (t_min is a Bonferroni bound; cells are correlated, so it is conservative)

## Baseline (random entry, train)

| label | n | gross R | cost R | net R | t net |
|---|---|---|---|---|---|
| long_B1_1_12 | 3810 | -0.0505 | 0.3413 | -0.3919 | -5.02 |
| long_B1_1_24 | 3807 | -0.0596 | 0.3413 | -0.4010 | -4.25 |
| long_B1_1p5_12 | 3775 | -0.0708 | 0.3420 | -0.4128 | -4.11 |
| long_B1_1p5_24 | 3768 | -0.1058 | 0.3420 | -0.4478 | -3.53 |
| long_B1_2_12 | 3750 | -0.0997 | 0.3424 | -0.4421 | -3.93 |
| long_B1_2_24 | 3732 | -0.1515 | 0.3423 | -0.4938 | -3.42 |
| long_B2_1_12 | 3780 | -0.1014 | 0.3419 | -0.4434 | -4.78 |
| long_B2_1_24 | 3756 | -0.1224 | 0.3426 | -0.4650 | -3.82 |
| long_B2_1p5_12 | 3743 | -0.1381 | 0.3426 | -0.4807 | -4.00 |
| long_B2_1p5_24 | 3690 | -0.1869 | 0.3435 | -0.5304 | -3.25 |
| long_B2_2_12 | 3717 | -0.1717 | 0.3429 | -0.5146 | -3.91 |
| long_B2_2_24 | 3646 | -0.2334 | 0.3434 | -0.5768 | -3.13 |
| ... 72 more labels in baseline.csv | | | | | |

Baseline check passed: no significant gross drift and no positive random-entry net R.

## Hits

No cell passed the hit rule. Per the spec this means: stop and report; consider another timeframe, instrument or feature family. Caveat (NOTES Q-003): an edge that exists only as a CONJUNCTION of conditions can be invisible to a one-feature view at this sample size, so NO_GO is a recommendation to review, not proof that no edge exists.

## Notes

- Hits use the GROSS lift over the baseline of the same label; the cell may still have negative net R after costs (see net mean R). CI columns in univariate.csv refer to net R.
- This stage selects nothing: thresholds and atoms are built in Stage 4. Use this output only to see where information may sit.
