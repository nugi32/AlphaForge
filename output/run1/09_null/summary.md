# Stage 9 - null test over the whole search (labels rotated inside the train segment)

- Real data: S = 0.00 (max cluster t among rules passing F1-F2), S_cont = 2.20, rules passing F1-F2: 0, all filters: 0.
- Replicates per variant: 100 (minimum 50); 200 of 200 done; complete: True; week = 120 bars; embargo = 24 bars.
- Deep scope (engines priced by the null statistic): apriori+trees - the null test repeats the FULL deep stage (Apriori expansion AND tree engine) inside every replicate.

## Variant `calendar` (100 replicates)
- S: threshold (q=0.95) = 0.00, null max = 0.00, p_global = 1.0000
- S_cont: threshold = 2.90, null max = 3.36, p_global = 0.1584
- Replicates with any rule passing F1-F2: 0%; mean rules passing per replicate: 0.00; false-discovery estimate = nan

## Variant `random` (100 replicates)
- S: threshold (q=0.95) = 0.00, null max = 0.00, p_global = 1.0000
- S_cont: threshold = 2.42, null max = 3.12, p_global = 0.1188
- Replicates with any rule passing F1-F2: 0%; mean rules passing per replicate: 0.00; false-discovery estimate = nan

**G1 (null test): FAIL** - fails: no rule passes F1-F2 on the real data; p_global_S=1.000 > 0.05; false discovery nan >= 0.5 or undefined

A resolution note: with R replicates the smallest possible p_global is 1/(R+1). The search's own thresholds make most null replicates empty (S = 0), so S_cont is the more informative continuous comparison.
