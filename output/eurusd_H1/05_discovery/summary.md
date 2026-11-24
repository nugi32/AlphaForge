# Stage 5 - rule discovery, coarse screen of atom pairs (train rows only)

- Atoms: 1197; legal pairs: 708093; train rows: 4034; tiles processed: 15; time: 2.77 s.
- Screening labels: long_T3, long_T12, long_B2_1_12, short_T3, short_T12, short_B2_1_12.
- Survivors (pair, direction): 0 (long 0, short 0).

Survivors are SCREENING candidates, not evidence: the filters are coarse (hold-deflated t, fold signs, lift over parents) and the search
is wide. Nothing here is a finding until the exact statistics (M5), the exit selection (M6) and the null test over the whole search (M7).

No pair passed the screen. On real data this is a valid result for depth 2; deeper rules or finer atoms are not tried automatically.
