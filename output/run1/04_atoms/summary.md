# Stage 4 atoms summary

- Atoms: 643 (stored masks: 339; `>` atoms are derived from the stored `<=` mask)
- By kind: {'quantile': 608, 'category': 33, 'event': 2}
- Features with atoms: 40; train rows used: 17771
- Mask storage: 1241.5 KiB packed for 30000 rows
- Dropped: 12 (see dropped_atoms.csv)

## Dropped by reason

- duplicate threshold (ties): 9
- support outside limits: 3

Thresholds come from TRAIN rows only and are fixed numbers from here on (validation and holdout apply them via `apply_atoms`).
