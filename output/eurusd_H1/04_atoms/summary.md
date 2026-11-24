# Stage 4 atoms summary

- Atoms: 1197 (stored masks: 616; `>` atoms are derived from the stored `<=` mask)
- By kind: {'quantile': 1162, 'category': 33, 'event': 2}
- Features with atoms: 72; train rows used: 4034
- Mask storage: 12138.3 KiB packed for 161423 rows
- Dropped: 22 (see dropped_atoms.csv)

## Dropped by reason

- duplicate threshold (ties): 19
- support outside limits: 3

Thresholds come from TRAIN rows only and are fixed numbers from here on (validation and holdout apply them via `apply_atoms`).
