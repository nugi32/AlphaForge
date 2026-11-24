# Stage 7 - candidate filters F2, F7, F4 (+ F6 soft) and dedup (best family of each candidate, train rows)

- F2: >= 4 positive folds, >= 70% positive Nones (with >= 10 episodes), leave-one-out >= 80%, no fold/period over 50% of the profit, 5 largest episodes <= 30%.
- F7: mean > 0 with all costs x 1.5.  F4: neighbours +-2 on the quantile ladder, median >= 50% of the original and >= 80% positive.
- Funnel: candidates 0 -> F2 0 -> + F7 0 -> + F4 0.
- Passed: 0; independent clusters after the second dedup: 0; time 0.0 s.

Still selected numbers: not evidence before the null test (M7) and validation (M8). F3 (exit neighbourhood) is checked in Stage 8.

No candidate passed.
