# Stage 6 - candidate filters and dedup (exact statistics on train rows)

- Filters: episodes >= 200 (gap = hold of the label family), cluster t >= 2.5, lift over BOTH parents t >= 1.5, each on >= 2 label families of the direction; dedup = Jaccard on trigger rows >= 0.6 (same direction).
- Funnel (candidates with >= 2 passing families): survivors 94 -> episodes 0 -> + t 0 -> + lift 0.
- Qualified candidates: 0; independent clusters (the number of distinct IDEAS): 0; time 0.94 s.

These are SELECTED numbers from a wide search: they are not evidence. Still missing: time stability beyond folds (F2), exit neighbourhood (F3), threshold plateau (F4),
cost stress (F7), exit selection (M6), null test over the whole search (M7), validation (M8).

No candidate passed. On real data this is a valid result for depth 2.
