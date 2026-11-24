# Edge Research V2

Outcome-first search for trading rules from MT5 indicator exports. The pipeline measures the NET trade result (in ATR units,
with the real exit rule and full costs) from the first stage on, instead of filtering on proxies.

Everything in this repository is written in English. The unchanged ingestion loader (`edge_research/ingestion/loader.py`) reads the CSV
produced by `dataExporter.mq5`.

## Status
Stages 0-9 are implemented and tested: data QA, feature table (64 stationary features with MT5 recipes), label table (time-stop and
bar-boundary exit families, long and short, gross and cost separated), univariate go/no-go, atoms, exhaustive atom-pair screen, and candidate
filters with dedup (exact episode statistics, lift over parents, Jaccard clustering), time-stability / plateau / cost-stress filters, exit selection on a smoothed surface, and a null test of the whole search by label rotation. Stage 10 (validation gates, holdout protocol), stage 11 (engine backtest), the outer walk-forward, and M10 (stage 12: MQL5 generator with a refuse-rather-than-approximate rule, an EA model checked against the engine, and the parity harness for gate G9) are implemented; the generated EA has NOT been compiled or run in a Strategy Tester yet (no MetaTrader in the development sandbox). See `CHECKPOINT.md` for the exact state and the next step.

## Quick start
```bash
pip install -e .[dev]
pytest -q
python scripts/make_synth.py --out data/synth_edge.csv --bars 30000 --edge
python scripts/run_stages_0_9.py --csv data/synth_edge.csv --spread-csv data/synth_edge_spread.csv --out output/run1


python scripts/run_mql5_pipeline.py audit --csv data/EURUSD/EURUSD_D1.csv
python scripts/run_mql5_pipeline.py run   --csv data/EURUSD/EURUSD_D1.csv --out output/eurusd_d1 --through 2


python scripts/run_mql5_pipeline.py run \
  --csv data/EURUSD/EURUSD_H1.csv \
  --out output/eurusd_H1 \
  --through 13
```
Validation (stage 10, holdout stays closed), engine backtest of accepted rules (stage 11), outer walk-forward, and the one-time holdout opening:
`python scripts/run_stages_0_10.py ...` / `run_stages_0_11.py ...` / `run_walkforward.py --csv ... --run output/run1` / `open_holdout.py --run output/run1 --yes-i-will-not-tune-after-this`.

Real data: `python scripts/run_stages_0_9.py --csv path/to/export.csv --spread-csv path/to/spread.csv --config my.yaml --out output/run1`
and read `output/run1/00_data_qa/report.md` before anything else.

MQL5 EAs for the accepted rules and their parity checks (stage 12, design 16):
`python scripts/gen_ea.py --run output/run1` -> `output/run1/12_ea/ER_<rule>.mq5`, `refused.json`, `REVIEW_CHECKLIST.md`; compile in MetaEditor, run the tester,
then `python scripts/check_parity.py --run output/run1 --rule R02441 --features ER_R02441_features.csv --trades ER_R02441_trades.csv`.

## Layout
| Path | Purpose |
|---|---|
| `edge_research/ingestion, checkpoint, utils, known_at` | Verbatim copies from the old project (do not edit) |
| `edge_research/synth` | Synthetic data with an optional planted edge (test harness, M0) |
| `edge_research/qa` | Stage 0 checks and report |
| `edge_research/features` | Stage 1 registry, table, look-ahead test, correlation clusters |
| `edge_research/labels` | Stage 2 cost model and label table |
| `edge_research/atoms` | Stage 4 atoms: train-quantile thresholds, bit-packed masks |
| `edge_research/discovery` | Stage 5: episodes, pair statistics by matrix products, coarse screen. Stage 6: `candidates.py`. Stage 7: `filters.py`. Stage 8: `exits.py` |
| `edge_research/nulltest` | Stage 9: null test over the whole search (`null.py`) |
| `edge_research/codegen` | M10: indicator table, feature recipes, plan, MQL5 renderer, lint, EA model (`ea_sim`), parity checks, stage 12 |
| `edge_research/eda`, `mt5_backtest` | Stage 3 univariate go/no-go; backtest engine core |
| `edge_research/splits.py`, `pipeline.py` | Chronological split with embargo; stage 0-9 runner with fingerprints |
| `config/default.yaml` | All parameters (starting points, not sacred numbers) |
| `docs/` | `SPEC_V2_CONDENSED.md`, `ROADMAP.md`, `NOTES.md`, `NEXT_SESSION_PROMPT.md` |
| `CHECKPOINT.md` | Session checkpoint: read first |

## Principles that tests enforce
No look-ahead in any feature; label values match hand calculations; long pays the entry-bar spread and short the exit-bar spread;
the vectorised swap equals the old loop implementation; the same data and config skip finished stages.
