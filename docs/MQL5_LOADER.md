# MQL5 loader mode: one CSV, every indicator processed

The pipeline reads the single CSV written by `mql5/dataExporter.mq5` (v1.03, 108 columns). Bars, per-bar spread, all indicators and the news columns come from that one file.

```bash
python scripts/run_mql5_pipeline.py audit  --csv data/EURUSD_D1_export_20261101.csv          # coverage + integrity findings, no run
python scripts/run_mql5_pipeline.py run    --csv data/EURUSD_D1_export_20261101.csv --out output/eurusd_d1 --through 2
python scripts/run_mql5_pipeline.py sample --out data/EURUSD_H1_export_sample.csv --bars 4000  # synthetic file in the exporter format (tests only)
```
`--through N` runs `run_stages_0_N` (2 = QA + features + labels ... 10 = + validation, 13 = all + report). Symbol and timeframe come from the file name; digits / point,
bars per year, the weekend-gap threshold and the fallback spread are derived from the file (see `edge_research/mql5_loader/runner.py`). Without `pip install -e .`, run from the repo root.

All-NaN exporter columns (including the seven `news_*` columns when the calendar was unavailable) are warned about and excluded from feature generation. If a short file cannot form the configured splits, `--through 2` still writes QA, feature, and label outputs; feature leakcheck and correlation clustering are skipped with a warning, and all remaining features are retained without correlation pruning. Later stages that need train/validation/holdout data still require enough rows for the configured embargo.

## What it does beyond "reading the file"
| step | where | result |
|---|---|---|
| header vs exporter manifest | `schema.py`, `loader.header_report` | missing / unlisted / reordered columns reported |
| envelope repair | `loader.repair_envelope` | `envelope_20_01_upper` was the LOWER band, `..._lower` all NaN (iEnvelopes buffer 0/1, exporter reads 1/2); rebuilt from SMA20 symmetry. Fix the source with `mql5/dataExporter_envelope_fix.patch` |
| integrity checks | `loader.integrity_report` | BB middle == SMA20, (upper-middle)/stddev == deviation, OHLC, volumes, spread coverage |
| spread | `integration.prepare_run_inputs` | `spread` column of the same file (0 = unknown -> last known -> fixed fallback) |
| features for every column | `features.py` | `spread_atr`, `spread_rel`, `real_volume_rel`, `atr_ratio_14_20`, `dist_ama_10_2_30`, `bb_mid_dist_*` + all stock features |
| guard | `coverage.py` | a column read by NO feature stops the run (`--allow-unprocessed` to override) |
| report | `01_features/column_coverage.md` | per column: used / redundant (merged by correlation clustering) / qa_excluded (constant, NaN) / gated |
| EA generator | `extension.py`, `schema.codegen_overrides` | exact handle + buffer for every indicator column; refusals carry the reason |

Backtest runs (stage 11) also write an equity + balance PNG per segment, with settings per pair: see `docs/BACKTEST_CHART.md`.

Outputs next to the stock ones: `_ingest/mql5_loader_report.json`, `_ingest/column_coverage_before.md`, `config_used.json`, `mql5_loader_summary.json`, `00_data_qa/excluded_columns.json`.
