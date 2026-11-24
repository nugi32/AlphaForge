# Per-symbol / per-timeframe tester overrides (design 18)

One global `costs:` block is not enough for real markets: spread, commission and swap differ per symbol, and the
tester mode is fixed per market. This folder holds ONE file per symbol and timeframe:

    config/MT5_TESTER/<SYMBOL>/<TF>.yaml      e.g. config/MT5_TESTER/EURUSD/H1.yaml

A file may set ONLY the `costs:`, `engine:` and `backtest:` sections; everything else stays global. The file is deep-merged over
`config/default.yaml` (+ any `--config`) wherever the config is loaded with a market context:

    python scripts/run_backtest.py --run output/run1 --rules output/run1/rules.json --symbol EURUSD --timeframe H1

A missing file means "no override" — the global costs are used. Example file:

```yaml
costs:
  point: 0.00001
  digits: 5
  fixed_spread_points: 7        # typical EURUSD spread in points; a spread CSV still wins when given
  slippage_points: 1
  commission_per_lot_per_side: 3.5
  swap_long_per_lot: -8.3
  swap_short_per_lot: 3.1
  swap_rollover_hour: 0
  swap_triple_weekday: 2        # Wednesday

engine:
  fidelity: open_prices_only    # the ONLY supported value; the parity contract is built on it
  no_weekend_hold: true
```

`engine.fidelity` (default `open_prices_only`) records the Strategy Tester mode the whole pipeline assumes; any other
value is rejected at config load and by `engine_settings`.

## Backtest output per pair: `backtest:` section

Every backtest run (stage 11 of the pipeline, `scripts/run_backtest.py`, `scripts/run_mql5_pipeline.py run --through 11`) writes, per segment
(`<run>/11_backtest/<segment>/`): `trades.csv`, `equity.csv`, `by_year.csv`, `by_month.csv`, `summary.json`, `report.md` and
**`equity_balance.png`** (equity + balance chart with an equity drawdown panel; `report.md` embeds it).

The per-pair file controls it:

```yaml
backtest:
  segments: [train, val]      # train | val | all (the holdout is opened only by the validation stage)
  account_currency: USD       # label of the money axis
  chart:
    enabled: true             # false = no PNG
    filename: equity_balance.png
    dpi: 150
    width: 14.0
    height: 8.0
    drawdown_panel: true
    crop_to_trades: true      # cut the flat stretch before the first entry / after the last exit
    stats_box: true
```

`_TEMPLATE.yaml` is a commented copy of all keys; copy it to `<SYMBOL>/<TF>.yaml`. The EURUSD / GBPUSD / XAUUSD files are EXAMPLES: replace the numbers with the
specification of YOUR broker. The runner puts the symbol into the chart title automatically (from the exporter file name).
