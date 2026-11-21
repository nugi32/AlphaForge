# Per-symbol / per-timeframe tester overrides (design 18)

One global `costs:` block is not enough for real markets: spread, commission and swap differ per symbol, and the
tester mode is fixed per market. This folder holds ONE file per symbol and timeframe:

    config/MT5_TESTER/<SYMBOL>/<TF>.yaml      e.g. config/MT5_TESTER/EURUSD/H1.yaml

A file may set ONLY the `costs:` and `engine:` sections; everything else stays global. The file is deep-merged over
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
