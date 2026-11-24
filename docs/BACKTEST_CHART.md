# Backtest output: equity + balance chart and per-pair backtest config

Every backtest run ends with a report AND a picture. `write_backtest` (called by stage 11, `scripts/run_backtest.py` and therefore by
`scripts/run_mql5_pipeline.py run --through 11|13`) writes, per segment, into `<run>/11_backtest/<segment>/`:

| file | content |
|---|---|
| `equity_balance.png` | balance (step line) + equity (line) + initial deposit; lower panel = equity drawdown %; key numbers above the plot |
| `report.md` | the metrics report; embeds the PNG |
| `trades.csv`, `equity.csv`, `by_year.csv`, `by_month.csv`, `summary.json` | as before |

`13_report/RUN_SUMMARY.md` embeds the PNG of every segment; `mql5_loader_summary.json` lists them under `backtest_outputs`
(`"status": "no backtest output ..."` when no rule was accepted by stage 10, because then stage 11 has nothing to backtest).

Code: `edge_research/mt5_backtest/plots.py` (matplotlib, Agg canvas, no pyplot; a missing matplotlib only warns and skips the PNG).

## Per-pair config

`config/MT5_TESTER/<SYMBOL>/<TF>.yaml` may now set three sections: `costs`, `engine` and **`backtest`**
(`segments`, `account_currency`, `chart.*`). See `config/MT5_TESTER/README.md`, the commented `config/MT5_TESTER/_TEMPLATE.yaml` and the
examples in `config/MT5_TESTER/_examples/` (EURUSD, GBPUSD, XAUUSD; NOT active until copied to `<SYMBOL>/<TF>.yaml`; the numbers are examples, take your broker's).

Layer order (later wins): `default.yaml` < values derived from the CSV < `mql5_loader.yaml` < `--config` < `MT5_TESTER/<SYMBOL>/<TF>.yaml` < CLI.
`11_backtest` is fingerprinted on `engine`, `backtest`, `costs`: changing a pair file re-runs stage 11 only.
