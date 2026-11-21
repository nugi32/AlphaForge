 tsunugi on  /run/media/tsunugi/Projects/new-edge-search/edge-research-v2 
# python scripts/make_synth.py --out data/synth_edge.csv --bars 30000 --edge
wrote data/synth_edge.csv (30000 bars, planted triggers: 218)
 tsunugi on  /run/media/tsunugi/Projects/new-edge-search/edge-research-v2 
# python scripts/run_stages_0_9.py --csv data/synth_edge.csv --spread-csv data/synth_edge_spread.csv --out output/run1
Found NaN values:
atr_14                  13
atr_20                  19
rsi_14                  14
cci_14                  13
mom_10                  10
stoch_14_3_3_k          17
wpr_14                  13
ma_5_sma                 4
ma_5_ema                 4
ma_10_sma                9
ma_10_ema                9
ma_20_sma               19
ma_20_ema               19
ma_50_sma               49
ma_50_ema               49
ma_100_sma              99
ma_100_ema              99
ma_200_sma             199
ma_200_ema             199
bb_20_2_upper           19
bb_20_2_middle          19
bb_20_2_lower           19
stddev_20               19
macd_12_26_9_main       25
macd_12_26_9_signal     33
adx_14                  26
dtype: int64


{
  "status": {
    "00_data_qa": "ran",
    "01_features": "ran",
    "02_labels": "ran",
    "03_eda": "ran",
    "04_atoms": "ran",
    "05_discovery": "ran",
    "05b_deep": "ran",
    "06_candidates": "ran",
    "07_filters": "ran",
    "08_exits": "ran",
    "09_null": "ran"
  },
  "n_rows": 30000,
  "n_features": 64,
  "first_valid_row": 199,
  "eda": {
    "verdict": "GO",
    "n_hits": 8,
    "n_consistent_hits": 8
  },
  "n_atoms": 643,
  "n_survivors": 0,
  "n_deep": 59,
  "n_candidates": 0,
  "n_clusters": 0,
  "n_deep_candidates": {},
  "n_filtered": 0,
  "n_filtered_clusters": 0,
  "n_frozen": 0,
  "n_pass_f3": 0,
  "g1_pass": false,
  "null_complete": true
}