# Exporter column coverage (before the run)

108 columns: 107 processed, 0 gated, 1 exempt, **0 unprocessed**

| column | group | status | features reading it |
|---|---|---|---|
| `time` | bar | exempt | bar timestamp: becomes the index of every table |
| `open` | bar | processed | candle_body, candle_lower_wick, candle_upper_wick, gap_atr |
| `high` | bar | processed | breakout_up_20, candle_close_pos, candle_range, candle_upper_wick, dist_hh_20, dist_hh_50 |
| `low` | bar | processed | breakout_down_20, candle_close_pos, candle_lower_wick, candle_range, dist_ll_20, dist_ll_50 |
| `close` | bar | processed | bb_mid_dist_bb_20_15, bb_mid_dist_bb_20_2, bb_mid_dist_bb_20_25, breakout_down_20, breakout_up_20, candle_body (+55) |
| `tick_volume` | bar | processed | tick_volume_rel |
| `real_volume` | bar | processed | real_volume_rel |
| `spread` | bar | processed | spread_atr, spread_rel |
| `ma_5_sma` | ma | processed | dist_ma_5_sma |
| `ma_5_ema` | ma | processed | dist_ma_5_ema |
| `ma_5_smma` | ma | processed | dist_ma_5_smma |
| `ma_5_lwma` | ma | processed | dist_ma_5_lwma |
| `ma_10_sma` | ma | processed | bars_since_ma10_20_cross_down, bars_since_ma10_20_cross_up, dist_ma_10_sma, rel_ma_10_sma_ma_20_sma |
| `ma_10_ema` | ma | processed | dist_ma_10_ema, rel_ma_10_ema_ma_20_ema |
| `ma_10_smma` | ma | processed | dist_ma_10_smma |
| `ma_10_lwma` | ma | processed | dist_ma_10_lwma |
| `ma_20_sma` | ma | processed | bars_since_ma10_20_cross_down, bars_since_ma10_20_cross_up, dist_ma_20_sma, rel_ma_10_sma_ma_20_sma, rel_ma_20_sma_ma_50_sma, slope_ma_20_sma_10 (+1) |
| `ma_20_ema` | ma | processed | dist_ma_20_ema, rel_ma_10_ema_ma_20_ema, rel_ma_20_ema_ma_50_ema, slope_ma_20_ema_10, slope_ma_20_ema_3 |
| `ma_20_smma` | ma | processed | dist_ma_20_smma |
| `ma_20_lwma` | ma | processed | dist_ma_20_lwma |
| `ma_50_sma` | ma | processed | dist_ma_50_sma, rel_ma_20_sma_ma_50_sma, rel_ma_50_sma_ma_200_sma, slope_ma_50_sma_10, slope_ma_50_sma_3 |
| `ma_50_ema` | ma | processed | dist_ma_50_ema, rel_ma_20_ema_ma_50_ema, rel_ma_50_ema_ma_200_ema, slope_ma_50_ema_10, slope_ma_50_ema_3 |
| `ma_50_smma` | ma | processed | dist_ma_50_smma |
| `ma_50_lwma` | ma | processed | dist_ma_50_lwma |
| `ma_100_sma` | ma | processed | dist_ma_100_sma |
| `ma_100_ema` | ma | processed | dist_ma_100_ema |
| `ma_100_smma` | ma | processed | dist_ma_100_smma |
| `ma_100_lwma` | ma | processed | dist_ma_100_lwma |
| `ma_200_sma` | ma | processed | dist_ma_200_sma, rel_ma_50_sma_ma_200_sma |
| `ma_200_ema` | ma | processed | dist_ma_200_ema, rel_ma_50_ema_ma_200_ema |
| `ma_200_smma` | ma | processed | dist_ma_200_smma |
| `ma_200_lwma` | ma | processed | dist_ma_200_lwma |
| `dema_10` | adaptive_ma | processed | dist_dema_10 |
| `dema_20` | adaptive_ma | processed | dist_dema_20 |
| `dema_50` | adaptive_ma | processed | dist_dema_50 |
| `tema_10` | adaptive_ma | processed | dist_tema_10 |
| `tema_20` | adaptive_ma | processed | dist_tema_20 |
| `tema_50` | adaptive_ma | processed | dist_tema_50 |
| `frama_14` | adaptive_ma | processed | dist_frama_14 |
| `frama_20` | adaptive_ma | processed | dist_frama_20 |
| `ama_10_2_30` | adaptive_ma | processed | dist_ama_10_2_30 |
| `vidya_9` | adaptive_ma | processed | dist_vidya_9 |
| `vidya_12` | adaptive_ma | processed | dist_vidya_12 |
| `sar_002_02` | trend | processed | dist_sar_002_02 |
| `sar_001_02` | trend | processed | dist_sar_001_02 |
| `sar_003_02` | trend | processed | dist_sar_003_02 |
| `adx_14` | trend | processed | adx_14 |
| `adxwilder_14` | trend | processed | adxwilder_14 |
| `rsi_7` | oscillator | processed | rsi_7 |
| `rsi_14` | oscillator | processed | bars_since_rsi14_above70, bars_since_rsi14_below30, rsi_14 |
| `rsi_21` | oscillator | processed | rsi_21 |
| `stoch_5_3_3_k` | oscillator | processed | stoch_5_3_3_k |
| `stoch_5_3_3_d` | oscillator | processed | stoch_5_3_3_d |
| `stoch_14_3_3_k` | oscillator | processed | stoch_14_3_3_k |
| `stoch_14_3_3_d` | oscillator | processed | stoch_14_3_3_d |
| `cci_14` | oscillator | processed | cci_14 |
| `cci_20` | oscillator | processed | cci_20 |
| `wpr_14` | oscillator | processed | wpr_14 |
| `mom_10` | oscillator | processed | mom_10_c |
| `mom_14` | oscillator | processed | mom_14_c |
| `macd_12_26_9_main` | oscillator | processed | macd_12_26_9_main_atr, macd_hist_atr |
| `macd_12_26_9_signal` | oscillator | processed | macd_12_26_9_signal_atr, macd_hist_atr |
| `osma_12_26_9` | oscillator | processed | osma_12_26_9_atr |
| `demarker_14` | oscillator | processed | demarker_14 |
| `rvi_10` | oscillator | processed | rvi_10 |
| `trix_14` | oscillator | processed | trix_14 |
| `ac` | oscillator | processed | ac_atr |
| `ao` | oscillator | processed | ao_atr |
| `bears_13` | oscillator | processed | bears_13_atr |
| `bulls_13` | oscillator | processed | bulls_13_atr |
| `atr_14` | volatility | processed | atr_ratio, atr_ratio_14_20 |
| `atr_20` | volatility | processed | atr_ratio_14_20 |
| `bb_20_2_upper` | volatility | processed | bbwidth_bb_20_2, pctb_bb_20_2 |
| `bb_20_2_middle` | volatility | processed | bb_mid_dist_bb_20_2 |
| `bb_20_2_lower` | volatility | processed | bbwidth_bb_20_2, pctb_bb_20_2 |
| `bb_20_15_upper` | volatility | processed | bbwidth_bb_20_15, pctb_bb_20_15 |
| `bb_20_15_middle` | volatility | processed | bb_mid_dist_bb_20_15 |
| `bb_20_15_lower` | volatility | processed | bbwidth_bb_20_15, pctb_bb_20_15 |
| `bb_20_25_upper` | volatility | processed | bbwidth_bb_20_25, pctb_bb_20_25 |
| `bb_20_25_middle` | volatility | processed | bb_mid_dist_bb_20_25 |
| `bb_20_25_lower` | volatility | processed | bbwidth_bb_20_25, pctb_bb_20_25 |
| `stddev_20` | volatility | processed | stddev_atr |
| `envelope_20_01_upper` | volatility | processed | env_pctb_envelope_20_01, env_width_envelope_20_01 |
| `envelope_20_01_lower` | volatility | processed | env_pctb_envelope_20_01, env_width_envelope_20_01 |
| `obv` | volume | processed | obv_chg6_z |
| `ad` | volume | processed | ad_chg6_z |
| `mfi_14` | volume | processed | mfi_14 |
| `force_13` | volume | processed | force_13_z |
| `chaikin_3_10` | volume | processed | chaikin_z |
| `volumes` | volume | processed | volumes_rel |
| `volregime_atr_pctl` | custom_volregime | processed | volregime_atr_pctl |
| `volregime_atr_ratio` | custom_volregime | processed | volregime_atr_ratio |
| `volregime_regime` | custom_volregime | processed | volregime_regime |
| `ps_dist_pdh_atr` | custom_pricestructure | processed | ps_dist_pdh_atr |
| `ps_dist_pdl_atr` | custom_pricestructure | processed | ps_dist_pdl_atr |
| `ps_dist_sess_h_atr` | custom_pricestructure | processed | ps_dist_sess_h_atr |
| `ps_dist_sess_l_atr` | custom_pricestructure | processed | ps_dist_sess_l_atr |
| `ps_dist_round_up_atr` | custom_pricestructure | processed | ps_dist_round_up_atr |
| `ps_dist_round_dn_atr` | custom_pricestructure | processed | ps_dist_round_dn_atr |
| `ps_gap_atr` | custom_pricestructure | processed | ps_gap_atr |
| `ps_gap_fill_pct` | custom_pricestructure | processed | ps_gap_fill_pct |
| `news_min_to_next` | news | processed | news_min_to_next |
| `news_next_importance` | news | processed | news_next_importance |
| `news_min_since_last` | news | processed | news_min_since_last |
| `news_last_importance` | news | processed | news_last_importance |
| `news_last_surprise` | news | processed | news_last_surprise |
| `news_in_window` | news | processed | news_in_window |
| `news_count_next_24h` | news | processed | news_count_next_24h |
