 tsunugi on  /run/media/tsunugi/Projects/new-edge-search/AlphaForge-main 
# python scripts/run_mql5_pipeline.py run \
  --csv data/EURUSD/EURUSD_H1.csv \
  --out output/eurusd_H1 \
  --through 13
Found NaN values:
envelope_20_01_lower    161423
ps_dist_sess_h_atr       10808
ps_dist_sess_l_atr       10808
news_min_to_next        161423
news_next_importance    161423
news_min_since_last     161423
news_last_importance    161423
news_last_surprise      161423
news_in_window          161423
news_count_next_24h     161423
dtype: int64
news_*: every value is NaN. The EA writes NaN when it could not load the calendar CSV in the tester. Run dataExporter.mq5 once on a LIVE chart to create economic_calendar.csv (Common\Files), then re-run the tester export. These all-NaN columns are excluded from feature generation; the pipeline continues with a warning.
Found NaN values:
envelope_20_01_lower    161423
ps_dist_sess_h_atr       10808
ps_dist_sess_l_atr       10808
news_min_to_next        161423
news_next_importance    161423
news_min_since_last     161423
news_last_importance    161423
news_last_surprise      161423
news_in_window          161423
news_count_next_24h     161423
dtype: int64
news_*: every value is NaN. The EA writes NaN when it could not load the calendar CSV in the tester. Run dataExporter.mq5 once on a LIVE chart to create economic_calendar.csv (Common\Files), then re-run the tester export. These all-NaN columns are excluded from feature generation; the pipeline continues with a warning.
Traceback (most recent call last):
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/scripts/run_mql5_pipeline.py", line 75, in <module>
    sys.exit(main())
             ~~~~^^
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/scripts/run_mql5_pipeline.py", line 67, in main
    print(json.dumps(run_pipeline(args.csv, args.out, args.through, max_replicates=args.max_replicates, **kw), indent=2, default=str))
                     ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/mql5_loader/runner.py", line 113, in run_pipeline
    result = fn(csv_path, out, cfg, sym, None, **kwargs)                              # spread_csv=None: the spread comes from the data table
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/pipeline.py", line 751, in run_stages_0_13
    out = run_stages_0_12(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/pipeline.py", line 741, in run_stages_0_12
    out = run_stages_0_11(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/pipeline.py", line 675, in run_stages_0_11
    out = run_stages_0_10(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/pipeline.py", line 633, in run_stages_0_10
    s10 = run_stage_10(run_dir, cfg, spread_csv)
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/pipeline.py", line 568, in run_stage_10
    rules = freeze_rules(chosen, aset, cfg, max_rules=cfg["validation"]["max_rules"], null_flags=_read_null_flags(run))
                                                                                                 ~~~~~~~~~~~~~~~~^^^^^
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/edge_research/pipeline.py", line 521, in _read_null_flags
    return pd.read_csv(p) if p.exists() else None
           ~~~~~~~~~~~^^^
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 1866, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 423, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 2721, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 3006, in _make_engine
    return mapping[engine](f, **self.options)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/run/media/tsunugi/Projects/new-edge-search/AlphaForge-main/.venv/lib/python3.14/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 107, in __init__
    self._reader = parsers.TextReader(src, **kwds)
                   ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "pandas/_libs/parsers.pyx", line 735, in pandas._libs.parsers.TextReader.__cinit__
pandas.errors.EmptyDataError: No columns to parse from file
 tsunugi on  /run/media/tsunugi/Projects/new-edge-search/AlphaForge-main 
# 