# market_inputs_top20_month.csv

App-level LLM demand with TPS, delay, price and Arena score, for the 20 models with the most tokens that have measured performance data.

- Rows: 8,868 (one row per date, app and model; dated versions of a model are summed); apps: 98 (anonymised as app_001, app_002, ... by token volume); dates: 2026-10-19 to 2026-11-15 (28 days).
- tokens: daily tokens from OpenRouter app pages. tps / ttft_ms: tokens per second and time to first token (ms), from OpenRouter endpoint statistics and the llm-bench-data probes.
- Probe values above 1,000 tokens/s were discarded; those model-days use the nearest OpenRouter snapshot value instead. Maximum TPS in this file is 953.
- price_*_per_mtok: OpenRouter listing price at collection time.
- elo_score: LMArena text/overall Arena Score from the latest snapshot on or before the date; for dates before the model's first snapshot the first available score is used (`elo_snapshot_date` gives the snapshot date); blank when no Arena entry matches the model.
- Sources: OpenRouter (unofficial frontend API), github.com/Jwrede/llm-bench-data, huggingface.co/datasets/lmarena-ai/leaderboard-dataset.
- Rebuild or extend with `data_collection/` and `crawler/`; the window is chosen by `python3 -m data_collection.make_release_top20`.
