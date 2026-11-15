# Data collection

| Script | Source | Gives | Cadence |
|---|---|---|---|
| `fetch_openrouter_perf.py` | OpenRouter `stats/*-comparison`, `stats/endpoint` | TPS, latency (TTFT), price per provider endpoint | OpenRouter keeps ~7 days, so run daily |
| `export_harvester.py` | `crawler/` SQLite store (`crawler/output/store/openrouter.sqlite`) | model usage, **app-level** usage per model | daily (harvester runs 09:15 Asia/Shanghai) |
| `build_market_inputs.py` | the two above + [llm-bench-data](https://github.com/Jwrede/llm-bench-data) | market inputs: demand + TPS + TTFT per (date, app, model) | after the others |
| `collect_daily.sh` | runs all four | | cron on the harvester machine |

`perf_source` in the output says where each TPS / TTFT came from:
`openrouter_snapshot`, `bench` (quality-filtered probes: TTFT>0, >=10 output tokens, daily median TPS <= `--tps-cap`, default 1000),
`openrouter_snapshot_static` (the model's snapshot median applied to earlier dates),
`author_median` and `global_median` (**estimates** borrowed from other models).
Report estimated rows as estimates.

Caveats: the OpenRouter endpoints are the website's own, undocumented, and may change;
model ids are matched after
removing a trailing `-YYYYMMDD`, so renamed slugs (`claude-4.5-sonnet` vs `claude-sonnet-4.5`) do not match;
historic prices are only known for models seen in a snapshot.
