# Crawler

Collects the public OpenRouter ranking and app-usage pages into a local SQLite store, so you can build the market
table yourself. Standard library only; needs `curl`.

```bash
cd crawler
scripts/run_daily.sh
```
Output: `crawler/output/store/openrouter.sqlite` (append-only history), `output/runs/<run>/` (raw responses,
normalised tables, an Excel workbook). Run it daily; OpenRouter only exposes recent windows.

Then, from the repository root, `data_collection/collect_daily.sh` exports the store, fetches endpoint speed and
latency, joins llm-bench-data probes and LMArena scores, and `python3 -m data_collection.make_release_top20`
cuts the released table. The pages are OpenRouter's own, undocumented, and may change; check OpenRouter's terms of
use before collecting or redistributing.
