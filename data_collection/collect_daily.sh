#!/usr/bin/env bash
# Daily collector. Run crawler/scripts/run_daily.sh first (it fills crawler/output/store),
# then this script, e.g. crontab:
#   45 9 * * * cd /path/to/PriLLM && PUSH=0 data_collection/collect_daily.sh >> data/collect.log 2>&1
# Nothing is pushed unless PUSH=1 (reviewed and enabled by you).
set -euo pipefail
cd "$(dirname "$0")/.."
DB="${HARVESTER_DB:-crawler/output/store/openrouter.sqlite}"
BENCH="${BENCH_DIR:-$HOME/.cache/llm-bench-data}"

mkdir -p data/harvester data/perf data/market data/elo

# 1. per-endpoint TPS / latency / price from OpenRouter (rolling 7 days -> must run at least weekly)
python3 -m data_collection.fetch_openrouter_perf --top "${TOP_MODELS:-80}" --out data/perf/openrouter_perf_daily.csv

# 2. usage + app-level demand from the harvester store (read-only), latest run wins per key
for t in app_model_daily model_usage_weekly models; do
  python3 data_collection/export_harvester.py --db "$DB" --table "$t" | gzip -9 > "data/harvester/$t.csv.gz"
done

# 3. public hourly probe history (TTFT / TPS), refreshed shallowly
if [ -d "$BENCH/.git" ]; then git -C "$BENCH" pull -q --ff-only; else git clone -q --depth 1 https://github.com/Jwrede/llm-bench-data "$BENCH"; fi

# 4. join into market inputs
python3 -m data_collection.build_market_inputs --bench-dir "$BENCH" --demand data/harvester/app_model_daily.csv.gz

# 5. LMArena Elo (needs pyarrow: ELO_PY=path/to/python with pyarrow)
if [ -n "${ELO_PY:-}" ]; then "$ELO_PY" -m data_collection.fetch_lmarena_elo --out data/elo/lmarena_text_overall.csv; fi
[ -f data/elo/lmarena_text_overall.csv ] && python3 -m data_collection.add_elo

if [ "${PUSH:-0}" = "1" ]; then
  git add data/perf data/harvester data/market
  git diff --cached --quiet || { git commit -m "data: daily update $(date -u +%F)"; git push; }
fi
