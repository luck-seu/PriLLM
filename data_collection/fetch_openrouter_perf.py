"""Snapshot OpenRouter's public per-endpoint throughput (tokens/s) and latency (ms) series.

The model page's "Performance" charts are backed by
  /api/frontend/v1/stats/throughput-comparison   and   .../latency-comparison
which return roughly the last 7 days, one value per day per provider endpoint.
OpenRouter keeps no longer history, so run this at least weekly (daily is better) and
let the output file accumulate. New runs overwrite overlapping (date, endpoint) rows.

Unofficial endpoints (used by the website): they may change without notice.

    python -m data_collection.fetch_openrouter_perf --top 60 --out data/perf/openrouter_perf_daily.csv
"""
import argparse, csv, json, os, time, urllib.parse, urllib.request

BASE = "https://openrouter.ai/api/frontend/v1"
FIELDS = ["date", "model_permaslug", "variant", "endpoint_id", "provider",
          "throughput_tps", "latency_ms", "price_prompt_per_mtok", "price_completion_per_mtok",
          "fetched_at"]


def get(path, **params):
    url = f"{BASE}/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "prillm-data-collector"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def top_models(n):
    rows = get("rankings/models", view="day")
    latest = max(r["date"] for r in rows)
    rows = [r for r in rows if r["date"] == latest]
    rows.sort(key=lambda r: r.get("rankingMetricValue") or 0, reverse=True)
    seen, out = set(), []
    for r in rows:
        key = (r["model_permaslug"], r.get("variant") or "standard")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:n]


def fetch_model(permaslug, variant, fetched_at):
    eps = get("stats/endpoint", permaslug=permaslug, variant=variant)
    endpoints = {e["id"]: e.get("provider_display_name") or e.get("provider_name") for e in eps}
    prices = {e["id"]: e.get("pricing") or {} for e in eps}

    def per_mtok(eid, k):  # OpenRouter quotes USD per token
        try:
            return round(float(prices[eid][k]) * 1e6, 6)
        except (KeyError, TypeError, ValueError):
            return ""
    tp = get("stats/throughput-comparison", permaslug=permaslug, variant=variant)
    lat = get("stats/latency-comparison", permaslug=permaslug, variant=variant)
    cells = {}
    for kind, series in (("throughput_tps", tp), ("latency_ms", lat)):
        for pt in series:
            for eid, v in (pt.get("y") or {}).items():
                cells.setdefault((pt["x"], eid), {})[kind] = v
    return [dict(date=d, model_permaslug=permaslug, variant=variant, endpoint_id=eid,
                 provider=endpoints.get(eid, ""), throughput_tps=v.get("throughput_tps", ""),
                 latency_ms=v.get("latency_ms", ""),
                 price_prompt_per_mtok=per_mtok(eid, "prompt"),
                 price_completion_per_mtok=per_mtok(eid, "completion"), fetched_at=fetched_at)
            for (d, eid), v in cells.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60, help="top-N models by yesterday's usage")
    ap.add_argument("--out", default="data/perf/openrouter_perf_daily.csv")
    a = ap.parse_args()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store = {}
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out, encoding="utf8")):
            store[(r["date"], r["model_permaslug"], r["variant"], r["endpoint_id"])] = r
    ok = fail = 0
    for slug, variant in top_models(a.top):
        try:
            for r in fetch_model(slug, variant, now):
                store[(r["date"], slug, variant, r["endpoint_id"])] = r
            ok += 1
        except Exception as e:  # keep going; report at the end
            fail += 1
            print(f"skip {slug}: {e}")
        time.sleep(0.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(store):
            w.writerow(store[k])
    print(f"models ok={ok} failed={fail}; rows={len(store)} -> {a.out}")


if __name__ == "__main__":
    main()
