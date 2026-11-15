"""Join app-level demand with per-model TPS / delay (TTFT) into market inputs.

Inputs
  data/harvester/app_model_daily*.csv.gz    demand: tokens per (date, app, model)   [harvester export + backfill]
  <bench_dir>/data/*/*.jsonl                 hourly probes (github.com/Jwrede/llm-bench-data)
  data/perf/openrouter_perf_daily.csv        OpenRouter per-endpoint 7-day snapshots + prices
Outputs (data/market/)
  model_perf_daily.csv   one TPS / TTFT per (date, model) with a `perf_source` column
  market_inputs.csv      demand rows with the model's TPS / TTFT attached

Priority for each (date, model):  openrouter_snapshot  >  bench  >  openrouter_snapshot_static
                                  >  author_median  >  global_median
`author_median` / `global_median` are *estimates* borrowed from other models; they are kept
apart from measurements by `perf_source` and must be described as estimates when published.
"""
import argparse, csv, glob, gzip, json, os, re, statistics as st
from collections import defaultdict

norm = lambda s: re.sub(r"-\d{8}$", "", s)
author = lambda s: s.split("/")[0]


def bench_daily(bench_dir, tps_cap=1000.0):
    per = defaultdict(lambda: ([], []))
    skipped = 0
    for f in sorted(glob.glob(os.path.join(bench_dir, "data", "*", "*.jsonl"))):
        day = os.path.basename(f)[:10]
        for line in open(f, encoding="utf8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            # quality filter: probes with TTFT 0 or very short outputs are not trustworthy
            if (r.get("status") != "healthy" or not r.get("ttft_ms") or r["ttft_ms"] <= 0
                    or (r.get("token_count") or 0) < 10 or not r.get("tokens_per_sec")):
                continue
            tps, ttft = per[(day, r["model"])]
            tps.append(r["tokens_per_sec"]); ttft.append(r["ttft_ms"])
    print(f"bench: unparsable lines skipped={skipped}")
    out = {k: (st.median(v[0]), st.median(v[1])) for k, v in per.items()}
    # probes above the cap are not serving speeds (OpenRouter endpoints peak at ~1,000 tok/s);
    # drop those model-days so the row falls back to the next source instead of using a clamped value
    kept = {k: v for k, v in out.items() if v[0] <= tps_cap}
    print(f"bench: dropped {len(out) - len(kept)} model-days with TPS > {tps_cap:g}")
    return kept


def snapshot(path):
    by = defaultdict(lambda: defaultdict(lambda: ([], [], [], [])))
    if not os.path.exists(path):
        return {}, {}, {}
    for r in csv.DictReader(open(path, encoding="utf8")):
        m = norm(r["model_permaslug"])
        t, l, pp, pc = by[m][r["date"]]
        for lst, k in ((t, "throughput_tps"), (l, "latency_ms"),
                       (pp, "price_prompt_per_mtok"), (pc, "price_completion_per_mtok")):
            if r[k] != "":
                lst.append(float(r[k]))
    daily, static, price = {}, {}, {}
    for m, days in by.items():
        allt, alll, allp, allc = [], [], [], []
        for d, (t, l, pp, pc) in days.items():
            if t and l:
                daily[(d, m)] = (st.median(t), st.median(l))
            allt += t; alll += l; allp += pp; allc += pc
        if allt and alll:
            static[m] = (st.median(allt), st.median(alll))
        price[m] = (st.median(allp) if allp else "", st.median(allc) if allc else "")
    return daily, static, price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, help="clone of Jwrede/llm-bench-data")
    ap.add_argument("--demand", nargs="+",
                    default=["data/harvester/app_model_daily.csv.gz",
                             "data/harvester/app_model_daily_backfill.csv.gz"],
                    help="demand tables; later files override earlier ones per (date, app, model)")
    ap.add_argument("--snapshots", default="data/perf/openrouter_perf_daily.csv")
    ap.add_argument("--tps-cap", type=float, default=1000.0,
                    help="bench model-days with median TPS above this are discarded (default 1000)")
    ap.add_argument("--out", default="data/market")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    merged = {}
    for path in a.demand:
        if os.path.exists(path):
            for r in csv.DictReader(gzip.open(path, "rt", encoding="utf8")):
                merged[(r["date"], r["app_id"], r["model_id"])] = r
    # dated versions of a model (x-20260423, x-20260731, "(batch)") share one normalised id:
    # sum their tokens so there is one row per (date, app, model)
    agg = {}
    for k in sorted(merged):
        r = dict(merged[k]); key = (r["date"], r["app_id"], norm(r["model_id"]))
        if key in agg:
            agg[key]["tokens"] = float(agg[key]["tokens"] or 0) + float(r["tokens"] or 0)
        else:
            r["tokens"] = float(r["tokens"] or 0); agg[key] = r
    demand = [r for r in agg.values() if r["tokens"] > 0]   # drop zero-token rows
    bench = bench_daily(a.bench_dir, a.tps_cap)
    snap_daily, snap_static, price = snapshot(a.snapshots)

    dates = sorted({r["date"] for r in demand})
    models = sorted({norm(r["model_id"]) for r in demand})
    measured = defaultdict(dict)  # date -> model -> (tps, ttft, source)
    for d in dates:
        for m in models:
            if (d, m) in snap_daily:
                measured[d][m] = (*snap_daily[(d, m)], "openrouter_snapshot")
            elif (d, m) in bench:
                measured[d][m] = (*bench[(d, m)], "bench")
            elif m in snap_static:
                measured[d][m] = (*snap_static[m], "openrouter_snapshot_static")

    perf = {}
    for d in dates:
        pool = measured[d]
        by_author = defaultdict(list)
        for m, (t, l, _) in pool.items():
            by_author[author(m)].append((t, l))
        g = list(pool.values())
        gmed = (st.median(x[0] for x in g), st.median(x[1] for x in g)) if g else None
        for m in models:
            if m in pool:
                perf[(d, m)] = pool[m]
            elif by_author.get(author(m)):
                v = by_author[author(m)]
                perf[(d, m)] = (st.median(x[0] for x in v), st.median(x[1] for x in v), "author_median")
            elif gmed:
                perf[(d, m)] = (*gmed, "global_median")

    with open(os.path.join(a.out, "model_perf_daily.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["date", "model", "tps", "ttft_ms", "price_prompt_per_mtok",
                    "price_completion_per_mtok", "perf_source"])
        for (d, m), (t, l, s) in sorted(perf.items()):
            p = price.get(m, ("", ""))
            w.writerow([d, m, round(t, 2), round(l, 1), p[0], p[1], s])
    n = 0
    src = defaultdict(float); tot = 0.0
    with open(os.path.join(a.out, "market_inputs.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["date", "app_id", "app_name", "model", "tokens", "tps", "ttft_ms",
                    "price_prompt_per_mtok", "price_completion_per_mtok", "perf_source"])
        for r in demand:
            m = norm(r["model_id"]); p = perf.get((r["date"], m))
            if not p:
                continue
            pr = price.get(m, ("", ""))
            w.writerow([r["date"], r["app_id"], r["app_name"], m, int(r["tokens"]),
                        round(p[0], 2), round(p[1], 1), pr[0], pr[1], p[2]])
            n += 1; tok = float(r["tokens"] or 0); src[p[2]] += tok; tot += tok
    print(f"market_inputs rows={n}")
    for s, v in sorted(src.items(), key=lambda x: -x[1]):
        print(f"  {s:28s} {100*v/tot:5.1f}% of tokens")


if __name__ == "__main__":
    main()
