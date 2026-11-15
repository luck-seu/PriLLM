"""Write the released market table: top models by tokens, one ~month window, apps anonymised.

    python3 -m data_collection.make_release_top20 --top 20 --days 28

Reads data/market/market_inputs.csv (build_market_inputs, then add_elo), keeps rows whose TPS/delay comes from a
measurement (OpenRouter snapshot or llm-bench-data probes), keeps the --top models by tokens, and cuts a window of
--days consecutive days. With no --start the window is the one in which all models appear and the largest share of
tokens carries an Arena score taken from a snapshot on or before the date. Apps are anonymised: app_name is dropped
and app_id becomes app_001, app_002, ... in order of tokens inside the window. perf_source is dropped.
"""
import argparse, csv
import datetime as dt
from collections import Counter, defaultdict

MEASURED = ("openrouter_snapshot", "bench", "openrouter_snapshot_static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="data/market/market_inputs.csv")
    ap.add_argument("--out", default="data/release/market_inputs_top20_month.csv")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--start", default="", help="first day of the window (YYYY-MM-DD); default: best-quality window")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.market, encoding="utf8")))
    tok = Counter()
    for r in rows:
        if r["perf_source"] in MEASURED:
            tok[r["model"]] += float(r["tokens"])
    top = {m for m, _ in tok.most_common(a.top)}
    sub = [r for r in rows if r["model"] in top and r["perf_source"] in MEASURED and r["tps"] and r["ttft_ms"]]

    by_date = defaultdict(list)
    for r in sub:
        by_date[r["date"]].append(r)
    dates = sorted(by_date)
    D = dt.date.fromisoformat
    def window(d0):
        end = D(d0) + dt.timedelta(days=a.days - 1)
        return [d for d in dates if D(d0) <= D(d) <= end]
    if a.start:
        start = a.start
    else:
        best = None
        for d0 in dates:
            w = window(d0)
            if len(w) < a.days:
                continue
            wr = [r for d in w for r in by_date[d]]
            t = sum(float(r["tokens"]) for r in wr)
            good = sum(float(r["tokens"]) for r in wr if r["elo_score"] and r["elo_snapshot_date"] <= r["date"])
            score = (len({r["model"] for r in wr}), good / t)
            if best is None or score > best[0]:
                best = (score, d0)
        start = best[1]
    keep = set(window(start))
    sub = [r for d in sorted(keep) for r in by_date[d]]

    app_tok = Counter()
    for r in sub:
        app_tok[r["app_id"]] += float(r["tokens"])
    alias = {k: f"app_{i:03d}" for i, (k, _) in enumerate(app_tok.most_common(), 1)}
    for r in sub:
        r["app_id"] = alias[r["app_id"]]
    sub.sort(key=lambda r: (r["date"], r["app_id"], r["model"]))
    cols = [c for c in rows[0] if c not in ("perf_source", "app_name")]
    with open(a.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(sub)
    print(f"{len(sub)} rows, {len(top)} models, {len(alias)} apps, {sorted(keep)[0]}..{sorted(keep)[-1]} -> {a.out}")


if __name__ == "__main__":
    main()
