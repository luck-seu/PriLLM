"""Attach LMArena text "Arena Score" (Elo-style rating) to the market tables.

Matching: an OpenRouter model id (author/name, dated suffix removed) matches an Arena model name when
their normalised base names are equal (also tried: Claude word order, and an Arena parameter-count suffix
such as "-120B-A12B" when it is the only candidate). Preview vs final releases are never merged. The base name drops parenthesised variants such as "(High)",
"(Max)", "(NVFP4)" and a dated suffix "-20260910". If several Arena variants match, the one with the
most votes in that snapshot is used. For each date the latest Arena snapshot published on or before it
is used; before a model's first snapshot its first available score is used (elo_snapshot_date shows which snapshot
the value comes from; --no-backfill leaves those dates empty). Models with no Arena match get an empty score.

    python3 -m data_collection.add_elo
Outputs: data/market/model_elo_map.csv (audit of every match) and elo_* columns in data/market/market_inputs.csv
"""
import argparse, bisect, csv, re
from collections import defaultdict


def base(name):
    s = re.sub(r"\([^)]*\)", "", name.lower())
    s = re.sub(r"-?20\d{6}", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="data/elo/lmarena_text_overall.csv")
    ap.add_argument("--market", default="data/market/market_inputs.csv")
    ap.add_argument("--no-backfill", action="store_true",
                    help="leave the score empty before a model's first Arena snapshot (default: use that first snapshot)")
    ap.add_argument("--map-out", default="data/market/model_elo_map.csv")
    a = ap.parse_args()

    by_base = defaultdict(lambda: defaultdict(list))  # base -> snapshot -> [(votes, name, rating)]
    for r in csv.DictReader(open(a.elo, encoding="utf8")):
        if r["rating"]:
            by_base[base(r["model_name"])][r["leaderboard_publish_date"]].append(
                (float(r["vote_count"] or 0), r["model_name"], float(r["rating"])))

    rows = list(csv.DictReader(open(a.market, encoding="utf8")))
    models = sorted({r["model"] for r in rows})

    def keys_of(m):
        author, short = m.split("/", 1)
        keys = [base(short), base(author + short)]
        # OpenRouter writes claude-4.6-sonnet, Arena writes Claude Sonnet 4.6
        g = re.match(r"claude-(\d+(?:\.\d+)?)-(opus|sonnet|haiku)(.*)$", short)
        if g:
            keys.append(base(f"claude-{g.group(2)}-{g.group(1)}{g.group(3)}"))
        return keys

    def snaps_of(m):
        for k in keys_of(m):
            if k in by_base:
                return by_base[k]
        # Arena appends the parameter count (e.g. ...-vl-120b-a12b); accept it only if that is the single candidate
        for k in keys_of(m):
            hits = [b for b in by_base if re.fullmatch(re.escape(k) + r"\d+b(?:a\d+b)?", b)]
            if len(hits) == 1:
                return by_base[hits[0]]
        return None

    series = {}  # model -> (sorted snapshot dates, values)
    for m in models:
        snaps = snaps_of(m)
        if snaps:
            ds = sorted(snaps)
            series[m] = (ds, [max(snaps[d]) for d in ds])  # most-voted variant per snapshot

    def lookup(m, date):
        if m not in series:
            return None
        ds, vals = series[m]
        i = bisect.bisect_right(ds, date) - 1
        if i < 0:
            return None if a.no_backfill else (ds[0], vals[0])  # before the first snapshot: first available score
        return (ds[i], vals[i])

    with open(a.map_out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["model", "matched", "arena_names", "first_snapshot", "last_snapshot", "last_rating"])
        for m in models:
            if m in series:
                ds, vals = series[m]
                names = sorted({v[1] for d in ds for v in snaps_of(m)[d]})
                w.writerow([m, 1, " | ".join(names), ds[0], ds[-1], round(vals[-1][2], 1)])
            else:
                w.writerow([m, 0, "", "", "", ""])

    cols = ["elo_score", "elo_snapshot_date", "elo_arena_name"]
    keep = [c for c in rows[0] if c not in cols]
    tot = hit = 0.0
    with open(a.market, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=keep + cols)
        w.writeheader()
        for r in rows:
            hv = lookup(r["model"], r["date"])
            if hv:
                v = hv[1]
                r.update(elo_score=round(v[2], 1), elo_snapshot_date=hv[0], elo_arena_name=v[1])
            else:
                r.update(elo_score="", elo_snapshot_date="", elo_arena_name="")
            t = float(r["tokens"]); tot += t; hit += t if r["elo_score"] != "" else 0
            w.writerow(r)
    print(f"models matched {len(series)}/{len(models)}; rows with ELO cover {100*hit/tot:.1f}% of tokens")


if __name__ == "__main__":
    main()
