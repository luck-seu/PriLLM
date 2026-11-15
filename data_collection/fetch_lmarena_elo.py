"""Download LMArena ("Arena") text-leaderboard ratings (Elo-style "Arena Score") from the official
Hugging Face dataset  lmarena-ai/leaderboard-dataset  (config `text`, category `overall`).

`full` = every published snapshot (`leaderboard_publish_date`), `latest` = newest only.
Needs pyarrow. Writes one row per model per snapshot.
Source / attribution: LMArena, https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset

    python3 -m data_collection.fetch_lmarena_elo --out data/elo/lmarena_text_overall.csv
"""
import argparse, csv, os, tempfile, urllib.request

import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/{config}/{split}-00000-of-00001.parquet"
FIELDS = ["leaderboard_publish_date", "model_name", "organization", "license", "rating",
          "rating_lower", "rating_upper", "vote_count", "rank", "category"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="text")
    ap.add_argument("--split", default="full", choices=["full", "latest"])
    ap.add_argument("--category", default="overall")
    ap.add_argument("--out", default="data/elo/lmarena_text_overall.csv")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lb.parquet")
        req = urllib.request.Request(URL.format(config=a.config, split=a.split),
                                     headers={"User-Agent": "prillm-data"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
        table = pq.read_table(path, columns=[c for c in FIELDS if c in pq.read_schema(path).names])
    rows = [r for r in table.to_pylist() if r.get("category") == a.category]
    rows.sort(key=lambda r: (str(r["leaderboard_publish_date"]), r.get("rank") or 1e9))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    dates = sorted({str(r["leaderboard_publish_date"]) for r in rows})
    print(f"{len(rows)} rows, {len(dates)} snapshots {dates[0]} -> {dates[-1]} -> {a.out}")


if __name__ == "__main__":
    main()
