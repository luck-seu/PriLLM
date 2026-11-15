"""Recover app-level usage history from the harvester's saved raw App pages.

Needs the harvester package WITH the 0001 patch (new `data.daily` chart format):
    PYTHONPATH=<patched>/src python3 backfill_app_history.py --root ~/project/openrouter_harvester \
        --since 2026-10-08 | gzip > app_model_daily_backfill.csv.gz
It never writes to the harvester store or output directory; it only prints CSV in the same
columns as `export_harvester.py --table app_model_daily` (latest run wins per date/app/model).
"""
import argparse, csv, json, sys
from pathlib import Path

from openrouter_harvester.flight import parse_react_flight
from openrouter_harvester.normalize import normalize_bundle

COLS = ["date", "app_id", "app_name", "model_id", "model_name", "author", "tokens", "share", "run_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--since", default="2026-10-08")
    ap.add_argument("--until", default="9999")
    a = ap.parse_args()
    out = {}
    runs = sorted(p for p in (Path(a.root) / "output" / "runs").iterdir()
                  if p.is_dir() and a.since <= p.name[:10] <= a.until)
    for run in runs:
        mf = run / "raw_manifest.json"
        if not mf.exists():
            print(f"skip {run.name}: no manifest", file=sys.stderr); continue
        raw = json.loads(mf.read_text(encoding="utf8"))
        reparsed = 0
        for d in raw.get("app_details") or []:
            sel = d.get("selected") or {}
            html = run / "raw" / f"app_{sel.get('app_id') or sel.get('app_slug') or 'unknown'}.html"
            if d.get("parsed", {}).get("chartData") or not html.exists():
                continue
            d["parsed"] = parse_react_flight(html.read_text(encoding="utf8", errors="replace"))
            reparsed += 1
        bundle = {**raw["core"], "app_details": raw.get("app_details") or [],
                  "selected_app_count": raw.get("selected_app_count")}
        rows = normalize_bundle(bundle, raw["captured_at"])["app_model_history"]
        for r in rows:
            out[(str(r["date"])[:10], str(r["app_id"]), r["model_id"])] = [
                str(r["date"])[:10], r["app_id"], r["app_name"], r["model_id"], r["model_name"],
                r["author"], r["token_usage"], r["share"], run.name]
        print(f"{run.name}: reparsed {reparsed} pages, {len(rows)} rows", file=sys.stderr)
    w = csv.writer(sys.stdout)
    w.writerow(COLS)
    for k in sorted(out):
        w.writerow(out[k])


if __name__ == "__main__":
    main()
