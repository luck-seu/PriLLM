"""Re-derive `app_model_history` for old harvester runs from their saved raw App pages and
append the rows to the store, exactly as a normal run (with the 0001 patch) would have.

Only touches table `app_model_history`, only for runs that currently have 0 rows in it, and
uses the harvester's own SQLiteStore writer. Dry run unless --apply.
    PYTHONPATH=<harvester>/src python3 import_app_history_to_store.py --root <harvester> --since 2026-10-08 [--apply]
"""
import argparse, json, sqlite3
from pathlib import Path

from openrouter_harvester.flight import parse_react_flight
from openrouter_harvester.normalize import normalize_bundle
from openrouter_harvester.storage import SQLiteStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--since", default="2026-10-08")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    db = root / "output" / "store" / "openrouter.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    have = {r[0]: r[1] for r in con.execute("select _run_id, count(*) from app_model_history group by _run_id")}
    known = {r[0] for r in con.execute("select run_id from harvest_runs")}
    con.close()
    store = SQLiteStore(db)
    total = done = 0
    for run in sorted(p for p in (root / "output" / "runs").iterdir() if p.is_dir() and p.name[:10] >= a.since):
        mf = run / "raw_manifest.json"
        if run.name not in known or not mf.exists():
            print(f"skip {run.name}: not a stored run / no manifest"); continue
        if have.get(run.name):
            print(f"skip {run.name}: already has {have[run.name]} rows"); continue
        raw = json.loads(mf.read_text(encoding="utf8"))
        for d in raw.get("app_details") or []:
            sel = d.get("selected") or {}
            html = run / "raw" / f"app_{sel.get('app_id') or sel.get('app_slug') or 'unknown'}.html"
            if not d.get("parsed", {}).get("chartData") and html.exists():
                d["parsed"] = parse_react_flight(html.read_text(encoding="utf8", errors="replace"))
        bundle = {**raw["core"], "app_details": raw.get("app_details") or [],
                  "selected_app_count": raw.get("selected_app_count")}
        rows = normalize_bundle(bundle, raw["captured_at"])["app_model_history"]
        total += len(rows)
        print(f"{run.name}: {len(rows)} rows" + ("" if a.apply else "  (dry run)"))
        if a.apply and rows:
            with __import__("sqlite3").connect(db) as c:
                store._append_table(c, "app_model_history", run.name, rows)
                c.commit()
            done += 1
    print(f"{'imported' if a.apply else 'would import'} {total} rows from {done if a.apply else 'N'} runs")


if __name__ == "__main__":
    main()
