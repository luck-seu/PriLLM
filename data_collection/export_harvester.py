"""Export tidy tables from the openrouter_harvester SQLite store (read-only).

Run on the machine that holds the store, e.g.
    python3 export_harvester.py --db ~/project/openrouter_harvester/output/store/openrouter.sqlite \
        --table app_model_daily | gzip > app_model_daily.csv.gz
Tables: app_model_daily (date, app, model, tokens), model_usage_weekly, models.
Only stdlib; no raw HTML/JSON snapshots are touched.
"""
import argparse, csv, sqlite3, sys

QUERIES = {
    "app_model_daily": """
        select substr(date,1,10) as date, app_id, app_name, model_id, model_name, author,
               token_usage as tokens, share, max(_run_id) as run_id
        from app_model_history group by substr(date,1,10), app_id, model_id
        order by 1, 2, 4""",  # store is append-only: keep the latest run's value per key
    "model_usage_weekly": """
        select date, model_id, model_name, author, token_usage as tokens, share, rank,
               max(_run_id) as run_id
        from model_history group by date, model_id order by 1, 7""",
    "models": """
        select model_id, max(model_name) model_name, max(author) author,
               max(context_length) context_length, max(supports_reasoning) supports_reasoning,
               max(created_at) created_at
        from models group by model_id""",
}

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--table", required=True, choices=sorted(QUERIES))
a = ap.parse_args()
con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
cur = con.execute(QUERIES[a.table])
w = csv.writer(sys.stdout)
w.writerow([d[0] for d in cur.description])
for row in cur:
    w.writerow(row)
