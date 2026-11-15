from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def _identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _sql_type(values: list[Any]) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "TEXT"
    if all(isinstance(value, bool) for value in present):
        return "INTEGER"
    if all(isinstance(value, (bool, int)) for value in present):
        return "INTEGER"
    if all(isinstance(value, (bool, int, float)) for value in present):
        return "REAL"
    return "TEXT"


def _sql_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def append_run(
        self,
        run_id: str,
        captured_at: str,
        status: str,
        counts: dict[str, int],
        issues: list[dict[str, Any]],
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS harvest_runs (
                    run_id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    counts_json TEXT NOT NULL,
                    issues_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO harvest_runs
                (run_id, captured_at, status, counts_json, issues_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    captured_at,
                    status,
                    json.dumps(counts, sort_keys=True),
                    json.dumps(issues, ensure_ascii=False),
                ),
            )
            for name, rows in tables.items():
                self._append_table(connection, name, run_id, rows)
            connection.commit()

    def _append_table(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        run_id: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        table = _identifier(table_name)
        columns = sorted({key for row in rows for key in row})
        column_types = {
            column: _sql_type([row.get(column) for row in rows])
            for column in columns
        }
        definitions = ", ".join(
            [f'"{_identifier(column)}" {column_types[column]}' for column in columns]
        )
        connection.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            f'("_run_id" TEXT NOT NULL{", " if definitions else ""}{definitions})'
        )
        existing = {
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        for column in columns:
            sql_column = _identifier(column)
            if sql_column not in existing:
                connection.execute(
                    f'ALTER TABLE "{table}" ADD COLUMN "{sql_column}" '
                    f'{column_types[column]}'
                )
        connection.execute(f'DELETE FROM "{table}" WHERE "_run_id" = ?', (run_id,))
        sql_columns = ["_run_id", *[_identifier(column) for column in columns]]
        placeholders = ",".join("?" for _ in sql_columns)
        quoted = ",".join(f'"{column}"' for column in sql_columns)
        values = [
            [run_id, *[_sql_value(row.get(column)) for column in columns]]
            for row in rows
        ]
        connection.executemany(
            f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
            values,
        )

