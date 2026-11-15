from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .contracts import (
    ContractIssue,
    failure_signature,
    status_from_issues,
    validate_raw,
    warning_signature,
)
from .flight import parse_react_flight
from .http import CurlClient, FetchResult
from .normalize import GROUPS, RANKING_PERIODS, normalize_bundle, select_apps, table_counts
from .storage import SQLiteStore
from .util import atomic_write_json, iso_now, run_id
from .xlsx import build_workbook

SITE_BASE = "https://openrouter.ai"
API_BASE = f"{SITE_BASE}/api/frontend/v1"
CORE_ENDPOINTS = {
    "model_daily": f"{API_BASE}/rankings/models?view=day",
    "model_chart": f"{API_BASE}/rankings/model-rankings-chart",
    "task_spend": f"{API_BASE}/rankings/task-spend",
    "marketplace": f"{API_BASE}/apps/marketplace/marketplace",
    "catalog": f"{API_BASE}/catalog/models",
}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _source_row(result: FetchResult, root: Path) -> dict[str, Any]:
    return {
        "name": result.name,
        "url": result.url,
        "sha256": result.sha256,
        "bytes": len(result.body),
        "content_type": result.content_type,
        "raw_path": _relative(result.path, root),
    }


def _fetch_core(
    client: CurlClient,
    root: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jobs: dict[str, str] = dict(CORE_ENDPOINTS)
    for group in GROUPS:
        for period in RANKING_PERIODS:
            jobs[f"category_{group}_{period}"] = (
                f"{API_BASE}/apps/marketplace/category?"
                f"group={quote(group)}&period={quote(period)}"
            )
        jobs[f"category_{group}_chart"] = (
            f"{API_BASE}/apps/marketplace/category-chart?group={quote(group)}"
        )

    results: dict[str, FetchResult] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(client.fetch_json, name, url): name
            for name, url in jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("Core source fetch failed: " + " | ".join(errors))

    raw: dict[str, Any] = {
        key: results[key].data() for key in CORE_ENDPOINTS
    }
    raw["categories"] = {}
    for group in GROUPS:
        raw["categories"][group] = {
            period: results[f"category_{group}_{period}"].data()
            for period in RANKING_PERIODS
        }
        raw["categories"][group]["chart"] = results[
            f"category_{group}_chart"
        ].data()
    sources = [_source_row(result, root) for result in results.values()]
    sources.sort(key=lambda row: row["name"])
    return raw, sources


def _detail_url(app: dict[str, Any]) -> str:
    if app.get("app_slug"):
        return f"{SITE_BASE}/apps/{quote(str(app['app_slug']))}"
    return f"{SITE_BASE}/apps?url={quote(str(app.get('origin_url') or ''), safe='')}"


def _fetch_details(
    client: CurlClient,
    root: Path,
    apps: list[dict[str, Any]],
    *,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    def one(app: dict[str, Any]) -> tuple[dict[str, Any], FetchResult | None]:
        page_url = _detail_url(app)
        name = f"app_{app.get('app_id') or app.get('app_slug') or 'unknown'}"
        try:
            result = client.fetch_html(name, page_url)
            parsed = parse_react_flight(result.body.decode("utf-8", errors="replace"))
            status = "ok" if parsed.get("summary") else "parse_failed"
            return (
                {
                    "selected": app,
                    "page_url": page_url,
                    "fetch_status": status,
                    "error": None if status == "ok" else "React Flight summary not found",
                    "parsed": parsed,
                },
                result,
            )
        except Exception as exc:  # noqa: BLE001
            return (
                {
                    "selected": app,
                    "page_url": page_url,
                    "fetch_status": "fetch_failed",
                    "error": str(exc),
                    "parsed": {
                        "summary": None,
                        "categoryRankings": [],
                        "chartData": [],
                        "appModelAnalytics": [],
                    },
                },
                None,
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, app) for app in apps]
        for future in as_completed(futures):
            detail, result = future.result()
            details.append(detail)
            if result is not None:
                sources.append(_source_row(result, root))
    details.sort(key=lambda item: str((item.get("selected") or {}).get("app_name") or ""))
    sources.sort(key=lambda row: row["name"])
    return details, sources


def _write_failure(
    root: Path,
    current_run_id: str,
    captured_at: str,
    run_dir: Path,
    issue: ContractIssue | list[ContractIssue],
    *,
    exception: str | None = None,
) -> dict[str, Any]:
    issues = issue if isinstance(issue, list) else [issue]
    signature = failure_signature(issues)
    report = {
        "run_id": current_run_id,
        "captured_at": captured_at,
        "status": "failed",
        "signature": signature,
        "run_path": _relative(run_dir, root),
        "issues": [item.to_dict() for item in issues],
        "exception": exception,
        "repair_classification": "unclassified",
    }
    failure_dir = root / "output" / "failures" / current_run_id
    atomic_write_json(failure_dir / "failure_report.json", report)
    status_dir = root / "output" / "status"
    atomic_write_json(status_dir / "latest_failure.json", report)
    atomic_write_json(status_dir / "latest.json", report)
    return report


def _update_latest(root: Path, run_dir: Path) -> None:
    output_dir = root / "output"
    target = Path("runs") / run_dir.name
    temporary = output_dir / ".latest.tmp"
    latest = output_dir / "latest"
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, latest)


def run_harvest(root: Path, *, workers: int = 6) -> int:
    current_run_id = run_id()
    captured_at = iso_now()
    run_dir = root / "output" / "runs" / current_run_id
    raw_dir = run_dir / "raw"
    client = CurlClient(raw_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw, sources = _fetch_core(client, root, workers=workers)
        selected = select_apps(raw["marketplace"], raw["categories"])
        details, detail_sources = _fetch_details(
            client,
            root,
            selected,
            workers=workers,
        )
        sources.extend(detail_sources)
        raw["app_details"] = details
        raw["selected_app_count"] = len(selected)
        issues = validate_raw(raw)
        status = status_from_issues(issues)
        issue_dicts = [issue.to_dict() for issue in issues]

        raw_manifest = {
            "captured_at": captured_at,
            "selected_app_count": len(selected),
            "sources": sources,
            "core": {
                "model_daily": raw["model_daily"],
                "model_chart": raw["model_chart"],
                "task_spend": raw["task_spend"],
                "marketplace": raw["marketplace"],
                "catalog": raw["catalog"],
                "categories": raw["categories"],
            },
            "app_details": details,
        }
        atomic_write_json(run_dir / "raw_manifest.json", raw_manifest)
        atomic_write_json(
            run_dir / "contract_report.json",
            {
                "run_id": current_run_id,
                "captured_at": captured_at,
                "status": status,
                "issues": issue_dicts,
            },
        )

        if status == "failed":
            _write_failure(root, current_run_id, captured_at, run_dir, issues)
            print(json.dumps({"run_id": current_run_id, "status": status}))
            return 2

        tables = normalize_bundle(raw, captured_at)
        counts = table_counts(tables)
        normalized = {
            "run_id": current_run_id,
            "captured_at": captured_at,
            "status": status,
            "counts": counts,
            "issues": issue_dicts,
            "sources": sources,
            "tables": tables,
        }
        atomic_write_json(run_dir / "normalized.json", normalized)
        database = root / "output" / "store" / "openrouter.sqlite"
        SQLiteStore(database).append_run(
            current_run_id,
            captured_at,
            status,
            counts,
            issue_dicts,
            tables,
        )
        workbook_path = run_dir / f"openrouter_public_data_{captured_at[:10]}.xlsx"
        build_workbook(
            workbook_path,
            captured_at=captured_at,
            status=status,
            sources=sources,
            issues=issue_dicts,
            tables=tables,
        )
        status_report = {
            "run_id": current_run_id,
            "captured_at": captured_at,
            "status": status,
            "counts": counts,
            "issues": issue_dicts,
            "run_path": _relative(run_dir, root),
            "workbook_path": _relative(workbook_path, root),
            "database_path": _relative(database, root),
        }
        if status == "degraded":
            status_report["warning_signature"] = warning_signature(issues)
        atomic_write_json(root / "output" / "status" / "latest.json", status_report)
        try:
            (root / "output" / "status" / "latest_failure.json").unlink()
        except FileNotFoundError:
            pass
        _update_latest(root, run_dir)
        print(json.dumps(status_report, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        trace = traceback.format_exc()
        atomic_write_json(
            run_dir / "exception.json",
            {"error": str(exc), "traceback": trace},
        )
        issue = ContractIssue(
            "error",
            "unhandled_exception",
            "harvest",
            str(exc),
        )
        report = _write_failure(
            root,
            current_run_id,
            captured_at,
            run_dir,
            issue,
            exception=trace,
        )
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 3


def probe(root: Path, *, workers: int = 6) -> int:
    probe_id = run_id()
    raw_dir = root / "output" / "probes" / probe_id / "raw"
    client = CurlClient(raw_dir, retries=2, timeout=45)
    try:
        raw, sources = _fetch_core(client, root, workers=workers)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    checks = {
        "model_daily": len(raw.get("model_daily") or []),
        "model_chart": len((raw.get("model_chart") or {}).get("data") or []),
        "task_count": len(
            ((raw.get("task_spend") or {}).get("tokens") or {}).get("tasks") or []
        ),
        "global_day": len(
            ((raw.get("marketplace") or {}).get("globalRankingMap") or {}).get("day")
            or []
        ),
        "catalog": len(raw.get("catalog") or []),
        "source_count": len(sources),
    }
    ok = (
        checks["model_daily"] >= 20
        and checks["model_chart"] >= 20
        and checks["task_count"] >= 10
        and checks["global_day"] >= 40
        and checks["catalog"] >= 100
    )
    print(json.dumps({"status": "success" if ok else "failed", "checks": checks}))
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing output/",
    )
    parser.add_argument("--workers", type=int, default=6)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run the full harvest and publish outputs")
    subparsers.add_parser("probe", help="Check core sources without App details")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.command == "run":
        return run_harvest(root, workers=args.workers)
    if args.command == "probe":
        return probe(root, workers=args.workers)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
