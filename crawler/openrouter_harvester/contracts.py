from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .normalize import GLOBAL_PERIODS, GROUPS, RANKING_PERIODS


@dataclass(frozen=True)
class ContractIssue:
    severity: str
    code: str
    source: str
    message: str
    observed: Any = None
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _issue(
    issues: list[ContractIssue],
    severity: str,
    code: str,
    source: str,
    message: str,
    observed: Any = None,
    expected: Any = None,
) -> None:
    issues.append(
        ContractIssue(severity, code, source, message, observed, expected)
    )


def validate_raw(raw: dict[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []

    model_daily = raw.get("model_daily")
    if not isinstance(model_daily, list):
        _issue(issues, "error", "shape", "model_daily", "Expected a list")
    elif len(model_daily) < 20:
        _issue(
            issues,
            "error",
            "min_rows",
            "model_daily",
            "Too few model ranking rows",
            len(model_daily),
            20,
        )
    elif (
        "model_permaslug" not in model_daily[0]
        or not (
            "total_tokens" in model_daily[0]
            or {
                "total_prompt_tokens",
                "total_completion_tokens",
            }.issubset(model_daily[0])
        )
    ):
        _issue(
            issues,
            "error",
            "required_fields",
            "model_daily",
            "Ranking row lost required fields",
            sorted(model_daily[0]),
            [
                "model_permaslug",
                "total_tokens OR (total_prompt_tokens + total_completion_tokens)",
            ],
        )

    chart = raw.get("model_chart") or {}
    chart_rows = chart.get("data") if isinstance(chart, dict) else None
    if not isinstance(chart_rows, list) or len(chart_rows) < 20:
        _issue(
            issues,
            "error",
            "min_points",
            "model_chart",
            "Model chart has too few points",
            len(chart_rows or []),
            20,
        )
    elif not isinstance(chart_rows[0].get("ys"), dict):
        _issue(
            issues,
            "error",
            "shape",
            "model_chart",
            "Chart point is missing ys mapping",
        )

    tasks = ((raw.get("task_spend") or {}).get("tokens") or {}).get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 10:
        _issue(
            issues,
            "error",
            "min_tasks",
            "task_spend",
            "Task-share payload has too few tasks",
            len(tasks or []),
            10,
        )
    elif not isinstance(tasks[0].get("models"), list):
        _issue(
            issues,
            "error",
            "shape",
            "task_spend",
            "Task record is missing models",
        )

    marketplace = raw.get("marketplace") or {}
    global_map = marketplace.get("globalRankingMap")
    if not isinstance(global_map, dict):
        _issue(
            issues,
            "error",
            "shape",
            "marketplace",
            "Missing globalRankingMap",
        )
    else:
        for period in GLOBAL_PERIODS:
            count = len(global_map.get(period) or [])
            if count < 40:
                _issue(
                    issues,
                    "error",
                    "min_rows",
                    f"marketplace:{period}",
                    "Global app ranking has fewer than 40 rows",
                    count,
                    40,
                )

    categories = raw.get("categories") or {}
    for group in GROUPS:
        group_data = categories.get(group) or {}
        for period in RANKING_PERIODS:
            count = len((group_data.get(period) or {}).get("apps") or [])
            if count < 1:
                _issue(
                    issues,
                    "error",
                    "empty_ranking",
                    f"category:{group}:{period}",
                    "App category ranking is empty",
                )
            elif count < 10:
                _issue(
                    issues,
                    "warning",
                    "short_ranking",
                    f"category:{group}:{period}",
                    "App category contains fewer than 10 entries",
                    count,
                    10,
                )
        chart_count = len((group_data.get("chart") or {}).get("chartData") or [])
        if chart_count < 20:
            _issue(
                issues,
                "error",
                "min_points",
                f"category_chart:{group}",
                "App category chart has too few points",
                chart_count,
                20,
            )

    catalog = raw.get("catalog")
    if not isinstance(catalog, list) or len(catalog) < 100:
        _issue(
            issues,
            "error",
            "min_rows",
            "catalog",
            "Model catalog has too few rows",
            len(catalog or []),
            100,
        )

    details = raw.get("app_details") or []
    selected_count = raw.get("selected_app_count") or len(details)
    success = sum(
        1
        for detail in details
        if detail.get("fetch_status") == "ok"
        and (detail.get("parsed") or {}).get("summary")
    )
    ratio = success / selected_count if selected_count else 0
    if selected_count < 20:
        _issue(
            issues,
            "error",
            "min_selected_apps",
            "app_details",
            "Selection set unexpectedly small",
            selected_count,
            20,
        )
    if ratio < 0.8:
        _issue(
            issues,
            "error",
            "success_ratio",
            "app_details",
            "Too many App detail pages failed",
            ratio,
            0.8,
        )
    elif ratio < 0.95:
        _issue(
            issues,
            "warning",
            "success_ratio",
            "app_details",
            "Some App detail pages failed",
            ratio,
            0.95,
        )

    # An App page can parse (summary found) while its usage chart silently fails to
    # extract (this happened on 2026-10-08 and went unnoticed for five weeks).
    with_chart = sum(
        1 for detail in details if (detail.get("parsed") or {}).get("chartData")
    )
    if success >= 20 and with_chart / success < 0.5:
        _issue(
            issues,
            "error",
            "app_chart_missing",
            "app_details",
            "App usage charts were not extracted from the detail pages",
            with_chart,
            success,
        )

    return issues


def status_from_issues(issues: list[ContractIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "failed"
    if any(issue.severity == "warning" for issue in issues):
        return "degraded"
    return "success"


def failure_signature(issues: list[ContractIssue]) -> str:
    return issue_signature(issues, severities={"error"})


def warning_signature(issues: list[ContractIssue]) -> str:
    return issue_signature(issues, severities={"warning"})


def issue_signature(
    issues: list[ContractIssue],
    *,
    severities: set[str] | None = None,
) -> str:
    stable = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "source": issue.source,
            "message": issue.message,
        }
        for issue in issues
        if severities is None or issue.severity in severities
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
