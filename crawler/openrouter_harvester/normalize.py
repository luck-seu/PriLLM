from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .util import canonical_model_id, guess_model_name

SITE_BASE = "https://openrouter.ai"
GLOBAL_PERIODS = ("day", "week", "month")
GROUPS = ("coding", "productivity", "creative", "entertainment")
RANKING_PERIODS = ("day", "week")


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _model_total_tokens(item: dict[str, Any]) -> int | float:
    direct = _number(item.get("total_tokens"))
    if direct is not None:
        return direct
    return (_number(item.get("total_prompt_tokens")) or 0) + (
        _number(item.get("total_completion_tokens")) or 0
    )


def normalize_app_analytics(entry: dict[str, Any] | None) -> dict[str, Any]:
    entry = entry or {}
    analytics = entry.get("appAnalytics") or entry
    app = analytics.get("app") or {}
    slug = app.get("slug")
    growth_percent = _number(entry.get("growthPercent"))
    return {
        "app_id": analytics.get("app_id") or app.get("id"),
        "app_name": app.get("title") or "",
        "app_slug": slug,
        "app_url": f"{SITE_BASE}/apps/{slug}" if slug else None,
        "origin_url": app.get("origin_url"),
        "main_url": app.get("main_url"),
        "description": app.get("description"),
        "categories": ",".join(app.get("categories") or []),
        "created_at": app.get("created_at"),
        "token_usage": _number(analytics.get("total_tokens")),
        "request_count": _number(analytics.get("total_requests")),
        "rank": analytics.get("rank"),
        "growth_percent": (
            float(growth_percent) / 100 if growth_percent is not None else None
        ),
    }


def app_key(app: dict[str, Any]) -> str:
    return str(
        app.get("app_id")
        or app.get("app_slug")
        or app.get("origin_url")
        or app.get("app_name")
    )


def select_apps(
    marketplace: dict[str, Any],
    categories: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def add(entry: dict[str, Any], reason: str) -> None:
        app = normalize_app_analytics(entry)
        key = app_key(app)
        if not key or key == "None":
            return
        current = selected.setdefault(key, app)
        reasons = set(filter(None, str(current.get("selection_reasons", "")).split(",")))
        reasons.add(reason)
        current["selection_reasons"] = ",".join(sorted(reasons))

    day_global = (marketplace.get("globalRankingMap") or {}).get("day") or []
    for entry in day_global[:40]:
        add(entry, "global:day:top40")
    for group in GROUPS:
        group_data = categories.get(group) or {}
        for period in RANKING_PERIODS:
            ranking = (group_data.get(period) or {}).get("apps") or []
            for entry in ranking[:10]:
                add(entry, f"{group}:{period}:top10")
    return list(selected.values())


def _model_index(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in catalog:
        for key in (
            item.get("slug"),
            item.get("permaslug"),
            canonical_model_id(item.get("slug")),
            canonical_model_id(item.get("permaslug")),
        ):
            if key:
                index[str(key)] = item
    return index


def _model_info(
    model_id: str,
    index: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    item = (
        index.get(model_id)
        or index.get(canonical_model_id(model_id))
        or {}
    )
    name = item.get("name") or item.get("short_name") or guess_model_name(model_id)
    author = item.get("author_display_name") or item.get("author") or (
        model_id.split("/", 1)[0] if "/" in model_id else ""
    )
    slug = item.get("slug") or canonical_model_id(model_id)
    return str(name or ""), str(author or ""), f"{SITE_BASE}/{slug}" if slug else ""


def _ranked_usage(
    ys: dict[str, Any],
    *,
    limit: int | None = None,
    exclude_others: bool = False,
) -> list[tuple[int, str, int | float, int | float, float]]:
    total = sum((_number(value) or 0) for value in ys.values())
    items = [
        (str(key), _number(value) or 0)
        for key, value in ys.items()
        if not (exclude_others and str(key) == "Others")
    ]
    items.sort(key=lambda item: float(item[1]), reverse=True)
    if limit is not None:
        items = items[:limit]
    return [
        (rank, key, value, total, float(value) / total if total else 0.0)
        for rank, (key, value) in enumerate(items, 1)
    ]


def normalize_bundle(raw: dict[str, Any], captured_at: str) -> dict[str, list[dict[str, Any]]]:
    model_daily = raw["model_daily"]
    model_chart = raw["model_chart"]
    task_spend = raw["task_spend"]
    marketplace = raw["marketplace"]
    catalog = raw["catalog"]
    categories = raw["categories"]
    app_details = raw["app_details"]
    model_index = _model_index(catalog)
    snapshot_date = captured_at[:10]

    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)

    ranked_models = sorted(
        model_daily,
        key=lambda item: float(_model_total_tokens(item)),
        reverse=True,
    )
    total_model_tokens = sum(
        float(_model_total_tokens(item)) for item in ranked_models
    )
    for rank, item in enumerate(ranked_models, 1):
        model_id = str(item.get("variant_permaslug") or item.get("model_permaslug") or "")
        name, author, url = _model_info(model_id, model_index)
        usage = _model_total_tokens(item)
        tables["model_rankings"].append(
            {
                "snapshot_date": item.get("date") or snapshot_date,
                "view": "day",
                "rank": rank,
                "model_id": model_id,
                "model_name": name,
                "author": author,
                "model_url": url,
                "variant": item.get("variant"),
                "token_usage": usage,
                "request_count": _number(item.get("count")),
                "share": float(usage) / total_model_tokens if total_model_tokens else 0,
                "change": _number(item.get("change")),
            }
        )

    for point in model_chart.get("data") or []:
        for rank, model_id, usage, total, share in _ranked_usage(point.get("ys") or {}):
            name, author, url = _model_info(model_id, model_index)
            tables["model_history"].append(
                {
                    "date": point.get("x"),
                    "grain": "week",
                    "rank": rank,
                    "model_id": model_id,
                    "model_name": name,
                    "author": author,
                    "model_url": url,
                    "token_usage": usage,
                    "total_usage": total,
                    "share": share,
                }
            )

    for metric in ("tokens", "spend"):
        metric_data = task_spend.get(metric) or {}
        for task in metric_data.get("tasks") or []:
            for rank, model in enumerate(task.get("models") or [], 1):
                model_id = str(model.get("model") or "")
                name, author, url = _model_info(model_id, model_index)
                tables["task_model_share"].append(
                    {
                        "snapshot_date": snapshot_date,
                        "metric": metric,
                        "window_days": metric_data.get("windowDays"),
                        "macro_category": task.get("macroCategory"),
                        "task": task.get("tag"),
                        "task_share_total": _number(task.get("spendShareOfTotal")),
                        "rank": rank,
                        "model_id": model_id,
                        "model_name": name,
                        "author": author,
                        "model_url": url,
                        "share_within_task": _number(model.get("share")),
                        "delta_pp": _number(model.get("deltaPp")),
                    }
                )

    for group, entries in (marketplace.get("leaderboards") or {}).items():
        for entry in entries or []:
            app = normalize_app_analytics(entry)
            tables["apps_root"].append({"section": group, **app})
    for section in ("popular", "trending"):
        for entry in marketplace.get(section) or []:
            app = normalize_app_analytics(entry)
            tables["apps_root"].append({"section": section, **app})

    for period in GLOBAL_PERIODS:
        entries = (marketplace.get("globalRankingMap") or {}).get(period) or []
        for entry in entries[:40]:
            app = normalize_app_analytics(entry)
            tables["app_global_rankings"].append(
                {"snapshot_date": snapshot_date, "period": period, **app}
            )

    app_entities: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        group_data = categories.get(group) or {}
        for period in RANKING_PERIODS:
            for entry in (group_data.get(period) or {}).get("apps") or []:
                app = normalize_app_analytics(entry)
                app_entities[app_key(app)] = app
                tables["app_category_rankings"].append(
                    {
                        "snapshot_date": snapshot_date,
                        "group": group,
                        "period": period,
                        **app,
                    }
                )
        chart = group_data.get("chart") or {}
        title_map = chart.get("titleMap") or {}
        for point in chart.get("chartData") or []:
            for rank, raw_app_id, usage, total, share in _ranked_usage(
                point.get("ys") or {},
                limit=10,
                exclude_others=True,
            ):
                tables["app_category_history"].append(
                    {
                        "date": point.get("x"),
                        "grain": "week",
                        "group": group,
                        "rank": rank,
                        "app_id": int(raw_app_id) if raw_app_id.isdigit() else raw_app_id,
                        "app_name": title_map.get(raw_app_id) or "",
                        "token_usage": usage,
                        "total_usage": total,
                        "share": share,
                    }
                )

    for detail in app_details:
        selected = detail.get("selected") or {}
        parsed = detail.get("parsed") or {}
        summary = parsed.get("summary") or {}
        summary_app = normalize_app_analytics(
            {
                "app_id": (summary.get("app") or {}).get("id"),
                "app": summary.get("app") or {},
            }
        )
        app = {**selected}
        for key, value in summary_app.items():
            if value not in (None, ""):
                app[key] = value
        app["fetch_status"] = detail.get("fetch_status")
        app["fetch_error"] = detail.get("error")
        app["page_url"] = detail.get("page_url")
        app["total_tokens"] = _number(summary.get("totalTokens"))
        app["global_rank"] = summary.get("rank")
        app["models_used"] = summary.get("modelsUsed")
        app["category_rankings"] = "; ".join(
            f"{item.get('label')} #{item.get('rank')}"
            for item in parsed.get("categoryRankings") or []
        )
        app_entities[app_key(app)] = app
        tables["app_details"].append(app)

        app_id = app.get("app_id")
        app_name = app.get("app_name")
        monthly = sorted(
            parsed.get("appModelAnalytics") or [],
            key=lambda item: float(_number(item.get("total_tokens")) or 0),
            reverse=True,
        )
        for rank, item in enumerate(monthly[:20], 1):
            model_id = str(item.get("model_permaslug") or "")
            name, author, url = _model_info(model_id, model_index)
            tables["app_model_rankings"].append(
                {
                    "snapshot_date": snapshot_date,
                    "app_id": app_id,
                    "app_name": app_name,
                    "rank": rank,
                    "model_id": model_id,
                    "model_name": name,
                    "author": author,
                    "model_url": url,
                    "token_usage": _number(item.get("total_tokens")),
                }
            )
        for point in parsed.get("chartData") or []:
            for rank, model_id, usage, total, share in _ranked_usage(
                point.get("ys") or {},
                limit=20,
                exclude_others=True,
            ):
                name, author, url = _model_info(model_id, model_index)
                tables["app_model_history"].append(
                    {
                        "date": point.get("x"),
                        "app_id": app_id,
                        "app_name": app_name,
                        "rank": rank,
                        "model_id": model_id,
                        "model_name": name,
                        "author": author,
                        "model_url": url,
                        "token_usage": usage,
                        "total_usage": total,
                        "share": share,
                    }
                )

    tables["apps"] = sorted(
        app_entities.values(),
        key=lambda item: (str(item.get("app_name") or "").lower(), app_key(item)),
    )

    seen_model_ids: set[str] = set()
    for item in catalog:
        model_id = str(item.get("permaslug") or item.get("slug") or "")
        if not model_id:
            continue
        seen_model_ids.add(model_id)
        tables["models"].append(
            {
                "model_id": model_id,
                "slug": item.get("slug"),
                "model_name": item.get("name") or item.get("short_name"),
                "author": item.get("author_display_name") or item.get("author"),
                "model_url": f"{SITE_BASE}/{item.get('slug')}" if item.get("slug") else None,
                "context_length": item.get("context_length"),
                "input_modalities": ",".join(item.get("input_modalities") or []),
                "output_modalities": ",".join(item.get("output_modalities") or []),
                "supports_reasoning": item.get("supports_reasoning"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }
        )

    observed: set[str] = set()
    for table_name in (
        "model_rankings",
        "model_history",
        "task_model_share",
        "app_model_rankings",
        "app_model_history",
    ):
        observed.update(
            str(row.get("model_id"))
            for row in tables[table_name]
            if row.get("model_id") and row.get("model_id") != "Others"
        )
    for model_id in sorted(observed):
        if model_id in seen_model_ids or canonical_model_id(model_id) in seen_model_ids:
            continue
        name, author, url = _model_info(model_id, model_index)
        tables["models"].append(
            {
                "model_id": model_id,
                "slug": canonical_model_id(model_id),
                "model_name": name,
                "author": author,
                "model_url": url,
                "context_length": None,
                "input_modalities": "",
                "output_modalities": "",
                "supports_reasoning": None,
                "created_at": None,
                "updated_at": None,
            }
        )

    return dict(tables)


def table_counts(tables: dict[str, Iterable[dict[str, Any]]]) -> dict[str, int]:
    return {name: len(list(rows)) for name, rows in tables.items()}
