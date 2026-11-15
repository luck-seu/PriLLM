from __future__ import annotations

import json
import re
from typing import Any


PUSH_RE = re.compile(
    r"<script[^>]*>self\.__next_f\.push\((.*?)\)</script>",
    flags=re.DOTALL,
)

_ROW_ID_RE = re.compile(rb"([0-9A-Fa-f]*):")
_LENGTH_FRAMED_TAGS = b"TAOobUSsLlGgMmV"
_LENGTH_ROW_RE = re.compile(
    rb"(?P<tag>[TAOobUSsLlGgMmV])(?P<size>[0-9A-Fa-f]+),"
)
_NEWLINE_FRAMED_TAGS = frozenset(b"#BCDEFHIPRWXrx")


def _walk(value: Any, found: dict[str, Any]) -> None:
    if isinstance(value, dict):
        keys = set(value)
        if (
            found["summary"] is None
            and {"app", "totalTokens"}.issubset(keys)
            and isinstance(value.get("app"), dict)
            and value["app"].get("title")
        ):
            found["summary"] = value
        # OpenRouter changed the App chart payload on 2026-10-08 from
        #   {"data": [{"x": ..., "ys": {...}}, ...]}
        # to
        #   {"data": {"daily": [...], "weekly": [...]}}
        # Accept both; the harvester stores the daily series.
        chart = value.get("data")
        if isinstance(chart, dict) and isinstance(chart.get("daily"), list):
            chart = chart["daily"]
        if (
            not found["chartData"]
            and isinstance(chart, list)
            and chart
            and isinstance(chart[0], dict)
            and {"x", "ys"}.issubset(chart[0])
        ):
            found["chartData"] = chart
        if not found["appModelAnalytics"] and isinstance(
            value.get("appModelAnalytics"), list
        ):
            found["appModelAnalytics"] = value["appModelAnalytics"]
        if (
            not found["categoryRankings"]
            and isinstance(value.get("rankings"), list)
            and value["rankings"]
            and isinstance(value["rankings"][0], dict)
            and value["rankings"][0].get("label")
        ):
            found["categoryRankings"] = value["rankings"]
        for child in value.values():
            _walk(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk(child, found)


def _extract_push_chunks(html: str) -> str:
    chunks: list[str] = []
    for match in PUSH_RE.finditer(html):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed React Flight push payload") from exc
        if (
            isinstance(payload, list)
            and len(payload) > 1
            and isinstance(payload[1], str)
        ):
            chunks.append(payload[1])
    return "".join(chunks)


def _framing_error(message: str, stream: bytes, pos: int) -> ValueError:
    return ValueError(f"{message} at byte offset {pos} in {len(stream)} bytes")


def parse_react_flight(html: str) -> dict[str, Any]:
    stream = _extract_push_chunks(html).encode("utf-8")

    found: dict[str, Any] = {
        "summary": None,
        "categoryRankings": [],
        "chartData": [],
        "appModelAnalytics": [],
    }

    pos = 0
    while pos < len(stream):
        if stream[pos : pos + 1] in (b"\r", b"\n"):
            pos += 1
            continue
        row_id_match = _ROW_ID_RE.match(stream, pos)
        if row_id_match is None:
            raise _framing_error("Expected React Flight row ID", stream, pos)
        pos = row_id_match.end()
        if pos >= len(stream):
            raise _framing_error("Truncated React Flight row tag", stream, pos)

        tag = stream[pos : pos + 1]
        if tag in _LENGTH_FRAMED_TAGS:
            length_match = _LENGTH_ROW_RE.match(stream, pos)
            if length_match is None:
                raise _framing_error("Malformed length-prefixed row header", stream, pos)
            size = int(length_match["size"], 16)
            payload_start = length_match.end()
            payload_end = payload_start + size
            if payload_end > len(stream):
                raise _framing_error("Truncated length-prefixed row", stream, pos)
            if tag == b"T":
                try:
                    stream[payload_start:payload_end].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise _framing_error("Invalid UTF-8 text row boundary", stream, pos) from exc
            pos = payload_end
            continue

        if tag[0] in _NEWLINE_FRAMED_TAGS:
            newline = stream.find(b"\n", pos + 1)
            if newline == -1:
                raise _framing_error("Truncated newline-framed metadata row", stream, pos)
            pos = newline + 1
            continue

        newline = stream.find(b"\n", pos)
        end = len(stream) if newline == -1 else newline
        try:
            value = json.loads(stream[pos:end])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _framing_error(
                "Truncated, malformed or unsupported JSON row framing", stream, pos
            ) from exc
        _walk(value, found)
        pos = end + (newline != -1)

    return found
