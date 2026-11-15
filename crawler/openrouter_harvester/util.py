from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def run_id(now: datetime | None = None) -> str:
    return (now or utc_now()).strftime("%Y-%m-%dT%H-%M-%SZ")


def iso_now(now: datetime | None = None) -> str:
    return (now or utc_now()).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:120] or "source"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_model_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "Others":
        return text
    return re.sub(r":(?:free|nitro|extended)$", "", text, flags=re.IGNORECASE)


def guess_model_name(model_id: str) -> str:
    if not model_id:
        return ""
    if model_id == "Others":
        return model_id
    leaf = canonical_model_id(model_id).rsplit("/", 1)[-1]
    leaf = re.sub(r"-\d{8}$", "", leaf)
    return " ".join(part[:1].upper() + part[1:] for part in leaf.split("-") if part)


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
        default=json_default,
    )
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_data(value: Any) -> Any:
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return value

