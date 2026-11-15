from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_bytes, safe_name, sha256_bytes, unwrap_data


@dataclass(frozen=True)
class FetchResult:
    name: str
    url: str
    path: Path
    body: bytes
    sha256: str
    content_type: str

    def json(self) -> Any:
        return json.loads(self.body)

    def data(self) -> Any:
        return unwrap_data(self.json())


class CurlClient:
    def __init__(
        self,
        raw_dir: Path,
        *,
        curl_bin: str = "/usr/bin/curl",
        user_agent: str = "Mozilla/5.0 (compatible; OpenRouterHarvester/1.0)",
        retries: int = 4,
        timeout: int = 120,
    ) -> None:
        self.raw_dir = raw_dir
        self.curl_bin = curl_bin
        self.user_agent = user_agent
        self.retries = retries
        self.timeout = timeout

    def fetch(self, name: str, url: str, *, suffix: str) -> FetchResult:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                completed = subprocess.run(
                    [
                        self.curl_bin,
                        "--compressed",
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        str(self.timeout),
                        "-A",
                        self.user_agent,
                        "-H",
                        "Accept: application/json,text/html;q=0.9,*/*;q=0.8",
                        "-fsSL",
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=self.timeout + 15,
                )
                body = completed.stdout
                if not body:
                    raise RuntimeError(f"Empty response from {url}")
                path = self.raw_dir / f"{safe_name(name)}.{suffix}"
                atomic_write_bytes(path, body)
                return FetchResult(
                    name=name,
                    url=url,
                    path=path,
                    body=body,
                    sha256=sha256_bytes(body),
                    content_type="application/json" if suffix == "json" else "text/html",
                )
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error

    def fetch_json(self, name: str, url: str) -> FetchResult:
        result = self.fetch(name, url, suffix="json")
        try:
            result.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name} returned invalid JSON: {exc}") from exc
        return result

    def fetch_html(self, name: str, url: str) -> FetchResult:
        return self.fetch(name, url, suffix="html")

