"""Shared helpers for auditable SFT-data pipeline runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(record: dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def usage_to_dict(usage: Any) -> dict[str, int | None]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


class JsonlTraceLogger:
    """Append-only LLM trace logger.

    The logger intentionally records prompts and responses, but never receives
    API keys or client configuration. Keep trace files in experiment output
    directories, not in source control.
    """

    def __init__(self, path: str | None = None, enabled: bool = False):
        self.path = path
        self.enabled = bool(enabled and path)
        if self.enabled:
            ensure_dir(os.path.dirname(os.path.abspath(path or "")))

    def log(
        self,
        *,
        stage: str,
        model: str,
        messages: list[dict[str, Any]] | None = None,
        payload: Any = None,
        raw_response: str | None = None,
        parsed: Any = None,
        usage: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self.path:
            return
        record = {
            "timestamp": utc_now(),
            "stage": stage,
            "model": model,
            "messages": json_safe(messages),
            "payload": json_safe(payload),
            "raw_response": raw_response,
            "parsed": json_safe(parsed),
            "usage": usage_to_dict(usage),
            "error": error,
            "metadata": json_safe(metadata or {}),
        }
        append_jsonl(record, self.path)
