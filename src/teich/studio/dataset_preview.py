"""Local Hugging Face-style dataset previews for Teich Studio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..converter import convert_traces_to_training_data
from ..loader import trace_is_complete
from .events import summarize_chat_row, summarize_trace_events

MAX_README_CHARS = 24_000
MAX_TRACE_FILES = 8
MAX_TRACE_EVENTS = 5_000
MAX_TRACE_DISPLAY = 80


def _jsonl_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".jsonl":
        return [root]
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*.jsonl")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in {"failures", "partials"} for part in relative_parts):
            continue
        files.append(path)
    return files


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _feature_type(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"dtype": "string", "_type": "Value"}
    if isinstance(value, bool):
        return {"dtype": "bool", "_type": "Value"}
    if isinstance(value, int):
        return {"dtype": "int64", "_type": "Value"}
    if isinstance(value, float):
        return {"dtype": "float64", "_type": "Value"}
    if isinstance(value, list):
        nested = _feature_type(value[0]) if value else {"dtype": "null", "_type": "Value"}
        return {"feature": nested, "_type": "Sequence"}
    if isinstance(value, dict):
        return {key: _feature_type(item) for key, item in sorted(value.items())}
    if value is None:
        return {"dtype": "null", "_type": "Value"}
    return {"dtype": type(value).__name__, "_type": "Value"}


def _features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, value)
    return [
        {"feature_idx": index, "name": name, "type": _feature_type(value)}
        for index, (name, value) in enumerate(sorted(columns.items()))
    ]


def _stringify_for_search(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).casefold()


def _row_preview(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    tools = row.get("tools") if isinstance(row.get("tools"), list) else []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "prompt": row.get("prompt") or _first_message_text(messages, "user"),
        "response": row.get("response") or _last_message_text(messages, "assistant"),
        "model": row.get("model") or metadata.get("model"),
        "message_count": len(messages),
        "tool_count": len(tools),
        "trace_type": metadata.get("trace_type"),
        "complete": trace_is_complete(row),
    }


def _first_message_text(messages: list[Any], role: str) -> str | None:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _last_message_text(messages: list[Any], role: str) -> str | None:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _column_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = sorted({key for row in rows for key in row.keys()})
    stats: list[dict[str, Any]] = []
    for column in columns:
        values = [row.get(column) for row in rows if column in row]
        scalar_counts: dict[str, int] = {}
        for value in values:
            if isinstance(value, str | int | float | bool) or value is None:
                label = "null" if value is None else str(value)
                scalar_counts[label] = scalar_counts.get(label, 0) + 1
        top_values = sorted(scalar_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        stats.append(
            {
                "name": column,
                "present": len(values),
                "missing": max(len(rows) - len(values), 0),
                "type": _feature_type(values[0]) if values else {"dtype": "null", "_type": "Value"},
                "top_values": [{"value": value, "count": count} for value, count in top_values],
            }
        )
    return stats


def _detect_trace_provider(events: list[dict[str, Any]]) -> str:
    for event in events[:5]:
        event_type = event.get("type")
        if event_type == "session_meta":
            return "codex"
        if event_type == "session":
            return "pi"
        if event_type == "external_session_meta":
            return "hermes"
        if event_type in {"response_item", "event_msg"}:
            return "codex"
        if event_type == "message" and isinstance(event.get("message"), dict):
            return "pi"
        if "sessionId" in event or event_type == "queue-operation":
            return "claude-code"
        if isinstance(event.get("messages"), list):
            return "chat"
    return "unknown"


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
                if len(events) >= MAX_TRACE_EVENTS:
                    break
    except OSError:
        return []
    return events


def _trace_preview(root: Path, path: Path) -> dict[str, Any]:
    events = _read_trace_events(path)
    provider = _detect_trace_provider(events)
    if provider == "chat":
        display: list[dict[str, Any]] = []
        for row in events:
            display.extend(summarize_chat_row(row))
    else:
        display = summarize_trace_events(provider, events)
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = path.name
    return {
        "name": name,
        "provider": provider,
        "event_count": len(events),
        "display": display[:MAX_TRACE_DISPLAY],
        "truncated": len(display) > MAX_TRACE_DISPLAY,
    }


def _readme_text(root: Path) -> str | None:
    readme = root / "README.md" if root.is_dir() else root.parent / "README.md"
    if not readme.exists():
        return None
    try:
        text = readme.read_text(encoding="utf-8")
    except OSError:
        return None
    return text[:MAX_README_CHARS]


def build_dataset_preview(
    root: Path,
    *,
    repo_id: str | None = None,
    offset: int = 0,
    limit: int = 100,
    search: str | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset output path not found: {root}")

    offset = max(offset, 0)
    limit = max(1, min(limit, 100))
    trace_files = _jsonl_files(root)
    file_rows = [
        {
            "name": path.relative_to(root).as_posix() if root.is_dir() else path.name,
            "size_bytes": path.stat().st_size,
            "rows": _line_count(path),
        }
        for path in trace_files
    ]

    errors: list[str] = []
    try:
        rows = convert_traces_to_training_data(root)
    except Exception as exc:
        rows = []
        errors.append(str(exc))

    query = (search or "").strip().casefold()
    indexed_rows = list(enumerate(rows))
    if query:
        indexed_rows = [(index, row) for index, row in indexed_rows if query in _stringify_for_search(row)]
    page = indexed_rows[offset:offset + limit]
    page_rows = [
        {"row_idx": index, "row": row, "preview": _row_preview(row)}
        for index, row in page
    ]
    selected_rows = [row for _, row in indexed_rows]
    complete_rows = sum(1 for row in selected_rows if trace_is_complete(row))

    return {
        "root": str(root),
        "repo_id": repo_id,
        "hf_embed_url": f"https://huggingface.co/datasets/{repo_id}/embed/viewer" if repo_id else None,
        "splits": [{"config": "default", "split": "train"}],
        "files": file_rows,
        "readme": _readme_text(root),
        "dataset": {
            "config": "default",
            "split": "train",
            "num_rows": len(selected_rows),
            "total_rows": len(rows),
            "offset": offset,
            "length": len(page_rows),
            "features": _features(selected_rows),
            "rows": page_rows,
            "search": search or "",
            "complete_rows": complete_rows,
            "incomplete_rows": max(len(selected_rows) - complete_rows, 0),
        },
        "statistics": _column_statistics(selected_rows),
        "trace_previews": [_trace_preview(root, path) for path in trace_files[:MAX_TRACE_FILES]],
        "errors": errors,
        "notes": [
            "Local preview uses Teich conversion directly. Hugging Face's hosted viewer adds Parquet-backed search, filtering, and SQL after upload."
        ],
    }
