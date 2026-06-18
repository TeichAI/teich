"""Local Hugging Face-style dataset previews for Teich Studio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..converter import convert_traces_to_training_data
from ..loader import trace_is_complete

MAX_README_CHARS = 24_000
MAX_TRACE_EVENTS = 5_000


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        path.write_text(text, encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


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
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "prompt": row.get("prompt") or _first_message_text(messages, "user"),
        "response": row.get("response") or _last_message_text(messages, "assistant"),
        "model": row.get("model") or metadata.get("model"),
        "message_count": _user_message_count(messages),
        "tool_count": _tool_call_count(messages),
        "trace_type": metadata.get("trace_type"),
        "complete": trace_is_complete(row),
    }


def _user_message_count(messages: list[Any]) -> int:
    return sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "user")


def _tool_call_count(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"assistant", "model"}:
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += sum(1 for tool_call in tool_calls if isinstance(tool_call, dict))
    return count


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


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    message = f"Cannot edit malformed JSONL file {path.name} at line {line_number}: {exc}"
                    raise ValueError(message) from exc
                if isinstance(event, dict):
                    events.append(event)
                if len(events) >= MAX_TRACE_EVENTS:
                    break
    except OSError:
        return []
    return events


def _resolve_source_path(root: Path, row: dict[str, Any]) -> Path | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source_file = metadata.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        return None

    root = root.expanduser().resolve()
    raw_source = Path(source_file)
    candidates: list[Path] = []
    if raw_source.is_absolute():
        candidates.append(raw_source.expanduser().resolve())
    elif root.is_file():
        candidates.append(root)
    else:
        direct = (root / raw_source).resolve()
        if direct.exists():
            candidates.append(direct)
        for path in _jsonl_files(root):
            relative = path.relative_to(root).as_posix()
            if path.name == source_file or relative == source_file:
                candidates.append(path.resolve())

    unique_candidates = sorted(set(candidates), key=lambda path: str(path))
    safe_candidates: list[Path] = []
    for candidate in unique_candidates:
        if root.is_file():
            if candidate == root:
                safe_candidates.append(candidate)
            continue
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        safe_candidates.append(candidate)

    if len(safe_candidates) == 1:
        return safe_candidates[0]
    if len(safe_candidates) > 1:
        raise ValueError(f"Source file {source_file!r} is ambiguous under {root}.")
    return None


def _row_source_line(row: dict[str, Any]) -> int | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source_line = metadata.get("source_line")
    if isinstance(source_line, int) and source_line > 0:
        return source_line
    return None


def _structured_row(row: Any) -> bool:
    return isinstance(row, dict) and (
        isinstance(row.get("messages"), list)
        or isinstance(row.get("prompt"), str)
        or isinstance(row.get("response"), str)
    )


def _source_usage_count(root: Path, rows: list[dict[str, Any]], source_path: Path) -> int:
    count = 0
    for row in rows:
        try:
            if _resolve_source_path(root, row) == source_path:
                count += 1
        except ValueError:
            continue
    return count


def _dataset_rows(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset output path not found: {root}")
    return root, convert_traces_to_training_data(root, skip_invalid_lines=True)


def dataset_row_context(root: Path, row_idx: int) -> dict[str, Any]:
    root, rows = _dataset_rows(root)
    if row_idx < 0 or row_idx >= len(rows):
        raise IndexError(f"Dataset row {row_idx} is out of range.")
    row = rows[row_idx]
    source_path = _resolve_source_path(root, row)
    source_file: str | None = None
    if source_path is not None:
        try:
            source_file = source_path.relative_to(root).as_posix()
        except ValueError:
            source_file = source_path.name
    return {
        "row_idx": row_idx,
        "row": row,
        "preview": _row_preview(row),
        "source_file": source_file,
        "can_edit": source_path is not None,
        "can_delete": source_path is not None,
    }


def update_dataset_row(root: Path, row_idx: int, updated_row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(updated_row, dict):
        raise TypeError("Updated row must be a JSON object.")
    root, rows = _dataset_rows(root)
    if row_idx < 0 or row_idx >= len(rows):
        raise IndexError(f"Dataset row {row_idx} is out of range.")
    current_row = rows[row_idx]
    source_path = _resolve_source_path(root, current_row)
    if source_path is None:
        raise ValueError("This dataset row does not have a resolvable source file.")

    row_to_write = dict(updated_row)
    metadata = dict(row_to_write.get("metadata")) if isinstance(row_to_write.get("metadata"), dict) else {}
    metadata.setdefault("source_file", source_path.name)
    row_to_write["metadata"] = metadata

    source_rows = _read_trace_events(source_path)
    source_line = _row_source_line(current_row)
    if source_line is not None and source_line <= len(source_rows) and _structured_row(source_rows[source_line - 1]):
        metadata.setdefault("source_line", source_line)
        source_rows[source_line - 1] = row_to_write
        _write_jsonl_rows(source_path, source_rows)
        mode = "line"
    elif _source_usage_count(root, rows, source_path) == 1:
        _write_jsonl_rows(source_path, [row_to_write])
        mode = "file"
    else:
        raise ValueError("This source file maps to multiple dataset rows; refusing to rewrite it as one row.")

    try:
        source_file = source_path.relative_to(root).as_posix()
    except ValueError:
        source_file = source_path.name
    return {"row_idx": row_idx, "source_file": source_file, "mode": mode, "row": row_to_write}


def delete_dataset_row(root: Path, row_idx: int) -> dict[str, Any]:
    root, rows = _dataset_rows(root)
    if row_idx < 0 or row_idx >= len(rows):
        raise IndexError(f"Dataset row {row_idx} is out of range.")
    row = rows[row_idx]
    source_path = _resolve_source_path(root, row)
    if source_path is None:
        raise ValueError("This dataset row does not have a resolvable source file.")

    source_rows = _read_trace_events(source_path)
    source_line = _row_source_line(row)
    if source_line is not None and source_line <= len(source_rows) and _structured_row(source_rows[source_line - 1]):
        del source_rows[source_line - 1]
        _write_jsonl_rows(source_path, source_rows)
        mode = "line"
    elif _source_usage_count(root, rows, source_path) == 1:
        source_path.unlink()
        metadata_path = source_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata_path.unlink()
        mode = "file"
    else:
        raise ValueError("This source file maps to multiple dataset rows; refusing to delete it as one row.")

    try:
        source_file = source_path.relative_to(root).as_posix()
    except ValueError:
        source_file = source_path.name
    return {"row_idx": row_idx, "source_file": source_file, "mode": mode}


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
        rows = convert_traces_to_training_data(root, skip_invalid_lines=True)
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
        "errors": errors,
        "notes": [
            "Local preview uses Teich conversion directly. Hugging Face's hosted viewer adds Parquet-backed search, filtering, and SQL after upload."
        ],
    }
