"""Local agent session discovery and extraction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import re
import shutil
import sqlite3
from typing import Any, Literal

ExtractProvider = Literal["claude", "codex", "hermes", "pi"]


@dataclass(frozen=True)
class ExtractResult:
    provider: ExtractProvider
    destination_dir: Path
    copied_files: list[Path] = field(default_factory=list)
    source_paths: list[Path] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.copied_files)


def default_session_sources(provider: ExtractProvider, home: Path | None = None) -> list[Path]:
    """Return likely local session stores for a supported provider."""
    home = home or Path.home()
    if provider == "codex":
        return _existing_unique_paths(
            [
                _env_path("CODEX_HOME", "sessions"),
                home / ".codex" / "sessions",
            ]
        )
    if provider == "claude":
        return _existing_unique_paths(
            [
                _env_path("CLAUDE_CONFIG_DIR", "projects"),
                _env_path("CLAUDE_HOME", "projects"),
                home / ".claude" / "projects",
            ]
        )
    if provider == "pi":
        return _existing_unique_paths(
            [
                _env_path("PI_SESSION_DIR"),
                _env_path("PI_CODING_AGENT_DIR", "sessions"),
                home / ".pi" / "agent" / "sessions",
                home / ".pi" / "sessions",
            ]
        )
    if provider == "hermes":
        return _existing_unique_paths(
            [
                _env_path("HERMES_STATE_DB"),
                _env_path("HERMES_HOME", "state.db"),
                home / ".hermes" / "state.db",
            ]
        )
    raise ValueError(f"Unsupported extract provider: {provider}")


def extract_local_sessions(
    provider: ExtractProvider,
    *,
    output_dir: Path = Path("data"),
    sources: Iterable[Path] | None = None,
    home: Path | None = None,
    model_filter: str | None = None,
    clear_destination: bool = False,
) -> ExtractResult:
    """Extract local sessions for provider into output_dir."""
    resolved_sources = _existing_unique_paths(list(sources) if sources is not None else default_session_sources(provider, home))
    destination_dir = output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    if clear_destination:
        _clear_extract_destination(destination_dir)
    if provider == "hermes":
        copied_files = _extract_hermes_state_dbs(resolved_sources, destination_dir, model_filter=model_filter)
    else:
        copied_files = _extract_jsonl_session_files(
            provider,
            resolved_sources,
            destination_dir,
            model_filter=model_filter,
        )
    return ExtractResult(
        provider=provider,
        destination_dir=destination_dir,
        copied_files=copied_files,
        source_paths=resolved_sources,
    )


def _env_path(name: str, *parts: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    for part in parts:
        path /= part
    return path


def _existing_unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path is None:
            continue
        expanded = path.expanduser()
        if not expanded.exists():
            continue
        try:
            key = expanded.resolve()
        except OSError:
            key = expanded
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _clear_extract_destination(destination_dir: Path) -> None:
    """Remove stale extract artifacts while leaving unrelated files alone."""
    for path in destination_dir.glob("*.jsonl"):
        if path.is_file():
            path.unlink()
    for artifact_name in ("README.md", "tools.json"):
        artifact_path = destination_dir / artifact_name
        if artifact_path.is_file():
            artifact_path.unlink()
    for provider in ("claude", "codex", "hermes", "pi"):
        legacy_dir = destination_dir / _provider_output_name(provider)
        if legacy_dir.is_dir():
            shutil.rmtree(legacy_dir)
        elif legacy_dir.exists():
            legacy_dir.unlink()


def _provider_output_name(provider: ExtractProvider) -> str:
    if provider == "claude":
        return "claude-code"
    return provider


def _jsonl_files(source: Path) -> list[Path]:
    if source.is_file() and source.suffix == ".jsonl":
        return [source]
    if not source.is_dir():
        return []
    return sorted(path for path in source.rglob("*.jsonl") if path.is_file())


def _extract_jsonl_session_files(
    provider: ExtractProvider,
    sources: list[Path],
    destination_dir: Path,
    *,
    model_filter: str | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for source in sources:
        for path in _jsonl_files(source):
            destination = _unique_destination(destination_dir, path.name)
            if _copy_provider_jsonl(provider, path, destination, model_filter=model_filter):
                copied.append(destination)
    return copied


def _copy_provider_jsonl(
    provider: ExtractProvider,
    source: Path,
    destination: Path,
    *,
    model_filter: str | None = None,
) -> bool:
    events = _read_jsonl_dict_events(source)
    if events is None:
        if model_filter:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True
    if model_filter and not trace_matches_model(events, model_filter):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    try:
        shutil.copystat(source, destination)
    except OSError:
        pass
    return True


def _read_jsonl_dict_events(path: Path) -> list[dict[str, Any]] | None:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    return None
                events.append(event)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return events


def _write_jsonl_dict_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _unique_destination(destination_dir: Path, file_name: str) -> Path:
    destination = destination_dir / file_name
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = destination.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _extract_hermes_state_dbs(
    sources: list[Path],
    destination_dir: Path,
    *,
    model_filter: str | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for source in sources:
        state_dbs = [source] if source.is_file() else sorted(source.rglob("state.db"))
        for state_db in state_dbs:
            copied.extend(_export_hermes_state_db(state_db, destination_dir, model_filter=model_filter))
    return copied


def _export_hermes_state_db(
    state_db: Path,
    destination_dir: Path,
    *,
    model_filter: str | None = None,
) -> list[Path]:
    if not state_db.exists():
        return []
    connection = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if not _has_table(connection, "sessions") or not _has_table(connection, "messages"):
            return []
        session_rows = connection.execute("SELECT * FROM sessions ORDER BY started_at ASC, id ASC").fetchall()
        exported: list[Path] = []
        for session_row in session_rows:
            session_id = str(_sqlite_row_get(session_row, "id", "") or "")
            if not session_id:
                continue
            message_rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            events = _hermes_external_trace_events(session_row, message_rows)
            if model_filter and not trace_matches_model(events, model_filter):
                continue
            destination = _unique_destination(destination_dir, f"hermes-agent-{_safe_file_id(session_id)}.jsonl")
            _write_jsonl_dict_events(destination, events)
            exported.append(destination)
        return exported
    finally:
        connection.close()


def _has_table(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _sqlite_row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _safe_file_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()) or "session"


def trace_matches_model(events: Iterable[dict[str, Any]], model_filter: str) -> bool:
    """Return whether a trace has a model-identifying field matching model_filter."""
    needle = _normalize_model_text(model_filter)
    if not needle:
        return True
    return any(_value_has_matching_model(event, needle) for event in events)


def _value_has_matching_model(value: Any, needle: str, *, key: str | None = None) -> bool:
    if isinstance(value, dict):
        return any(_value_has_matching_model(item, needle, key=str(item_key)) for item_key, item in value.items())
    if isinstance(value, list):
        return any(_value_has_matching_model(item, needle, key=key) for item in value)
    if isinstance(value, str) and key and _is_model_key(key):
        return needle in _normalize_model_text(value)
    return False


def _is_model_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", key.lower())
    if normalized in {
        "model",
        "modelid",
        "modelname",
        "modelslug",
        "modelversion",
        "selectedmodel",
        "requestedmodel",
        "defaultmodel",
        "subagentmodel",
    }:
        return True
    return normalized.endswith("model") and normalized not in {"modelconfig"}


def _normalize_model_text(value: str) -> str:
    return value.strip().casefold()


def _timestamp(value: Any) -> str:
    if isinstance(value, (int, float)):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_or_original(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    if value.startswith("\x00json:"):
        value = value[len("\x00json:"):]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _hermes_external_trace_events(
    session_row: sqlite3.Row,
    message_rows: list[sqlite3.Row],
) -> list[dict[str, Any]]:
    session = dict(session_row)
    input_tokens = _sqlite_row_get(session_row, "input_tokens")
    output_tokens = _sqlite_row_get(session_row, "output_tokens")
    total_tokens = _sqlite_row_get(session_row, "total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    payload: dict[str, Any] = {
        **session,
        "source": "hermes-agent",
        "hermes_source": _sqlite_row_get(session_row, "source"),
        "timestamp": _timestamp(_sqlite_row_get(session_row, "started_at")),
        "model": _sqlite_row_get(session_row, "model"),
        "total_tokens": total_tokens,
        "estimated_cost_usd": _sqlite_row_get(session_row, "estimated_cost_usd"),
        "actual_cost_usd": _sqlite_row_get(session_row, "actual_cost_usd"),
    }
    model_config = _json_or_original(payload.get("model_config"))
    if model_config is not None:
        payload["model_config"] = model_config
    events: list[dict[str, Any]] = [
        {
            "timestamp": payload["timestamp"],
            "type": "external_session_meta",
            "payload": payload,
        }
    ]
    for message_row in message_rows:
        event = _hermes_external_message_event(message_row)
        if event is not None:
            events.append(event)
    return events


def _hermes_external_message_event(row: sqlite3.Row) -> dict[str, Any] | None:
    role = _sqlite_row_get(row, "role")
    if not isinstance(role, str) or not role.strip():
        return None
    content = _json_or_original(_sqlite_row_get(row, "content")) or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    event: dict[str, Any] = {
        "timestamp": _timestamp(_sqlite_row_get(row, "timestamp")),
        "type": "external_message",
        "role": role.strip(),
        "content": content,
    }
    for key in (
        "tool_call_id",
        "tool_calls",
        "token_count",
        "finish_reason",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ):
        value = _json_or_original(_sqlite_row_get(row, key))
        if value is not None and value != "":
            event[key] = value
    tool_name = _sqlite_row_get(row, "tool_name") or _sqlite_row_get(row, "name")
    if isinstance(tool_name, str) and tool_name.strip():
        event["name"] = tool_name.strip()
    return event
