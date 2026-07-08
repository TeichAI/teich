"""Local agent session discovery and extraction helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import re
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any, Literal

from .anonymize import TraceAnonymizer, anonymize_file
from .tool_schema import CURSOR_BUILTIN_TOOLS

ExtractProvider = Literal["claude", "codex", "cursor", "hermes", "pi"]
CURSOR_EXTRACTION_NOTICE = (
    "Cursor extraction may take a while depending on how much local Cursor data is being scanned."
)
CURSOR_RECORD_TERMS = (
    "aichat",
    "ai_chat",
    "aiService",
    "chat",
    "composer",
    "conversation",
)
CURSOR_SQLITE_TIMEOUT_SECONDS = 2.0
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ExtractResult:
    provider: ExtractProvider
    destination_dir: Path
    copied_files: list[Path] = field(default_factory=list)
    source_paths: list[Path] = field(default_factory=list)
    anonymize_totals: dict[str, int] | None = None

    @property
    def count(self) -> int:
        return len(self.copied_files)


class _InlineAnonymizer:
    """Anonymize traces as they are written, one TraceAnonymizer per file."""

    def __init__(self) -> None:
        self.totals: dict[str, int] = {"email": 0, "username": 0, "api_key": 0}

    def _add(self, counts: dict[str, int]) -> None:
        for key, count in counts.items():
            self.totals[key] = self.totals.get(key, 0) + count

    def copy_file(self, source: Path, destination: Path) -> None:
        report = anonymize_file(source, destination)
        self._add(report.replacements)

    def anonymize_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anonymizer = TraceAnonymizer()
        events = [anonymizer.anonymize_value(event) for event in events]
        self._add(anonymizer.counts)
        return events


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
    if provider == "cursor":
        return _existing_unique_paths(_cursor_default_sources(home))
    raise ValueError(f"Unsupported extract provider: {provider}")


def extract_local_sessions(
    provider: ExtractProvider,
    *,
    output_dir: Path = Path("data"),
    sources: Iterable[Path] | None = None,
    home: Path | None = None,
    model_filter: str | None = None,
    clear_destination: bool = False,
    progress: ProgressCallback | None = None,
    anonymize: bool = False,
) -> ExtractResult:
    """Extract local sessions for provider into output_dir."""
    source_candidates = list(sources) if sources is not None else default_session_sources(provider, home)
    resolved_sources = _existing_unique_paths(source_candidates)
    destination_dir = output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    if clear_destination:
        _clear_extract_destination(destination_dir)
    anonymizer = _InlineAnonymizer() if anonymize else None
    if provider == "hermes":
        copied_files = _extract_hermes_state_dbs(
            resolved_sources, destination_dir, model_filter=model_filter, anonymizer=anonymizer
        )
    elif provider == "cursor":
        copied_files = _extract_cursor_databases(
            resolved_sources,
            destination_dir,
            model_filter=model_filter,
            progress=progress,
            anonymizer=anonymizer,
        )
    else:
        copied_files = _extract_jsonl_session_files(
            provider,
            resolved_sources,
            destination_dir,
            model_filter=model_filter,
            anonymizer=anonymizer,
        )
    return ExtractResult(
        provider=provider,
        destination_dir=destination_dir,
        copied_files=copied_files,
        source_paths=resolved_sources,
        anonymize_totals=anonymizer.totals if anonymizer is not None else None,
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
        try:
            exists = expanded.exists()
        except OSError:
            continue
        if not exists:
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
    for path in destination_dir.rglob("*.jsonl"):
        if path.is_file():
            path.unlink()
    for artifact_name in ("README.md", "tools.json"):
        artifact_path = destination_dir / artifact_name
        if artifact_path.is_file():
            artifact_path.unlink()
    for provider in ("claude", "codex", "cursor", "hermes", "pi"):
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


def _running_in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        kernel_release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").casefold()
    except OSError:
        return False
    return "microsoft" in kernel_release or "wsl" in kernel_release


def _cursor_default_sources(home: Path) -> list[Path | None]:
    sources: list[Path | None] = [
        _env_path("CURSOR_GLOBAL_STORAGE_DB"),
        _env_path("CURSOR_WORKSPACE_STORAGE"),
        _env_path("CURSOR_USER_DATA_DIR", "globalStorage", "state.vscdb"),
        _env_path("CURSOR_USER_DATA_DIR", "workspaceStorage"),
        home / ".cursor" / "projects",
        home / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb",
        home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb",
        home / ".cursor-server" / "data" / "User" / "globalStorage" / "state.vscdb",
    ]
    sources.extend(_cursor_recent_workspace_state_dbs(home / ".config" / "Cursor" / "User" / "workspaceStorage"))
    sources.extend(
        _cursor_recent_workspace_state_dbs(
            home / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"
        )
    )
    sources.extend(_cursor_recent_workspace_state_dbs(home / ".cursor-server" / "data" / "User" / "workspaceStorage"))
    if not _running_in_wsl():
        appdata = os.environ.get("APPDATA")
        if appdata:
            roaming = Path(appdata).expanduser()
            cursor_user_dir = roaming / "Cursor" / "User"
            sources.extend(
                [
                    cursor_user_dir / "globalStorage" / "state.vscdb",
                    *_cursor_recent_workspace_state_dbs(cursor_user_dir / "workspaceStorage"),
                ]
            )
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            roaming = Path(userprofile).expanduser() / "AppData" / "Roaming"
            cursor_user_dir = roaming / "Cursor" / "User"
            sources.extend(
                [
                    cursor_user_dir / "globalStorage" / "state.vscdb",
                    *_cursor_recent_workspace_state_dbs(cursor_user_dir / "workspaceStorage"),
                ]
            )
    return sources


def _cursor_recent_workspace_state_dbs(workspace_storage_dir: Path) -> list[Path]:
    try:
        if not workspace_storage_dir.is_dir():
            return []
        state_dbs = [path for path in workspace_storage_dir.glob("*/state.vscdb") if path.is_file()]
    except OSError:
        return []
    return sorted(state_dbs, key=_safe_path_mtime, reverse=True)


def _safe_path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _cursor_state_dbs(source: Path) -> list[Path]:
    if source.is_file() and source.name.endswith((".vscdb", ".db", ".sqlite", ".sqlite3")):
        return [source]
    if not source.is_dir():
        return []
    try:
        candidates = [path for path in source.rglob("state.vscdb") if path.is_file()]
        seen = set(candidates)
        for path in source.rglob("*.vscdb"):
            if path.is_file() and path not in seen:
                seen.add(path)
                candidates.append(path)
    except OSError:
        return []
    return sorted(candidates, key=_safe_path_mtime, reverse=True)


def _extract_cursor_databases(
    sources: list[Path],
    destination_dir: Path,
    *,
    model_filter: str | None = None,
    progress: ProgressCallback | None = None,
    anonymizer: _InlineAnonymizer | None = None,
) -> list[Path]:
    indexed_rows: list[tuple[int, dict[str, Any]]] = []
    row_index = 0
    seen: set[Path] = set()
    for source in sources:
        state_dbs = _cursor_state_dbs(source)
        _emit_progress(
            progress,
            "extract_progress",
            f"Found {len(state_dbs)} Cursor database(s) under {source}.",
            source=str(source),
            databases=len(state_dbs),
        )
        for state_db in state_dbs:
            try:
                key = state_db.resolve()
            except OSError:
                key = state_db
            if key in seen:
                continue
            seen.add(key)
            _emit_progress(progress, "extract_progress", f"Scanning Cursor database {state_db}...", source=str(state_db))
            db_rows = _cursor_database_rows(state_db, progress=progress)
            for row in db_rows:
                clean_row = _sanitize_cursor_row(row)
                if clean_row is not None:
                    indexed_rows.append((row_index, clean_row))
                row_index += 1
            _emit_progress(
                progress,
                "extract_progress",
                f"Scanned Cursor database {state_db}: {len(db_rows)} candidate row(s).",
                source=str(state_db),
                rows=len(db_rows),
            )
    rows = _dedupe_cursor_rows(indexed_rows)
    if model_filter:
        rows = [row for row in rows if trace_matches_model([row], model_filter)]
    if not rows:
        return _extract_cursor_project_files(
            sources, destination_dir, model_filter=model_filter, anonymizer=anonymizer
        )
    copied: list[Path] = []
    for row_index, row in enumerate(rows, start=1):
        destination = _unique_destination(destination_dir, _cursor_session_file_name(row, row_index))
        _write_jsonl_dict_events(destination, _cursor_row_to_transcript_events(row, row_index), anonymizer=anonymizer)
        copied.append(destination)
    return copied


def _extract_cursor_project_files(
    sources: list[Path],
    destination_dir: Path,
    *,
    model_filter: str | None = None,
    anonymizer: _InlineAnonymizer | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for source in sources:
        for transcript in _cursor_project_transcript_files(source):
            events = _read_jsonl_dict_events(transcript)
            if events is None:
                continue
            if model_filter and not trace_matches_model(events, model_filter):
                continue
            project_dir = _cursor_project_dir_for_transcript(transcript)
            if project_dir is None:
                relative = transcript.name
            else:
                try:
                    relative = str(transcript.relative_to(project_dir.parent))
                except ValueError:
                    relative = transcript.name
            destination = _unique_nested_destination(destination_dir / relative)
            _write_jsonl_dict_events(
                destination, _cursor_native_transcript_events(events, transcript, project_dir), anonymizer=anonymizer
            )
            copied.append(destination)
    return copied


def _cursor_project_transcript_files(source: Path) -> list[Path]:
    if source.is_file() and source.suffix == ".jsonl" and "agent-transcripts" in source.parts:
        return [source]
    if not source.is_dir():
        return []
    try:
        if source.name == "agent-transcripts":
            return sorted(path for path in source.rglob("*.jsonl") if path.is_file())
        return sorted(path for path in source.rglob("agent-transcripts/**/*.jsonl") if path.is_file())
    except OSError:
        return []


def _cursor_project_dir_for_transcript(transcript: Path) -> Path | None:
    parts = transcript.parts
    if "agent-transcripts" not in parts:
        return None
    index = parts.index("agent-transcripts")
    if index == 0:
        return None
    return Path(*parts[:index])


def _cursor_native_transcript_events(
    events: list[dict[str, Any]],
    transcript: Path,
    project_dir: Path | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = [
        {
            "type": "cursor_session_meta",
            "source": "cursor",
            "session_id": transcript.stem,
            "project": project_dir.name if project_dir is not None else None,
            "source_file": str(transcript),
        }
    ]
    tools = _cursor_project_tools(project_dir) if project_dir is not None else []
    if tools:
        enriched.append({"type": "cursor_available_tools", "tools": tools})
    enriched.extend(events)
    return enriched


def _cursor_project_tools(project_dir: Path) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    tools_dir = project_dir / "mcps"
    if tools_dir.is_dir():
        for tool_path in sorted(tools_dir.glob("*/tools/*.json")):
            try:
                tool = json.loads(tool_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            normalized = _cursor_normalize_mcp_tool(tool)
            if normalized is not None:
                tools.append(normalized)
    return _cursor_merge_tools(_cursor_builtin_tools(), tools)


def _cursor_normalize_mcp_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    function: dict[str, Any] = {"name": name.strip()}
    description = tool.get("description")
    if isinstance(description, str) and description.strip():
        function["description"] = description.strip()
    arguments = tool.get("arguments")
    if isinstance(arguments, dict):
        function["parameters"] = arguments
    return {"type": "function", "function": function}


def _unique_nested_destination(destination: Path) -> Path:
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


def _emit_progress(
    progress: ProgressCallback | None,
    kind: str,
    text: str,
    **fields: Any,
) -> None:
    if progress is None:
        return
    progress({"kind": kind, "text": text, **fields})


def _extract_jsonl_session_files(
    provider: ExtractProvider,
    sources: list[Path],
    destination_dir: Path,
    *,
    model_filter: str | None = None,
    anonymizer: _InlineAnonymizer | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for source in sources:
        for path in _jsonl_files(source):
            destination = _unique_destination(destination_dir, path.name)
            if _copy_provider_jsonl(provider, path, destination, model_filter=model_filter, anonymizer=anonymizer):
                copied.append(destination)
    return copied


def _copy_provider_jsonl(
    provider: ExtractProvider,
    source: Path,
    destination: Path,
    *,
    model_filter: str | None = None,
    anonymizer: _InlineAnonymizer | None = None,
) -> bool:
    events = _read_jsonl_dict_events(source)
    if events is None:
        if model_filter:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        if anonymizer is not None:
            anonymizer.copy_file(source, destination)
        else:
            shutil.copy2(source, destination)
        return True
    if model_filter and not trace_matches_model(events, model_filter):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if anonymizer is not None:
        anonymizer.copy_file(source, destination)
        return True
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


def _write_jsonl_dict_events(
    path: Path,
    events: list[dict[str, Any]],
    anonymizer: _InlineAnonymizer | None = None,
) -> None:
    if anonymizer is not None:
        events = anonymizer.anonymize_events(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            line = line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            handle.write(line + "\n")


def _write_individual_jsonl_rows(
    destination_dir: Path,
    rows: list[dict[str, Any]],
    file_name_for_row: Callable[[dict[str, Any], int], str],
    anonymizer: _InlineAnonymizer | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for row_index, row in enumerate(rows, start=1):
        destination = _unique_destination(destination_dir, file_name_for_row(row, row_index))
        _write_jsonl_dict_events(destination, [row], anonymizer=anonymizer)
        copied.append(destination)
    return copied


def _safe_file_stem(value: Any, *, fallback: str, max_length: int = 96) -> str:
    text = str(value).strip() if value is not None else ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("._-")
    if not text:
        text = fallback
    if len(text) > max_length:
        text = text[:max_length].rstrip("._-")
    return text or fallback


def _jsonl_file_name(prefix: str, identifier: Any, row_index: int) -> str:
    fallback = f"{prefix}-{row_index:04d}"
    stem = _safe_file_stem(identifier, fallback=fallback)
    if not stem.lower().startswith(f"{prefix}-"):
        stem = f"{prefix}-{stem}"
    return f"{stem}.jsonl"


def _cursor_session_file_name(row: dict[str, Any], row_index: int) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    identifier = (
        metadata.get("cursor_composer_id")
        or metadata.get("cursor_key")
        or (
            f"{metadata.get('cursor_table')}-{metadata.get('cursor_row')}"
            if metadata.get("cursor_table") is not None or metadata.get("cursor_row") is not None
            else None
        )
        or row.get("prompt")
    )
    return _jsonl_file_name("cursor", identifier, row_index)


def _hermes_session_file_name(row: dict[str, Any], row_index: int) -> str:
    return _jsonl_file_name("hermes", row.get("id") or row.get("session_id") or row.get("title"), row_index)


def _cursor_row_to_transcript_events(row: dict[str, Any], row_index: int) -> list[dict[str, Any]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    session_id = (
        metadata.get("cursor_composer_id")
        or metadata.get("cursor_key")
        or f"cursor-{row_index:04d}"
    )
    events: list[dict[str, Any]] = [
        {
            "type": "cursor_session_meta",
            "source": "cursor",
            "session_id": str(session_id),
            "model": row.get("model") or metadata.get("model"),
            "source_db": metadata.get("source_db"),
            "cursor_scope": metadata.get("cursor_scope"),
            "cursor_workspace_id": metadata.get("cursor_workspace_id"),
            "cursor_table": metadata.get("cursor_table"),
            "cursor_key": metadata.get("cursor_key"),
            "cursor_storage_kind": metadata.get("cursor_storage_kind"),
        }
    ]
    tools = row.get("tools")
    if isinstance(tools, list) and tools:
        events.append({"type": "cursor_available_tools", "tools": tools})
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    for message in messages:
        if not isinstance(message, dict):
            continue
        event = _cursor_message_to_transcript_event(message)
        if event is not None:
            events.append(event)
    events.append({"type": "turn_ended", "status": "success"})
    return events


def _cursor_message_to_transcript_event(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    content = message.get("content")
    text = content if isinstance(content, str) else _cursor_string_content(content)
    if role == "user":
        content_blocks = _cursor_transcript_content_blocks(content)
        if not content_blocks:
            return None
        return {
            "role": "user",
            "message": {"content": content_blocks},
        }
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        reasoning = message.get("reasoning_content") or message.get("thinking")
        if isinstance(reasoning, str) and reasoning.strip():
            blocks.append({"type": "thinking", "text": reasoning.strip()})
        if text.strip():
            blocks.append({"type": "text", "text": text})
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                block = _cursor_tool_call_to_content_block(tool_call)
                if block is not None:
                    blocks.append(block)
        if not blocks:
            return None
        return {"role": "assistant", "message": {"content": blocks}}
    if role == "tool":
        tool_result: dict[str, Any] = {
            "type": "tool_result",
            "content": text,
        }
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            tool_result["tool_use_id"] = tool_call_id
        name = message.get("name")
        if isinstance(name, str) and name:
            tool_result["name"] = name
        if message.get("is_error") is True:
            tool_result["is_error"] = True
        return {"role": "user", "message": {"content": [tool_result]}}
    if role == "system" and text.strip():
        return {"role": "system", "message": {"content": [{"type": "text", "text": text}]}}
    return None


def _cursor_tool_call_to_content_block(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = _cursor_tool_call_arguments(tool_call)
    block: dict[str, Any] = {
        "type": "tool_use",
        "name": name,
        "input": arguments,
    }
    tool_call_id = tool_call.get("id")
    if isinstance(tool_call_id, str) and tool_call_id:
        block["id"] = tool_call_id
    return block


def _cursor_database_rows(state_db: Path, *, progress: ProgressCallback | None = None) -> list[dict[str, Any]]:
    try:
        with TemporaryDirectory(prefix="teich-cursor-") as temp_dir:
            snapshot_db = _snapshot_sqlite_db(state_db, Path(temp_dir))
            return _cursor_database_rows_from_snapshot(snapshot_db, state_db, progress=progress)
    except OSError:
        return _cursor_database_rows_from_snapshot(state_db, state_db, progress=progress)


def _snapshot_sqlite_db(state_db: Path, temp_dir: Path) -> Path:
    snapshot_db = temp_dir / state_db.name
    shutil.copy2(state_db, snapshot_db)
    for suffix in ("-wal", "-shm"):
        sidecar = state_db.with_name(state_db.name + suffix)
        if sidecar.is_file():
            shutil.copy2(sidecar, snapshot_db.with_name(snapshot_db.name + suffix))
    return snapshot_db


def _cursor_database_rows_from_snapshot(
    snapshot_db: Path,
    source_db: Path,
    *,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    try:
        connection = sqlite3.connect(
            f"file:{snapshot_db}?mode=ro",
            uri=True,
            timeout=CURSOR_SQLITE_TIMEOUT_SECONDS,
        )
    except sqlite3.Error:
        return []
    connection.row_factory = sqlite3.Row
    try:
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(CURSOR_SQLITE_TIMEOUT_SECONDS * 1000)}")
        except sqlite3.Error:
            pass
        table_names = [
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
            if isinstance(row["name"], str) and not str(row["name"]).startswith("sqlite_")
        ]
        rows: list[dict[str, Any]] = []
        for table_index, table_name in enumerate(table_names, start=1):
            _emit_progress(
                progress,
                "extract_progress",
                f"Scanning Cursor table {table_index}/{len(table_names)} in {source_db.name}: {table_name}",
                source=str(source_db),
                table=table_name,
                table_index=table_index,
                table_count=len(table_names),
            )
            rows.extend(_cursor_table_rows(connection, source_db, table_name, progress=progress))
            rows.extend(_cursor_composer_data_rows(connection, source_db, table_name, progress=progress))
        return rows
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def _cursor_table_rows(
    connection: sqlite3.Connection,
    state_db: Path,
    table_name: str,
    *,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    try:
        columns = [
            str(row["name"])
            for row in connection.execute(
                f"PRAGMA table_info({_sqlite_identifier(table_name)})"
            ).fetchall()
            if isinstance(row["name"], str)
        ]
    except sqlite3.Error:
        return []
    if not columns:
        return []
    table_is_candidate = _cursor_text_is_candidate(table_name) or any(
        _cursor_text_is_candidate(column) for column in columns
    )
    selected_columns = ", ".join(_sqlite_identifier(column) for column in columns)
    where_clause, params = _cursor_candidate_row_filter(columns)
    if not table_is_candidate and where_clause is None:
        return []
    query = f"SELECT {selected_columns} FROM {_sqlite_identifier(table_name)}"
    query_params: list[str] = []
    if where_clause is not None:
        query += f" WHERE {where_clause}"
        query_params = params
    try:
        raw_rows = connection.execute(query, query_params)
    except sqlite3.Error:
        return []

    rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(raw_rows, start=1):
        if row_index % 10000 == 0:
            _emit_progress(
                progress,
                "extract_progress",
                f"Scanned {row_index} row(s) in Cursor table {table_name}...",
                source=str(state_db),
                table=table_name,
                scanned_rows=row_index,
            )
        row_data = {
            column: _decode_cursor_sqlite_value(raw_row[column])
            for column in columns
        }
        key = _cursor_record_key(row_data)
        if not table_is_candidate and not _cursor_record_is_candidate(key, row_data):
            continue
        value = _cursor_primary_payload(row_data)
        messages = _cursor_extract_messages(value)
        explicit_prompt = _cursor_explicit_prompt(value)
        response = _cursor_response(messages) or _cursor_explicit_response(value)
        model = _cursor_find_model(value) or _cursor_find_model(row_data)
        tools = _cursor_extract_tools(value)
        if not messages and not response and not tools:
            continue
        prompt = _cursor_prompt(messages, key, explicit_prompt=explicit_prompt)
        metadata = {
            "trace_type": "cursor",
            "source": "cursor",
            "source_db": str(state_db),
            "cursor_scope": _cursor_db_scope(state_db),
            "cursor_workspace_id": _cursor_workspace_id(state_db),
            "cursor_table": table_name,
            "cursor_row": row_index,
            "cursor_key": key,
        }
        if model:
            metadata["model"] = model
        rows.append(
            {
                "messages": messages,
                "prompt": prompt,
                "response": response,
                "model": model,
                "tools": tools,
                "metadata": metadata,
                "raw_cursor": row_data,
            }
        )
    return rows


def _sanitize_cursor_row(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    clean_messages = [_cursor_clean_message(message) for message in messages]
    clean_messages = [message for message in clean_messages if message is not None]
    clean_messages = _cursor_drop_adjacent_duplicate_messages(clean_messages)
    if not any(_cursor_message_is_user_prompt(message) for message in clean_messages):
        fallback_prompt = _cursor_fallback_prompt(row)
        if fallback_prompt:
            clean_messages.insert(0, {"role": "user", "content": fallback_prompt})
    clean_messages = _cursor_trim_messages_to_trainable_conversation(clean_messages)
    if not clean_messages:
        return None
    if not any(_cursor_message_is_user_prompt(message) for message in clean_messages):
        return None
    if not any(_cursor_message_is_trainable_assistant(message) for message in clean_messages):
        return None

    clean_row = dict(row)
    clean_row["messages"] = clean_messages
    clean_row["prompt"] = _cursor_prompt(
        clean_messages,
        str((row.get("metadata") or {}).get("cursor_key") or ""),
        explicit_prompt=row.get("prompt") if isinstance(row.get("prompt"), str) else None,
    )
    clean_row["response"] = _cursor_response(clean_messages)
    merged_tools = _cursor_merge_tools(
        _cursor_builtin_tools(),
        row.get("tools") if isinstance(row.get("tools"), list) else [],
        _cursor_tools_from_messages(clean_messages),
    )
    clean_row["tools"] = merged_tools
    metadata = dict(row.get("metadata")) if isinstance(row.get("metadata"), dict) else {}
    metadata["cursor_message_count"] = len(clean_messages)
    clean_row["metadata"] = metadata
    return clean_row


def _cursor_fallback_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        return ""
    prompt = _cursor_string_content(prompt).strip()
    if not prompt or prompt.startswith(("composerData:", "bubbleId:")):
        return ""
    return prompt


def _cursor_clean_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    role = _cursor_normalize_role(message.get("role"))
    content = _cursor_content_with_preserved_media(message.get("content"))
    clean: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        reasoning_content = _cursor_string_content(message.get("reasoning_content") or message.get("thinking")).strip()
        if reasoning_content:
            clean["reasoning_content"] = reasoning_content
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_tool_calls = [_cursor_normalize_tool_call(tool_call) for tool_call in tool_calls]
            normalized_tool_calls = [tool_call for tool_call in normalized_tool_calls if tool_call is not None]
            if normalized_tool_calls:
                clean["tool_calls"] = normalized_tool_calls
    elif role == "tool":
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            clean["tool_call_id"] = tool_call_id.strip()
        name = message.get("name")
        if isinstance(name, str) and name.strip():
            clean["name"] = name.strip()
        if message.get("is_error") is True:
            clean["is_error"] = True
    if not _cursor_message_has_signal(clean):
        return None
    return clean


def _cursor_message_has_signal(message: dict[str, Any]) -> bool:
    if isinstance(message.get("content"), str) and message["content"].strip():
        return True
    if isinstance(message.get("content"), list):
        return any(_cursor_content_block_has_signal(block) for block in message["content"])
    if message.get("role") == "assistant":
        return bool(message.get("tool_calls")) or (
            isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip()
        )
    return False


def _cursor_message_is_user_prompt(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and _cursor_message_has_signal(message)


def _cursor_message_is_trainable_assistant(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    if isinstance(message.get("content"), str) and message["content"].strip():
        return True
    if isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip():
        return True
    return bool(message.get("tool_calls"))


def _cursor_drop_adjacent_duplicate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    previous_signature: str | None = None
    for message in messages:
        signature = _cursor_message_signature(message)
        if signature == previous_signature:
            continue
        deduped.append(message)
        previous_signature = signature
    return deduped


def _cursor_trim_messages_to_trainable_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_user_index = next(
        (index for index, message in enumerate(messages) if _cursor_message_is_user_prompt(message)),
        None,
    )
    if first_user_index is None:
        return []
    start_index = first_user_index
    while start_index > 0 and messages[start_index - 1].get("role") == "system":
        start_index -= 1
    messages = messages[start_index:]

    last_trainable_assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if _cursor_message_is_trainable_assistant(messages[index])
        ),
        None,
    )
    if last_trainable_assistant_index is None:
        return []
    return messages[: last_trainable_assistant_index + 1]


def _dedupe_cursor_rows(indexed_rows: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    kept: list[tuple[int, dict[str, Any], tuple[str, ...], int]] = []
    exact_sequences: set[tuple[str, ...]] = set()
    kept_by_signature: dict[str, list[tuple[str, ...]]] = {}
    for original_index, row in sorted(indexed_rows, key=lambda item: len(item[1].get("messages") or []), reverse=True):
        sequence = _cursor_row_sequence_tuple(row)
        if not sequence:
            continue
        if sequence in exact_sequences:
            continue
        candidates = kept_by_signature.get(sequence[0], [])
        if any(_cursor_sequence_tuple_contains(kept_sequence, sequence) for kept_sequence in candidates):
            continue
        exact_sequences.add(sequence)
        for signature in set(sequence):
            kept_by_signature.setdefault(signature, []).append(sequence)
        kept.append((original_index, row, sequence, len(row.get("messages") or [])))
    return [row for _, row, _, _ in sorted(kept, key=lambda item: item[0])]


def _cursor_row_sequence_tuple(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_cursor_message_signature(message) for message in row.get("messages") or [])


def _cursor_sequence_tuple_contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not haystack or not needle:
        return False
    if len(needle) > len(haystack):
        return False
    if haystack == needle:
        return True
    last_start = len(haystack) - len(needle)
    return any(haystack[start:start + len(needle)] == needle for start in range(last_start + 1))


def _cursor_message_signature(message: dict[str, Any]) -> str:
    return json.dumps(
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": message.get("tool_calls"),
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name"),
            "is_error": message.get("is_error"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _cursor_normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = name.strip()
    return {
        "id": call_id.strip(),
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


def _cursor_builtin_tools() -> list[dict[str, Any]]:
    return deepcopy(CURSOR_BUILTIN_TOOLS)


def _cursor_merge_tools(*tool_groups: list[Any]) -> list[dict[str, Any]]:
    merged_by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for tool_group in tool_groups:
        for tool in tool_group:
            normalized_tool = _cursor_normalize_tool(tool)
            if normalized_tool is None:
                continue
            name = normalized_tool.get("function", {}).get("name")
            if not isinstance(name, str):
                continue
            if name in merged_by_name:
                merged_by_name[name] = _cursor_merge_tool_schema(merged_by_name[name], normalized_tool)
                continue
            order.append(name)
            merged_by_name[name] = deepcopy(normalized_tool)
    return [merged_by_name[name] for name in order]


def _cursor_merge_tool_schema(existing_tool: dict[str, Any], incoming_tool: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(existing_tool)
    existing_function = merged.get("function")
    incoming_function = incoming_tool.get("function")
    if not isinstance(existing_function, dict) or not isinstance(incoming_function, dict):
        return merged
    if not existing_function.get("description") and isinstance(incoming_function.get("description"), str):
        existing_function["description"] = incoming_function["description"]
    incoming_parameters = incoming_function.get("parameters")
    if isinstance(incoming_parameters, dict):
        existing_parameters = existing_function.get("parameters")
        existing_function["parameters"] = _cursor_merge_json_schema(existing_parameters, incoming_parameters)
    return merged


def _cursor_composer_data_rows(
    connection: sqlite3.Connection,
    state_db: Path,
    table_name: str,
    *,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    key_column, value_column = _cursor_key_value_columns(connection, table_name)
    if key_column is None or value_column is None:
        return []
    try:
        composer_rows = connection.execute(
            (
                f"SELECT {_sqlite_identifier(key_column)}, {_sqlite_identifier(value_column)} "
                f"FROM {_sqlite_identifier(table_name)} "
                f"WHERE {_sqlite_identifier(key_column)} LIKE 'composerData:%' "
                f"ORDER BY {_sqlite_identifier(key_column)}"
            )
        )
    except sqlite3.Error:
        return []
    rows: list[dict[str, Any]] = []
    for row_index, composer_row in enumerate(composer_rows, start=1):
        if row_index % 1000 == 0:
            _emit_progress(
                progress,
                "extract_progress",
                f"Reconstructed {row_index} Cursor composer row(s) from {table_name}...",
                source=str(state_db),
                table=table_name,
                composer_rows=row_index,
            )
        key = str(composer_row[0])
        composer_data = _decode_cursor_sqlite_value(composer_row[1])
        if not isinstance(composer_data, dict):
            continue
        composer_id = str(composer_data.get("composerId") or key.split(":", 1)[1])
        headers = composer_data.get("fullConversationHeadersOnly")
        if not isinstance(headers, list) or not headers:
            continue
        bubble_keys: list[str] = []
        seen_bubble_keys: set[str] = set()
        for header in headers:
            if not isinstance(header, dict):
                continue
            bubble_id = header.get("bubbleId")
            if not isinstance(bubble_id, str) or not bubble_id:
                continue
            bubble_key = f"bubbleId:{composer_id}:{bubble_id}"
            if bubble_key in seen_bubble_keys:
                continue
            seen_bubble_keys.add(bubble_key)
            bubble_keys.append(bubble_key)
        bubble_map = _cursor_fetch_keyed_rows(
            connection,
            table_name,
            key_column,
            value_column,
            bubble_keys,
        )
        messages, used_bubbles = _cursor_messages_from_composer_headers(composer_id, headers, bubble_map)
        if not messages:
            continue
        response = _cursor_response(messages)
        if not response and not any(message.get("tool_calls") for message in messages):
            continue
        model = _cursor_find_model(composer_data) or _cursor_find_model(headers) or _cursor_find_model(used_bubbles)
        prompt = _cursor_prompt(messages, key, explicit_prompt=_cursor_explicit_prompt(composer_data))
        metadata = {
            "trace_type": "cursor",
            "source": "cursor",
            "source_db": str(state_db),
            "cursor_scope": _cursor_db_scope(state_db),
            "cursor_workspace_id": _cursor_workspace_id(state_db),
            "cursor_table": table_name,
            "cursor_row": row_index,
            "cursor_key": key,
            "cursor_storage_kind": "composerData",
            "cursor_composer_id": composer_id,
            "cursor_status": composer_data.get("status"),
            "cursor_headers": len(headers),
            "cursor_bubbles": len(used_bubbles),
        }
        if model:
            metadata["model"] = model
        rows.append(
            {
                "messages": messages,
                "prompt": prompt,
                "response": response,
                "model": model,
                "tools": _cursor_tools_from_messages(messages),
                "metadata": metadata,
                "raw_cursor": {
                    "composer_data": _cursor_relevant_composer_data(composer_data),
                    "bubble_ids": [
                        bubble["bubbleId"]
                        for bubble in used_bubbles
                        if isinstance(bubble.get("bubbleId"), str)
                    ],
                },
            }
        )
    return rows


def _cursor_fetch_keyed_rows(
    connection: sqlite3.Connection,
    table_name: str,
    key_column: str,
    value_column: str,
    keys: list[str],
) -> dict[str, Any]:
    if not keys:
        return {}
    rows: dict[str, Any] = {}
    chunk_size = 250
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start:start + chunk_size]
        placeholders = ", ".join("?" for _ in chunk)
        query = (
            f"SELECT {_sqlite_identifier(key_column)}, {_sqlite_identifier(value_column)} "
            f"FROM {_sqlite_identifier(table_name)} "
            f"WHERE {_sqlite_identifier(key_column)} IN ({placeholders})"
        )
        try:
            fetched = connection.execute(query, chunk).fetchall()
        except sqlite3.Error:
            continue
        for row in fetched:
            key = row[0]
            if isinstance(key, str):
                rows[key] = _decode_cursor_sqlite_value(row[1])
    return rows


def _cursor_key_value_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[str | None, str | None]:
    try:
        columns = [
            str(row["name"])
            for row in connection.execute(
                f"PRAGMA table_info({_sqlite_identifier(table_name)})"
            ).fetchall()
            if isinstance(row["name"], str)
        ]
    except sqlite3.Error:
        return None, None
    key_column = next((column for column in columns if _cursor_text_column_can_identify_record(column)), None)
    value_column = next(
        (
            column
            for column in columns
            if re.sub(r"[^a-z0-9]+", "", column.casefold()) in {"value", "data", "json", "payload", "state"}
        ),
        None,
    )
    return key_column, value_column


def _cursor_messages_from_composer_headers(
    composer_id: str,
    headers: list[Any],
    bubble_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    used_bubbles: list[dict[str, Any]] = []
    for header in headers:
        if not isinstance(header, dict):
            continue
        bubble_id = header.get("bubbleId")
        if not isinstance(bubble_id, str) or not bubble_id:
            continue
        bubble_key = f"bubbleId:{composer_id}:{bubble_id}"
        bubble = bubble_map.get(bubble_key)
        if not isinstance(bubble, dict):
            continue
        used_bubbles.append(bubble)
        messages.extend(_cursor_messages_from_bubble(header, bubble))
    return messages, used_bubbles


def _cursor_messages_from_bubble(header: dict[str, Any], bubble: dict[str, Any]) -> list[dict[str, Any]]:
    bubble_type = bubble.get("type", header.get("type"))
    content = _cursor_bubble_content(bubble, bubble_type)
    thinking = _cursor_bubble_thinking(bubble)
    tool_call = _cursor_bubble_tool_call(bubble)

    if bubble_type == 1:
        return [{"role": "user", "content": content}] if content else []

    messages: list[dict[str, Any]] = []
    if content or thinking or tool_call:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if thinking:
            message["reasoning_content"] = thinking
        if tool_call:
            message["tool_calls"] = [tool_call]
        messages.append(message)

    tool_result = _cursor_bubble_tool_result_message(bubble, tool_call)
    if tool_result is not None:
        messages.append(tool_result)
    return messages


def _cursor_bubble_content(bubble: dict[str, Any], bubble_type: Any) -> str:
    for key in ("text", "content", "message", "plainText", "rawText", "markdown"):
        text = _cursor_string_content(bubble.get(key)).strip()
        if text:
            return text
    if bubble_type == 1:
        for key in ("richText", "prompt", "query", "input", "userMessage"):
            text = _cursor_string_content(bubble.get(key)).strip()
            if text:
                return text
    return ""


def _cursor_bubble_thinking(bubble: dict[str, Any]) -> str:
    thinking = bubble.get("thinking")
    if isinstance(thinking, dict):
        return _cursor_string_content(thinking.get("text")).strip()
    return ""


def _cursor_bubble_tool_call(bubble: dict[str, Any]) -> dict[str, Any] | None:
    tool_data = bubble.get("toolFormerData")
    if not isinstance(tool_data, dict):
        return None
    name = tool_data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = tool_data.get("rawArgs")
    if not isinstance(arguments, str):
        arguments = tool_data.get("params")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    tool_call_id = tool_data.get("toolCallId") or bubble.get("bubbleId")
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        tool_call_id = name.strip()
    return {
        "id": tool_call_id.strip(),
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


def _cursor_bubble_tool_result_message(
    bubble: dict[str, Any],
    tool_call: dict[str, Any] | None,
) -> dict[str, Any] | None:
    tool_data = bubble.get("toolFormerData")
    if not isinstance(tool_data, dict):
        return None
    result = tool_data.get("result")
    error = tool_data.get("error")
    if result in (None, "") and error in (None, ""):
        return None
    tool_name = _cursor_nested_tool_call_name(tool_call) or tool_data.get("name")
    tool_call_id = tool_data.get("toolCallId") or (tool_call or {}).get("id") or bubble.get("bubbleId")
    content = _cursor_string_content(error if error not in (None, "") else result)
    message: dict[str, Any] = {
        "role": "tool",
        "content": content,
    }
    if isinstance(tool_call_id, str) and tool_call_id.strip():
        message["tool_call_id"] = tool_call_id.strip()
    if isinstance(tool_name, str) and tool_name.strip():
        message["name"] = tool_name.strip()
    if error not in (None, ""):
        message["is_error"] = True
    return message


def _cursor_nested_tool_call_name(tool_call: dict[str, Any] | None) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _cursor_tools_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed_arguments: dict[str, list[dict[str, Any]]] = {}
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            name = _cursor_nested_tool_call_name(tool_call)
            if not name:
                continue
            observed_arguments.setdefault(name, []).append(_cursor_tool_call_arguments(tool_call))
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": _cursor_schema_from_observed_arguments(arguments),
            },
        }
        for name, arguments in sorted(observed_arguments.items())
    ]


def _cursor_tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return arguments if isinstance(arguments, dict) else {}


def _cursor_schema_from_observed_arguments(arguments: list[dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required_counts: dict[str, int] = {}
    for argument_set in arguments:
        for key, value in argument_set.items():
            if not isinstance(key, str) or not key:
                continue
            required_counts[key] = required_counts.get(key, 0) + 1
            properties[key] = _cursor_merge_json_schema(properties.get(key), _cursor_json_schema_for_value(value))
    schema: dict[str, Any] = {"type": "object"}
    if properties:
        schema["properties"] = dict(sorted(properties.items()))
        required = sorted(
            key
            for key, count in required_counts.items()
            if count == len(arguments)
        )
        if required:
            schema["required"] = required
    return schema


def _cursor_json_schema_for_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schema: dict[str, Any] | None = None
        for item in value:
            item_schema = _cursor_merge_json_schema(item_schema, _cursor_json_schema_for_value(item))
        return {"type": "array", "items": item_schema or {}}
    if isinstance(value, dict):
        return _cursor_schema_from_observed_arguments([value])
    if value is None:
        return {"type": "null"}
    return {}


def _cursor_merge_json_schema(existing: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing, dict) or not existing:
        return incoming
    if not incoming:
        return existing
    existing_type = existing.get("type")
    incoming_type = incoming.get("type")
    if existing_type == incoming_type:
        if existing_type == "object":
            properties = dict(existing.get("properties")) if isinstance(existing.get("properties"), dict) else {}
            incoming_properties = incoming.get("properties")
            if isinstance(incoming_properties, dict):
                for key, value in incoming_properties.items():
                    properties[key] = _cursor_merge_json_schema(properties.get(key), value)
            merged = {"type": "object"}
            if properties:
                merged["properties"] = dict(sorted(properties.items()))
            existing_required = (
                {item for item in existing.get("required") if isinstance(item, str)}
                if isinstance(existing.get("required"), list)
                else None
            )
            incoming_required = (
                {item for item in incoming.get("required") if isinstance(item, str)}
                if isinstance(incoming.get("required"), list)
                else None
            )
            if existing_required is None:
                required = sorted(incoming_required or set())
            elif incoming_required is None:
                required = sorted(existing_required)
            else:
                required = sorted(existing_required & incoming_required)
            if required:
                merged["required"] = required
            for key in ("additionalProperties", "anyOf", "oneOf", "description", "enum"):
                if key in existing:
                    merged[key] = existing[key]
                elif key in incoming:
                    merged[key] = incoming[key]
            return merged
        if existing_type == "array":
            return {
                "type": "array",
                "items": _cursor_merge_json_schema(existing.get("items"), incoming.get("items") or {}),
            }
        return existing
    types: list[str] = []
    for schema_type in (existing_type, incoming_type):
        if isinstance(schema_type, list):
            types.extend(str(item) for item in schema_type)
        elif isinstance(schema_type, str):
            types.append(schema_type)
    return {"type": sorted(set(types))} if types else {}


def _cursor_relevant_composer_data(composer_data: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "_v",
        "composerId",
        "status",
        "createdAt",
        "lastUpdatedAt",
        "fullConversationHeadersOnly",
        "capabilities",
        "codebaseSearchSettings",
    )
    return {key: composer_data[key] for key in keep_keys if key in composer_data}


def _sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _cursor_candidate_row_filter(columns: list[str]) -> tuple[str | None, list[str]]:
    key_columns = [
        column
        for column in columns
        if _cursor_text_column_can_identify_record(column)
    ]
    if not key_columns:
        return None, []
    terms = [f"%{term.casefold()}%" for term in CURSOR_RECORD_TERMS]
    clauses: list[str] = []
    params: list[str] = []
    for column in key_columns:
        column_expr = f"LOWER(CAST({_sqlite_identifier(column)} AS TEXT))"
        for term in terms:
            clauses.append(f"{column_expr} LIKE ?")
            params.append(term)
    return " OR ".join(clauses), params


def _cursor_text_column_can_identify_record(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", column.casefold())
    return normalized in {"id", "key", "itemkey", "name"} or normalized.endswith("key")


def _decode_cursor_sqlite_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "bytes", "size": len(value)}
    decode_depth = 0
    while isinstance(value, str) and decode_depth < 5:
        stripped = value.strip()
        if stripped.startswith("\x00json:"):
            stripped = stripped[len("\x00json:"):].strip()
        if not stripped.startswith(("{", "[", '"')):
            break
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            break
        if decoded == value:
            break
        value = decoded
        decode_depth += 1
    return value


def _cursor_record_key(row_data: dict[str, Any]) -> str:
    for key_name in ("key", "name", "id", "itemKey", "item_key"):
        value = row_data.get(key_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _cursor_primary_payload(row_data: dict[str, Any]) -> Any:
    for key_name in ("value", "data", "json", "payload", "state"):
        value = row_data.get(key_name)
        if value is not None:
            return value
    return row_data


def _cursor_text_is_candidate(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold()
    return any(term.casefold() in normalized for term in CURSOR_RECORD_TERMS)


def _cursor_record_is_candidate(key: str, value: Any) -> bool:
    if _cursor_text_is_candidate(key):
        return True
    return _cursor_value_has_candidate_shape(value)


def _cursor_value_has_candidate_shape(value: Any) -> bool:
    if isinstance(value, dict):
        if any(_cursor_text_is_candidate(key) for key in value):
            return True
        if any(key in value for key in ("messages", "conversation", "conversations", "bubbles", "toolCalls")):
            return True
        return any(_cursor_value_has_candidate_shape(item) for item in value.values())
    if isinstance(value, list):
        return _cursor_messages_from_list(value) is not None or any(
            _cursor_value_has_candidate_shape(item) for item in value
        )
    return False


def _cursor_extract_messages(value: Any) -> list[dict[str, Any]]:
    message_list = _cursor_find_message_list(value)
    if not message_list:
        return []
    messages: list[dict[str, Any]] = []
    for item in message_list:
        message = _cursor_normalize_message(item)
        if message is not None:
            messages.append(message)
    return messages


def _cursor_find_message_list(value: Any) -> list[Any] | None:
    direct = _cursor_messages_from_list(value)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        preferred_keys = (
            "messages",
            "conversation",
            "conversations",
            "bubbles",
            "chat",
            "composer",
            "turns",
            "items",
        )
        for key in preferred_keys:
            found = _cursor_find_message_list(value.get(key))
            if found is not None:
                return found
        for item in value.values():
            found = _cursor_find_message_list(item)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _cursor_find_message_list(item)
            if found is not None:
                return found
    return None


def _cursor_messages_from_list(value: Any) -> list[Any] | None:
    if not isinstance(value, list) or not value:
        return None
    content_keys = {
        "content",
        "input",
        "markdown",
        "message",
        "parts",
        "plainText",
        "prompt",
        "query",
        "rawText",
        "richText",
        "text",
        "toolCalls",
        "tool_calls",
        "userMessage",
        "value",
    }
    message_like = [
        item
        for item in value
        if isinstance(item, dict)
        and (
            any(key in item for key in ("role", "from", "speaker", "type", "author"))
            and any(key in item for key in content_keys)
        )
    ]
    return value if message_like else None


def _cursor_normalize_message(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    role = _cursor_normalize_role(
        item.get("role")
        or item.get("from")
        or item.get("speaker")
        or item.get("author")
        or item.get("type")
    )
    content = _cursor_message_content(item)
    tool_calls = item.get("toolCalls") or item.get("tool_calls")
    if not content and not tool_calls:
        return None
    message: dict[str, Any] = {"role": role, "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _cursor_normalize_role(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"user", "human", "request", "prompt"}:
        return "user"
    if normalized in {"assistant", "ai", "agent", "gpt", "bot", "response"}:
        return "assistant"
    if normalized in {"tool", "tool_result", "toolresult", "function"}:
        return "tool"
    if normalized == "system":
        return "system"
    return "assistant" if "assistant" in normalized or "ai" in normalized else "user"


def _cursor_message_content(item: dict[str, Any]) -> Any:
    value = _cursor_message_content_value(item)
    return _cursor_content_with_preserved_media(value)


def _cursor_message_content_value(item: dict[str, Any]) -> Any:
    for key in (
        "content",
        "text",
        "message",
        "value",
        "markdown",
        "rawText",
        "richText",
        "plainText",
        "prompt",
        "query",
        "input",
        "userMessage",
    ):
        value = item.get(key)
        text = _cursor_string_content(value)
        if text:
            return value
        blocks = _cursor_content_blocks_with_media(value)
        if blocks is not None:
            return value
    return item.get("parts")


def _cursor_string_content(value: Any) -> str:
    if isinstance(value, str):
        rich_text = _cursor_rich_text_content(value)
        if rich_text is not None:
            return rich_text
        return value
    if isinstance(value, list):
        parts = [_cursor_string_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if _cursor_is_media_content_block(value):
            return ""
        rich_text = _cursor_rich_text_content(value)
        if rich_text is not None:
            return rich_text
        for key in ("text", "content", "value", "message"):
            text = _cursor_string_content(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _cursor_content_with_preserved_media(value: Any) -> str | list[dict[str, Any]]:
    blocks = _cursor_content_blocks_with_media(value)
    if blocks is not None:
        return blocks
    return _cursor_string_content(value).strip()


def _cursor_content_blocks_with_media(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        return None
    items = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
    blocks: list[dict[str, Any]] = []
    has_media = False
    for item in items:
        if isinstance(item, str):
            text = _cursor_rich_text_content(item) or item
            if text.strip():
                blocks.append({"type": "text", "text": text})
            continue
        if not isinstance(item, dict):
            continue
        if _cursor_is_media_content_block(item):
            blocks.append(deepcopy(item))
            has_media = True
            continue
        text = _cursor_string_content(item).strip()
        if text:
            blocks.append({"type": "text", "text": text})
    return blocks if has_media else None


def _cursor_transcript_content_blocks(value: Any) -> list[dict[str, Any]]:
    blocks = _cursor_content_blocks_with_media(value)
    if blocks is not None:
        return blocks
    text = _cursor_string_content(value).strip()
    return [{"type": "text", "text": text}] if text else []


def _cursor_content_block_has_signal(block: Any) -> bool:
    if isinstance(block, str):
        return bool(block.strip())
    if not isinstance(block, dict):
        return False
    if _cursor_is_media_content_block(block):
        return True
    return bool(_cursor_string_content(block).strip())


def _cursor_is_media_content_block(value: dict[str, Any]) -> bool:
    block_type = value.get("type")
    if isinstance(block_type, str) and block_type.lower() in {"image", "input_image", "image_url"}:
        return True
    image_url = value.get("image_url")
    if isinstance(image_url, str) and image_url:
        return True
    if isinstance(image_url, dict) and any(isinstance(image_url.get(key), str) for key in ("url", "data")):
        return True
    for key in ("url", "uri", "data", "data_url", "dataUrl"):
        item = value.get(key)
        if isinstance(item, str) and item.startswith("data:image/"):
            return True
    media_type = _cursor_media_type(value)
    if isinstance(media_type, str) and media_type.lower().startswith("image/"):
        return True
    source = value.get("source")
    if isinstance(source, dict):
        source_type = source.get("type")
        source_media_type = _cursor_media_type(source)
        if isinstance(source_media_type, str) and source_media_type.lower().startswith("image/"):
            return True
        if isinstance(source_type, str) and source_type.lower() == "base64" and isinstance(source.get("data"), str):
            return True
    return False


def _cursor_media_type(value: dict[str, Any]) -> str | None:
    for key in ("media_type", "mime_type", "mimeType", "mime"):
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _cursor_rich_text_content(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        candidates = [stripped]
        if '\\"' in stripped and stripped.startswith(("{\\", "[\\")):
            candidates.append(stripped.replace('\\"', '"'))
        value = None
        for candidate in candidates:
            if not candidate.startswith(("{", "[")) or not any(
                term in candidate for term in ('"root"', '"children"', '"type"')
            ):
                continue
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            return None
    if isinstance(value, dict):
        root = value.get("root")
        if isinstance(root, dict):
            return _cursor_rich_text_node_content(root).strip()
        if _cursor_looks_like_rich_text_node(value):
            return _cursor_rich_text_node_content(value).strip()
    if isinstance(value, list) and any(_cursor_looks_like_rich_text_node(item) for item in value):
        parts = [_cursor_rich_text_node_content(item).strip() for item in value]
        return "\n".join(part for part in parts if part)
    return None


def _cursor_looks_like_rich_text_node(value: Any) -> bool:
    return isinstance(value, dict) and (
        "root" in value
        or ("children" in value and "type" in value)
        or (value.get("type") == "text" and "text" in value)
    )


def _cursor_rich_text_node_content(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    node_type = str(node.get("type") or "").casefold()
    if node_type == "text":
        text = node.get("text")
        return text if isinstance(text, str) else ""
    if node_type in {"linebreak", "line-break"}:
        return "\n"
    children = node.get("children")
    if not isinstance(children, list):
        return ""
    parts = [_cursor_rich_text_node_content(child) for child in children]
    parts = [part for part in parts if part]
    if node_type in {"root", "list", "table", "tablebody", "tablerow"}:
        return "\n".join(part.strip() for part in parts if part.strip())
    return "".join(parts).strip()


def _cursor_prompt(messages: list[dict[str, Any]], key: str, *, explicit_prompt: str | None = None) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str) and message["content"].strip():
            return message["content"].strip()
    if explicit_prompt:
        return explicit_prompt
    return key


def _cursor_explicit_prompt(value: Any) -> str | None:
    for prompt_key in ("prompt", "query", "input", "userMessage", "text", "title", "name"):
        prompt = _cursor_find_string_key(value, prompt_key)
        if prompt:
            return prompt
    return None


def _cursor_response(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str) and message["content"].strip():
            return message["content"]
    return ""


def _cursor_explicit_response(value: Any) -> str:
    for response_key in ("response", "answer", "assistantMessage", "output"):
        response = _cursor_find_string_key(value, response_key)
        if response:
            return response
    return ""


def _cursor_find_model(value: Any) -> str | None:
    candidates = _cursor_model_candidates(value)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _cursor_model_candidates(
    value: Any,
    *,
    key: str | None = None,
    in_model_context: bool = False,
) -> list[tuple[int, str]]:
    key_is_model = key is not None and _is_model_key(key)
    model_context = in_model_context or key_is_model or (key is not None and _is_model_context_key(key))
    if isinstance(value, dict):
        candidates: list[tuple[int, str]] = []
        for item_key, item in value.items():
            candidates.extend(
                _cursor_model_candidates(item, key=str(item_key), in_model_context=model_context)
            )
        return candidates
    if isinstance(value, list):
        candidates = []
        for item in value:
            candidates.extend(_cursor_model_candidates(item, key=key, in_model_context=model_context))
        return candidates
    if not isinstance(value, str):
        return []
    text = _cursor_string_content(value).strip()
    if not text:
        return []
    if key_is_model:
        return [(0, text)]
    if not model_context or not _string_looks_like_model_value(text):
        return []
    normalized_key = re.sub(r"[^a-z0-9]+", "", (key or "").casefold())
    if normalized_key in {"id", "name", "slug", "displayname", "display", "title"}:
        return [(1, text)]
    return [(2, text)]


def _cursor_find_string_key(value: Any, target_key: str) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == target_key and isinstance(item, str) and item.strip():
                text = _cursor_string_content(item).strip()
                if text:
                    return text
        for item in value.values():
            found = _cursor_find_string_key(item, target_key)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _cursor_find_string_key(item, target_key)
            if found:
                return found
    return None


def _cursor_extract_tools(value: Any) -> list[dict[str, Any]]:
    tools = _cursor_find_key_value(value, "tools")
    if isinstance(tools, list):
        normalized = [_cursor_normalize_tool(tool) for tool in tools]
        return [tool for tool in normalized if tool is not None]
    return []


def _cursor_normalize_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    function_schema: dict[str, Any] = {"name": name.strip()}
    description = tool.get("description")
    if isinstance(description, str) and description.strip():
        function_schema["description"] = description.strip()
    parameters = tool.get("parameters") or tool.get("input_schema") or tool.get("schema")
    if isinstance(parameters, dict):
        function_schema["parameters"] = parameters
    return {"type": "function", "function": function_schema}


def _cursor_find_key_value(value: Any, target_key: str) -> Any:
    if isinstance(value, dict):
        if target_key in value:
            return value[target_key]
        for item in value.values():
            found = _cursor_find_key_value(item, target_key)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _cursor_find_key_value(item, target_key)
            if found is not None:
                return found
    return None


def _cursor_db_scope(state_db: Path) -> str:
    parts = {part.casefold() for part in state_db.parts}
    if "globalstorage" in parts:
        return "global"
    if "workspacestorage" in parts:
        return "workspace"
    return "unknown"


def _cursor_workspace_id(state_db: Path) -> str | None:
    if state_db.parent.name and state_db.parent.name != "globalStorage":
        return state_db.parent.name
    return None


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
    anonymizer: _InlineAnonymizer | None = None,
) -> list[Path]:
    copied: list[Path] = []
    for source in sources:
        state_dbs = [source] if source.is_file() else sorted(source.rglob("state.db"))
        for state_db in state_dbs:
            rows = _hermes_state_db_session_rows(state_db, model_filter=model_filter)
            copied.extend(
                _write_individual_jsonl_rows(destination_dir, rows, _hermes_session_file_name, anonymizer=anonymizer)
            )
    return copied


def _hermes_state_db_session_rows(
    state_db: Path,
    *,
    model_filter: str | None = None,
) -> list[dict[str, Any]]:
    if not state_db.exists():
        return []
    connection = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if not _has_table(connection, "sessions") or not _has_table(connection, "messages"):
            return []
        session_rows = connection.execute("SELECT * FROM sessions ORDER BY started_at ASC, id ASC").fetchall()
        exported: list[dict[str, Any]] = []
        for session_row in session_rows:
            session_id = str(_sqlite_row_get(session_row, "id", "") or "")
            if not session_id:
                continue
            message_rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            messages = [_hermes_message_export(message_row) for message_row in message_rows]
            row = _hermes_session_export(session_row, messages, state_db)
            if model_filter and not trace_matches_model([row], model_filter):
                continue
            exported.append(row)
        return exported
    finally:
        connection.close()


def _hermes_message_export(row: sqlite3.Row) -> dict[str, Any]:
    message = dict(row)
    message["content"] = _json_or_original(message.get("content")) or ""
    for key in (
        "tool_calls",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ):
        value = _json_or_original(message.get(key))
        if value is not None and value != "":
            message[key] = value
    return message


def _hermes_session_export(
    row: sqlite3.Row,
    messages: list[dict[str, Any]],
    state_db: Path,
) -> dict[str, Any]:
    session = dict(row)
    hermes_source = _sqlite_row_get(row, "source")
    input_tokens = _sqlite_row_get(row, "input_tokens")
    output_tokens = _sqlite_row_get(row, "output_tokens")
    total_tokens = _sqlite_row_get(row, "total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    actual_cost = _sqlite_row_get(row, "actual_cost_usd")
    estimated_cost = _sqlite_row_get(row, "estimated_cost_usd")
    total_cost = actual_cost if actual_cost is not None else estimated_cost
    if total_cost is None:
        total_cost = _sqlite_row_get(row, "total_cost")
    session.update(
        {
            "source": hermes_source or "cli",
            "timestamp": _timestamp(_sqlite_row_get(row, "started_at")),
            "teich_export_status": "completed",
            "teich_partial": False,
            "hermes_source": hermes_source,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "actual_cost_usd": actual_cost,
            "total_cost": total_cost,
            "source_db": str(state_db),
            "messages": messages,
        }
    )
    model_config = _json_or_original(session.get("model_config"))
    if model_config is not None:
        session["model_config"] = model_config
    return session


def _has_table(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _sqlite_row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def trace_matches_model(events: Iterable[dict[str, Any]], model_filter: str) -> bool:
    """Return whether a trace has a model-identifying field matching model_filter."""
    needle = _normalize_model_text(model_filter)
    if not needle:
        return True
    return any(_value_has_matching_model(event, needle) for event in events)


def _value_has_matching_model(
    value: Any,
    needle: str,
    *,
    key: str | None = None,
    in_model_context: bool = False,
) -> bool:
    key_is_model = key is not None and _is_model_key(key)
    model_context = in_model_context or key_is_model or (key is not None and _is_model_context_key(key))
    if isinstance(value, dict):
        return any(
            _value_has_matching_model(item, needle, key=str(item_key), in_model_context=model_context)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_value_has_matching_model(item, needle, key=key, in_model_context=model_context) for item in value)
    if isinstance(value, str):
        normalized = _normalize_model_text(value)
        if key_is_model:
            return needle in normalized
        if model_context and _string_looks_like_model_value(value):
            return needle in normalized
    return False


def _is_model_context_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", key.lower())
    return normalized in {
        "activemodel",
        "activemodelinfo",
        "activemodelmetadata",
        "availablemodels",
        "currentmodel",
        "currentmodelinfo",
        "currentmodelmetadata",
        "defaultmodel",
        "modelconfig",
        "modelconfiguration",
        "modeldata",
        "modeldetails",
        "modelinfo",
        "modellist",
        "modelmap",
        "modelmetadata",
        "modelsettings",
        "models",
        "requestedmodel",
        "selectedmodel",
        "selectedmodelinfo",
        "selectedmodelmetadata",
        "selectedmodels",
        "subagentmodel",
    }


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
    if normalized.endswith("model") and normalized not in {"modelconfig"}:
        return True
    return "model" in normalized and normalized.endswith(("id", "name", "slug", "version"))


def _string_looks_like_model_value(value: str) -> bool:
    normalized = _normalize_model_text(value)
    if not normalized or len(normalized) > 200:
        return False
    model_terms = (
        "anthropic",
        "claude",
        "cursor",
        "deepseek",
        "fable",
        "gemini",
        "gpt",
        "grok",
        "haiku",
        "llama",
        "mistral",
        "o1",
        "o3",
        "o4",
        "opus",
        "qwen",
        "sonnet",
    )
    return "/" in normalized or "-" in normalized or any(term in normalized for term in model_terms)


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
