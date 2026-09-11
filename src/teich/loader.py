from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Json, List, Value
from huggingface_hub import snapshot_download

from .converter import convert_traces_to_training_data


def _trace_directory(root: Path, split: str | None) -> Path:
    if split:
        candidate = root / split
        if candidate.is_dir():
            return candidate
    return root


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    try:
        return Dataset.from_list(rows, on_mixed_types="use_json")
    except TypeError as exc:
        if "on_mixed_types" not in str(exc):
            raise
    features = Features(
        {
            "prompt": Value("string"),
            "messages": List(Json()),
            "tools": List(Json()),
            "metadata": Json(),
        }
    )
    return Dataset.from_list(rows, features=features)


def _normalize_tools_snapshot(tools: Any) -> list[dict[str, Any]]:
    if isinstance(tools, list):
        return [tool for tool in tools if isinstance(tool, dict)]
    return []


def _load_tools_snapshot_from_readme(root: Path) -> list[dict[str, Any]]:
    readme_path = root / "README.md"
    if not readme_path.is_file():
        return []
    try:
        readme = readme_path.read_text(encoding="utf-8")
    except OSError:
        return []
    summary = "<summary>Training-ready tool schema snapshot</summary>"
    summary_index = readme.find(summary)
    if summary_index < 0:
        return []
    fence_start = readme.find("```json", summary_index)
    if fence_start < 0:
        return []
    json_start = readme.find("\n", fence_start)
    if json_start < 0:
        return []
    fence_end = readme.find("```", json_start + 1)
    if fence_end < 0:
        return []
    try:
        tools = json.loads(readme[json_start:fence_end].strip())
    except json.JSONDecodeError:
        return []
    return _normalize_tools_snapshot(tools)


def _load_tools_snapshot(root: Path) -> list[dict[str, Any]]:
    tools_from_readme = _load_tools_snapshot_from_readme(root)
    if tools_from_readme:
        return tools_from_readme
    candidates = [root / "tools.json"]
    if root.is_dir():
        candidates.extend(path for path in root.rglob("tools.json") if path.is_file())
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            tools = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        normalized = _normalize_tools_snapshot(tools)
        if normalized:
            return normalized
    return []


def _apply_tools_snapshot(rows: list[dict], tools: list[dict[str, Any]]) -> list[dict]:
    if not tools:
        return rows
    updated_rows: list[dict] = []
    for row in rows:
        row_tools = row.get("tools")
        if isinstance(row_tools, list) and row_tools:
            updated_rows.append(row)
            continue
        updated = dict(row)
        updated["tools"] = tools
        updated_rows.append(updated)
    return updated_rows


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _row_has_training_signal(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"assistant", "model"}:
            continue
        if _message_text(message.get("content")):
            return True
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return True
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def _filter_rows_with_training_signal(rows: list[dict]) -> list[dict]:
    return [row for row in rows if _row_has_training_signal(row)]


def trace_is_complete(row: dict[str, Any]) -> bool:
    messages = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(messages, list):
        return False
    relevant_roles = [
        message.get("role")
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"assistant", "model", "tool"}
    ]
    if not relevant_roles:
        return False
    return relevant_roles[-1] != "tool"


def _filter_complete_trace_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if trace_is_complete(row)]


def _resolve_hf_token(token: str | None, hf_token: str | None) -> str | None:
    if token is not None and hf_token is not None and token != hf_token:
        raise ValueError("Pass only one of token or hf_token, or pass the same value for both.")
    return token if token is not None else hf_token


def load_traces(
    source: str | Path,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    max_examples: int | None = None,
    drop_incomplete_traces: bool = True,
) -> Dataset:
    if max_examples is not None and (
        not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 0
    ):
        raise ValueError("max_examples must be a non-negative integer.")
    effective_token = _resolve_hf_token(token, hf_token)
    source_path = Path(source)
    if source_path.exists():
        root = source_path
    elif isinstance(source, Path):
        raise FileNotFoundError(f"Local trace path not found: {source_path}")
    else:
        root = Path(
            snapshot_download(
                repo_id=str(source),
                repo_type="dataset",
                revision=revision,
                token=effective_token,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_dir=str(local_dir) if local_dir is not None else None,
                allow_patterns=[
                    "*.jsonl", "**/*.jsonl", "*.jsonl.zstd", "**/*.jsonl.zstd",
                    "*.jsonl.zst", "**/*.jsonl.zst", "README.md", "tools.json", "**/tools.json",
                ],
            )
        )
    traces_dir = root if root.is_file() else _trace_directory(root, split)
    rows = convert_traces_to_training_data(traces_dir)
    if not rows:
        location = traces_dir if traces_dir != root else root
        if split and traces_dir == root and root.is_dir():
            raise ValueError(f"No trace files found in {location} for split '{split}'.")
        raise ValueError(f"No JSONL trace or training data files found in {location}.")
    rows = _filter_rows_with_training_signal(rows)
    if not rows:
        location = traces_dir if traces_dir != root else root
        raise ValueError(f"No trace or training data rows with assistant training signal found in {location}.")
    if drop_incomplete_traces:
        rows = _filter_complete_trace_rows(rows)
        if not rows:
            location = traces_dir if traces_dir != root else root
            raise ValueError(f"No complete trace or training data rows found in {location}.")
    snapshot_root = root if root.is_dir() else root.parent
    rows = _apply_tools_snapshot(rows, _load_tools_snapshot(snapshot_root))
    dataset = _dataset_from_rows(rows)
    if max_examples is not None:
        dataset = dataset.shuffle(seed=3407)
        limit = min(max_examples, dataset.num_rows)
        dataset = dataset.select(range(limit))
    return dataset
