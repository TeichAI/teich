from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .converter import convert_traces_to_training_data

README_SAMPLE_MAX_CHARS = 4_000
README_SAMPLE_STRING_MAX_CHARS = 600
README_SAMPLE_MAX_ITEMS = 6
README_SAMPLE_MAX_DEPTH = 5
README_INLINE_TOOLS_MAX_CHARS = 80_000
TEICH_TRAINING_DOCS_URL = "https://github.com/TeichAI/teich/blob/main/docs/training.md"
TEICH_PREPARE_DOCS_URL = "https://github.com/TeichAI/teich/blob/main/docs/prepare-data.md"
EXTRACTION_PROVIDERS = {"claude", "codex", "cursor", "hermes", "pi"}
NON_DATA_README_SCAN_DIR_NAMES = {"failures", "partials", "sandbox", "__pycache__"}


def normalize_extraction_provider(provider: str | None) -> str | None:
    if not isinstance(provider, str):
        return None
    normalized = provider.strip().lower().replace("_", "-")
    if normalized == "claude-code":
        normalized = "claude"
    return normalized if normalized in EXTRACTION_PROVIDERS else None


def extraction_readme_tags(provider: str) -> list[str]:
    normalized = normalize_extraction_provider(provider) or provider.strip().lower().replace("_", "-")
    ordered_tags = ["agent-traces", "format:agent-traces", normalized, "distillation", "teich"]
    tags: list[str] = []
    seen: set[str] = set()
    for tag in ordered_tags:
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def extraction_provider_from_existing_readme(readme_path: Path) -> str | None:
    try:
        text = readme_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"\bteich\s+extract\s+([A-Za-z0-9_-]+)\b", text)
    if match:
        return normalize_extraction_provider(match.group(1))
    return None


def extraction_provider_from_dataset_rows(traces_dir: Path) -> str | None:
    if not traces_dir.exists():
        return None
    for trace_file in sorted(traces_dir.rglob("*.jsonl")):
        if not trace_file.is_file():
            continue
        try:
            relative_parts = trace_file.relative_to(traces_dir).parts
        except ValueError:
            relative_parts = trace_file.parts
        if any(part in NON_DATA_README_SCAN_DIR_NAMES for part in relative_parts):
            continue
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    if line_index >= 100:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metadata = row.get("metadata") if isinstance(row, dict) else None
                    if not isinstance(metadata, dict):
                        metadata = {}
                    if metadata.get("source") == "cursor" or any(
                        key in metadata for key in ("cursor_scope", "cursor_table", "cursor_workspace_id")
                    ):
                        return "cursor"
        except OSError:
            continue
    return None


def extraction_provider_for_existing_dataset(traces_dir: Path) -> str | None:
    return extraction_provider_from_existing_readme(traces_dir / "README.md") or extraction_provider_from_dataset_rows(
        traces_dir
    )


def _path_is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _merge_tool_parameters(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    object_schemas = [schema for schema in schemas if isinstance(schema, dict) and schema]
    if not object_schemas:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    if len(object_schemas) == 1:
        return object_schemas[0]
    properties: dict[str, list[dict[str, Any]]] = {}
    required_sets: list[set[str]] = []
    additional_properties = False
    for schema in object_schemas:
        schema_properties = schema.get("properties")
        if isinstance(schema_properties, dict):
            for key, value in schema_properties.items():
                if isinstance(value, dict):
                    properties.setdefault(key, []).append(value)
        required = schema.get("required")
        if isinstance(required, list):
            required_sets.append({item for item in required if isinstance(item, str)})
        else:
            required_sets.append(set())
        if schema.get("additionalProperties", True) is not False:
            additional_properties = True
    merged_properties: dict[str, dict[str, Any]] = {}
    for key, values in sorted(properties.items()):
        unique_values: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            identity = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if identity in seen:
                continue
            seen.add(identity)
            unique_values.append(value)
        if len(unique_values) == 1:
            merged_properties[key] = unique_values[0]
        else:
            merged_properties[key] = {"anyOf": unique_values}
    merged: dict[str, Any] = {
        "type": "object",
        "properties": merged_properties,
        "additionalProperties": additional_properties,
    }
    if required_sets:
        required = sorted(set.intersection(*required_sets))
        if required:
            merged["required"] = required
    return merged


def _dataset_tools(trace_files: Iterable[Path]) -> list[dict[str, Any]]:
    merged_by_name: dict[str, dict[str, Any]] = {}
    for trace_file in trace_files:
        try:
            examples = convert_traces_to_training_data(trace_file, skip_invalid_lines=True)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for example in examples:
            tools = example.get("tools") if isinstance(example, dict) else None
            if not isinstance(tools, list):
                continue
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                function = tool.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                entry = merged_by_name.setdefault(name, {"type": "function", "function": {"name": name}})
                merged_function = entry["function"]
                if not isinstance(merged_function, dict):
                    continue
                description = function.get("description")
                if isinstance(description, str) and description and "description" not in merged_function:
                    merged_function["description"] = description
                schema = function.get("parameters")
                if isinstance(schema, dict):
                    existing_schema = merged_function.get("parameters")
                    schema_list = [existing_schema] if isinstance(existing_schema, dict) else []
                    schema_list.append(schema)
                    merged_function["parameters"] = _merge_tool_parameters(schema_list)
    return [merged_by_name[name] for name in sorted(merged_by_name)]


def _frontmatter(pretty_name: str, tags: list[str]) -> str:
    lines = [
        "---",
        f'pretty_name: "{pretty_name}"',
        "task_categories:",
        "- text-generation",
    ]
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f'- "{tag}"')
    lines.extend(
        [
            "configs:",
            "- config_name: default",
            "  data_files:",
            "  - split: train",
            '    path: "**/*.jsonl"',
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}... [truncated {omitted} chars]"


def _readme_sample_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= README_SAMPLE_MAX_DEPTH:
        return "[truncated: nested value]"
    if isinstance(value, str):
        return _truncate_text(value, README_SAMPLE_STRING_MAX_CHARS)
    if isinstance(value, list):
        items = [_readme_sample_value(item, depth=depth + 1) for item in value[:README_SAMPLE_MAX_ITEMS]]
        if len(value) > README_SAMPLE_MAX_ITEMS:
            items.append(f"[truncated: {len(value) - README_SAMPLE_MAX_ITEMS} items omitted]")
        return items
    if isinstance(value, dict):
        items = list(value.items())
        sampled = {
            str(key): _readme_sample_value(item, depth=depth + 1)
            for key, item in items[:README_SAMPLE_MAX_ITEMS]
        }
        if len(items) > README_SAMPLE_MAX_ITEMS:
            sampled["__truncated__"] = f"{len(items) - README_SAMPLE_MAX_ITEMS} keys omitted"
        return sampled
    return value


def _readme_sample_line(raw_line: str) -> str:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError:
        return json.dumps({"raw": _truncate_text(raw_line, README_SAMPLE_STRING_MAX_CHARS)}, ensure_ascii=False)
    return json.dumps(_readme_sample_value(value), ensure_ascii=False)


def _sample_lines(trace_files: Iterable[Path], sample_size: int = 1) -> list[str]:
    total_chars = 0
    samples: list[str] = []
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    if not line.strip():
                        continue
                    sample = _readme_sample_line(line)
                    projected_total = total_chars + len(sample) + (1 if samples else 0)
                    if projected_total > README_SAMPLE_MAX_CHARS:
                        return samples
                    samples.append(sample)
                    total_chars = projected_total
                    if len(samples) >= sample_size:
                        return samples
        except OSError:
            continue
    return samples


def _row_count(trace_files: Iterable[Path]) -> int:
    total = 0
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return total


def _sample_entry(trace_files: Iterable[Path]) -> dict[str, Any] | None:
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        return entry
                    break
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _is_structured_dataset(trace_files: Iterable[Path]) -> bool:
    sample_entry = _sample_entry(trace_files)
    return (
        isinstance(sample_entry, dict)
        and isinstance(sample_entry.get("messages"), list)
        and sample_entry.get("source") != "cli"
    )


def _is_agent_trace_row_dataset(trace_files: Iterable[Path]) -> bool:
    sample_entry = _sample_entry(trace_files)
    return isinstance(sample_entry, dict) and isinstance(sample_entry.get("traces"), list)


def _tools_json(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, indent=2, ensure_ascii=False)


def _should_externalize_tools(tools: list[dict[str, Any]]) -> bool:
    return bool(tools) and len(_tools_json(tools)) > README_INLINE_TOOLS_MAX_CHARS


def _tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _tools_details_block(tools: list[dict[str, Any]]) -> list[str]:
    tools_json = _tools_json(tools)
    if len(tools_json) > README_INLINE_TOOLS_MAX_CHARS:
        tool_names = sorted(name for tool in tools if (name := _tool_name(tool)))
        return [
            "## Tool schema snapshot",
            "",
            (
                "The complete dataset-level tool schema snapshot was written to `tools.json` because it is "
                "too large to embed safely in the Hugging Face dataset card."
            ),
            "",
            "<details>",
            "<summary>Tool names in snapshot</summary>",
            "",
            "```json",
            json.dumps(
                {
                    "tool_count": len(tools),
                    "tools": tool_names,
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "</details>",
            "",
        ]
    return [
        "## Tool schema snapshot",
        "",
        "<details>",
        "<summary>Training-ready tool schema snapshot</summary>",
        "",
        "```json",
        tools_json,
        "```",
        "",
        "</details>",
        "",
    ]


def _extraction_snippet(provider: str) -> list[str]:
    provider = normalize_extraction_provider(provider) or provider.strip().lower().replace("_", "-") or "claude"
    return [
        "## Create A Similar Dataset",
        "",
        "This dataset was staged from existing local agent sessions with `teich extract`.",
        "To build your own local dataset, install Teich and point it at one of the supported providers:",
        "",
        "```bash",
        f"teich extract {provider} --out data",
        "```",
        "",
        "Use `--sessions-dir /path/to/store` when your sessions are not in the default location,",
        "and `--model <substring>` when you only want sessions whose model metadata matches a value.",
        "",
    ]


def build_traces_readme(
    *,
    pretty_name: str,
    trace_files: list[Path],
    tags: list[str],
    model_id: str | None = None,
    repo_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    extraction_provider: str | None = None,
) -> str:
    structured_dataset = _is_structured_dataset(trace_files)
    agent_trace_rows = _is_agent_trace_row_dataset(trace_files)
    dataset_tools = tools if tools is not None else _dataset_tools(trace_files)
    dataset_reference = repo_id or "username/repo"
    row_count = _row_count(trace_files)
    sample_lines = _sample_lines(trace_files)
    sample_block = "\n".join(sample_lines)
    lines = [
        _frontmatter(pretty_name, tags),
        'This dataset was generated using [teich](https://github.com/TeichAI/teich) by [TeichAI](https://huggingface.co/TeichAI) <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6837935ac3b7ffe0d2559ce9/-AxyvV4wfUY8uo87kNKkK.png" width="20" height="20" style="display: inline-block; vertical-align: middle; margin: 0 3px;">',
        "",
        f"# {pretty_name}",
        "",
        (
            "This directory contains newline-delimited JSON training examples generated by teich."
            if structured_dataset
            else "This directory contains raw agent trace files generated by teich."
        ),
        "",
        f"Rows: {row_count}" if structured_dataset or agent_trace_rows else f"JSONL files: {len(trace_files)}",
        "",
    ]
    if model_id:
        lines.extend([f"Model metadata: `{model_id}`", ""])
    if dataset_tools:
        lines.extend(
            [
                "## Training-ready tools",
                "",
                "Generated agent traces carry configured or recovered tool schemas so tools remain available for training even when a session did not call them.",
                "Native Claude Code imports recover schemas for Claude Code and Claude Desktop built-ins, plus conservative name-derived MCP schemas, when the raw transcript only records tool names or calls.",
                (
                    "A complete dataset-level `tools` schema snapshot is embedded in the collapsed section at the bottom of this README."
                    if not _should_externalize_tools(dataset_tools)
                    else "A complete dataset-level `tools` schema snapshot is stored in `tools.json` to keep this dataset card upload-safe."
                ),
                "`load_traces` applies the dataset snapshot to each loaded example as a fallback `tools` field.",
                "",
            ]
        )
    if extraction_provider:
        lines.extend(_extraction_snippet(extraction_provider))
    lines.extend(
        [
            "## Format",
            "",
        ]
    )
    if structured_dataset:
        lines.extend(
            [
                "Each file is newline-delimited JSON where every line is already a training example.",
                "Chat-only datasets include `messages` plus convenience fields like optional `system`, `prompt`, `follow_up_prompts`, `thinking`, `response`, and `responses`.",
                "Tool datasets can include the same normalized `messages` structure together with a `tools` field.",
                "",
            ]
        )
    elif agent_trace_rows:
        lines.extend(
            [
                "Each file is newline-delimited JSON where every line is one preserved agent session.",
                "Rows use `id`, `task`, `traces`, `tools`, and `metadata`:",
                "",
                "- `id`: provider session id",
                "- `task`: first user prompt extracted from the trace",
                "- `traces`: native conversation messages in `from` / `value` form",
                "- `tools`: the tool schema available to that session, including tools that were not called",
                "- `metadata`: provider, model, first-message timestamp, token, cost, parent-session, and export details",
                "- `metadata.first_message_timestamp`: first timestamp-bearing source user message when the trace format provides one",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Each file is newline-delimited JSON representing a single captured agent session.",
                "The trace schema is designed for upload-first preservation so you can keep the original session history and convert it later for training.",
                "Teich normalizes split assistant fragments during trace copy and conversion so the semantic order is reasoning first, optional assistant text second, and tool calls last.",
                "Native Claude Code conversion also preserves runtime context such as skills, MCP instructions, hook context, permission state, date changes, and session recaps as masked `system` messages when the raw transcript provides them.",
                "",
                "Common top-level event groups:",
                "",
                "- `session_meta`",
                "- `turn_context`",
                "- `event_msg`",
                "- `response_item`",
                "- `session`",
                "- `message`",
                "- `session_info`",
                "- `model_change`",
                "- `thinking_level_change`",
                "- `external_session_meta`",
                "- `external_message`",
                "- `external_stderr`",
                "",
            ]
        )
    if sample_block:
        lines.extend(["## Example", "", "```json", sample_block, "```", ""])
    lines.extend(
        [
            "## Training",
            "",
            f"Use this dataset as `{dataset_reference}` with Teich's data preparation and training utilities.",
            "If you do not want Teich to handle chat-template formatting or masking, run `teich convert` "
            "to write standalone OpenAI-style JSONL rows with `prompt`, `messages`, `tools`, and `metadata`.",
            f"Training setup details evolve over time, so the maintained guide lives in the [Teich training docs]({TEICH_TRAINING_DOCS_URL}).",
            f"For loading, mixing, converting, and validating Teich datasets, see [Preparing Data]({TEICH_PREPARE_DOCS_URL}).",
            "",
        ]
    )
    if dataset_tools:
        lines.extend(_tools_details_block(dataset_tools))
    return "\n".join(lines)


def write_traces_readme(
    traces_dir: Path,
    *,
    pretty_name: str,
    tags: list[str],
    model_id: str | None = None,
    repo_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    excluded_dirs: list[Path] | None = None,
    extraction_provider: str | None = None,
    trace_files: list[Path] | None = None,
) -> Path:
    # Callers that just wrote the traces can pass the file list to skip the
    # directory rescan; the exclusion filters still apply either way.
    candidates = trace_files if trace_files is not None else traces_dir.rglob("*.jsonl")
    trace_files = sorted(
        path
        for path in candidates
        if path.is_file()
        and not {"partials", "failures"}.intersection(path.relative_to(traces_dir).parts)
        and not any(_path_is_relative_to(path, excluded_dir) for excluded_dir in excluded_dirs or [])
    )
    dataset_tools = tools if tools is not None else _dataset_tools(trace_files)
    readme_path = traces_dir / "README.md"
    readme_path.write_text(
        build_traces_readme(
            pretty_name=pretty_name,
            trace_files=trace_files,
            tags=tags,
            model_id=model_id,
            repo_id=repo_id,
            tools=dataset_tools,
            extraction_provider=extraction_provider,
        ),
        encoding="utf-8",
    )
    tools_path = traces_dir / "tools.json"
    if _should_externalize_tools(dataset_tools):
        tools_path.write_text(_tools_json(dataset_tools) + "\n", encoding="utf-8")
    elif tools_path.exists():
        tools_path.unlink()
    return readme_path
