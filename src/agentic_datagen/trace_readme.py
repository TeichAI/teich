from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .converter import convert_trace_to_training_example


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
            example = convert_trace_to_training_example(trace_file)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for tool in example.tools:
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
    lines = ["---", f'pretty_name: "{pretty_name}"']
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
            '    path: "*.jsonl"',
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def _sample_lines(trace_files: Iterable[Path], sample_size: int = 3) -> list[str]:
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                lines = [line.rstrip("\n") for line in handle if line.strip()]
        except OSError:
            continue
        if lines:
            return lines[:sample_size]
    return []


def build_traces_readme(*, pretty_name: str, trace_files: list[Path], tags: list[str], model_id: str | None = None) -> str:
    sample_lines = _sample_lines(trace_files)
    tools_block = json.dumps(_dataset_tools(trace_files), indent=2, ensure_ascii=False)
    sample_block = "\n".join(sample_lines) if sample_lines else json.dumps(
        {
            "type": "session_meta",
            "payload": {
                "id": "example-session",
                "model_provider": "codex",
            },
        },
        ensure_ascii=False,
    )
    return "\n".join(
        [
            _frontmatter(pretty_name, tags),
            f"# {pretty_name}",
            "",
            "This directory contains raw agent trace files generated by teich.",
            "",
            f"All assistant responses were generated by **{model_id or 'unknown model'}**.",
            "",
            f"Trace files: {len(trace_files)}",
            "",
            "## Training-ready tools",
            "",
            "Use this `tools` payload when rendering converted examples through your training chat template.",
            "The same structure is emitted on each converted example as the `tools` field.",
            "",
            "```json",
            tools_block,
            "```",
            "",
            "## Format",
            "",
            "Each file is newline-delimited JSON representing a single captured agent session.",
            "The trace schema is designed for upload-first preservation so you can keep the original session history and convert it later for training.",
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
            "",
            "## Example",
            "",
            "```json",
            sample_block,
            "```",
            "",
            "## Conversion",
            "",
            "You can convert these raw traces into training examples with:",
            "",
            "```python",
            "from pathlib import Path",
            "from teich import convert_traces_to_training_data",
            "",
            "examples = convert_traces_to_training_data(Path('.'))",
            "```",
            "",
        ]
    )


def write_traces_readme(
    traces_dir: Path,
    *,
    pretty_name: str,
    tags: list[str],
    model_id: str | None = None,
    readme_file_name: str = "README.md",
) -> Path:
    trace_files = sorted(
        path for path in traces_dir.glob("*.jsonl") if path.is_file()
    )
    readme_path = traces_dir / readme_file_name
    readme_path.write_text(
        build_traces_readme(
            pretty_name=pretty_name,
            trace_files=trace_files,
            tags=tags,
            model_id=model_id,
        ),
        encoding="utf-8",
    )
    return readme_path
