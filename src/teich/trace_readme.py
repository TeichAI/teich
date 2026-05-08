from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .converter import convert_traces_to_training_data


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
            examples = convert_traces_to_training_data(trace_file)
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
    return isinstance(sample_entry, dict) and isinstance(sample_entry.get("messages"), list)


def build_traces_readme(
    *,
    pretty_name: str,
    trace_files: list[Path],
    tags: list[str],
    model_id: str | None = None,
    repo_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    structured_dataset = _is_structured_dataset(trace_files)
    dataset_tools = tools if tools is not None else _dataset_tools(trace_files)
    dataset_reference = repo_id or "username/repo"
    row_count = _row_count(trace_files)
    sample_lines = _sample_lines(trace_files)
    sample_block = "\n".join(sample_lines) if sample_lines else json.dumps(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                {"role": "user", "content": "Hello", "thinking": None},
                {"role": "assistant", "content": "Hi!", "thinking": None},
            ],
            "prompt": "Hello",
            "response": "Hi!",
            "model": model_id or "unknown model",
        } if structured_dataset else {
            "type": "session_meta",
            "payload": {
                "id": "example-session",
                "model_provider": "codex",
            },
        },
        ensure_ascii=False,
    )
    lines = [
        _frontmatter(pretty_name, tags),
        'This dataset was generated using [teich](https://github.com/TeichAI/teich) by [TeichAI](https://huggingface.co/TeichAI) <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6837935ac3b7ffe0d2559ce9/-AxyvV4wfUY8uo87kNKkK.png" width="20" height="20" style="display: inline-block; vertical-align: middle; margin: 0 3px;">',
        "",
        "Prepare these datasets for supervised fine-tuning in just a few lines of code — see the **Conversion** section below.",
        "",
        f"# {pretty_name}",
        "",
        (
            "This directory contains newline-delimited JSON training examples generated by teich."
            if structured_dataset
            else "This directory contains raw agent trace files generated by teich."
        ),
        "",
        f"All assistant responses were generated by **{model_id or 'unknown model'}**.",
        "",
        f"Rows: {row_count}" if structured_dataset else f"JSONL files: {len(trace_files)}",
        "",
    ]
    if dataset_tools:
        lines.extend(
            [
                "## Training-ready tools",
                "",
                "A complete configured `tools` schema snapshot is available in `tools.json`.",
                "Use it when rendering loaded examples through your training chat template.",
                "`load_traces` applies this snapshot to each loaded example as the `tools` field.",
                "",
            ]
        )
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
                "Chat-only datasets include `messages` plus convenience fields like `system`, `prompt`, `thinking`, and `response`.",
                "Tool datasets can include the same normalized `messages` structure together with a `tools` field.",
                "",
            ]
        )
    else:
        lines.extend(
            [
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
            ]
        )
    lines.extend(
        [
            "## Example",
            "",
            "```json",
            sample_block,
            "```",
            "",
            "## Conversion",
            "",
            "### Recommended: prepare for SFT",
            "",
            "Load this dataset as trainer-friendly text, then let `SFTTrainer` tokenize it",
            "before applying Teich's multi-turn/tool-aware response-only mask:",
            "",
            "```python",
            "from trl import SFTConfig, SFTTrainer",
            "from teich import mask_data, prepare_data",
            "",
            "train_dataset = prepare_data(",
            f"    '{dataset_reference}',",
            "    tokenizer,",
            "    max_length=32768,",
            "    drop_oversized_examples=True,",
            "    chat_template_kwargs={'enable_thinking': True},",
            ")",
            "",
            "trainer = SFTTrainer(",
            "    model=model,",
            "    train_dataset=train_dataset,",
            "    args=SFTConfig(",
            "        dataset_text_field='text',",
            "        dataset_num_proc=1,",
            "        max_length=32768,",
            "        packing=False,",
            "    ),",
            ")",
            "trainer = mask_data(trainer, tokenizer=tokenizer)",
            "```",
            "",
            "`mask_data` keeps the normal trainer configuration flow while applying Teich's",
            "assistant/tool-call labels after trainer tokenization. Keep `packing=False` for this flow.",
            "",
            "You can combine this dataset with other Teich chat-only or tool-call datasets by",
            "passing a list of dataset IDs, local paths, or loaded `datasets.Dataset` objects:",
            "",
            "```python",
            "train_dataset = prepare_data(",
            f"    ['{dataset_reference}', 'username/other-teich-dataset'],",
            "    tokenizer,",
            "    max_length=32768,",
            "    drop_oversized_examples=True,",
            ")",
            "```",
            "",
            "### Advanced: load, format, and mask manually",
            "",
            "```python",
            "from teich import load_traces, format_and_mask",
            "",
            f"dataset = load_traces('{dataset_reference}')",
            "training_data = format_and_mask(dataset, tokenizer, max_length=32768, strict=True)",
            "```",
            "",
            "### Manual: render loaded examples with your tokenizer",
            "",
            "If you want to drive the tokenizer yourself, `load_traces` returns rows with normalized",
            "`messages` ready for `tokenizer.apply_chat_template(...)`:",
            "",
            "```python",
            "from teich import load_traces",
            "",
            f"dataset = load_traces('{dataset_reference}')",
            "example = dataset[0]",
            "rendered = tokenizer.apply_chat_template(",
            "    example['messages'],",
        ]
    )
    if dataset_tools:
        lines.append("    tools=example.get('tools') or [],")
    lines.extend(
        [
            "    tokenize=False,",
            "    add_generation_prompt=False,",
            "    enable_thinking=True,",
            ")",
            "tokenized = tokenizer(rendered, truncation=True, max_length=32768)",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_traces_readme(
    traces_dir: Path,
    *,
    pretty_name: str,
    tags: list[str],
    model_id: str | None = None,
    repo_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Path:
    trace_files = sorted(
        path for path in traces_dir.glob("*.jsonl") if path.is_file()
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
        ),
        encoding="utf-8",
    )
    tools_path = traces_dir / "tools.json"
    if dataset_tools:
        tools_path.write_text(
            json.dumps(dataset_tools, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif tools_path.exists():
        tools_path.unlink()
    return readme_path
