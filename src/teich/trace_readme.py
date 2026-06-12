from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .converter import convert_traces_to_training_data


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
    return (
        isinstance(sample_entry, dict)
        and isinstance(sample_entry.get("messages"), list)
        and sample_entry.get("source") != "cli"
    )


def _is_agent_trace_row_dataset(trace_files: Iterable[Path]) -> bool:
    sample_entry = _sample_entry(trace_files)
    return isinstance(sample_entry, dict) and isinstance(sample_entry.get("traces"), list)


def _tools_details_block(tools: list[dict[str, Any]]) -> list[str]:
    return [
        "## Tool schema snapshot",
        "",
        "<details>",
        "<summary>Training-ready tool schema snapshot</summary>",
        "",
        "```json",
        json.dumps(tools, indent=2, ensure_ascii=False),
        "```",
        "",
        "</details>",
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
) -> str:
    structured_dataset = _is_structured_dataset(trace_files)
    agent_trace_rows = _is_agent_trace_row_dataset(trace_files)
    dataset_tools = tools if tools is not None else _dataset_tools(trace_files)
    dataset_reference = repo_id or "username/repo"
    row_count = _row_count(trace_files)
    sample_lines = _sample_lines(trace_files)
    sample_block = "\n".join(sample_lines) if sample_lines else json.dumps(
        {
            "messages": [
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
        f"Rows: {row_count}" if structured_dataset or agent_trace_rows else f"JSONL files: {len(trace_files)}",
        "",
    ]
    if dataset_tools:
        lines.extend(
            [
                "## Training-ready tools",
                "",
                "Generated agent traces carry configured or recovered tool schemas so tools remain available for training even when a session did not call them.",
                "Native Claude Code imports recover schemas for Claude Code and Claude Desktop built-ins, plus conservative name-derived MCP schemas, when the raw transcript only records tool names or calls.",
                "A complete dataset-level `tools` schema snapshot is also embedded in the collapsed section at the bottom of this README.",
                "`load_traces` applies the dataset snapshot to each loaded example as a fallback `tools` field.",
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
            "### Recommended: train with Unsloth and TRL `SFTTrainer`",
            "",
            "Use the trainer-first path: `prepare_data` renders trainer-friendly `text` rows with Teich supervision metadata,",
            "`SFTTrainer` tokenizes them, then `mask_data` applies Teich's multi-turn/tool-aware response-only labels:",
            "`oversized_policy='trim_followups'` lets multi-turn rows drop final follow-ups before oversized rows are discarded.",
            "",
            "```python",
            "import os",
            "",
            "from unsloth import FastLanguageModel",
            "from trl import SFTConfig, SFTTrainer",
            "",
            "from teich import mask_data, prepare_data",
            "",
            "MAX_SEQ_LEN = 32768",
            "MODEL_NAME = 'unsloth/Qwen3.5-0.8B'",
            "CHAT_TEMPLATE_KWARGS = {'enable_thinking': True}",
            "PUSH_TO_HUB_REPO_ID = 'username/teich-sft-model'",
            "HF_TOKEN = os.environ.get('HF_TOKEN') or ''",
            "",
            "model, tokenizer = FastLanguageModel.from_pretrained(",
            "    model_name=MODEL_NAME,",
            "    max_seq_length=MAX_SEQ_LEN,",
            "    load_in_4bit=False,",
            "    load_in_8bit=False,",
            "    full_finetuning=False,",
            ")",
            "",
            "model = FastLanguageModel.get_peft_model(",
            "    model,",
            "    r=32,",
            "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'out_proj'],",
            "    lora_alpha=64,",
            "    lora_dropout=0,",
            "    bias='none',",
            "    use_gradient_checkpointing='unsloth',",
            "    random_state=3407,",
            "    use_rslora=False,",
            "    loftq_config=None,",
            ")",
            "",
            "train_dataset = prepare_data(",
            f"    '{dataset_reference}',",
            "    tokenizer,",
            "    split='train',",
            "    max_examples=500,",
            "    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,",
            "    max_length=MAX_SEQ_LEN,",
            "    oversized_policy='trim_followups',",
            "    tokenize=True,",
            "    strict=True,",
            ")",
            "",
            "trainer = SFTTrainer(",
            "    model=model,",
            "    tokenizer=tokenizer,",
            "    train_dataset=train_dataset,",
            "    eval_dataset=None,",
            "    args=SFTConfig(",
            "        dataset_text_field='text',",
            "        dataset_num_proc=1,",
            "        max_length=MAX_SEQ_LEN,",
            "        packing=False,",
            "        per_device_train_batch_size=1,",
            "        gradient_accumulation_steps=4,",
            "        warmup_steps=5,",
            "        num_train_epochs=1,",
            "        learning_rate=2e-4,",
            "        logging_steps=1,",
            "        optim='muon',",
            "        optim_target_modules='all-linear',",
            "        weight_decay=0.001,",
            "        lr_scheduler_type='linear',",
            "        output_dir='outputs',",
            "        seed=3407,",
            "        report_to='none',",
            "    ),",
            ")",
            "trainer = mask_data(",
            "    trainer,",
            "    tokenizer=tokenizer,",
            "    train_on_reasoning=True,",
            "    train_on_final_answers=True,",
            "    train_on_tools=True,",
            ")",
            "",
            "trainer_stats = trainer.train(resume_from_checkpoint=False)",
            "",
            "model.push_to_hub_merged(PUSH_TO_HUB_REPO_ID, tokenizer, save_method='merged_16bit', token=HF_TOKEN)",
            "```",
            "",
            "`mask_data` keeps the normal trainer configuration flow while applying Teich's",
            "assistant/tool-call labels after trainer tokenization. Keep `packing=False` for this flow.",
            "If you want standard next-token training without Teich response-only labels, call `prepare_data(..., teich_masking=False)` and skip `mask_data()`.",
            "",
            "For preparation audits, call `prepare_data(..., return_report=True)` to receive a `PrepareReport` with",
            "dropped rows, oversized rows, trimmed rows, max token lengths, and row ids. Use `preserve_columns=True`",
            "or `preserve_columns=['metadata', 'raw_index', 'source_key']` when you want those fields kept in the prepared dataset.",
            "`validate_tools=True` checks assistant tool-call names and required arguments against each row's declared tools before rendering.",
            "",
            "You can combine this dataset with other Teich chat-only or tool-call datasets by",
            "passing a list of dataset IDs, local paths, or loaded `datasets.Dataset` objects:",
            "",
            "```python",
            "train_dataset = prepare_data(",
            f"    ['{dataset_reference}', 'username/other-teich-dataset'],",
            "    tokenizer,",
            "    max_length=MAX_SEQ_LEN,",
            "    oversized_policy='trim_followups',",
            "    tokenize=True,",
            "    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,",
            ")",
            "```",
            "",
            "For weighted mixes, pass a source mapping with `percentage`, `weight`, or per-source `max_examples`.",
            "Explicit ratios stay true: if a source cannot fill its share after filtering, Teich scales the total row count down instead of backfilling from another source.",
            "Global `chat_template_kwargs` are the default; source-level `chat_template_kwargs` override those keys for that dataset only.",
            "",
            "```python",
            "train_dataset = prepare_data(",
            "    {",
            "        'max_examples': 2_000,",
            f"        'agent': {{'source': '{dataset_reference}', 'percentage': 80}},",
            "        'chat': {",
            "            'source': 'username/other-teich-dataset',",
            "            'percentage': 20,",
            "            'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False},",
            "        },",
            "    },",
            "    tokenizer,",
            "    max_length=MAX_SEQ_LEN,",
            "    oversized_policy='trim_followups',",
            "    tokenize=True,",
            "    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,",
            ")",
            "```",
            "",
            "### Fallback: render loaded examples with your tokenizer",
            "",
            "Use `load_traces` directly only when you want to own the remaining training pipeline yourself:",
            "chat-template rendering, filtering, tokenization, label masking, packing policy, and auditing.",
            "`load_traces` returns rows with normalized `messages` ready for `tokenizer.apply_chat_template(...)`:",
            "",
            "```python",
            "from teich import load_traces, row_fits_context, validate_tool_calls",
            "",
            f"dataset = load_traces('{dataset_reference}')",
            "example = dataset[0]",
            "# load_traces drops rows ending on tool results by default; pass",
            "# drop_incomplete_traces=False only to inspect or repair incomplete rows.",
            "validate_tool_calls(example).raise_for_errors()",
            "assert row_fits_context(example, tokenizer, 32768, {'enable_thinking': True})",
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
) -> Path:
    trace_files = sorted(
        path
        for path in traces_dir.rglob("*.jsonl")
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
        ),
        encoding="utf-8",
    )
    tools_path = traces_dir / "tools.json"
    if tools_path.exists():
        tools_path.unlink()
    return readme_path
