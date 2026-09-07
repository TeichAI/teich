# Preparing Data

Use `prepare_data()` when you already have data and want it rendered for a target tokenizer.

Supported sources:

- local JSONL files
- folders of JSONL traces
- DeepSeek Harness (`dsh`) session `.jsonl`, `.jsonl.zstd`, and `.jsonl.zst` files, or folders containing them
- Hugging Face dataset repos
- already-loaded `datasets.Dataset` objects
- source mixes with explicit ratios

## Basic Usage

```python
from teich import prepare_data

train_dataset = prepare_data(
    "TeichAI/Claude-Opus-4.6-Reasoning-887x",
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
)
```

`prepare_data()` returns rendered `text`, Teich span metadata, and optionally `input_ids` / `attention_mask`.

Call [Training](training.md)'s `mask_data()` step after constructing your trainer to convert those spans into token-level labels.

## Reports and Provenance

For audit-friendly preparation, request a report and keep provenance columns:

```python
train_dataset, prep_report = prepare_data(
    "TeichAI/Claude-Opus-4.6-Reasoning-887x",
    tokenizer,
    max_length=32768,
    oversized_policy="drop",
    preserve_columns=True,
    return_report=True,
    tokenize=True,
)

print(prep_report.max_token_length)
print(prep_report.oversized_rows[:3])
```

`PrepareReport` includes dropped rows, oversized rows, trimmed rows, token lengths,
max token length, kept-row ids, and returned row count. With a live Gemma 4
template it also reports per-row mode counts in `gemma4_modes` and migrated
leading `<|think|>` markers in `gemma4_legacy_triggers_normalized`. When
`reasoning_policy="strip"` is used, it reports affected rows and messages in
`reasoning_stripped_rows` and `reasoning_stripped_messages`.

Gemma 4 defaults to per-row auto mode when `enable_thinking` is omitted:
reasoning-bearing rows enable thinking and history preservation, while direct
rows use the loaded model's non-thinking protocol. See [Live Gemma 4 Models](training.md#live-gemma-4-models)
for explicit overrides and validation rules.

Original columns are removed after formatting unless `preserve_columns=True` or an explicit list is passed. The default provenance set is `source`, `metadata`, `raw_index`, and `source_key`.

## Oversized Rows

When `max_length` is set, use one of:

- `oversized_policy="drop"`: drop oversized rows
- `oversized_policy="trim_followups"`: for multi-turn rows, keep the longest
  complete conversation prefix that fits. Teich searches the possible
  follow-up boundaries logarithmically, avoiding a full rerender and
  retokenization for every removed turn.
- `oversized_policy="error"`: raise instead of filtering

The older `drop_oversized_examples` and `trim_oversized_followups` flags still work as compatibility aliases, but `oversized_policy` is the preferred API.

## Mixed Sources

Mix datasets with true ratios:

```python
train_dataset = prepare_data(
    {
        "max_examples": 1000,
        "reasoning-agent": {
            "source": "badlogicgames/pi-mono",
            "percentage": 80,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
        "instruct-chat": {
            "source": "TeichAI/polaris-alpha-1000x",
            "percentage": 20,
            "reasoning_policy": "strip",
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        },
    },
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
)
```

`percentage`, `proportion`, and `weight` are treated as true ratios.

If one source cannot fill its share after filtering or context-window drops, Teich scales the total row count down instead of silently changing the realized mix.

Global `chat_template_kwargs` and `reasoning_policy` values are the defaults for
every source. Source-level values override them for that dataset only.

You can also pass a simple list of sources:

```python
train_dataset = prepare_data(
    ["username/chat-traces", "username/tool-traces"],
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True},
)
```

## Tool Validation

Teich can fail early on undeclared or malformed tool calls:

```python
train_dataset = prepare_data(
    "./output",
    tokenizer,
    validate_tools=True,
    strict=True,
)
```

`validate_tools=True` checks tool-call names and required arguments against each row's declared `tools`.

## Plain Next-Token Training

If you do not want Teich response-only labels, turn masking metadata off:

```python
train_dataset = prepare_data(
    "./data.jsonl",
    tokenizer,
    teich_masking=False,
)
```

Rows contain rendered `text` only, plus tokens if `tokenize=True`.

## Export OpenAI-Style Training JSONL

If you want a training-friendly JSONL file without requiring Teich in the training environment, convert raw or extracted traces first:

```bash
teich convert ./data --out teich-training.jsonl
```

Each output line has:

- `prompt`
- `messages`
- `tools`
- `metadata`

This preserves tool schemas, tool calls, reasoning fields, and provenance in standalone OpenAI-style message rows. It does not render a tokenizer-specific chat template and does not create token-level labels; use `prepare_data()` and `mask_data()` for that path.

## Export ShareGPT-Style JSONL

Teich does not currently have a dedicated ShareGPT export command. Use `load_traces()` to normalize extracted data into `messages`, then write the `conversations` shape expected by ShareGPT-style trainers:

```python
import json
from pathlib import Path

from teich import load_traces


ROLE_MAP = {
    "system": "system",
    "developer": "system",
    "user": "human",
    "assistant": "gpt",
    "model": "gpt",
}


def message_text(content):
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"].strip())
    return "\n".join(part for part in parts if part).strip()


dataset = load_traces("./data")
output_path = Path("sharegpt.jsonl")

with output_path.open("w", encoding="utf-8") as handle:
    for row in dataset:
        conversations = []
        for message in row["messages"]:
            role = ROLE_MAP.get(message.get("role"))
            if role is None:
                continue
            text = message_text(message.get("content"))
            reasoning = message.get("reasoning_content")
            if role == "gpt" and isinstance(reasoning, str) and reasoning.strip():
                text = f"<think>\n{reasoning.strip()}\n</think>\n{text}".strip()
            if text:
                conversations.append({"from": role, "value": text})
        if any(turn["from"] == "gpt" for turn in conversations):
            handle.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
```

This simple export is best for text chat data. ShareGPT-style schemas usually do not preserve Teich's full agent surface, including tool schemas, tool result messages, tool-call arguments, provenance metadata, and typed supervision spans. Keep the native Teich JSONL format when training agent/tool-use models with `prepare_data()` and `mask_data()`.

## Manual Flow with `load_traces`

Use `load_traces()` directly when you want to own chat-template rendering, filtering, tokenization, label masking, and packing policy.

```python
from teich import load_traces, row_fits_context, validate_tool_calls

dataset = load_traces("./output")
example = dataset[0]

validate_tool_calls(example).raise_for_errors()
if not row_fits_context(example, tokenizer, 32768, {"enable_thinking": True}):
    raise ValueError("example does not fit the target context window")

rendered = tokenizer.apply_chat_template(
    example["messages"],
    tools=example.get("tools") or [],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
)
tokenized = tokenizer(rendered, truncation=True, max_length=32768)
```

`load_traces()` drops rows that end on a tool result by default, because those traces are incomplete without a follow-up assistant turn. Pass `drop_incomplete_traces=False` only when you intentionally want to inspect or repair those rows.

## Preflight Helpers

- `row_fits_context(row, tokenizer, max_length, chat_template_kwargs)`: render and measure one row
- `validate_tool_calls(row)`: check declared tool names and required arguments
- `trace_is_complete(row)`: flag rows that end on a tool result
- `detect_trace_type(events)`: identify supported raw trace events, or return `None`
