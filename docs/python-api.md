# Python API

Teich's public API is designed around three levels:

1. high-level training prep with `prepare_data()` and `mask_data()`
2. trace loading and conversion with `load_traces()`
3. preflight helpers for validation, fitting, and previews

## Imports

```python
from teich import (
    prepare_data,
    mask_data,
    load_traces,
    detect_trace_type,
    validate_tool_calls,
    row_fits_context,
    trace_is_complete,
    preview_sft_example,
    Config,
    TrainingExample,
)
```

## `prepare_data()`

Recommended entry point for training.

```python
train_dataset = prepare_data(
    source_or_dataset,
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True},
)
```

Accepts:

- local file or folder
- Hugging Face dataset id
- `datasets.Dataset`
- list of sources
- source mix mapping with explicit ratios

Returns a trainer-friendly dataset with rendered `text`, Teich supervision spans, and optionally `input_ids` / `attention_mask`.

Useful options:

- `split`
- `revision`
- `token` / `hf_token`
- `cache_dir`
- `local_dir`
- `max_examples`
- `max_length`
- `oversized_policy`
- `preserve_columns`
- `return_report`
- `validate_tools`
- `strict`
- `teich_masking`
- `tokenize`
- `chat_template_kwargs`
- `reasoning_policy` (`"keep"` or `"strip"`)

See [Preparing Data](prepare-data.md).

## `mask_data()`

Apply response-only labels to a trainer after trainer tokenization.

```python
trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    train_on_reasoning=True,
    train_on_final_answers=True,
    train_on_tools=True,
)
```

By default, Teich supervises assistant reasoning, final answers, and tool calls. Prompt/context tokens stay `-100`.

Policy options:

- `train_on_reasoning`
- `train_on_final_answers`
- `train_on_tools`
- `train_on_user`
- `train_on_system`
- `train_on_developer`
- `train_on_tool_responses`
- `max_supervised_tokens`
- `audit`
- `text_column`

See [Training](training.md).

## `load_traces()`

Load and convert raw traces without running the full preparation pipeline.

```python
dataset = load_traces("./output")
```

Use this when you want to own rendering, filtering, tokenization, masking, and packing yourself.

By default, rows that end on a tool result are dropped because they are incomplete. Pass `drop_incomplete_traces=False` only for inspection or repair workflows.

## `detect_trace_type()`

Detect supported parsed raw trace events.

```python
from teich import detect_trace_type

trace_type = detect_trace_type(events)
```

Returns one of:

- `codex`
- `claude_code`
- `deepseek_harness`
- `droid`
- `pi`
- `openclaw`
- `hermes`
- `external_agent`
- `None`

Factory `droid` CLI sessions are supported as a conversion-only source. Point `prepare_data()` or `load_traces()` at session JSONL files from `~/.factory/sessions/...`; Teich reads the adjacent `<session-id>.settings.json` sidecar for model and token usage metadata when present.

DeepSeek Harness (`dsh`) is also a conversion-only source. `load_traces()` and `prepare_data()` accept native session `.jsonl`, `.jsonl.zstd`, or `.jsonl.zst` files and directories containing them. Parsed native events are detected as `deepseek_harness`; see [DeepSeek Harness data](data-format.md#deepseek-harness).

## Validation Helpers

### `validate_tool_calls()`

```python
result = validate_tool_calls(example)
result.raise_for_errors()
```

Checks that assistant tool calls reference declared tools and include required arguments.

### `row_fits_context()`

```python
fits = row_fits_context(
    example,
    tokenizer,
    max_length=32768,
    chat_template_kwargs={"enable_thinking": True},
)
```

Renders one row with the target chat template and checks whether it fits the target context window.

### `trace_is_complete()`

```python
if not trace_is_complete(example):
    ...
```

Returns `False` when a row ends on a tool result without a follow-up assistant turn.

## Preview Helpers

Use `preview_sft_example()` before training or the dataset preview helper attached by `mask_data()`.

```python
from teich import preview_sft_example

preview = preview_sft_example(tokenizer, input_ids, labels)
print(preview)
```

After `mask_data()`:

```python
print(trainer.train_dataset.preview(0, tokenizer))
```

Previewing is the quickest way to confirm that reasoning, tool calls, and final answers are supervised while context is masked.

## Config Objects

`Config` loads generation config:

```python
from teich import Config

config = Config.from_yaml("config.yaml")
```

`TrainingExample` is the typed representation used internally for converted rows.
