# Data Format

Teich normalizes supported sources into structured training examples.

To write standalone OpenAI-style training JSONL from raw or extracted traces, run:

```bash
teich convert data --out teich-training.jsonl
```

Core fields:

- `prompt`: initial task description
- `follow_up_prompts`: optional additional user turns after the initial prompt
- `messages`: chat history
- `tools`: tool schemas available to the session, including tools that were not called
- `category`: optional prompt category, also retained in `metadata.category`
- `metadata`: session info, model, timestamps, usage, and provenance when available

## Messages

`messages` follows an OpenAI-style chat shape:

```json
[
  {"role": "user", "content": "Build a todo app"},
  {
    "role": "assistant",
    "content": "I will inspect the project.",
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "Read",
          "arguments": {"file_path": "README.md"}
        }
      }
    ]
  },
  {"role": "tool", "tool_call_id": "call_1", "name": "Read", "content": "..."}
]
```

Supported roles include:

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

Assistant messages can include:

- `content`: text response
- `reasoning_content`: reasoning traces when the source provides them
- `tool_calls`: function calls with arguments

Some providers split a single model turn across multiple native events. Teich normalizes those fragments so semantic order is:

1. `reasoning_content`
2. optional assistant `content`
3. `tool_calls`

Reasoning that arrives after assistant text or a tool-call fragment is moved back in front of the output it explains.

## Tools

`tools` contains function schemas available to the session:

```json
[
  {
    "type": "function",
    "function": {
      "name": "Read",
      "description": "Read a file.",
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"}
        },
        "required": ["file_path"],
        "additionalProperties": true
      }
    }
  }
]
```

Teich preserves configured tool snapshots where the source provides them. This matters because a training row can need the tool schema even when the model did not call that tool.

Tool schema sources include:

- configured tools embedded in generated traces
- generated dataset `README.md` snapshots
- `tools.json` snapshots
- native provider tool declarations
- fallback inference from observed tool calls when no explicit schema exists

## Metadata

Common metadata keys:

- `source_file`
- `source_line`
- `session_id`
- `trace_type`
- `model_provider`
- `model`
- `cwd`
- `cli_version`
- `turn_count`
- `usage`
- `total_cost_usd`
- `first_message_timestamp`

When the source format exposes per-message timestamps, converted rows include `metadata.first_message_timestamp` from the first timestamp-bearing source event that becomes a user message. It is not synthesized from session-start metadata.

## Native Claude Code Context

Native Claude Code and Claude Desktop traces can contain runtime context that the model saw but that is not ordinary user text.

Teich preserves this as masked `system` messages and mirrors it into `metadata.system_prompt`.

Examples include:

- Claude Desktop skill listings
- MCP instruction deltas
- deferred tool declarations
- command permission context
- date changes
- hook context
- away summaries
- session recaps

Local slash-command artifacts such as `/model` are filtered. `/goal` contributes the actual user goal text. Queued prompts become real user turns.

Advertised native Claude Code / Claude Desktop tools receive schemas even when a tool is only declared through deferred-tool context.

## DeepSeek Harness

Teich parses native DeepSeek Harness (`dsh`) session `.jsonl`, `.jsonl.zstd`, and `.jsonl.zst` files through `teich convert`, `load_traces()`, and `prepare_data()`. Compatible v0, v1, and v2 sessions are detected automatically and converted rows use `metadata.trace_type: "deepseek_harness"`.

```bash
teich convert /path/to/session.jsonl.zstd --out teich-training.jsonl
```

Native user and assistant messages become chat messages, assistant reasoning becomes `reasoning_content`, and tool calls and results retain their matching call IDs. Failed tool results preserve `isError: true` as `is_error: true`. Streaming chunks and tool execution events are omitted because their content is already represented by completed messages.

Conversion applies `surfaceOp` replacements to reconstruct the final effective conversation. The latest `request/header` snapshot supplies system instructions, tool schemas, and configuration when present. Directory loading selects the highest canonical `session.vN` file for each session header ID within a directory to avoid counting migration copies as separate rollouts. Files with different session IDs remain separate rollouts. Pass an individual file to select a specific version.

Unresolved image and file attachments raise an error instead of producing incomplete text-only training data. The supported native format is documented in the [upstream session persistence specification](https://github.com/deepseek-ai/deepseek-harness/blob/d347e703908d0406b7a7ef80e3a0e594d86b2215/packages/session/session-persistence-jsonl/README.md).

See the [synthetic native session](../examples/example_deepseek_session.jsonl) for a small example. This is support for parsing existing rollouts; DeepSeek Harness is not a generation or extraction provider.

## Structured Chat Rows

The `chat` provider writes structured training rows directly instead of raw traces.

Single-turn rows can include:

- `messages`
- `prompt`
- `thinking`
- `response`
- `model`

Existing chat-completion JSONL can omit `messages` when each row has `prompt`
plus `response` or `thinking`. `teich convert` builds the system, user, and
assistant messages for every line, maps assistant `thinking` to
`reasoning_content`, and keeps supported capture/provenance fields in
`metadata`. Provider-only opaque thinking signatures are not copied into the
training row.

Multi-turn follow-up rows can also include:

- `follow_up_prompts`
- `responses`

For these multi-turn chat-only rows, `messages` is authoritative. Teich omits
the single-turn `prompt`, `thinking`, and `response` convenience columns because
they do not describe the full conversation.

`system` is prompt-specific when provided. If a prompt row does not include `system`, Teich does not inject a default system prompt.

## Incomplete Traces

Rows ending on a tool result are incomplete without a follow-up assistant turn.

`load_traces()` drops those rows by default. Pass `drop_incomplete_traces=False` only when you intentionally want to inspect or repair them.

## Generated Dataset Cards

Generated datasets include a compact `README.md` with:

- Teich attribution
- row or file counts
- provider metadata when available; generated-run cards may also include model metadata
- a bounded example row
- links to the maintained training and data-preparation docs
- extraction guidance when the dataset came from `teich extract`
- a tool-schema summary when tools are available

Small tool snapshots are embedded in the dataset card. Large snapshots are written to `tools.json` so Hugging Face dataset-card validation does not have to process a massive YAML/Markdown payload.

The generated card intentionally avoids trainer-specific code blocks. Training APIs, model defaults, and library setup change over time; the maintained guidance lives in [Training](training.md) and [Preparing Data](prepare-data.md).

When you want a trainer-ready JSONL file without relying on Teich formatting or masking at training time, run:

```bash
teich convert data --out teich-training.jsonl
```

Generated dataset guidance is produced by `src/teich/trace_readme.py`, so behavior changes should update both top-level docs and that template.
