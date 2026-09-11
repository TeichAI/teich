<div align="center">
  <img src="assets/teich.svg" alt="Teich logo" width="132">
  <h1>Teich</h1>
  <p><strong>Agent data infrastructure for generation, normalization, formatting, response masking, and training audits.</strong></p>
  <p>
    <a href="https://pepy.tech/projects/teich"><img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/teich?label=downloads&color=green"></a>
    <a href="https://pypi.org/project/teich/"><img alt="PyPI" src="https://img.shields.io/pypi/v/teich?label=pypi&color=black"></a>
    <a href="https://pypi.org/project/teich/"><img alt="Python versions" src="https://img.shields.io/badge/python-%3E%3D3.10-green"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/teich?color=black"></a>
  </p>
</div>

Teich turns raw agent sessions, chat datasets, local JSONL, Hugging Face datasets, and in-memory `datasets.Dataset` objects into auditable SFT data.

It handles the parts that usually break training runs:

- normalizing traces into OpenAI-style `messages` and `tools`
- preserving tool schemas, reasoning, metadata, and provenance
- rendering through your target tokenizer's chat template
- recording typed supervision spans before tokenization
- applying response-only labels after TRL / Unsloth trainer tokenization
- reporting dropped, oversized, trimmed, malformed, and fully masked rows

Use it as a trace generator, a dataset loader, a chat-template renderer, a masking layer, or the whole pipeline.

## Install

```bash
pip install teich
```

Or run it without installing:

```bash
uvx teich --help
```

Agent trace generation needs Docker and an API key for the configured provider. Preparing an existing local or Hugging Face dataset does not need Docker.

Prefer a browser workflow?

```bash
teich studio
```

See [Teich Studio](docs/studio.md).

## Quickstart: Prepare Existing Data

If your dataset already has `messages`, Teich can usually prepare it directly.

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

With a live Gemma 4 tokenizer, omit `chat_template_kwargs` to let Teich choose
thinking or non-thinking independently for each row. See [Training](docs/training.md#live-gemma-4-models)
for auto-mode rules and the E4B, 26B-A4B, and 31B template contract.
Qwen 3.8 keeps its own template-native defaults, including historical reasoning
preservation and `reasoning_effort`; see [Live Qwen 3.8 Models](docs/training.md#live-qwen-38-models).
Granite 4.2 and Qwen 3.6 have their own live-template contracts and runnable
examples under [Granite 4.2](docs/training.md#live-granite-42-models) and
[Qwen 3.6](docs/training.md#live-qwen-36-models).

Then create your trainer and call `mask_data()`:

```python
from teich import mask_data

trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    train_on_reasoning=True,
    train_on_final_answers=True,
    train_on_tools=True,
)
```

More detail: [Preparing Data](docs/prepare-data.md) and [Training](docs/training.md).

## Quickstart: Generate New Traces

```bash
teich init my-project
cd my-project
```

Add prompts to `prompts.jsonl`:

```jsonl
{"prompt":"Build a simple todo list app in React"}
{"github_repo":"armand0e/perplexica-mcp","prompt":"Add a small usability improvement and update the tests"}
{"prompt":"Draft a compact project plan","follow_up_prompts":["Revise it for a solo developer","Add a risk checklist"]}
```

Set your provider key and run:

```bash
export OPENAI_API_KEY=sk-...
teich generate -c config.yaml
```

Teich writes raw traces, converted training rows, sandbox snapshots, a compact dataset card, and sometimes `tools.json` under `output/`. Use `--resume` to skip prompts that already completed.

More detail: [Generation](docs/generation.md).

## Quickstart: Extract Local Sessions

If you already have local agent sessions, Teich can stage them as an anonymized dataset in one command:

```bash
teich extract claude --model fable-5
```

`extract` supports `claude`, `codex`, `cursor`, `pi`, and `hermes`. It writes anonymized traces to `data/` by default using provider-native or recovered session JSONL files. The generated Hugging Face dataset metadata matches `**/*.jsonl`, so providers such as Cursor can preserve nested project transcript paths. It generates a dataset `README.md`, and then asks whether to upload the folder to Hugging Face. Use `--out` / `--output` to choose another folder.

If the agent store is somewhere other than the default home-directory location, pass it explicitly. `--sessions-dir` accepts either the agent root, such as `.claude`, `.codex`, `.pi`, or `.hermes`, or the native store under it, such as `.claude/projects`, `.codex/sessions`, `.hermes/state.db`, or Cursor's `workspaceStorage` / `globalStorage/state.vscdb`:

```bash
teich extract claude --sessions-dir /path/to/.claude --out data
teich extract claude --sessions-dir /path/to/.claude/projects --out data
teich extract codex --sessions-dir /path/to/.codex --out data
teich extract codex --sessions-dir /path/to/.codex/sessions --out data
teich extract pi --sessions-dir /path/to/.pi --out data
teich extract pi --sessions-dir /path/to/.pi/agent/sessions --out data
teich extract pi --sessions-dir /path/to/.pi/sessions --out data
teich extract hermes --sessions-dir /path/to/.hermes --out data
teich extract hermes --sessions-dir /path/to/.hermes/state.db --out data
teich extract cursor --sessions-dir /path/to/Cursor/User/workspaceStorage --out data
teich extract cursor --sessions-dir /path/to/Cursor/User/globalStorage/state.vscdb --out data
```

Extraction anonymizes staged traces by default. To keep the raw extracted data unchanged, pass `--no-anon` or `--no-anonymize` and review the output carefully before sharing or uploading it.

To convert raw or extracted traces into standalone OpenAI-style JSONL rows that can be consumed without Teich at training time:

```bash
teich convert data --out teich-training.jsonl
```

This writes standalone OpenAI-style rows with `prompt`, `messages`, `tools`, `metadata`, and an optional captured `system` field. Use `prepare_data()` and `mask_data()` when you want Teich to handle tokenizer-specific formatting and response-only labels.

DeepSeek Harness (`dsh`) session files are also supported by `teich convert`, `load_traces()`, and `prepare_data()`, with automatic detection of native `.jsonl`, `.jsonl.zstd`, and `.jsonl.zst` traces. See [DeepSeek Harness data](docs/data-format.md#deepseek-harness) and the [synthetic session example](examples/example_deepseek_session.jsonl). This support covers parsing existing rollouts; DeepSeek Harness is not a generation or extraction provider.

## What Teich Supports

| Use case | Start here |
| --- | --- |
| Find command examples and options | [CLI Reference](docs/cli.md) |
| Configure and steer runs in a browser | [Teich Studio](docs/studio.md) |
| Generate Codex, Pi, Claude Code, Hermes, or chat data | [Generation](docs/generation.md) |
| Load local files, folders, Hugging Face datasets, or `datasets.Dataset` objects | [Preparing Data](docs/prepare-data.md) |
| Train with TRL / Unsloth while keeping response-only labels correct | [Training](docs/training.md) |
| Understand `messages`, `tools`, metadata, and native trace behavior | [Data Format](docs/data-format.md) |
| Use `prepare_data`, `mask_data`, `load_traces`, and validation helpers | [Python API](docs/python-api.md) |
| See the full generation, preparation, and masking pipeline | [Pipeline Flow](docs/pipeline.md) |

## Why Teich

Most SFT pipelines flatten agent data too early. That loses tool schemas, tool results, reasoning boundaries, provenance, and the exact assistant spans you meant to train on.

Teich keeps the data structured until the last practical moment:

```text
prompts / traces / JSONL / HF datasets / Dataset objects
        -> load_traces() or prepare_data()
        -> normalized messages + tools
        -> tokenizer chat template rendering
        -> trainer-friendly text + Teich supervision spans
        -> SFTTrainer tokenization
        -> mask_data()
        -> audited input_ids + labels
```

This makes multi-turn, tool-call, reasoning, and mixed-source datasets trainable without relying on brittle single-span masking.

## Common Commands

```bash
# Create a generation project
teich init my-project

# Generate data from config.yaml
teich generate -c config.yaml

# Resume an interrupted batch
teich generate -c config.yaml --resume

# Extract, anonymize, and stage local Claude Code traces
teich extract claude --model fable-5 --out data

# Convert staged raw traces to standalone OpenAI-style training JSONL
teich convert data --out teich-training.jsonl

# Launch the local browser UI
teich studio

# Use a local OpenAI-compatible endpoint
TEICH_PROVIDER=LMstudio \
TEICH_MODEL=gemma-4 \
TEICH_BASE_URL=http://localhost:1234/v1 \
TEICH_API_KEY=llm \
teich generate -c config.yaml
```

## Minimal Config

```yaml
agent:
  provider: codex  # codex, pi, claude-code, hermes, or chat

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.jsonl

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
  failures_dir: ./failures

publish:
  repo_id: username/my-dataset
  private: false
```

`agent.provider: chat` writes structured chat rows directly and does not require Docker. Agent providers preserve raw or native traces as source-of-truth artifacts.

To run Codex on your ChatGPT subscription instead of an API key, set `agent.codex.use_host_auth: true` (Teich shares your host `codex login` across containers), and enable Codex fast mode with `model.service_tier: fast`. See [Generation](docs/generation.md#using-your-chatgpt-subscription-host-auth).

To run Claude Code on your Claude subscription (Pro/Max), export a `claude setup-token` token as `CLAUDE_CODE_OAUTH_TOKEN` (or set `agent.claude.oauth_token`) — it activates automatically and bills your plan's rate limits, not API credits. Subscription request starts are paced 45 seconds apart by default; configure `agent.claude.subscription_request_delay_seconds` or set it to `0` to disable pacing. Batch and Studio Claude Code sessions use a real interactive PTY so readable thinking summaries can reach Claude's native trace; `agent.claude.always_thinking` and `agent.claude.show_thinking_summaries` default to `true` and can be disabled explicitly. Claude Code batch runs also support Teich-managed `agent.claude.fallback_model` retries, while `model.reasoning_effort` (`--effort`) and `agent.claude.max_thinking_tokens` apply to batch and Studio sessions. Codex supports `model.reasoning_summary: detailed` plus `model.reasoning_summaries_enabled: true` for its richest readable summaries. Set `capture_harness_context.enabled: true` to make one local fake-provider preflight that records the client-visible Codex/Claude system instructions and tool schemas without using provider quota; `teich capture-context` can save the same context directly. See [Generation](docs/generation.md#providers) and the runnable [`Codex`](examples/config.codex-reasoning.yaml) / [`Claude Code`](examples/config.claude-code-thinking.yaml) configs.

## Python Entry Points

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
)
```

See [Python API](docs/python-api.md) for the full public surface.

## Status

Teich is alpha. The core trace, preparation, masking, and audit workflow is usable, but APIs may evolve as more agent formats and training flows are added.

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest --ignore=tests/test_integration.py -q
```

## License

Apache-2.0
