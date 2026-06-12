<div align="center">
  <img src="assets/teich.svg" alt="Teich logo" width="132">
  <h1>Teich</h1>
  <p><strong>Agent SFT data infrastructure for generation, normalization, chat-template rendering, response masking, and training audits.</strong></p>
  <p>
    <a href="https://pepy.tech/projects/teich"><img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/teich?label=downloads&color=green"></a>
    <a href="https://pypi.org/project/teich/"><img alt="PyPI" src="https://img.shields.io/pypi/v/teich?label=pypi&color=black"></a>
    <a href="https://pypi.org/project/teich/"><img alt="Python versions" src="https://img.shields.io/badge/python-%3E%3D3.10-green"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/teich?color=black"></a>
  </p>
</div>

Teich is not only a dataset generator. It is a bridge between messy agent/chat data and model-specific supervised fine-tuning.

Start from any of these:

- Fresh Codex, Pi, Claude Code, or Hermes traces
- Text-only chat generations
- Local JSONL files or folders
- Hugging Face datasets
- Already-loaded `datasets.Dataset` objects

Then Teich handles the training-sensitive parts:

- Normalize sources into OpenAI-style `messages` / `tools`
- Render through the target tokenizer chat template
- Preserve typed supervision spans before tokenization
- Apply response-only labels after trainer tokenization

That means the same package can:

- Generate new agent traces or chat-only distillation data.
- Load and normalize existing local or Hub datasets.
- Mix chat-only and tool-call datasets with explicit ratios.
- Preserve raw traces as source-of-truth artifacts.
- Render with arbitrary tokenizer chat templates.
- Mask assistant reasoning, final answers, and tool calls while keeping prompts/tool responses ignored.
- Report dropped, oversized, and trimmed rows without rerunning audits.
- Preserve provenance columns like `metadata`, `raw_index`, and `source_key` when requested.
- Validate tool-call names and required arguments against each row's declared tools.
- Audit labels before training so fully masked or misaligned rows fail early.

## Mental Model

```text
prompts / traces / JSONL / HF datasets / Dataset objects
        ↓
load_traces() or prepare_data()
        ↓
normalized messages + tools
        ↓
tokenizer chat template rendering
        ↓
trainer-friendly text + Teich supervision spans
        ↓
SFTTrainer tokenization
        ↓
mask_data()
        ↓
audited input_ids + labels
```

Use only the pieces you need:

- Already have a dataset? Skip generation and go straight to `prepare_data()`.
- Want raw trace preservation? Use the CLI.
- Want standard next-token training? Use `prepare_data(..., teich_masking=False)` and skip `mask_data()`.

## Entry Points

| Goal | Use |
| --- | --- |
| Generate coding-agent traces | `teich generate` with `agent.provider: codex`, `pi`, `claude-code`, or `hermes` |
| Generate text-only chat rows | `teich generate` with `agent.provider: chat` |
| Detect supported raw trace events | `detect_trace_type()` |
| Load raw traces manually | `load_traces()` |
| Prepare local/HF/mixed datasets for training | `prepare_data()` |
| Apply response-only labels after TRL/Unsloth tokenization | `mask_data()` |
| Inspect supervised vs masked tokens | `preview_sft_example()` / `trainer.train_dataset.preview()` |

## Install

```bash
pip install teich
```

To create a new generation project:

```bash
teich init my-project && cd my-project
teich generate -c config.yaml
```

Or use [astral-uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uvx teich init my-project && cd my-project
uvx teich generate -c config.yaml
```

> Edit `config.yaml` and `prompts.jsonl` before running a real generation batch.

## Core Capabilities

- **Trace-first data collection**: Run real coding agents and keep raw session traces as the source of truth.
- **Dataset-first training**: Load existing JSONL files, folders, Hugging Face repos, or `datasets.Dataset` objects without using the generator.
- **Multi-provider generation**: Works with Docker-backed Codex, Pi, Claude Code, Hermes, and a direct OpenAI-compatible `chat` mode.
- **Structured conversion**: Converts traces into chat messages with tool calls, reasoning, tool results, metadata, and configured tool snapshots.
- **Universal masking surface**: Supports assistant reasoning, final answers, tool calls, user/system/developer text, and tool responses as independently configurable masking targets.
- **Multi-turn and tool-aware labels**: Avoids Unsloth-style single-span masking pitfalls by storing typed spans before tokenization and aligning them after trainer tokenization.
- **Source mixing**: Mix local paths, Hub datasets, and in-memory datasets; explicit percentages stay true by scaling to the limiting source instead of silently changing ratios.
- **Hugging Face integration**: Publishes dataset cards with embedded tool-schema snapshots, and loads local or Hub datasets through one API.

## 📥 Prerequisites

Requirements for agent trace generation:

- Docker
- API key for the configured provider, such as OpenAI, OpenRouter, or Anthropic. Local OpenAI-compatible endpoints are also supported where the selected runner can use them.

The bundled Docker runtime runs agent CLIs as the non-root `codex` user, but includes passwordless `apt` / `apt-get` wrappers so generated agents can install missing system packages when a task needs them.

`agent.provider: chat` does not require Docker.

The Python utilities also work without Docker if you already have traces or structured JSONL datasets.

Training examples use your existing finetuning stack. For the TRL example below, install compatible versions of `transformers`, `trl`, and your model-loading stack separately.

## Common Workflows

### Prepare an existing dataset for training

You do not need to generate data with Teich first.

If a local file, folder, Hugging Face dataset, or `datasets.Dataset` has a `messages` column, Teich can usually prepare it directly.

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

`prepare_data()` returns rendered `text`, Teich span metadata, and optionally `input_ids` / `attention_mask`. With `oversized_policy="trim_followups"`, multi-turn rows that exceed `max_length` can drop the final user follow-up and everything after it before the whole row is discarded. Call `mask_data()` after constructing your trainer to convert those spans into labels.

For audit-friendly preparation, request a report and provenance columns:

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

### Mix agent and chat datasets

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

Explicit `percentage`, `proportion`, and `weight` values are treated as true ratios.

If one source cannot fill its share after filtering or context-window drops, Teich scales the total row count down instead of silently changing the realized mix.

Global `chat_template_kwargs` are the default for every source. A source-level `chat_template_kwargs` mapping overrides those keys for that dataset only, which lets one `prepare_data()` call mix reasoning and instruct datasets safely.

### Generate new data from prompts

```bash
# Initialize project
teich init my-project
cd my-project

# Add prompts to prompts.jsonl, then:
export OPENAI_API_KEY=sk-...
teich generate -c config.yaml
```

Outputs:

- `codex` / `pi`: normalized copies of native agent session JSONL files in `output/`, sandboxes in `sandbox/`, and a `README.md`
- `claude-code`: native Claude Code transcript JSONL copied from `.claude/projects/...`, sandboxes in `sandbox/`, and a `README.md`
- `hermes`: one Teich external trace JSONL per Hermes `state.db` session, including delegated subagent sessions as separate files linked by `parent_session_id`, sandboxes in `sandbox/`, and a `README.md`
- `chat`: text-only JSONL training rows in `output/` and a dataset `README.md`

Only completed runs are kept at the top level of `output/`. Failed or interrupted agent traces are moved to `failures/` for debugging, and Teich excludes `failures/` plus legacy `partials/` directories from resume detection, conversion, README generation, and Hugging Face uploads.

Generation progress reports provider/model usage when Teich can retrieve it. For OpenRouter, Teich first queries the provider's generation stats API for native token and cost accounting, then falls back to harness-reported usage. If neither source is available, Teich prints `N/A` instead of treating the missing value as zero.

If `publish.repo_id` is configured, Teich also creates or updates the matching Hugging Face **dataset** repo.

Uploaded artifacts include:

- Generated JSONL
- Dataset `README.md`
- Configured tool-schema snapshots in generated traces and in the dataset card when tools are present

If a long run is interrupted, use:

```bash
teich generate -c config.yaml --resume
```

Teich will scan existing outputs and skip prompts that already converted into completed training examples.

Prompt files can be JSONL/NDJSON, JSON, CSV, or plain text.

JSONL is recommended because it handles long multiline prompts, repository metadata, and follow-up turns without CSV escaping problems.

Recommended `prompts.jsonl`:

```jsonl
{"prompt":"Build a simple todo list app in React"}
{"github_repo":"armand0e/perplexica-mcp","prompt":"Add a small usability improvement and update the tests"}
{"system":"Answer as a concise project manager.","prompt":"Draft a compact project plan"}
{"prompt":"Draft a compact project plan","follow_up_prompts":["Revise it for a solo developer","Add a risk checklist"]}
```

`system` is optional and prompt-specific. If a row does not include `system`, Teich does not inject a default system prompt.

`follow_up_prompts` is supported across providers. `chat` sends each follow-up as a real additional user turn in one generated training row. Agent runners keep one Docker container alive for the full prompt sequence, run the initial prompt, then resume or continue the same saved agent session for each follow-up so workspace edits, tool caches, and in-container installs remain available across turns.

### Provider notes

- `codex` copies the native Codex session JSONL out of the mounted `CODEX_HOME/sessions` directory, then normalizes known Codex event-shape edge cases so reasoning summaries are visible and split assistant turns render as thinking before text/tool use. Teich appends configured `tool_schema` metadata so tools remain available for training even if the model did not call them.
- `pi` copies the native Pi session JSONL out of the mounted `/home/codex/pi-sessions` directory, then normalizes and validates tool-call structure before writing output. Teich appends prompt-level system metadata and configured tool metadata as `custom` events. For OpenRouter, Teich forces Pi onto the chat/completions wire path because Pi's OpenRouter Responses adapter can stall before the first session event.
- `openclaw` is recognized for imported raw traces when the first session event has `.openclaw` in its `cwd`. OpenClaw is not a Teich runner yet, so Teich only identifies and converts the raw events with `metadata.trace_type = "openclaw"` without applying Pi runner metadata snapshots.
- `claude-code` copies Claude Code's native transcript JSONL from `.claude/projects/...` so the output keeps Claude's own `user`, `assistant`, `system`, and `result` event format. Split assistant fragments are normalized so thinking appears before the text or tool use it explains. During conversion, Claude Code runtime context such as skill listings, MCP instruction deltas, permission context, date changes, hook context, and away summaries becomes masked `system` messages and `metadata.system_prompt`; local slash-command artifacts such as `/model` stay out of training messages, `/goal` contributes its actual user goal text, queued prompts become real user turns, and advertised native Claude Code / Claude Desktop tools receive schemas even when a tool is only declared through deferred-tool context. With OpenRouter non-Claude models, Teich runs a local in-container proxy: Claude Code sees a Claude surrogate model name, while the proxy rewrites outbound requests back to the configured model. The native assistant/result events keep the provider-returned model and usage fields when Claude Code records them.
- `hermes` runs with built-in toolsets `safe,terminal,file,skills,memory,session_search,delegation`, then exports each Hermes Agent `state.db` session as a Teich external trace: an `external_session_meta` event followed by explicit `external_message` events. Hermes' internal `system_prompt`, enabled toolsets, and configured tools remain metadata on each trace. Delegated subagents remain separate trace files rather than being merged into the orchestrator session; child traces include `parent_session_id`.
- `chat` calls an OpenAI-compatible API directly and writes structured training rows instead of raw agent traces.

When converting preserved traces to training rows, Teich normalizes split assistant fragments into model-turn order: `reasoning_content` first, optional assistant `content` second, and `tool_calls` last. If a backend emits thinking after a text or tool-call fragment, Teich moves that thinking back in front of the output it explains.

### Generate a text-only chat dataset

```yaml
agent:
  provider: chat

model:
  model: gpt-4.1-mini

api:
  provider: openai
  wire_api: responses
```

Each generated JSONL line will look like:

```json
{"messages":[{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini"}
```

With follow-ups, the same row contains:

- Alternating `user` and `assistant` messages
- `follow_up_prompts`
- Per-turn `responses`
- Final `response`

### Train with Unsloth and TRL `SFTTrainer`

Use the trainer-first path:

1. `prepare_data` renders trainer-friendly `text` rows with Teich supervision metadata.
2. `SFTTrainer` tokenizes them.
3. `mask_data` applies multi-turn/tool-aware response-only labels to the trainer dataset.

```python
import os

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from teich import mask_data, prepare_data

MAX_SEQ_LEN = 32768
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}
PUSH_TO_HUB_REPO_ID = "username/teich-sft-model"
HF_TOKEN = os.environ.get("HF_TOKEN") or ""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

train_dataset = prepare_data(
    "TeichAI/lordx64-claude-opus-4.7-max-cleaned",
    tokenizer,
    split="train",
    max_examples=500,
    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    max_length=MAX_SEQ_LEN,
    oversized_policy="trim_followups",
    tokenize=True,
    strict=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=MAX_SEQ_LEN,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="muon",
        optim_target_modules="all-linear",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        output_dir="outputs",
        seed=3407,
        report_to="none",
    ),
)
trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    train_on_reasoning=True,
    train_on_final_answers=True,
    train_on_tools=True,
)

trainer_stats = trainer.train(resume_from_checkpoint=False)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")

model.push_to_hub_merged(PUSH_TO_HUB_REPO_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
```

`prepare_data`:

- Loads local folders, local files, Hugging Face datasets, source mixes, or `datasets.Dataset` objects.
- Applies the tokenizer chat template.
- Applies `oversized_policy="drop"`, `"trim_followups"`, or `"error"` when `max_length` is set. The older `drop_oversized_examples` and `trim_oversized_followups` flags are still accepted as compatibility aliases.
- Returns trainer-friendly `text` rows with typed Teich span metadata.
- Can return a `PrepareReport` with dropped rows, oversized rows, trimmed rows, token lengths, and row ids via `return_report=True`.
- Can preserve source provenance columns with `preserve_columns=True` or an explicit list like `["metadata", "raw_index", "source_key"]`.
- Can fail early on undeclared or malformed tool calls with `validate_tools=True`.
- Supports `teich_masking=False` for plain next-token training without Teich response-only labels.

For Unsloth / TRL, pass `tokenize=True` so trainer setup treats the dataset as already tokenized and preserves Teich span metadata until `mask_data()` runs.

`mask_data`:

- Follows the same trainer-first shape as Unsloth's response-only helper.
- Uses Teich span metadata so multi-turn tool calls and tool responses are masked correctly.
- Trains on assistant reasoning, final answers, and tool calls by default.
- Keeps user/system/developer/tool-response text masked by default.
- Returns a compact trainer dataset with only `input_ids` and `labels`.

You can override the default policy with `train_on_reasoning`, `train_on_final_answers`, `train_on_tools`, `train_on_user`, `train_on_system`, `train_on_developer`, and `train_on_tool_responses`.

Keep `packing=False` for this flow because packed datasets merge row boundaries before masking. For long-context runs, `max_supervised_tokens` defaults to the trainer's `max_length` to cap the number of trainable answer tokens per row.

To combine datasets, pass a list of dataset IDs, local paths, or loaded `datasets.Dataset` objects:

```python
train_dataset = prepare_data(
    ["username/chat-traces", "username/tool-traces"],
    tokenizer,
    max_length=MAX_SEQ_LEN,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
)
```

For weighted mixes, explicit `percentage`, `proportion`, and `weight` values are treated as true ratios.

If one source cannot fill its share after filtering or context-window drops, Teich scales the total row count down instead of silently changing the realized mix.

### Fallback manual flow with `load_traces`

Use `load_traces` directly when you want to own the rest of the training pipeline yourself:

- Chat-template rendering
- Filtering
- Tokenization
- Label masking
- Packing policy
- Auditing

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

## 📋 Configuration

`config.yaml`:

```yaml
agent:
  provider: codex  # or pi, claude-code, hermes, or chat

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.jsonl

prompts: []
# For chat provider follow-up turns:
# prompts:
#   - prompt: "Draft a compact project plan"
#     follow_up_prompts:
#       - "Revise it for a solo developer"
#       - "Add a risk checklist"

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
  failures_dir: ./failures
  pretty_name: "My Agent Traces"

publish:
  repo_id: armand0e/my-dataset
  hf_token: hf_xxx
  private: false
```

Dataset tags are auto-generated from the provider and model:

- `codex` / `pi` / `claude-code` / `hermes`: `agent-traces`, `format:agent-traces`, `<provider>`, `distillation`, `<model>`, `teich`
- `chat`: `conversational`, `distillation`, `teich`, `<model>`

If `publish.hf_token` is omitted, Teich also accepts `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `TEICH_HF_TOKEN` from the environment.

### Local providers (LM Studio, Ollama)

```bash
export TEICH_PROVIDER=LMstudio
export TEICH_MODEL=gemma-4
export TEICH_BASE_URL=http://localhost:1234/v1
export TEICH_API_KEY=llm

teich generate -c config.yaml
```

## 🏗️ Data Structure

Training examples include:

- `prompt`: initial task description
- `follow_up_prompts`: optional additional chat turns generated after the initial prompt
- `messages`: chat history (system, user, assistant, tool)
- `tools`: tool schemas available to the session, including tools that were not called
- `metadata`: session info, model, timestamps, and usage when available

When the source format exposes per-message timestamps, converted rows include `metadata.first_message_timestamp` from the first timestamp-bearing source event that becomes a user message.

Native Claude Code imports may include masked `system` messages that preserve the runtime context Claude saw, including Claude Desktop skills, MCP instructions, hook context, permission state, and session recaps. That context also appears as `metadata.system_prompt` for auditability.

Structured chat datasets can also include convenience top-level fields like:

- `system` when provided by the prompt row
- `follow_up_prompts`
- `thinking`
- `response`
- `responses`
- `model`

Assistant messages capture:

- `content`: text response
- `reasoning_content`: chain-of-thought traces
- `tool_calls`: function calls with arguments

Some providers split a single model turn across multiple native events. Teich normalizes those fragments during raw trace copy and conversion so the semantic order is `reasoning_content`, optional assistant `content`, then `tool_calls`.

## 🔧 Python API

```python
from teich import (
    prepare_data,        # Recommended: render trainer-friendly text rows
    mask_data,           # Recommended: apply Teich labels after SFTTrainer tokenization
    detect_trace_type,   # Detect supported parsed trace events, or return None
    load_traces,         # Fallback: load rows for fully manual processing
    row_fits_context,    # Public chat-template render + token fit check for one row
    validate_tool_calls, # Validate tool-call names and required arguments
    trace_is_complete,   # Check that a row does not end on a tool result
    preview_sft_example, # Preview supervised vs masked tokens
    Config,              # Load config.yaml
    TrainingExample,     # Typed training example
)
```

`detect_trace_type(events)` returns `codex`, `claude_code`, `droid`, `pi`, `openclaw`, `hermes`, or `external_agent` for supported parsed trace events, and `None` for ordinary JSON rows.

Factory `droid` CLI sessions are supported as a conversion-only source: point `prepare_data()` or `load_traces()` at session JSONL files from `~/.factory/sessions/...` and Teich converts them directly, reading the adjacent `<session-id>.settings.json` sidecar for model and token usage metadata.

`README.md` is the package readme used for PyPI, so these examples are the canonical public package docs.

## 📦 Trace-First Workflow

Teich preserves the **raw or native captured agent session** as the source of truth:

1. **Collect**: Run agents on real tasks → raw `.jsonl` traces
2. **Inspect/Share**: Traces are human-readable and uploadable
3. **Convert**: Transform to structured examples when ready
4. **Prepare**: Use `prepare_data()` + `mask_data()` to apply model-specific templates and labels through the trainer-first flow

If you choose `agent.provider: chat`, Teich skips the trace-preservation step and writes structured text-only JSONL rows directly.

This means you can:

- Re-convert with different logic later
- Share raw traces before releasing training data
- Train on the same sessions with different model templates

## 🛠️ Development

```bash
uv pip install -e ".[dev]"
uv run pytest --ignore=tests/test_integration.py -q
```

## 📌 Status

Teich is **alpha**. The core workflow is stable and usable. APIs may evolve as more agent types and training workflows are added.

## 📄 License

Apache-2.0
