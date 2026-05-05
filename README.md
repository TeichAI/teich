# Teich

Turn coding agent sessions into training data.

**Generate** → **Convert** → **Train**

Run `codex` or `pi` to capture raw traces, or use `chat` mode to generate text-only training rows directly.

Easily format, filter, combine and mask any supported dataset(s)

## ⚡ Quick Start

```bash
pip install teich
```

```bash
teich init my-project && cd my-project
teich generate -c config.yaml
```
Or use [astral-uv](https://docs.astral.sh/uv/getting-started/installation/)
```bash
uvx teich init my-project && cd my-project
uvx teich generate -c config.yaml
```

> Be sure to edit your config.yaml and prompts.csv file as needed
## ⭐ What Teich Does

- **Trace-first data collection**: Run real coding agents and keep the raw session traces when you want full fidelity
- **Multi-agent support**: Works with Codex, Pi, and a text-only `chat` mode
- **Structured output**: Converts traces into chat messages with tool calls, reasoning, and tool results, or emits ready-to-train chat rows directly
- **SFT-ready formatting**: Applies chat templates and creates assistant masks for supervised fine-tuning
- **Hugging Face integration**: Load raw traces or structured JSONL datasets from local folders, files, or dataset repos

## 📥 Install

```bash
pip install teich
```

Requirements for agent trace generation:

- Docker
- OpenAI/OpenRouter API key (or local OpenAI-compatible endpoint)

`agent.provider: chat` does not require Docker. The Python utilities also work without Docker if you already have traces or structured JSONL datasets.

## 🚀 Usage

### Generate traces from prompts

```bash
# Initialize project
teich init my-project
cd my-project

# Add prompts to prompts.csv, then:
export OPENAI_API_KEY=sk-...
teich generate -c config.yaml
```

Outputs:

- `codex` / `pi`: raw traces in `output/`, sandboxes in `sandbox/`, and a `README.md`
- `chat`: text-only JSONL training rows in `output/` and a dataset `README.md`

If `publish.repo_id` is configured, Teich also creates or updates the matching Hugging Face **dataset** repo and uploads the generated JSONL, README, and `tools.json` automatically.

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
{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"system":"You are a helpful assistant","prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini"}
```

### Load and format for training

```python
from teich import load_traces, format_and_mask

# Load from local folder, local file, or HF dataset
tool_dataset = load_traces("badlogicgames/pi-mono", split="train")
chat_dataset = load_traces("./chat-output/chat.jsonl")

# Apply chat template and create masks across multiple datasets
training_data = format_and_mask(
    [tool_dataset, chat_dataset],
    tokenizer,
    chat_template_kwargs={"enable_thinking": True}
)

# Preview a formatted example
print(training_data.preview())
```

## 📋 Configuration

`config.yaml`:

```yaml
agent:
  provider: codex  # or pi or chat

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.csv

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
  pretty_name: "My Agent Traces"

publish:
  repo_id: armand0e/my-dataset
  hf_token: hf_xxx
  private: false
```

Dataset tags are auto-generated from the provider and model:

- `codex` / `pi`: `agent-traces`, `<provider>`, `distillation`, `<model>`, `teich`
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
- `messages`: chat history (system, user, assistant, tool)
- `tools`: tool schemas used in the session
- `metadata`: session info, model, timestamps, and usage when available

Structured chat datasets can also include convenience top-level fields like:

- `system`
- `thinking`
- `response`
- `model`

Assistant messages capture:

- `content`: text response
- `reasoning_content`: chain-of-thought traces
- `tool_calls`: function calls with arguments

## 🔧 Python API

```python
from teich import (
    load_traces,           # Load from folder or HF dataset
    format_and_mask,        # Apply chat template + assistant masks
    convert_traces_to_training_data,  # Convert raw traces to examples
    Config,                 # Load config.yaml
    TrainingExample         # Typed training example
)
```

## 📦 Trace-First Workflow

Teich preserves the **raw agent session** as the source of truth:

1. **Collect**: Run agents on real tasks → raw `.jsonl` traces
2. **Inspect/Share**: Traces are human-readable and uploadable
3. **Convert**: Transform to structured examples when ready
4. **Format**: Apply model-specific chat templates for training

If you choose `agent.provider: chat`, Teich skips the trace-preservation step and writes structured text-only JSONL rows directly.

This means you can:

- Re-convert with different logic later
- Share raw traces before releasing training data
- Train on the same sessions with different model templates

## 🛠️ Development

```bash
uv pip install -e ".[dev]"
pytest tests/test_formatter.py tests/test_loader.py -q
```

## 📌 Status

Teich is **alpha**. The core workflow is stable and usable. APIs may evolve as more agent types and training workflows are added.

## 📄 License

Apache-2.0
