# Teich

Turn coding agent sessions into training data.

**Generate** → **Convert** → **Train**

Run Codex or Pi, capture raw traces, and convert them into structured training examples for fine-tuning.

## ⚡ Quick Start

```bash
pip install teich
```

```bash
teich init my-project && cd my-project
teich generate -c config.yaml
```

## ⭐ What Teich Does

- **Trace-first data collection**: Run real coding agents, keep the raw session traces
- **Multi-agent support**: Works with Codex and Pi
- **Structured output**: Converts traces into chat messages with tool calls, reasoning, and tool results
- **SFT-ready formatting**: Applies chat templates and creates assistant masks for supervised fine-tuning
- **Hugging Face integration**: Load traces from local folders or dataset repos like `badlogicgames/pi-mono`

## 📥 Install

```bash
pip install teich
```

Requirements for trace generation:
- Docker
- OpenAI API key (or local OpenAI-compatible endpoint)

The Python utilities work without Docker if you already have traces.

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

Outputs: raw traces in `output/`, sandboxes in `sandbox/`, and a `README.md`.

### Convert traces to training data

```python
from teich import convert_traces_to_training_data
from pathlib import Path

examples = convert_traces_to_training_data(Path("./output"))
print(examples[0]["messages"])
```

### Load and format for training

```python
from teich import load_traces, format_and_mask

# Load from local folder or HF dataset
dataset = load_traces("badlogicgames/pi-mono", split="train")

# Apply chat template and create masks
training_data = format_and_mask(
    dataset,
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
  provider: codex  # or pi

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.csv

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
```

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
- `metadata`: session info, model, timestamps

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

Unlike synthetic datasets, Teich preserves the **raw agent session** as the source of truth:

1. **Collect**: Run agents on real tasks → raw `.jsonl` traces
2. **Inspect/Share**: Traces are human-readable and uploadable
3. **Convert**: Transform to structured examples when ready
4. **Format**: Apply model-specific chat templates for training

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
