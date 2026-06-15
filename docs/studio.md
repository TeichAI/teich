# Teich Studio

Teich Studio is a local browser UI for configuring, generating, inspecting, and steering Teich datasets.

Use it when you want a more interactive workflow than editing `config.yaml` and `prompts.jsonl` by hand.

## Launch

From a Teich project directory:

```bash
teich studio
```

Or point Studio at a project path:

```bash
teich studio ./my-project
```

Options:

```bash
teich studio --host 127.0.0.1 --port 8420 --no-open
```

Studio initializes the project if needed, starts a local FastAPI server, and opens the browser at:

```text
http://127.0.0.1:8420
```

If dependencies are missing, install the Studio extra:

```bash
pip install "teich[studio]"
```

## What Studio Does

Studio gives you browser controls for the same files and runners used by the CLI:

- edit `config.yaml`
- add, edit, import, and validate `prompts.jsonl`
- start and resume batch generation
- extract existing local agent sessions from browser-selected source paths
- watch generation progress and provider/model usage
- inspect generated traces
- open interactive agent sessions
- save an interactive session as a trace for training

It writes to the same project layout as `teich generate`:

- `output/`: completed traces and dataset card
- `sandbox/`: workspace snapshots
- `failures/`: failed or interrupted traces for debugging
- `config.yaml`: generation config
- `prompts.jsonl`: prompt rows

## Interactive Sessions

For agent providers (`pi`, `codex`, `claude-code`, `hermes`), Studio starts the same persistent Docker runtime used by batch generation, launches the agent CLI inside a real PTY, and streams the terminal to your browser over WebSocket.

That means you can:

- steer an agent manually
- inspect its terminal output
- keep workspace edits and in-container installs available across turns
- save the resulting native trace into the dataset output

For `chat`, Studio uses a message-by-message loop over the configured OpenAI-compatible API instead of a terminal.

Saved Studio traces go through the same converter path as batch traces, including provider-specific normalization, tool-schema preservation, and incomplete-trace filtering.

## Batch Generation

The batch tab is a UI wrapper over:

```bash
teich generate -c config.yaml
```

Use it to start a batch, monitor status, and resume a run without leaving the browser.

## Browser Extraction

The Extract tab is a UI wrapper over:

```bash
teich extract PROVIDER --sessions-dir PATH --out ./output
```

Use it to stage existing local sessions from `claude`, `codex`, `pi`, or `hermes` without leaving the browser. The source box accepts either the provider home folder or the provider's direct data path:

- Claude: `.claude` or `.claude/projects`
- Codex: `.codex` or `.codex/sessions`
- Pi: `.pi`, `.pi/agent/sessions`, or `.pi/sessions`
- Hermes: `.hermes` or `.hermes/state.db`

Studio can fill in detected default paths for the selected provider. Extraction writes into the configured output folder, generates a dataset `README.md`, and anonymizes staged traces by default. Check **Skip anonymization** only when you are intentionally keeping raw local values.

## Dataset Preview

The Dataset Preview tab lets you inspect the configured output folder before upload. It shows:

- JSONL files and row counts
- Hugging Face-style features for the converted training rows
- searchable row previews
- row JSON details
- local session/trace previews with messages, reasoning, tool calls, and tool results
- the generated dataset card preview when `README.md` is present

If `publish.repo_id` is set in `config.yaml`, Studio also exposes the official Hugging Face embedded Dataset Viewer URL:

```text
https://huggingface.co/datasets/<owner>/<dataset>/embed/viewer
```

That official embed works for datasets already available on the Hub. For unpublished local output, Studio uses Teich's local converter to approximate the parts of the viewer that matter before upload. The full Hugging Face viewer backend is hosted by Hugging Face and adds Parquet-backed row serving, search, filtering, SQL, and statistics after the dataset is uploaded and processed.

## Requirements

For agent providers:

- Docker
- provider API key or local endpoint config
- a model supported by the selected provider

For `chat`:

- provider API key or local OpenAI-compatible endpoint
- no Docker requirement

Studio binds to `127.0.0.1` by default. Use `--host` only when you intentionally want to expose the server beyond localhost.

## Troubleshooting

Port already in use:

```bash
teich studio --port 8421
```

Browser did not open:

```bash
teich studio --no-open
```

Then open the printed URL manually.

Missing FastAPI or Uvicorn:

```bash
pip install "teich[studio]"
```

Docker or provider errors are surfaced in the Studio UI and mirror the same runner behavior used by the CLI.
