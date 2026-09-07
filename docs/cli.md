# CLI Reference

Use `teich --help` for the command list and `teich COMMAND --help` for command-specific options.

## Project Setup

Create a starter project:

```bash
teich init my-project
cd my-project
```

This writes `config.yaml` and `prompts.jsonl`. Edit both, set an API key such as `TEICH_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY`, then run generation.

## Generate

Run prompts from a config file:

```bash
teich generate -c config.yaml
```

Useful options:

- `--config`, `-c`: config path, default `config.yaml`
- `--output`, `-o`: override `output.traces_dir`
- `--concurrency`, `-j`: number of prompts to run in parallel
- `--resume`: skip prompts that already have completed outputs

Generated runs write raw traces, converted JSONL rows, and a compact dataset `README.md` under `output/` by default. Agent providers also write workspace snapshots under `sandbox/`; failed or interrupted traces go under `failures/`. Very large dataset-level tool snapshots are written to `tools.json` instead of being embedded in the dataset card.

## Extract

Extract existing local sessions:

```bash
teich extract claude --model fable-5 --out data
```

The first argument is the provider: `claude`, `codex`, `cursor`, `pi`, or `hermes`.

Useful options:

- `--sessions-dir`, `-s`: explicit agent root, native session folder, `state.db`, or JSONL file. Can be passed more than once.
- `--model`, `-m`: substring match against model metadata. For example, `fable-5` matches `claude-fable-5`.
- `--output`, `--out`, `-o`: staged dataset directory, default `data`
- `--no-anon`, `--no-anonymize`: skip automatic anonymization
- `--private`: if you choose the Hugging Face upload prompt, create the dataset as private

Default stores:

```text
claude  CLAUDE_CONFIG_DIR/projects, CLAUDE_HOME/projects, ~/.claude/projects
codex   CODEX_HOME/sessions, ~/.codex/sessions
pi      PI_SESSION_DIR, PI_CODING_AGENT_DIR/sessions, ~/.pi/agent/sessions, ~/.pi/sessions
hermes  HERMES_STATE_DB, HERMES_HOME/state.db, ~/.hermes/state.db
cursor  CURSOR_WORKSPACE_STORAGE, CURSOR_GLOBAL_STORAGE_DB, Cursor/User/workspaceStorage,
        Cursor/User/globalStorage/state.vscdb, ~/.cursor-server/data/User/workspaceStorage
```

Explicit paths can point at either the agent root or the native store:

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

Extraction anonymizes staged traces by default. Review the staged files before upload; `--no-anon` keeps raw values unchanged.

Generated extraction readmes include a short `teich extract` snippet so dataset users can stage their own local sessions with the same provider family.

## Convert

Convert raw or extracted traces into standalone OpenAI-style training JSONL:

```bash
teich convert data --out teich-training.jsonl
```

Each output line contains `prompt`, `messages`, `tools`, `metadata`, and an optional captured `system` field. Use this when your training stack can consume standalone OpenAI-style message rows without importing Teich. Use `prepare_data()` and `mask_data()` when you want tokenizer-specific rendering and exact response-only labels.

DeepSeek Harness (`dsh`) sessions are detected automatically, including compressed `.jsonl.zstd` and `.jsonl.zst` files:

```bash
teich convert /path/to/session.jsonl.zstd --out teich-training.jsonl
```

You can also pass a directory of sessions. See [DeepSeek Harness data](data-format.md#deepseek-harness) for supported content. DeepSeek Harness support is limited to parsing existing rollouts; it is not an option for `teich generate` or `teich extract`.

## Anonymize

Scrub common secrets and local usernames:

```bash
teich anonymize output --output output_anonymized
teich anonymize data --in-place
```

Anonymization replaces known credential formats, high-confidence secret assignments, personal email addresses, contextual PII, home-directory usernames, and embedded base64 media with deterministic dummy values. Media replacements are tiny decoder-valid payloads; unsupported subtypes are relabeled to the actual image, audio, or video placeholder format so multimodal loaders do not receive false MIME declarations. Reserved example-domain addresses, known public bot addresses, provider thinking signatures, placeholders, and common public IDs are preserved to avoid corrupting training data. User, assistant, reasoning, and structured tool content all pass through the same high-confidence privacy scanner. Reported totals are replacement occurrences rather than estimates of unique secrets. It is a best-effort pass; review data before publishing.

## Studio

Launch the browser UI:

```bash
teich studio
teich studio ./my-project
teich studio --host 127.0.0.1 --port 8420 --no-open
```

Studio edits the same `config.yaml` and `prompts.jsonl` files used by `teich generate`.

## Pool

`teich pool upload` is reserved for a future Teich community pool backend. For Hugging Face dataset uploads today, use `teich generate` or `teich extract` and confirm the upload prompt.
