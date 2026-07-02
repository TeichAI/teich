from pathlib import Path
import json

import pytest

from teich.trace_readme import (
    README_INLINE_TOOLS_MAX_CHARS,
    build_traces_readme,
    size_category,
    write_traces_readme,
)


def test_build_traces_readme_includes_model_and_references_tools(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        '{"type":"session_meta","payload":{"id":"session1"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build a todo app"}]}}\n'
        '{"type":"response_item","payload":{"type":"tool_schema","name":"bash","schema":{"description":"Run shell commands","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}}\n'
        '{"type":"response_item","payload":{"type":"function_call","name":"bash","call_id":"call_1","arguments":"{\\"command\\":\\"ls\\"}"}}\n',
        encoding="utf-8",
    )

    readme = build_traces_readme(
        pretty_name="Agentic Training Traces",
        trace_files=[trace_file],
        tags=["agent-traces", "codex"],
        model_id="codex-mini-latest",
    )

    assert "This dataset was generated using [teich](https://github.com/TeichAI/teich) by [TeichAI](https://huggingface.co/TeichAI)" in readme
    assert 'tags:' in readme
    assert '- "agent-traces"' in readme
    assert '- "codex"' in readme
    assert "Model metadata: `codex-mini-latest`" in readme
    assert "## Training-ready tools" in readme
    assert "tools.json" not in readme
    assert "<details>" in readme
    assert "<summary>Training-ready tool schema snapshot</summary>" in readme
    assert '"name": "bash"' in readme
    assert "## Training" in readme
    assert "run `teich convert`" in readme
    assert "OpenAI-style JSONL rows with `prompt`, `messages`, `tools`, and `metadata`" in readme
    assert "https://github.com/TeichAI/teich/blob/main/docs/training.md" in readme
    assert "https://github.com/TeichAI/teich/blob/main/docs/prepare-data.md" in readme
    assert "from unsloth import FastLanguageModel" not in readme
    assert "import torch" not in readme
    assert "MODEL_NAME = 'unsloth/Qwen3.5-0.8B'" not in readme
    assert "TRAIN_ON_REASONING" not in readme
    assert "Preparing Data" in readme
    assert "mask_data" not in readme
    assert "dataset_text_field='text'" not in readme
    assert "max_length=MAX_SEQ_LEN" not in readme
    assert "max_examples=500" not in readme
    assert "tokenize=True" not in readme
    assert "packing=False" not in readme
    assert "trainer_stats = trainer.train(resume_from_checkpoint=False)" not in readme
    assert "torch.cuda.get_device_properties(0)" not in readme
    assert "Peak reserved memory" not in readme
    assert "train_on_reasoning=True" not in readme
    assert "model.push_to_hub_merged" not in readme
    assert "load_traces" in readme
    assert "format_and_mask" not in readme
    assert "tokenizer.apply_chat_template" not in readme
    assert "tools=example.get('tools') or []" not in readme
    assert "dataset = load_traces('username/repo')" not in readme
    assert "train_dataset = prepare_data(" not in readme
    assert "You can combine this dataset with other Teich chat-only or tool-call datasets" not in readme
    assert "['username/repo', 'username/other-teich-dataset']" not in readme
    assert "For weighted mixes" not in readme
    assert "Explicit ratios stay true" not in readme
    assert "source-level `chat_template_kwargs` override those keys" not in readme
    assert "'agent': {'source': 'username/repo', 'percentage': 80}" not in readme
    assert "'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False}" not in readme
    assert "convert_traces_to_training_data" not in readme


def test_write_traces_readme_embeds_tools_snapshot_in_readme(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        '{"type":"session_meta","payload":{"id":"session1"}}\n'
        '{"type":"response_item","payload":{"type":"tool_schema","name":"bash","schema":{"description":"Run shell commands","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}}}\n'
        '{"type":"response_item","payload":{"type":"function_call","name":"bash","call_id":"call_1","arguments":"{\\"command\\":\\"ls\\"}"}}\n',
        encoding="utf-8",
    )

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Agentic Training Traces",
        tags=["agent-traces"],
        model_id="test-model",
    )

    assert readme_path.exists()
    readme = readme_path.read_text(encoding="utf-8")
    assert not (tmp_path / "tools.json").exists()
    assert "<details>" in readme
    assert "<summary>Training-ready tool schema snapshot</summary>" in readme
    assert '"name": "bash"' in readme
    assert '"description": "Run shell commands"' in readme
    assert '"additionalProperties": false' in readme


def test_write_traces_readme_externalizes_large_tools_snapshot(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        '{"type":"response_item","payload":{"type":"function_call","name":"huge_tool","call_id":"call_1","arguments":"{}"}}\n',
        encoding="utf-8",
    )
    huge_description = "x" * (README_INLINE_TOOLS_MAX_CHARS + 1)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "huge_tool",
                "description": huge_description,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "description": huge_description}},
                },
            },
        }
    ]

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Agentic Training Traces",
        tags=["agent-traces"],
        model_id="test-model",
        tools=tools,
    )

    readme = readme_path.read_text(encoding="utf-8")
    tools_json = (tmp_path / "tools.json").read_text(encoding="utf-8")
    assert "The complete dataset-level tool schema snapshot was written to `tools.json`" in readme
    assert "<summary>Tool names in snapshot</summary>" in readme
    assert "<summary>Training-ready tool schema snapshot</summary>" not in readme
    assert huge_description not in readme
    assert huge_description in tools_json


def test_write_traces_readme_omits_example_when_sample_is_too_large(tmp_path: Path):
    trace_file = tmp_path / "huge-session.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "x" * 10_000},
            {"role": "assistant", "content": "y" * 10_000},
            {"role": "tool", "content": "z" * 10_000},
            {"role": "assistant", "content": "a" * 10_000},
            {"role": "user", "content": "b" * 10_000},
            {"role": "assistant", "content": "c" * 10_000},
        ],
        "metadata": {f"key_{index}": "m" * 10_000 for index in range(6)},
    }
    trace_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Large Agent Traces",
        tags=["agent-traces"],
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert "## Example" not in readme
    assert "__truncated__" not in readme
    assert "Additional sample content omitted" not in readme
    assert "## Training" in readme


def test_write_traces_readme_includes_extraction_snippet_when_provider_is_set(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        '{"type":"session_meta","payload":{"id":"session1"}}\n',
        encoding="utf-8",
    )

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Extracted Cursor Traces",
        tags=["agent-traces", "cursor"],
        model_id="cursor-local-sessions",
        extraction_provider="cursor",
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert "## Create A Similar Dataset" in readme
    assert "teich extract cursor --out data" in readme
    assert "--sessions-dir /path/to/store" in readme
    assert "--model <substring>" in readme


def test_write_traces_readme_collects_tools_around_malformed_rows(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        '{"type":"session_meta","payload":{"id":"session1"}}\n'
        '{not json\n'
        '{"type":"response_item","payload":{"type":"tool_schema","name":"bash","schema":{"description":"Run shell commands","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}}\n',
        encoding="utf-8",
    )

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Agentic Training Traces",
        tags=["agent-traces"],
        model_id="test-model",
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert "## Training-ready tools" in readme
    assert '"name": "bash"' in readme


def test_write_traces_readme_skips_configured_failures_dir(tmp_path: Path):
    failed_dir = tmp_path / "failed-traces"
    failed_dir.mkdir()
    (failed_dir / "failed.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"failed"}}\n'
        '{"type":"response_item","payload":{"type":"tool_schema","name":"bad_tool","schema":{"description":"Failed tool","parameters":{"type":"object","properties":{}}}}}\n',
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"ok"}}\n'
        '{"type":"response_item","payload":{"type":"tool_schema","name":"bash","schema":{"description":"Run shell commands","parameters":{"type":"object","properties":{}}}}}\n',
        encoding="utf-8",
    )

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Agentic Training Traces",
        tags=["agent-traces"],
        excluded_dirs=[failed_dir],
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert '"name": "bash"' in readme
    assert "bad_tool" not in readme


def test_write_traces_readme_embeds_claude_code_builtin_tools(tmp_path: Path):
    trace_file = tmp_path / "claude-code.jsonl"
    trace_file.write_text(
        '{"type":"user","message":{"role":"user","content":"Inspect repo"},"sessionId":"claude-session"}\n'
        '{"type":"assistant","message":{"role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"I will inspect it."}]},"sessionId":"claude-session"}\n',
        encoding="utf-8",
    )

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Claude Code Traces",
        tags=["claude-code"],
        model_id="claude-sonnet-4-6",
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert "## Training-ready tools" in readme
    assert '"name": "Bash"' in readme
    assert '"name": "TodoWrite"' in readme
    assert '"command"' in readme
    assert '"todos"' in readme


def test_write_traces_readme_for_structured_chat_dataset_skips_tools_json(tmp_path: Path):
    trace_file = tmp_path / "chat.jsonl"
    trace_file.write_text(
        '{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"system":"You are a helpful assistant","prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini"}\n',
        encoding="utf-8",
    )

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Chat Dataset",
        tags=["chat"],
        model_id="gpt-4.1-mini",
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert "task_categories:\n- text-generation\n" in readme
    assert '- "chat"' in readme
    assert "newline-delimited JSON training examples generated by teich" in readme
    assert "Rows: 1" in readme
    assert "JSONL files:" not in readme
    assert "dataset = load_traces('username/repo')" not in readme
    assert "train_dataset = prepare_data(" not in readme
    assert "You can combine this dataset with other Teich chat-only or tool-call datasets" not in readme
    assert "['username/repo', 'username/other-teich-dataset']" not in readme
    assert "trainer = mask_data(" not in readme
    assert "train_on_reasoning=True" not in readme
    assert "TRAIN_ON_REASONING" not in readme
    assert "prepare_data(..., teich_masking=False)" not in readme
    assert "run `teich convert`" in readme
    assert "https://github.com/TeichAI/teich/blob/main/docs/training.md" in readme
    assert "https://github.com/TeichAI/teich/blob/main/docs/prepare-data.md" in readme
    assert "tools=example.get('tools') or []" not in readme
    assert "Chat-only datasets include `messages` plus convenience fields like optional `system`, `prompt`, `follow_up_prompts`, `thinking`, `response`, and `responses`." in readme
    assert "## Training-ready tools" not in readme
    assert not (tmp_path / "tools.json").exists()


def test_write_traces_readme_uses_configured_tools_and_repo_id(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        '{"type":"response_item","payload":{"type":"function_call","name":"observed","call_id":"call_1","arguments":"{}"}}\n',
        encoding="utf-8",
    )
    configured_tools = [
        {
            "type": "function",
            "function": {
                "name": "unobserved",
                "description": "Available but not called",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]

    readme_path = write_traces_readme(
        tmp_path,
        pretty_name="Agentic Training Traces",
        tags=["agent-traces"],
        model_id="test-model",
        repo_id="armand0e/teich-test",
        tools=configured_tools,
    )

    readme = readme_path.read_text(encoding="utf-8")
    assert "Use this dataset as `armand0e/teich-test`" in readme
    assert "dataset = load_traces('armand0e/teich-test')" not in readme
    assert "    'armand0e/teich-test'," not in readme
    assert not (tmp_path / "tools.json").exists()
    assert '"name": "unobserved"' in readme
    assert '"description": "Available but not called"' in readme


def _write_structured_row(path: Path, **extra) -> None:
    row = {"prompt": "hi", "messages": [{"role": "assistant", "content": "ok"}], **extra}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_readme_is_reward_aware_from_verification_sidecars(tmp_path: Path):
    # Two verified rows with sidecars; the card summarizes their pass/fail counts.
    _write_structured_row(tmp_path / "task-a.jsonl", passed=True, reward=1.0)
    _write_structured_row(tmp_path / "task-b.jsonl", passed=False, reward=0.0)
    verification = tmp_path / "verification"
    verification.mkdir()
    (verification / "task-a.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (verification / "task-b.json").write_text(
        json.dumps({"passed": False, "reward": 0.0}), encoding="utf-8"
    )
    # A harbor intermediate that must NOT be treated as data or pollute the card (a nested
    # `bench` dir is excluded by name; by default bench_dir is a sibling, never under output).
    bench_sessions = tmp_path / "bench" / "sessions" / "add-bug"
    bench_sessions.mkdir(parents=True)
    (bench_sessions / "pi.jsonl").write_text('{"type":"session","id":"x"}\n', encoding="utf-8")

    readme_path = write_traces_readme(
        tmp_path, pretty_name="Verifiable Traces", tags=["agent-traces"]
    )
    readme = readme_path.read_text(encoding="utf-8")
    assert "## Reward labels" in readme
    assert '- "reward-labeled"' in readme
    assert "Verified tasks: 2 (1 passed / 1 failed)." in readme
    assert "1 of 2 carry an explicit numeric score" in readme  # task-b has reward 0.0
    assert "Rows: 2" in readme  # the bench/ session file is excluded from the row count
    # The train split is a top-level glob that excludes the nested bench/ session.
    assert "- split: train" in readme
    assert 'path: "*.jsonl"' in readme
    assert "**/*.jsonl" not in readme  # top-level glob, not recursive
    assert "bench/sessions/add-bug/pi.jsonl" not in readme


def test_card_data_files_reach_nested_extractions(tmp_path: Path):
    # Cursor extraction writes transcripts under nested project-relative dirs. A bare top-level
    # `*.jsonl` would advertise an empty dataset while the card still counts these rows, so the
    # HF data_files config must include a recursive glob for each data-bearing subdir.
    nested = tmp_path / "c-Users-test-project" / "agent-transcripts" / "session-1"
    nested.mkdir(parents=True)
    _write_structured_row(nested / "session-1.jsonl")
    # a non-data dir (excluded by name) must NOT be advertised
    (tmp_path / "failures").mkdir()
    _write_structured_row(tmp_path / "failures" / "oops.jsonl")

    readme = write_traces_readme(tmp_path, pretty_name="Cursor Traces", tags=["agent-traces"]).read_text(
        encoding="utf-8"
    )
    assert "- split: train" in readme
    assert 'path: "*.jsonl"' in readme  # top-level files still advertised
    assert 'path: "c-Users-test-project/**/*.jsonl"' in readme  # nested extraction reached
    assert "failures/" not in readme  # non-data dir not advertised
    assert "Rows: 1" in readme  # the nested row is counted (and now reachable)


def test_reward_stats_from_bench_metadata_sidecars(tmp_path: Path):
    # Bench harvest writes metadata/<stem>.json (split + numeric reward), not verification/.
    # The card's reward summary must read those so bench datasets show pass/fail counts.
    for split, stem in (("passed", "bench-ds-a"), ("failed", "bench-ds-b")):
        (tmp_path / split).mkdir(exist_ok=True)
        _write_structured_row(tmp_path / split / f"{stem}.jsonl")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "bench-ds-a.json").write_text(
        json.dumps({"task": "a", "split": "passed", "reward": 1.0}), encoding="utf-8"
    )
    (metadata / "bench-ds-b.json").write_text(
        json.dumps({"task": "b", "split": "failed", "reward": 0.0}), encoding="utf-8"
    )
    readme = write_traces_readme(tmp_path, pretty_name="Bench", tags=["agent-traces"]).read_text(
        encoding="utf-8"
    )
    assert "## Reward labels" in readme
    assert "Verified tasks: 2 (1 passed / 1 failed)." in readme
    assert "2 of 2 carry an explicit numeric score" in readme


def test_reward_stats_counts_borderline_bench_tasks(tmp_path: Path):
    # A partial score (0<r<1) routes to borderline with a numeric reward; it must be counted as a
    # verified task, not dropped (an all-borderline run must still show the reward section).
    rows = (("passed", "bench-a", 1.0), ("failed", "bench-b", 0.0), ("borderline", "bench-c", 0.6))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for split, stem, reward in rows:
        (tmp_path / split).mkdir(exist_ok=True)
        _write_structured_row(tmp_path / split / f"{stem}.jsonl")
        (metadata / f"{stem}.json").write_text(
            json.dumps({"split": split, "reward": reward}), encoding="utf-8"
        )
    readme = write_traces_readme(tmp_path, pretty_name="Bench", tags=["agent-traces"]).read_text(
        encoding="utf-8"
    )
    assert "Verified tasks: 3 (1 passed / 1 failed / 1 borderline)." in readme
    assert "3 of 3 carry an explicit numeric score" in readme


def test_card_data_files_exclude_custom_failures_dir(tmp_path: Path):
    # A custom output.failures_dir located under traces_dir (basename not in the reserved set)
    # must not be advertised as a train split — those are failed/interrupted runs, ignored on
    # upload — while a real data-bearing subdir still is.
    (tmp_path / "failed-runs").mkdir()
    _write_structured_row(tmp_path / "failed-runs" / "bad.jsonl")
    (tmp_path / "proj").mkdir()
    _write_structured_row(tmp_path / "proj" / "good.jsonl")

    readme = write_traces_readme(
        tmp_path, pretty_name="X", tags=["agent-traces"], excluded_dirs=[tmp_path / "failed-runs"]
    ).read_text(encoding="utf-8")
    assert 'path: "proj/**/*.jsonl"' in readme  # real nested data advertised
    assert "failed-runs" not in readme  # custom failures dir not advertised


def test_readme_has_no_reward_section_without_verification(tmp_path: Path):
    _write_structured_row(tmp_path / "plain.jsonl")
    readme_path = write_traces_readme(tmp_path, pretty_name="Plain Traces", tags=["agent-traces"])
    readme = readme_path.read_text(encoding="utf-8")
    assert "## Reward labels" not in readme
    assert "reward-labeled" not in readme
    # No routing folders -> single top-level train glob (not the recursive **/*.jsonl).
    assert "- split: train" in readme and 'path: "*.jsonl"' in readme
    assert "**/*.jsonl" not in readme


def test_card_splits_reflect_routing_folders(tmp_path: Path):
    for split in ("passed", "failed", "borderline"):
        (tmp_path / split).mkdir()
        _write_structured_row(tmp_path / split / f"bench-{split}.jsonl")
    readme = write_traces_readme(tmp_path, pretty_name="Bench", tags=["agent-traces"]).read_text(encoding="utf-8")
    for split in ("passed", "failed", "borderline"):
        assert f"- split: {split}" in readme
        assert f'path: "{split}/*.jsonl"' in readme
    assert "split: train" not in readme
    assert '- "reward-labeled"' in readme  # routed datasets are reward-labeled


def test_card_omits_empty_routing_splits(tmp_path: Path):
    # Only passed/ has files -> only a passed split is declared (no empty failed/borderline).
    (tmp_path / "passed").mkdir()
    _write_structured_row(tmp_path / "passed" / "bench-a.jsonl")
    readme = write_traces_readme(tmp_path, pretty_name="Bench", tags=["x"]).read_text(encoding="utf-8")
    assert "- split: passed" in readme
    assert "- split: failed" not in readme and "- split: borderline" not in readme


# Golden byte-for-byte lock for the Jinja-rendered card. The body below is identical to the
# pre-refactor (hand-assembled) build_traces_readme output; the only structural addition is the
# auto-populated ``size_categories`` block (always-on now). Any unintended drift in the template
# or context wiring will fail this exact-equality check.
_GOLDEN_CHAT_CARD = '''---
pretty_name: "Chat Dataset"
task_categories:
- text-generation
tags:
- "chat"
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.jsonl"
size_categories:
- n<1K
---

This dataset was generated using [teich](https://github.com/TeichAI/teich) by [TeichAI](https://huggingface.co/TeichAI) <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6837935ac3b7ffe0d2559ce9/-AxyvV4wfUY8uo87kNKkK.png" width="20" height="20" style="display: inline-block; vertical-align: middle; margin: 0 3px;">

# Chat Dataset

This directory contains newline-delimited JSON training examples generated by teich.

Rows: 1

Model metadata: `gpt-4.1-mini`

## Format

Each file is newline-delimited JSON where every line is already a training example.
Chat-only datasets include `messages` plus convenience fields like optional `system`, `prompt`, `follow_up_prompts`, `thinking`, `response`, and `responses`.
Tool datasets can include the same normalized `messages` structure together with a `tools` field.

## Example

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant", "thinking": null}, {"role": "user", "content": "Hello", "thinking": null}, {"role": "assistant", "content": "Hi!", "thinking": "I should greet the user."}], "system": "You are a helpful assistant", "prompt": "Hello", "thinking": "I should greet the user.", "response": "Hi!", "model": "gpt-4.1-mini"}
```

## Training

Use this dataset as `username/repo` with Teich's data preparation and training utilities.
If you do not want Teich to handle chat-template formatting or masking, run `teich convert` to write standalone OpenAI-style JSONL rows with `prompt`, `messages`, `tools`, and `metadata`.
Training setup details evolve over time, so the maintained guide lives in the [Teich training docs](https://github.com/TeichAI/teich/blob/main/docs/training.md).
For loading, mixing, converting, and validating Teich datasets, see [Preparing Data](https://github.com/TeichAI/teich/blob/main/docs/prepare-data.md).
'''


def test_rendered_card_matches_golden_byte_for_byte(tmp_path: Path):
    trace_file = tmp_path / "chat.jsonl"
    trace_file.write_text(
        '{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},'
        '{"role":"user","content":"Hello","thinking":null},'
        '{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],'
        '"system":"You are a helpful assistant","prompt":"Hello",'
        '"thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini"}\n',
        encoding="utf-8",
    )
    readme = write_traces_readme(
        tmp_path, pretty_name="Chat Dataset", tags=["chat"], model_id="gpt-4.1-mini"
    ).read_text(encoding="utf-8")
    assert readme == _GOLDEN_CHAT_CARD


@pytest.mark.parametrize(
    "row_count, expected",
    [
        (0, None),
        (-5, None),
        (999, "n<1K"),
        (1000, "1K<n<10K"),
        (9999, "1K<n<10K"),
        (10_000, "10K<n<100K"),
        (999_999, "100K<n<1M"),
        (1_000_000, "1M<n<10M"),
        (1_000_000_000_000, "n>1T"),
        (5_000_000_000_000, "n>1T"),
    ],
)
def test_size_category_bucket_boundaries(row_count, expected):
    assert size_category(row_count) == expected


def test_card_includes_license_when_set(tmp_path: Path):
    _write_structured_row(tmp_path / "row.jsonl")
    readme = write_traces_readme(
        tmp_path, pretty_name="Licensed", tags=["x"], license="apache-2.0"
    ).read_text(encoding="utf-8")
    assert "license: apache-2.0\n" in readme
    # Placed inside the frontmatter block, before the closing fence.
    frontmatter = readme.split("---", 2)[1]
    assert "license: apache-2.0" in frontmatter


def test_card_omits_license_when_unset(tmp_path: Path):
    _write_structured_row(tmp_path / "row.jsonl")
    readme = write_traces_readme(tmp_path, pretty_name="Unlicensed", tags=["x"]).read_text(
        encoding="utf-8"
    )
    assert "license:" not in readme


def test_card_extra_keys_appear_and_reserved_keys_dropped(tmp_path: Path):
    _write_structured_row(tmp_path / "row.jsonl")
    readme = write_traces_readme(
        tmp_path,
        pretty_name="Extra",
        tags=["x"],
        card_extra={
            "annotations_creators": ["machine-generated"],
            "language": ["en"],
            # Reserved keys the card owns must be dropped, not allowed to shadow.
            "tags": ["should-not-appear"],
            "pretty_name": "should-not-appear",
            "size_categories": ["should-not-appear"],
            "license": "should-not-appear",
        },
    ).read_text(encoding="utf-8")
    frontmatter = readme.split("---", 2)[1]
    assert "annotations_creators:" in frontmatter
    assert "machine-generated" in frontmatter
    assert "language:" in frontmatter
    assert "should-not-appear" not in readme


def test_card_extra_empty_dict_adds_no_lines(tmp_path: Path):
    _write_structured_row(tmp_path / "row.jsonl")
    baseline = write_traces_readme(tmp_path, pretty_name="Plain", tags=["x"]).read_text(
        encoding="utf-8"
    )
    with_empty = write_traces_readme(
        tmp_path, pretty_name="Plain", tags=["x"], card_extra={}
    ).read_text(encoding="utf-8")
    assert baseline == with_empty


def test_custom_readme_template_override_renders(tmp_path: Path):
    template = tmp_path / "card.md.j2"
    template.write_text(
        "---\npretty_name: \"{{ pretty_name }}\"\n---\nCustom card for {{ pretty_name }}, rows {{ rows_line }}.\n",
        encoding="utf-8",
    )
    _write_structured_row(tmp_path / "row.jsonl")
    readme = write_traces_readme(
        tmp_path, pretty_name="Override", tags=["x"], readme_template=template
    ).read_text(encoding="utf-8")
    assert readme == '---\npretty_name: "Override"\n---\nCustom card for Override, rows Rows: 1.\n'
