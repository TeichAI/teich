from pathlib import Path
import json

from teich.trace_readme import README_INLINE_TOOLS_MAX_CHARS, build_traces_readme, write_traces_readme


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
