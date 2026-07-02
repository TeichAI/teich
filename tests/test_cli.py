"""Tests for CLI module."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from teich.cli import BatchProgressReporter, CONFIG_TEMPLATE, app
from teich.runner import SessionProgressUpdate, TraceMetrics

runner = CliRunner()


def test_init_command(tmp_path: Path):
    """Test init command creates files."""
    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    assert "Created" in result.output

    # Check files were created
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "prompts.jsonl").exists()


def test_init_command_existing_files(tmp_path: Path):
    """Test init command handles existing files."""
    # Create existing file
    (tmp_path / "config.yaml").write_text("existing")

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    assert "Already exists" in result.output


def test_checked_in_config_example_matches_init_template():
    example_path = Path(__file__).resolve().parent.parent / "config.example.yaml"
    assert example_path.read_text(encoding="utf-8") == CONFIG_TEMPLATE


def test_generate_command_missing_config():
    """Test generate command fails with missing config."""
    result = runner.invoke(app, ["generate", "-c", "/nonexistent/config.yaml"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_generate_rejects_unknown_mode(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("prompts:\n  - hello\n", encoding="utf-8")
    result = runner.invoke(app, ["generate", "-c", str(config_file), "--mode", "wat"])
    assert result.exit_code == 1
    assert "Unknown --mode" in result.output


def test_generate_bench_mode_requires_source(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("agent:\n  provider: codex\n", encoding="utf-8")  # no bench.sources
    result = runner.invoke(app, ["generate", "-c", str(config_file), "--mode", "bench"])
    assert result.exit_code == 1
    assert "bench.sources" in result.output


def test_generate_bench_refuses_to_mix_with_prompts_output(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "some-trace.jsonl").write_text('{"messages": []}\n', encoding="utf-8")  # prompts data
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"agent:\n  provider: pi\nbench:\n  sources:\n    - {{type: harbor, source: {tmp_path}/tasks}}\noutput:\n  traces_dir: {output}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["generate", "-c", str(config_file), "--mode", "bench"])
    assert result.exit_code == 1
    assert "already contains prompts-mode data" in " ".join(result.output.split())


def test_generate_bench_writes_readme_for_partial_dataset_on_failure(tmp_path: Path, monkeypatch):
    # An earlier bench source may harvest rows before a later one fails; the partial dataset must
    # still get a README (like prompt mode) even though the CLI exits non-zero.
    output = tmp_path / "output"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"agent:\n  provider: pi\nbench:\n  sources:\n    - {{type: harbor, source: {tmp_path}/tasks}}\noutput:\n  traces_dir: {output}\n",
        encoding="utf-8",
    )

    def _partial_then_fail(cfg, **kwargs):
        row = cfg.output.traces_dir / "passed" / "bench-x.jsonl"
        row.parent.mkdir(parents=True, exist_ok=True)
        row.write_text('{"messages": []}\n', encoding="utf-8")
        raise RuntimeError("a later bench source blew up")

    monkeypatch.setattr("teich.bench.run_bench", _partial_then_fail)
    result = runner.invoke(app, ["generate", "-c", str(config_file), "--mode", "bench"])
    assert result.exit_code == 1
    assert (output / "README.md").exists()  # partial dataset still documented


def test_generate_prompts_refuses_to_mix_with_bench_output(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "bench-add-bug.jsonl").write_text('{"messages": []}\n', encoding="utf-8")  # bench data
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"agent:\n  provider: chat\nprompts:\n  - hello\noutput:\n  traces_dir: {output}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["generate", "-c", str(config_file)])
    assert result.exit_code == 1
    assert "already contains bench-mode data" in " ".join(result.output.split())


def test_existing_dataset_modes_classifies_rows(tmp_path: Path):
    from teich.cli import _existing_dataset_modes

    (tmp_path / "bench-x.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    (tmp_path / "organic.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    (tmp_path / "passed").mkdir()
    (tmp_path / "passed" / "routed.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    # Intermediates / empties must not count as dataset rows (a nested `bench` dir is excluded
    # by name; normally bench_dir is a sibling of output and never under traces_dir at all).
    (tmp_path / "bench" / "sessions").mkdir(parents=True)
    (tmp_path / "bench" / "sessions" / "pi.jsonl").write_text('{"type":"session"}\n', encoding="utf-8")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    assert _existing_dataset_modes(tmp_path) == {"bench", "prompts"}


def test_convert_command_writes_openai_style_training_jsonl(tmp_path: Path):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    (traces_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "session-1", "model": "codex-fable-5"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Build a CLI"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Built it."}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "teich.jsonl"

    result = runner.invoke(app, ["convert", str(traces_dir), "--output", str(output_file)])

    assert result.exit_code == 0, result.output
    assert "Converted 1 trace row to" in result.output
    row = json.loads(output_file.read_text(encoding="utf-8"))
    assert row["prompt"] == "Build a CLI"
    assert row["messages"] == [
        {"role": "user", "content": "Build a CLI"},
        {"role": "assistant", "content": "Built it."},
    ]
    assert row["metadata"]["session_id"] == "session-1"
    assert row["metadata"]["model"] == "codex-fable-5"
    assert "tools" in row


def test_convert_command_rejects_missing_input(tmp_path: Path):
    result = runner.invoke(app, ["convert", str(tmp_path / "missing"), "--output", str(tmp_path / "out.jsonl")])

    assert result.exit_code == 1
    assert "Input path not found" in result.output


def test_generate_command_no_prompts(tmp_path: Path):
    """Test generate command fails when no prompts configured."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  model: codex-mini-latest
prompts: []
""")

    result = runner.invoke(app, ["generate", "-c", str(config_file)])

    assert result.exit_code == 1
    assert "No prompts configured" in result.output


def test_generate_command_accepts_follow_up_prompts_for_codex_provider(tmp_path: Path):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        json.dumps({"prompt": "Build app", "follow_up_prompts": ["Add tests"]}) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: codex
model:
  model: codex-mini-latest
prompts_file: {prompts_file}
output:
  traces_dir: {tmp_path}/output
openai_api_key: sk-test
""")

    with patch('teich.cli.CodexRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/session1.jsonl"]
        mock_runner.return_value = mock_instance

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "session1.jsonl").write_text(
            '{"type":"session_meta","payload":{"id":"session1","base_instructions":{"text":"You are a coding agent."}}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build app"}]}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}}\n',
            encoding="utf-8",
        )

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

    assert result.exit_code == 0
    assert "Success" in result.output
    kwargs = mock_instance.run_all.call_args.kwargs
    assert kwargs["prompt_inputs"][0].follow_up_prompts == ["Add tests"]


def test_generate_command_success_codex(tmp_path: Path):
    """Test generate command runs successfully with codex provider."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: codex
model:
  model: codex-mini-latest
prompts:
  - Build a todo app
output:
  traces_dir: {tmp_path}/output
openai_api_key: sk-test
""")

    with patch('teich.cli.CodexRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/session1.jsonl"]
        mock_runner.return_value = mock_instance

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "session1.jsonl").write_text(
            '{"type":"session_meta","payload":{"id":"session1","base_instructions":{"text":"You are a coding agent.\\n\\nAvailable tools:\\n- bash: Run shell commands\\n"}}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build a todo app"}]}}\n'
            '{"type":"response_item","payload":{"type":"function_call","name":"bash","call_id":"call_1","arguments":"{\\"command\\":\\"ls\\"}"}}\n'
        )

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

        assert result.exit_code == 0
        assert "Success" in result.output
        assert "Generated 1 JSONL files" in result.output
        assert "tokens=" in result.output
        assert "api_tokens=" not in result.output
        assert "est_total_tokens=" not in result.output
        assert (output_dir / "README.md").exists()
        readme = (output_dir / "README.md").read_text(encoding="utf-8")
        assert "Model metadata: `codex-mini-latest`" in readme
        assert '- "agent-traces"' in readme
        assert '- "codex"' in readme
        assert '- "distillation"' in readme
        assert '- "codex-mini-latest"' in readme
        assert '- "teich"' in readme
        assert "## Training-ready tools" in readme
        assert "tools.json" not in readme
        assert "<details>" in readme
        assert "<summary>Training-ready tool schema snapshot</summary>" in readme
        assert '"name": "bash"' in readme
        assert '"command"' in readme
        assert "from unsloth import FastLanguageModel" not in readme
        assert "train_dataset = prepare_data(" not in readme
        assert "https://github.com/TeichAI/teich/blob/main/docs/training.md" in readme
        assert not (output_dir / "tools.json").exists()


def test_generate_warns_on_host_auth_concurrency(tmp_path: Path):
    """Host-auth runs warn about host re-login and concurrent-refresh races."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: codex
  codex:
    use_host_auth: true
model:
  model: gpt-5.5
prompts:
  - Build a todo app
  - Another prompt
output:
  traces_dir: {tmp_path}/output
max_concurrency: 2
""")

    with patch('teich.cli.CodexRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/session1.jsonl"]
        mock_runner.return_value = mock_instance

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "session1.jsonl").write_text(
            '{"type":"session_meta","payload":{"id":"session1"}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build a todo app"}]}}\n'
        )

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

    assert result.exit_code == 0
    assert "host Codex login" in result.output
    assert "concurren" in result.output.lower()


def test_generate_no_host_auth_warning_when_disabled(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: codex
model:
  model: codex-mini-latest
prompts:
  - Build a todo app
output:
  traces_dir: {tmp_path}/output
openai_api_key: sk-test
max_concurrency: 2
""")

    with patch('teich.cli.CodexRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/session1.jsonl"]
        mock_runner.return_value = mock_instance
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "session1.jsonl").write_text(
            '{"type":"session_meta","payload":{"id":"session1"}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build a todo app"}]}}\n'
        )
        result = runner.invoke(app, ["generate", "-c", str(config_file)])

    assert result.exit_code == 0
    assert "host Codex login" not in result.output


def test_generate_command_success_chat(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
output:
  traces_dir: {tmp_path}/output
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")

    with patch('teich.cli.ChatRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/chat-session.jsonl"]
        mock_runner.return_value = mock_instance

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chat-session.jsonl").write_text(
            '{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"system":"You are a helpful assistant","prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini","provider":"openai"}\n',
            encoding="utf-8",
        )

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

        assert result.exit_code == 0
        assert "Success" in result.output
        assert (output_dir / "README.md").exists()
        readme = (output_dir / "README.md").read_text(encoding="utf-8")
        assert '- "conversational"' in readme
        assert '- "distillation"' in readme
        assert '- "teich"' in readme
        assert '- "gpt-4.1-mini"' in readme
        assert "newline-delimited JSON training examples generated by teich" in readme
        assert "Chat-only datasets include `messages` plus convenience fields like optional `system`, `prompt`, `follow_up_prompts`, `thinking`, `response`, and `responses`." in readme
        assert not (output_dir / "tools.json").exists()


def test_generate_command_resume_skips_completed_chat_prompts(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
  - Who are you?
output:
  traces_dir: {tmp_path}/output
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat.jsonl").write_text(
        json.dumps({"prompt": "Hello", "response": "Hi", "messages": [{"role": "assistant", "content": "Hi"}]})
        + "\n",
        encoding="utf-8",
    )

    with patch('teich.cli.ChatRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [output_dir / "chat.jsonl"]
        mock_runner.return_value = mock_instance

        result = runner.invoke(app, ["generate", "-c", str(config_file), "--resume"])

        assert result.exit_code == 0
        kwargs = mock_instance.run_all.call_args.kwargs
        assert kwargs["resume"] is True
        assert [prompt_input.prompt for prompt_input in kwargs["prompt_inputs"]] == ["Who are you?"]
        assert "skipping 1 configured prompts" in result.output


def test_generate_command_writes_readme_for_completed_outputs_on_failure(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
  - Who are you?
output:
  traces_dir: {tmp_path}/output
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat.jsonl").write_text(
        json.dumps({"prompt": "Hello", "response": "Hi", "messages": [{"role": "assistant", "content": "Hi"}]})
        + "\n",
        encoding="utf-8",
    )

    with patch('teich.cli.ChatRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.side_effect = RuntimeError("boom")
        mock_runner.return_value = mock_instance

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

        assert result.exit_code == 1
        assert "Wrote README for completed outputs" in result.output
        assert "Error: boom" in result.output
        assert (output_dir / "README.md").exists()


def test_generate_command_publishes_dataset_when_publish_repo_is_configured(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
output:
  traces_dir: {tmp_path}/output
publish:
  repo_id: armand0e/test-dataset
  hf_token: hf-test123
  private: true
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")

    with patch('teich.cli.ChatRunner') as mock_runner, patch('teich.cli.HfApi') as mock_api_cls:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/chat-session.jsonl"]
        mock_runner.return_value = mock_instance

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chat-session.jsonl").write_text(
            '{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"system":"You are a helpful assistant","prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini","provider":"openai"}\n',
            encoding="utf-8",
        )
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/test-dataset"
        mock_api_cls.return_value = mock_api

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

        assert result.exit_code == 0
        assert "Published dataset: https://huggingface.co/datasets/armand0e/test-dataset" in result.output
        mock_api_cls.assert_called_once_with(token="hf-test123")
        mock_api.create_repo.assert_called_once_with(
            repo_id="armand0e/test-dataset",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        mock_api.delete_file.assert_called_once_with(
            path_in_repo="tools.json",
            repo_id="armand0e/test-dataset",
            repo_type="dataset",
            commit_message="Remove legacy teich tools snapshot",
        )
        mock_api.upload_large_folder.assert_called_once_with(
            repo_id="armand0e/test-dataset",
            folder_path=str(output_dir),
            repo_type="dataset",
            private=True,
            ignore_patterns=["partials/**", "failures/**", "bench/**", "README.md", "tools.json"],
        )
        mock_api.upload_folder.assert_called_once_with(
            folder_path=str(output_dir),
            repo_id="armand0e/test-dataset",
            repo_type="dataset",
            commit_message="Upload teich dataset metadata",
            allow_patterns=["README.md"],
            ignore_patterns=["partials/**", "failures/**", "bench/**"],
        )


def test_generate_command_prompts_before_publishing_completed_outputs_and_defaults_no(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
  - Who are you?
output:
  traces_dir: {tmp_path}/output
publish:
  repo_id: armand0e/test-dataset
  hf_token: hf-test123
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat.jsonl").write_text(
        json.dumps({"prompt": "Hello", "response": "Hi", "messages": [{"role": "assistant", "content": "Hi"}]})
        + "\n",
        encoding="utf-8",
    )

    with patch('teich.cli.ChatRunner') as mock_runner, patch('teich.cli.HfApi') as mock_api_cls:
        mock_instance = MagicMock()
        mock_instance.run_all.side_effect = RuntimeError("timeout")
        mock_runner.return_value = mock_instance

        result = runner.invoke(app, ["generate", "-c", str(config_file)], input="\n")

        assert result.exit_code == 1
        assert "Wrote README for completed outputs" in result.output
        assert "Upload successful traces to Hugging Face dataset armand0e/test-dataset?" in result.output
        assert "Skipping Hugging Face upload for completed outputs" in result.output
        mock_api_cls.assert_not_called()


def test_generate_command_does_not_prompt_to_publish_after_keyboard_interrupt(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
output:
  traces_dir: {tmp_path}/output
publish:
  repo_id: armand0e/test-dataset
  hf_token: hf-test123
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat.jsonl").write_text(
        json.dumps({"prompt": "Hello", "response": "Hi", "messages": [{"role": "assistant", "content": "Hi"}]})
        + "\n",
        encoding="utf-8",
    )

    with patch('teich.cli.ChatRunner') as mock_runner, patch('teich.cli.HfApi') as mock_api_cls:
        mock_instance = MagicMock()
        mock_instance.run_all.side_effect = KeyboardInterrupt
        mock_runner.return_value = mock_instance

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

        assert result.exit_code == 130
        assert "Wrote README for completed outputs" in result.output
        assert "Upload successful traces to Hugging Face dataset" not in result.output
        assert "Skipping Hugging Face upload for interrupted run" in result.output
        assert (
            'hf upload armand0e/test-dataset . . --repo-type dataset --exclude "partials/**" --exclude "failures/**"'
            in result.output
        )
        assert "Interrupted. Completed outputs remain on disk" in result.output
        mock_api_cls.assert_not_called()


def test_generate_command_can_publish_completed_outputs_after_failure(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
  - Who are you?
output:
  traces_dir: {tmp_path}/output
publish:
  repo_id: armand0e/test-dataset
  hf_token: hf-test123
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat.jsonl").write_text(
        json.dumps({"prompt": "Hello", "response": "Hi", "messages": [{"role": "assistant", "content": "Hi"}]})
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "partials").mkdir()
    (output_dir / "partials" / "partial.jsonl").write_text("partial\n", encoding="utf-8")
    (output_dir / "failures").mkdir()
    (output_dir / "failures" / "failed.jsonl").write_text("failed\n", encoding="utf-8")

    with patch('teich.cli.ChatRunner') as mock_runner, patch('teich.cli.HfApi') as mock_api_cls:
        mock_instance = MagicMock()
        mock_instance.run_all.side_effect = RuntimeError("timeout")
        mock_runner.return_value = mock_instance
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/test-dataset"
        mock_api_cls.return_value = mock_api

        result = runner.invoke(app, ["generate", "-c", str(config_file)], input="y\n")

        assert result.exit_code == 1
        assert "Published completed outputs:" in result.output
        assert "https://huggingface.co/datasets/armand0e/test-dataset" in result.output
        mock_api.delete_file.assert_called_once_with(
            path_in_repo="tools.json",
            repo_id="armand0e/test-dataset",
            repo_type="dataset",
            commit_message="Remove legacy teich tools snapshot",
        )
        mock_api.upload_large_folder.assert_called_once_with(
            repo_id="armand0e/test-dataset",
            folder_path=str(output_dir),
            repo_type="dataset",
            private=False,
            ignore_patterns=["partials/**", "failures/**", "bench/**", "README.md", "tools.json"],
        )
        mock_api.upload_folder.assert_called_once_with(
            folder_path=str(output_dir),
            repo_id="armand0e/test-dataset",
            repo_type="dataset",
            commit_message="Upload teich dataset metadata",
            allow_patterns=["README.md"],
            ignore_patterns=["partials/**", "failures/**", "bench/**"],
        )


def test_generate_command_deduplicates_configured_prompts_before_running(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
agent:
  provider: chat
model:
  model: gpt-4.1-mini
prompts:
  - Hello
  - Hello
  - Who are you?
output:
  traces_dir: {tmp_path}/output
api:
  provider: openai
  api_key: sk-test
  wire_api: responses
""")

    with patch('teich.cli.ChatRunner') as mock_runner:
        mock_instance = MagicMock()
        mock_instance.run_all.return_value = [tmp_path / "output/chat.jsonl"]
        mock_runner.return_value = mock_instance
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chat.jsonl").write_text(
            json.dumps({"prompt": "Hello", "response": "Hi", "messages": [{"role": "assistant", "content": "Hi"}]})
            + "\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["generate", "-c", str(config_file)])

        assert result.exit_code == 0
        kwargs = mock_instance.run_all.call_args.kwargs
        assert [prompt_input.prompt for prompt_input in kwargs["prompt_inputs"]] == ["Hello", "Who are you?"]


def test_batch_progress_reporter_only_shows_queued_and_running_rows():
    console = Console(record=True, width=160)
    reporter = BatchProgressReporter(console)

    reporter.update(
        SessionProgressUpdate(
            prompt_id="queued-1",
            prompt_index=1,
            total_prompts=3,
            prompt="Queued prompt",
            prompt_preview="Queued prompt",
            status="queued",
        )
    )
    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=2,
            total_prompts=3,
            prompt="Running prompt",
            prompt_preview="Running prompt",
            status="running",
            metrics=TraceMetrics(total_tokens=42, total_cost=0.125, model="codex-mini-latest"),
        )
    )
    reporter.update(
        SessionProgressUpdate(
            prompt_id="completed-1",
            prompt_index=3,
            total_prompts=3,
            prompt="Completed prompt",
            prompt_preview="Completed prompt",
            status="completed",
            metrics=TraceMetrics(total_tokens=99, total_cost=0.5, model="codex-mini-latest"),
        )
    )

    console.print(reporter._render())
    output = console.export_text()

    assert "queued=1 running=1 completed=1 failed=0" in output
    assert "tokens=141" in output
    assert "Queued prompt" in output
    assert "Running prompt" in output
    assert "Completed prompt" not in output
    assert "api tokens" not in output
    assert "est. total" not in output


def test_batch_progress_reporter_marks_unknown_usage_as_na():
    console = Console(record=True, width=160)
    reporter = BatchProgressReporter(console)

    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=1,
            total_prompts=1,
            prompt="Running prompt",
            prompt_preview="Running prompt",
            status="running",
            metrics=TraceMetrics(model="minimax/minimax-m2.5:free"),
        )
    )

    console.print(reporter._render())
    output = console.export_text()

    assert "tokens=N/A cost=N/A" in output
    assert "minimax/minimax-m2.5:free" in output


def test_batch_progress_reporter_accumulates_running_metric_deltas():
    reporter = BatchProgressReporter(Console(record=True, width=160))

    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=1,
            total_prompts=1,
            prompt="Build app",
            prompt_preview="Build app",
            status="running",
            metrics=TraceMetrics(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                total_cost=0.10,
                model="Opus-Agent",
            ),
            metrics_delta=True,
        )
    )
    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=1,
            total_prompts=1,
            prompt="Build app",
            prompt_preview="Build app",
            status="running",
            metrics=TraceMetrics(
                input_tokens=5,
                output_tokens=7,
                total_tokens=12,
                total_cost=0.05,
                model="Opus-Agent",
            ),
            metrics_delta=True,
        )
    )

    totals = reporter.snapshot_totals()

    assert totals["input_tokens"] == 15
    assert totals["output_tokens"] == 27
    assert totals["total_tokens"] == 42
    assert totals["total_cost"] == pytest.approx(0.15)


def test_batch_progress_reporter_replaces_deltas_with_completed_metrics():
    reporter = BatchProgressReporter(Console(record=True, width=160))

    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=1,
            total_prompts=1,
            prompt="Build app",
            prompt_preview="Build app",
            status="running",
            metrics=TraceMetrics(total_tokens=30, total_cost=0.10, model="Opus-Agent"),
            metrics_delta=True,
        )
    )
    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=1,
            total_prompts=1,
            prompt="Build app",
            prompt_preview="Build app",
            status="completed",
            metrics=TraceMetrics(total_tokens=35, total_cost=0.12, model="Opus-Agent"),
        )
    )

    totals = reporter.snapshot_totals()

    assert totals["total_tokens"] == 35
    assert totals["total_cost"] == pytest.approx(0.12)


def test_batch_progress_reporter_refreshes_elapsed_without_status_update():
    reporter = BatchProgressReporter(Console(record=True, width=160))
    fake_live = MagicMock()
    reporter._live = fake_live
    reporter.update(
        SessionProgressUpdate(
            prompt_id="running-1",
            prompt_index=1,
            total_prompts=1,
            prompt="Running prompt",
            prompt_preview="Running prompt",
            status="running",
            started_at=datetime.now(timezone.utc) - timedelta(seconds=65),
        )
    )

    reporter._refresh_live_once()

    assert fake_live.update.call_count >= 2
    refreshed_renderable = fake_live.update.call_args.args[0]
    console = Console(record=True, width=160)
    console.print(refreshed_renderable)
    assert "01:05" in console.export_text()
