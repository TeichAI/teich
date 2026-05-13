"""Tests for runner module."""

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import teich.runner as runner_module
from teich.config import APIConfig, Config, MCPConfig, ModelConfig, PromptInput
from teich.runner import ChatRunner, ClaudeCodeRunner, CodexRunner, HermesRunner, PiRunner, pending_prompt_inputs_for_resume


def test_codex_runner_init():
    """Test CodexRunner initialization."""
    config = Config()

    with patch.object(CodexRunner, '_ensure_image') as mock_ensure:
        runner = CodexRunner(config)
        mock_ensure.assert_called_once()
        assert runner.image_name == "teich-runtime:v3"
        assert runner.config == config


def test_runtime_image_rebuilds_when_dockerfile_is_newer(tmp_path: Path):
    dockerfile = tmp_path / "codex-runtime.Dockerfile"
    dockerfile.write_text("FROM node:22-slim\n", encoding="utf-8")
    newer_time = datetime.now(timezone.utc)
    dockerfile.touch()

    with patch.object(CodexRunner, "_runtime_dockerfile_path", return_value=dockerfile), \
         patch.object(CodexRunner, "_image_created_at", return_value=newer_time - timedelta(minutes=5)), \
         patch.object(CodexRunner, "_build_image") as mock_build, \
         patch("teich.runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="existing-image\n")
        CodexRunner(Config())

    mock_build.assert_called_once()


def test_runtime_dockerfile_path_prefers_packaged_file(tmp_path: Path):
    package_dir = tmp_path / "site-packages" / "teich"
    package_dir.mkdir(parents=True)
    packaged_dockerfile = package_dir / "docker" / "codex-runtime.Dockerfile"
    packaged_dockerfile.parent.mkdir(parents=True)
    packaged_dockerfile.write_text("FROM node:22-slim\n", encoding="utf-8")
    fake_runner_file = package_dir / "runner.py"
    fake_runner_file.write_text("# test stub\n", encoding="utf-8")

    with patch.object(runner_module, "__file__", str(fake_runner_file)):
        dockerfile_path = CodexRunner._runtime_dockerfile_path()

    assert dockerfile_path == packaged_dockerfile


def test_codex_config_setup(tmp_path: Path):
    """Test Codex config.toml is written correctly."""
    config = Config(
        mcp_servers=[
            MCPConfig(name="filesystem", command="npx", args=["-y", "mcp-server"], env={}),
        ]
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)

    config_file = codex_home / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert '[mcp_servers."filesystem"]' in content
    assert 'command = "npx"' in content
    assert 'args = ["-y", "mcp-server"]' in content


def test_run_session_command_generation():
    """Test that codex exec command is generated correctly."""
    config = Config(
        model=ModelConfig(model="o4-mini", approval_mode="suggest"),
        openai_api_key="sk-test123",
        timeout_seconds=300,
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    with patch.object(runner, '_run_process') as mock_run_process, \
         patch.object(runner, '_extract_session_file') as mock_extract, \
         patch.object(runner, '_copy_workspace_snapshot') as mock_copy_workspace:

        mock_extract.return_value = Path("/output/session.jsonl")

        runner.run_session("Build a todo app", "test-session-123")

        # Verify the docker command was called
        call_args = mock_run_process.call_args
        cmd = call_args[0][0]

        # Check key elements of command
        assert "docker" in cmd
        assert "run" in cmd
        assert "--rm" in cmd
        assert "-i" in cmd
        assert "--name" in cmd
        assert "teich-codex-test-session-123" in cmd
        assert "-e" in cmd
        assert "OPENAI_API_KEY=sk-test123" in cmd
        assert "codex" in cmd
        assert "exec" in cmd
        assert "--model" in cmd
        assert "o4-mini" in cmd
        assert "--ask-for-approval" in cmd
        assert "on-request" in cmd
        assert "--sandbox" in cmd
        assert config.model.sandbox in cmd
        assert "--skip-git-repo-check" in cmd
        assert cmd.index("--ask-for-approval") < cmd.index("exec")
        assert cmd[-1] == "-"
        assert mock_run_process.call_args.args[6] == "teich-codex-test-session-123"
        assert mock_run_process.call_args.args[7] == "Build a todo app"
        mock_copy_workspace.assert_called_once()


def test_run_session_timeout():
    """Test that timeout is passed correctly."""
    config = Config(timeout_seconds=60)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    with patch.object(runner, '_run_process') as mock_run_process:
        from subprocess import TimeoutExpired
        mock_run_process.side_effect = TimeoutExpired(cmd="codex", timeout=60)

        with pytest.raises(RuntimeError, match="timed out"):
            runner.run_session("Test prompt")


def test_docker_runners_keep_long_prompt_out_of_host_command_line(tmp_path: Path):
    long_prompt = "x" * 40000

    codex_config = Config(output={"traces_dir": tmp_path / "codex-output", "sandbox_dir": tmp_path / "codex-sandbox"})
    with patch.object(CodexRunner, '_ensure_image'):
        codex_runner = CodexRunner(codex_config)
    with patch.object(codex_runner, '_run_process') as mock_run_process, \
         patch.object(codex_runner, '_extract_session_file', return_value=tmp_path / "codex-output" / "trace.jsonl"), \
         patch.object(codex_runner, '_copy_workspace_snapshot'):
        codex_runner.run_session(long_prompt, "codex-session")

    codex_command = mock_run_process.call_args.args[0]
    assert long_prompt not in codex_command
    assert codex_command[-1] == "-"
    assert mock_run_process.call_args.args[7] == long_prompt

    pi_config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "pi-output", "sandbox_dir": tmp_path / "pi-sandbox"})
    with patch.object(PiRunner, '_ensure_image'):
        pi_runner = PiRunner(pi_config)
    captured_workspace = None

    def capture_project_settings(workspace: Path) -> None:
        nonlocal captured_workspace
        captured_workspace = workspace

    def assert_pi_prompt_file_before_cleanup(*args, **kwargs) -> None:
        assert captured_workspace is not None
        assert (captured_workspace / ".teich-prompt.txt").read_text(encoding="utf-8") == long_prompt

    with patch.object(pi_runner, '_run_process', side_effect=assert_pi_prompt_file_before_cleanup) as mock_pi_run_process, \
         patch.object(pi_runner, '_extract_session_file', return_value=tmp_path / "pi-output" / "trace.jsonl"), \
         patch.object(pi_runner, '_copy_workspace_snapshot'), \
         patch.object(pi_runner, '_write_pi_agent_settings'), \
         patch.object(pi_runner, '_write_pi_project_settings', side_effect=capture_project_settings):
        pi_runner.run_session(long_prompt, "pi-session")

    pi_command = mock_pi_run_process.call_args.args[0]
    assert long_prompt not in pi_command
    assert "@/workspace/.teich-prompt.txt" in pi_command


def test_claude_code_runner_uses_stream_json_and_prompt_file(tmp_path: Path):
    long_prompt = "x" * 40000
    config = Config(
        agent={"provider": "claude-code"},
        api=APIConfig(provider="anthropic", api_key="sk-ant-test"),
        model=ModelConfig(model="claude-sonnet-4-6", approval_policy="never"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    with patch.object(runner, "_run_external_process", return_value=('{"type":"result","result":"done"}\n', "")) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        trace_path = runner.run_session(long_prompt, "claude-session")

    command = mock_run.call_args.args[0]
    command_text = " ".join(command)
    assert long_prompt not in command
    assert "claude" in command_text
    assert "--output-format stream-json" in command_text
    assert "--permission-mode bypassPermissions" in command_text
    assert "< /workspace/.teich-prompt.txt" in command_text
    assert "ANTHROPIC_API_KEY=sk-ant-test" in command
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "external_session_meta"
    assert rows[1]["type"] == "external_message"
    assert rows[1]["role"] == "user"
    assert rows[2]["type"] == "result"


def test_claude_code_runner_uses_openrouter_anthropic_skin_auth(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        api=APIConfig(
            provider="openrouter",
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        ),
        model=ModelConfig(model="minimax/minimax-m2.5:free", approval_policy="never"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    with patch.object(runner, "_run_external_process", return_value=('{"type":"result","result":"done"}\n', "")) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        trace_path = runner.run_session("smoke", "claude-openrouter-session")

    command = mock_run.call_args.args[0]
    command_text = " ".join(command)
    assert "ANTHROPIC_AUTH_TOKEN=sk-or-test" in command
    assert "ANTHROPIC_API_KEY=" in command
    assert "ANTHROPIC_API_KEY=sk-or-test" not in command
    assert "OPENROUTER_API_KEY=sk-or-test" in command
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:17891" in command
    assert "TEICH_CLAUDE_PROXY_TARGET=https://openrouter.ai/api/v1" in command
    assert "TEICH_CLAUDE_PROXY_TARGET_MODEL=minimax/minimax-m2.5:free" in command
    assert "--model claude-sonnet-4-6" in command_text
    assert "--model minimax/minimax-m2.5:free" not in command_text
    assert "node /home/codex/.claude/claude_openrouter_proxy.js" in command_text
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["payload"]["model"] == "minimax/minimax-m2.5:free"


def test_claude_code_runner_uses_continue_for_followups(tmp_path: Path):
    config = Config(
        agent={"provider": "claude"},
        model=ModelConfig(model="claude-sonnet-4-6"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)
    with patch.object(runner, "_start_container") as mock_start, \
         patch.object(runner, "_run_external_process", return_value=('{"type":"result","result":"done"}\n', "")) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"), \
         patch.object(runner, "_remove_container") as mock_remove:
        runner.run_session(prompt_input.prompt, "claude-followups", prompt_input=prompt_input)

    mock_start.assert_called_once()
    assert mock_run.call_count == 2
    first_command = " ".join(mock_run.call_args_list[0].args[0])
    second_command = " ".join(mock_run.call_args_list[1].args[0])
    assert "--continue" not in first_command
    assert "--continue" in second_command
    mock_remove.assert_called_once_with("teich-claude-claude-followups")


def test_hermes_runner_uses_chat_query_and_prompt_file(tmp_path: Path):
    long_prompt = "x" * 40000
    config = Config(
        agent={"provider": "hermes"},
        api=APIConfig(provider="openrouter", api_key="sk-or-test"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)
    with patch.object(runner, "_run_external_process", return_value=("done", "")) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        trace_path = runner.run_session(long_prompt, "hermes-session")

    command = mock_run.call_args.args[0]
    command_text = " ".join(command)
    assert long_prompt not in command
    assert "hermes chat --provider openrouter" in command_text
    assert "--model codex-mini-latest" in command_text
    assert "--ignore-user-config" in command_text
    assert '--source teich -q "$(cat /workspace/.teich-prompt.txt)"' in command_text
    assert "OPENROUTER_API_KEY=sk-or-test" in command
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["payload"]["source"] == "hermes-agent"
    assert rows[-1]["role"] == "assistant"
    assert rows[-1]["content"] == "done"


def test_run_process_removes_named_container_on_failure():
    config = Config()

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    process = MagicMock()
    process.poll.return_value = None
    with patch("teich.runner.subprocess.Popen", return_value=process) as mock_popen, \
         patch.object(runner, "_monitor_process", side_effect=KeyboardInterrupt), \
         patch("teich.runner.subprocess.run") as mock_run:
        with pytest.raises(KeyboardInterrupt):
            runner._run_process(
                ["docker", "run", "--name", "teich-codex-test", "teich-runtime:v3"],
                "test-session",
                datetime.now(timezone.utc),
                container_name="teich-codex-test",
            )

    mock_popen.assert_called_once()
    process.terminate.assert_called_once()
    mock_run.assert_called_with(
        ["docker", "rm", "-f", "teich-codex-test"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_codex_run_session_discards_partial_trace_on_failure(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    workspace_root = tmp_path / "workspace-root"
    workspace = workspace_root
    codex_home = tmp_path / "codex-home"
    sessions_dir = codex_home / "sessions"
    workspace.mkdir()
    sessions_dir.mkdir(parents=True)

    def fail_after_writing_trace(*args, **kwargs):
        (sessions_dir / "partial.jsonl").write_text(
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Test prompt"}]}}\n',
            encoding="utf-8",
        )
        raise RuntimeError("boom")

    with patch.object(runner, "_prepare_workspace", return_value=(workspace_root, workspace)), \
         patch("teich.runner.tempfile.mkdtemp", return_value=str(codex_home)), \
         patch.object(runner, "_run_process", side_effect=fail_after_writing_trace):
        with pytest.raises(RuntimeError, match="boom"):
            runner.run_session("Test prompt", "test-session")

    assert not (tmp_path / "output" / "partials").exists()


def test_run_all_with_no_prompts():
    """Test that run_all raises error when no prompts."""
    config = Config()

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    with pytest.raises(ValueError, match="No prompts configured"):
        runner.run_all()


def test_run_all_executes_all_prompts(tmp_path: Path):
    """Test that run_all processes all prompts."""
    config = Config(prompts=["Prompt 1", "Prompt 2", "Prompt 3"])

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    output_files = []
    for index in range(1, 4):
        trace_file = tmp_path / f"{index}.jsonl"
        trace_file.write_text(
            json.dumps({"type": "session_meta", "payload": {"id": f"session-{index}"}}) + "\n",
            encoding="utf-8",
        )
        output_files.append(trace_file)

    with patch.object(runner, 'run_session') as mock_run:
        mock_run.side_effect = output_files

        results = runner.run_all()

        assert len(results) == 3
        assert mock_run.call_count == 3
        assert results[0] == output_files[0]


def test_codex_run_session_runs_follow_up_prompts_by_resuming_session():
    config = Config()

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests", "Polish UI"])
    container_workspace = None
    codex_home = None

    def mounted_path(command: list[str], target: str) -> Path:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        match = next(mount for mount in mounts if mount.endswith(f":{target}"))
        return Path(match.rsplit(":", maxsplit=1)[0])

    def start_container(command: list[str]) -> None:
        nonlocal container_workspace, codex_home
        container_workspace = mounted_path(command, "/workspace")
        codex_home = mounted_path(command, "/home/codex/.codex")

    def record_process(command, *args) -> None:
        assert command[:3] == ["docker", "exec", "-i"]
        assert "--user" in command
        assert "-w" in command
        assert command[command.index("-w") + 1] == "/workspace"
        assert "teich-codex-test-session" in command
        assert container_workspace is not None
        stdin_text = args[6]
        if stdin_text == "Build app":
            (container_workspace / "created-by-first-turn.txt").write_text("persisted", encoding="utf-8")
        else:
            assert (container_workspace / "created-by-first-turn.txt").read_text(encoding="utf-8") == "persisted"

    with patch.object(runner, '_start_container', side_effect=start_container) as mock_start_container, \
         patch.object(runner, '_run_process', side_effect=record_process) as mock_run_process, \
         patch.object(runner, '_remove_container') as mock_remove_container, \
         patch.object(runner, '_extract_session_file') as mock_extract, \
         patch.object(runner, '_copy_workspace_snapshot'):

        mock_extract.return_value = Path("/output/session.jsonl")

        runner.run_session("Build app", "test-session", prompt_input=prompt_input)

    assert mock_run_process.call_count == 3
    stdin_texts = [call.args[7] for call in mock_run_process.call_args_list]
    assert stdin_texts == ["Build app", "Add tests", "Polish UI"]
    first_command = mock_run_process.call_args_list[0].args[0]
    second_command = mock_run_process.call_args_list[1].args[0]
    assert "resume" not in first_command
    assert "resume" in second_command
    assert "--last" in second_command
    assert second_command[second_command.index("--sandbox") + 1] == "danger-full-access"
    codex_index = second_command.index("codex")
    codex_exec_index = second_command.index("exec", codex_index)
    assert second_command.index("--sandbox") < codex_exec_index
    assert mock_start_container.call_count == 1
    assert codex_home is not None
    assert mock_remove_container.call_args.args == ("teich-codex-test-session",)


def test_pi_run_session_runs_follow_up_prompts_by_continuing_session(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    captured_workspace = None
    prompt_file_values = []
    commands = []
    container_workspace = None
    container_session_dir = None

    def mounted_path(command: list[str], target: str) -> Path:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        match = next(mount for mount in mounts if mount.endswith(f":{target}"))
        return Path(match.rsplit(":", maxsplit=1)[0])

    def start_container(command: list[str]) -> None:
        nonlocal container_workspace, container_session_dir
        container_workspace = mounted_path(command, "/workspace")
        container_session_dir = mounted_path(command, "/home/codex/pi-sessions")

    def capture_project_settings(workspace: Path) -> None:
        nonlocal captured_workspace
        captured_workspace = workspace

    def record_prompt_file(command, *args, **kwargs) -> None:
        assert captured_workspace is not None
        assert command[:3] == ["docker", "exec", "-i"]
        assert "--user" in command
        assert "-w" in command
        assert command[command.index("-w") + 1] == "/workspace"
        assert "teich-pi-pi-session" in command
        assert container_workspace is not None
        commands.append(command)
        prompt_file_values.append((captured_workspace / ".teich-prompt.txt").read_text(encoding="utf-8"))
        if prompt_file_values[-1] == "Build app":
            (container_workspace / "created-by-first-turn.txt").write_text("persisted", encoding="utf-8")
        else:
            assert (container_workspace / "created-by-first-turn.txt").read_text(encoding="utf-8") == "persisted"

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])

    with patch.object(runner, "_start_container", side_effect=start_container) as mock_start_container, \
         patch.object(runner, "_write_pi_project_settings", side_effect=capture_project_settings), \
         patch.object(runner, "_run_process", side_effect=record_prompt_file), \
         patch.object(runner, "_remove_container") as mock_remove_container, \
         patch.object(runner, "_extract_session_file", return_value=tmp_path / "output" / "pi.jsonl"), \
         patch.object(runner, "_copy_workspace_snapshot"):
        runner.run_session("Build app", "pi-session", prompt_input=prompt_input)

    assert prompt_file_values == ["Build app", "Add tests"]
    assert "--continue" not in commands[0]
    assert "--continue" in commands[1]
    assert mock_start_container.call_count == 1
    assert container_session_dir is not None
    assert mock_remove_container.call_args.args == ("teich-pi-pi-session",)


def test_run_all_reports_progress_and_preserves_prompt_order(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2"], max_concurrency=2)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    updates = []

    def fake_run_session(
        prompt: str,
        session_id: str | None = None,
        progress_callback=None,
        progress_base=None,
        prompt_input=None,
    ) -> Path:
        if prompt == "Prompt 1":
            time.sleep(0.05)
        trace_file = tmp_path / f"{prompt.replace(' ', '_').lower()}.jsonl"
        trace_file.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session", "id": session_id}),
                    json.dumps({"type": "model_change", "modelId": "deepseek/deepseek-v4-pro"}),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "model": "deepseek/deepseek-v4-pro",
                                "usage": {
                                    "input": 10,
                                    "output": 5,
                                    "cacheRead": 2,
                                    "cacheWrite": 0,
                                    "totalTokens": 17,
                                    "cost": {"total": 0.25},
                                },
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return trace_file

    with patch.object(runner, 'run_session', side_effect=fake_run_session):
        results = runner.run_all(max_concurrency=2, progress_callback=updates.append)

    assert [path.name for path in results] == ["prompt_1.jsonl", "prompt_2.jsonl"]
    assert [update.status for update in updates].count("queued") == 2
    assert [update.status for update in updates].count("running") == 2
    completed = [update for update in updates if update.status == "completed"]
    assert len(completed) == 2
    assert all(update.metrics is not None for update in completed)
    assert completed[0].metrics.total_tokens == 17
    assert completed[0].metrics.total_cost == 0.25


def test_run_all_queues_prompts_lazily_as_workers_free_up(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2", "Prompt 3"], max_concurrency=2)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    updates = []
    update_thread_names = []
    started = []
    started_lock = threading.Lock()
    first_batch_started = threading.Event()
    release_first_batch = threading.Event()

    def record_update(update):
        update_thread_names.append(threading.current_thread().name)
        updates.append(update)

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, progress_callback):
        with started_lock:
            started.append(prompt_id)
            if len(started) == 2:
                first_batch_started.set()
        if prompt_id in {"prompt-1", "prompt-2"}:
            assert release_first_batch.wait(timeout=2)
        return tmp_path / f"{prompt_id}.jsonl"

    with patch.object(runner, '_run_prompt_task', side_effect=fake_task):
        result_holder = {}

        def run():
            result_holder["results"] = runner.run_all(max_concurrency=2, progress_callback=record_update)

        thread = threading.Thread(target=run)
        thread.start()
        assert first_batch_started.wait(timeout=2)
        time.sleep(0.05)
        assert [update.prompt_id for update in updates if update.status == "queued"] == ["prompt-1", "prompt-2"]
        release_first_batch.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert [path.name for path in result_holder["results"]] == ["prompt-1.jsonl", "prompt-2.jsonl", "prompt-3.jsonl"]
    assert [update.prompt_id for update in updates if update.status == "queued"] == ["prompt-1", "prompt-2", "prompt-3"]
    queued_update_thread_names = [
        thread_name
        for update, thread_name in zip(updates, update_thread_names, strict=True)
        if update.status == "queued"
    ]
    assert all(thread_name.startswith("teich-prompt-worker-") for thread_name in queued_update_thread_names)


def test_run_all_stops_claiming_agent_prompts_after_failure(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2", "Prompt 3"], max_concurrency=1)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    trace_paths = {"Prompt 1": tmp_path / "first.jsonl"}

    def fake_run_session(
        prompt: str,
        session_id: str | None = None,
        progress_callback=None,
        progress_base=None,
        prompt_input=None,
    ) -> Path:
        if prompt == "Prompt 2":
            raise RuntimeError("boom")
        trace_path = trace_paths[prompt]
        trace_path.write_text(json.dumps({"prompt": prompt}) + "\n", encoding="utf-8")
        return trace_path

    with patch.object(runner, 'run_session', side_effect=fake_run_session):
        with pytest.raises(RuntimeError, match="boom"):
            runner.run_all(max_concurrency=1)

    assert trace_paths["Prompt 1"].exists()
    assert json.loads(trace_paths["Prompt 1"].read_text(encoding="utf-8")) == {"prompt": "Prompt 1"}
    assert not (tmp_path / "third.jsonl").exists()


def test_run_all_propagates_keyboard_interrupt_without_waiting_for_agent_workers(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2"], max_concurrency=2)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    started = threading.Event()
    release = threading.Event()

    def blocked_run_session(
        prompt: str,
        session_id: str | None = None,
        progress_callback=None,
        progress_base=None,
        prompt_input=None,
    ) -> Path:
        started.set()
        release.wait(timeout=2)
        return tmp_path / f"{prompt}.jsonl"

    def interrupt_after_worker_starts(*args, **kwargs):
        assert started.wait(timeout=1)
        raise KeyboardInterrupt

    try:
        with patch.object(runner, 'run_session', side_effect=blocked_run_session):
            with patch("teich.runner.threading.Thread.join", side_effect=interrupt_after_worker_starts):
                start = time.monotonic()
                with pytest.raises(KeyboardInterrupt):
                    runner.run_all(max_concurrency=2)
                elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < 0.5


def test_summarize_trace_file_uses_pi_usage_payload(tmp_path: Path):
    trace_file = tmp_path / "pi-trace.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "model_change", "modelId": "deepseek/deepseek-v4-pro"}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "model": "deepseek/deepseek-v4-pro",
                            "usage": {
                                "input": 12,
                                "output": 8,
                                "cacheRead": 3,
                                "cacheWrite": 1,
                                "totalTokens": 24,
                                "cost": {"total": 0.5},
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.model == "deepseek/deepseek-v4-pro"
    assert metrics.input_tokens == 12
    assert metrics.output_tokens == 8
    assert metrics.cache_read_tokens == 3
    assert metrics.cache_write_tokens == 1
    assert metrics.total_tokens == 24
    assert metrics.est_total_tokens == 24
    assert metrics.total_cost == 0.5


def test_summarize_trace_file_reads_structured_chat_usage(tmp_path: Path):
    trace_file = tmp_path / "chat.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                    {"role": "user", "content": "Hello", "thinking": None},
                    {"role": "assistant", "content": "Hi!", "thinking": "I should greet the user."},
                ],
                "prompt": "Hello",
                "response": "Hi!",
                "model": "gpt-4.1-mini",
                "provider": "openai",
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "openai"
    assert metrics.model == "gpt-4.1-mini"
    assert metrics.input_tokens == 4
    assert metrics.output_tokens == 3
    assert metrics.total_tokens == 7
    assert metrics.est_total_tokens == 7


def test_summarize_trace_file_prefers_codex_total_usage(tmp_path: Path):
    trace_file = tmp_path / "codex-trace.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"model_provider": "openrouter"},
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "last_token_usage": {
                                    "input_tokens": 2,
                                    "output_tokens": 3,
                                    "reasoning_output_tokens": 4,
                                    "cached_input_tokens": 5,
                                    "total_tokens": 14,
                                },
                                "total_token_usage": {
                                    "input_tokens": 20,
                                    "output_tokens": 30,
                                    "reasoning_output_tokens": 40,
                                    "cached_input_tokens": 50,
                                    "total_tokens": 140,
                                },
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "openrouter"
    assert metrics.input_tokens == 20
    assert metrics.output_tokens == 30
    assert metrics.reasoning_tokens == 40
    assert metrics.cache_read_tokens == 50
    assert metrics.total_tokens == 140
    assert metrics.est_total_tokens == 14


def test_summarize_trace_file_estimates_codex_total_from_usage_delta(tmp_path: Path):
    trace_file = tmp_path / "codex-trace-delta.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"model_provider": "openrouter"},
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 100,
                                    "output_tokens": 20,
                                    "reasoning_output_tokens": 5,
                                    "cached_input_tokens": 40,
                                    "total_tokens": 125,
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 160,
                                    "output_tokens": 35,
                                    "reasoning_output_tokens": 9,
                                    "cached_input_tokens": 70,
                                    "total_tokens": 204,
                                }
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.total_tokens == 204
    assert metrics.input_tokens == 160
    assert metrics.output_tokens == 35
    assert metrics.reasoning_tokens == 9
    assert metrics.cache_read_tokens == 70
    assert metrics.est_total_tokens == 79


def test_monitor_process_fails_fast_on_live_pi_tool_call_corruption(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    trace_file = session_dir / "session.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-1",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {}},
                                {"type": "toolCall", "id": "", "name": "", "arguments": {}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "tool-1",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": "call-1",
                            "toolName": "bash",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Validation failed for tool \"bash\":\n  - command: must have required properties command\n\nReceived arguments:\n{}",
                                }
                            ],
                            "isError": True,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    process = MagicMock()
    process.poll.return_value = None
    process.args = ["docker", "run"]
    stdout_handle = io.StringIO()
    stderr_handle = io.StringIO()

    with pytest.raises(RuntimeError, match="malformed tool calls/results"):
        runner._monitor_process(
            process,
            "session-id",
            datetime.fromtimestamp(0, tz=timezone.utc),
            session_dir,
            None,
            None,
            stdout_handle,
            stderr_handle,
        )

    process.kill.assert_called_once()
    process.wait.assert_called_once()


def test_pi_trace_with_model_error_is_rejected_before_export(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    source = tmp_path / "bad-session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps({"type": "model_change", "modelId": "deepseek/deepseek-v4-pro"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-error",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "api": "openai-completions",
                            "model": "deepseek/deepseek-v4-pro",
                            "usage": {"input": 0, "output": 0, "totalTokens": 0},
                            "stopReason": "error",
                            "errorMessage": "401 User not found.",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "output" / "bad-session.jsonl"

    with pytest.raises(RuntimeError, match="model/provider error: 401 User not found"):
        runner._copy_normalized_session_file(source, destination)

    assert not destination.exists()


def test_custom_api_provider_command():
    """Test that custom API provider generates correct command."""
    config = Config(
        model=ModelConfig(model="anthropic/claude-3.5-sonnet"),
        api=APIConfig(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test123",
        ),
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    with patch.object(runner, '_run_process') as mock_run_process, \
         patch.object(runner, '_extract_session_file') as mock_extract, \
         patch.object(runner, '_copy_workspace_snapshot'):

        mock_extract.return_value = Path("/output/session.jsonl")

        runner.run_session("Test prompt", "test-session")

        # Verify the command includes custom provider config
        call_args = mock_run_process.call_args[0][0]
        cmd_str = ' '.join(call_args)

        assert 'model_providers.openrouter' in cmd_str
        assert 'openrouter.ai/api/v1' in cmd_str
        assert 'model_provider="openrouter"' in cmd_str
        assert 'env_key="OPENROUTER_API_KEY"' in cmd_str
        assert 'wire_api="responses"' in cmd_str
        assert 'OPENROUTER_API_KEY=sk-or-test123' in call_args
        assert '--oss' not in call_args


def test_openai_custom_base_url_uses_non_reserved_provider_alias():
    """Test that custom openai-compatible endpoints do not override reserved built-in provider IDs."""
    config = Config(
        model=ModelConfig(model="gemma-4"),
        api=APIConfig(
            provider="openai",
            base_url="https://lm.gptbox.dev/v1",
            api_key="llm",
        ),
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    with patch.object(runner, '_run_process') as mock_run_process, \
         patch.object(runner, '_extract_session_file') as mock_extract, \
         patch.object(runner, '_copy_workspace_snapshot'):

        mock_extract.return_value = Path('/output/session.jsonl')

        runner.run_session('Test prompt', 'test-session')

        cmd_str = ' '.join(mock_run_process.call_args[0][0])
        assert 'model_providers.openai_compatible' in cmd_str
        assert 'model_provider="openai_compatible"' in cmd_str
        assert 'model_providers.openai=' not in cmd_str


def test_lmstudio_provider_uses_native_oss_flags():
    """Test that LM Studio uses Codex's native OSS provider flags."""
    config = Config(
        model=ModelConfig(model="gemma-4"),
        api=APIConfig(
            provider="LMstudio",
            base_url="http://localhost:1234/v1",
            api_key="llm",
        ),
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    with patch.object(runner, '_run_process') as mock_run_process, \
         patch.object(runner, '_extract_session_file') as mock_extract, \
         patch.object(runner, '_copy_workspace_snapshot'):

        mock_extract.return_value = Path('/output/session.jsonl')

        runner.run_session('Test prompt', 'test-session')

        call_args = mock_run_process.call_args[0][0]
        cmd_str = ' '.join(call_args)
        assert '--oss' in cmd_str
        assert '--local-provider' in cmd_str
        assert 'lmstudio' in cmd_str
        assert 'host.docker.internal:host-gateway' in cmd_str
        assert 'local_provider_proxy.js' in cmd_str
        assert 'model_providers.lmstudio' not in cmd_str


def test_run_session_copies_workspace_into_named_sandbox(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    captured_destination = None

    def capture_copy(workspace: Path, destination: Path) -> None:
        nonlocal captured_destination
        captured_destination = destination

    with patch.object(runner, '_run_process'), \
         patch.object(runner, '_extract_session_file', return_value=tmp_path / 'output' / 'trace-1.jsonl'), \
         patch.object(runner, '_copy_workspace_snapshot', side_effect=capture_copy):
        result = runner.run_session('Test prompt', 'test-session')

    assert result == tmp_path / 'output' / 'trace-1.jsonl'
    assert captured_destination == tmp_path / 'sandbox' / 'trace-1.jsonl'


def test_run_session_clones_github_repo_into_codex_workspace(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    cloned_destination = None
    captured_workspace = None

    def capture_clone(_github_repo: str, destination: Path) -> None:
        nonlocal cloned_destination
        cloned_destination = destination
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "README.md").write_text("repo", encoding="utf-8")

    def capture_copy(workspace: Path, _destination: Path) -> None:
        nonlocal captured_workspace
        captured_workspace = workspace

    prompt_input = PromptInput(github_repo="armand0e/perplexica-mcp", prompt="Fix the issue")

    with patch.object(runner, '_clone_github_repo', side_effect=capture_clone) as mock_clone, \
         patch.object(runner, '_run_process'), \
         patch.object(runner, '_extract_session_file', return_value=tmp_path / 'output' / 'trace-1.jsonl'), \
         patch.object(runner, '_copy_workspace_snapshot', side_effect=capture_copy):
        runner.run_session('Fix the issue', 'test-session', prompt_input=prompt_input)

    assert mock_clone.called
    assert cloned_destination is not None
    assert cloned_destination.name == "perplexica-mcp"
    assert captured_workspace == cloned_destination


def test_run_session_clones_github_repo_into_pi_workspace(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    cloned_destination = None
    captured_workspace = None

    def capture_clone(_github_repo: str, destination: Path) -> None:
        nonlocal cloned_destination
        cloned_destination = destination
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "package.json").write_text("{}", encoding="utf-8")

    def capture_project_settings(workspace: Path) -> None:
        nonlocal captured_workspace
        captured_workspace = workspace

    prompt_input = PromptInput(github_repo="armand0e/perplexica-mcp", prompt="Fix the issue")

    with patch.object(runner, '_clone_github_repo', side_effect=capture_clone) as mock_clone, \
         patch.object(runner, '_run_process'), \
         patch.object(runner, '_extract_session_file', return_value=tmp_path / 'output' / 'trace-1.jsonl'), \
         patch.object(runner, '_copy_workspace_snapshot'), \
         patch.object(runner, '_write_pi_agent_settings'), \
         patch.object(runner, '_write_pi_project_settings', side_effect=capture_project_settings):
        runner.run_session('Fix the issue', 'test-session', prompt_input=prompt_input)

    assert mock_clone.called
    assert cloned_destination is not None
    assert cloned_destination.name == "perplexica-mcp"
    assert captured_workspace == cloned_destination


def test_copy_normalized_session_file_populates_reasoning_summary(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "reasoning",
                            "summary": [],
                            "content": [
                                {
                                    "type": "reasoning_text",
                                    "text": "Use a direct factorial implementation.",
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    CodexRunner._copy_normalized_session_file(source, destination)

    events = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines() if line]
    assert events[0]["payload"]["summary"] == [
        {"type": "summary_text", "text": "Use a direct factorial implementation."}
    ]
    assert events[1]["payload"]["type"] == "message"


def test_copy_normalized_session_file_converts_codex_custom_tool_events(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call",
                            "status": "completed",
                            "call_id": "call_patch",
                            "name": "apply_patch",
                            "input": "*** Begin Patch\n*** Add File: app.py\n+print('hi')\n*** End Patch\n",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call_output",
                            "call_id": "call_patch",
                            "output": json.dumps({"output": "Success\n", "metadata": {"exit_code": 0}}),
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    CodexRunner._copy_normalized_session_file(source, destination)

    events = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines() if line]
    assert events[0]["payload"]["type"] == "function_call"
    assert events[0]["payload"]["arguments"] == json.dumps(
        {"patch": "*** Begin Patch\n*** Add File: app.py\n+print('hi')\n*** End Patch\n"},
        ensure_ascii=False,
    )
    assert events[0]["payload"]["status"] == "completed"
    assert events[1]["payload"] == {
        "call_id": "call_patch",
        "type": "function_call_output",
        "output": "Success\n",
    }


def test_pi_runner_builds_command_and_project_settings(tmp_path: Path):
    config = Config(
        agent={"provider": "pi"},
        model=ModelConfig(model="claude-sonnet-4-20250514", reasoning_effort="high"),
        api=APIConfig(
            provider="anthropic",
            base_url="https://proxy.example.com/v1",
            api_key="sk-ant-test",
        ),
    )
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    runner._write_pi_agent_settings(tmp_path)
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    extension_source = (tmp_path / "extensions" / "teich_system_prompt.ts").read_text(encoding="utf-8")
    models = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))

    assert settings == {
        "defaultProvider": "teich-anthropic",
        "defaultModel": "claude-sonnet-4-20250514",
        "defaultThinkingLevel": "high",
    }
    assert 'before_agent_start' in extension_source
    assert 'ctx.getSystemPrompt()' in extension_source
    assert models == {
        "providers": {
            "teich-anthropic": {
                "baseUrl": "https://proxy.example.com/v1",
                "api": "openai-responses",
                "authHeader": True,
                "apiKey": "sk-ant-test",
                "models": [
                    {
                        "id": "claude-sonnet-4-20250514",
                        "maxTokens": 131072,
                    }
                ],
            }
        }
    }

    with patch.object(PiRunner, "_resolve_pi_executable", return_value="@mariozechner/pi-coding-agent"):
        command = runner._build_pi_command(
            "Inspect the repo",
            tmp_path / "workspace",
            tmp_path,
            tmp_path / "sessions",
            "teich-pi-test-session",
        )
        assert command == [
            "docker",
            "run",
            "--rm",
            "--name",
            "teich-pi-test-session",
            "--user",
            "codex",
            "-e",
            "HOME=/home/codex",
            "-e",
            "PI_CODING_AGENT_DIR=/home/codex/.pi/agent",
            "-v",
            f"{tmp_path / 'workspace'}:/workspace",
            "-v",
            f"{tmp_path}:/home/codex/.pi/agent",
            "-v",
            f"{tmp_path / 'sessions'}:/home/codex/pi-sessions",
            "-w",
            "/workspace",
            "teich-runtime:v3",
            "npx",
            "-y",
            "@mariozechner/pi-coding-agent",
            "--mode",
            "json",
            "--session-dir",
            "/home/codex/pi-sessions",
            "--provider",
            "teich-anthropic",
            "--model",
            "claude-sonnet-4-20250514",
            "--thinking",
            "high",
            "--print",
            "@/workspace/.teich-prompt.txt",
        ]


def test_pi_runner_resolves_pi_executable_from_path():
    assert PiRunner._resolve_pi_executable() == "@mariozechner/pi-coding-agent"


def test_pi_run_session_discards_partial_trace_on_failure(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    workspace_root = tmp_path / "workspace-root"
    workspace = workspace_root
    agent_dir = tmp_path / "pi-agent"
    session_dir = tmp_path / "pi-sessions"
    workspace.mkdir()
    agent_dir.mkdir()
    session_dir.mkdir()

    def fail_after_writing_trace(*args, **kwargs):
        (session_dir / "partial.jsonl").write_text(
            '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Test prompt"}]}}\n',
            encoding="utf-8",
        )
        raise RuntimeError("boom")

    with patch.object(runner, "_prepare_workspace", return_value=(workspace_root, workspace)), \
         patch("teich.runner.tempfile.mkdtemp", side_effect=[str(agent_dir), str(session_dir)]), \
         patch.object(runner, "_write_pi_agent_settings"), \
         patch.object(runner, "_write_pi_project_settings"), \
         patch.object(runner, "_run_process", side_effect=fail_after_writing_trace):
        with pytest.raises(RuntimeError, match="boom"):
            runner.run_session("Test prompt", "test-session")

    assert not (tmp_path / "output" / "partials").exists()


def test_chat_runner_writes_structured_dataset_row_from_responses_api(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini", reasoning_effort="medium"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)

    payload = {
        "model": "gpt-4.1-mini",
        "output": [
            {"type": "reasoning", "summary": [{"text": "I should greet the user."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]},
        ],
        "output_text": "Hi!",
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
    }

    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("teich.runner.urlopen", return_value=response) as mock_urlopen:
        result = runner.run_session("Hello", "chat-session")

    assert result == tmp_path / "output" / "chat-session.jsonl"
    row = json.loads(result.read_text(encoding="utf-8").strip())
    assert row["prompt"] == "Hello"
    assert row["response"] == "Hi!"
    assert row["thinking"] == "I should greet the user."
    assert row["messages"][2]["thinking"] == "I should greet the user."
    assert row["usage"]["totalTokens"] == 7
    request = mock_urlopen.call_args.args[0]
    assert request.full_url == "https://api.openai.com/v1/responses"


def test_chat_runner_supports_follow_up_prompts_as_multiturn_rows(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    payloads = [
        {
            "model": "gpt-4.1-mini",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Initial answer"}]}],
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        },
        {
            "model": "gpt-4.1-mini",
            "output": [
                {"type": "reasoning", "summary": [{"text": "Need to revise."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Follow-up answer"}]},
            ],
            "usage": {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
        },
    ]
    responses = []
    for payload in payloads:
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        responses.append(response)

    with patch("teich.runner.urlopen", side_effect=responses) as mock_urlopen:
        result = runner.run_session(
            "Build a todo app",
            "chat-session",
            prompt_input=PromptInput(prompt="Build a todo app", follow_up_prompts=["Now add tests"]),
        )

    row = json.loads(result.read_text(encoding="utf-8").strip())
    assert row["prompt"] == "Build a todo app"
    assert row["follow_up_prompts"] == ["Now add tests"]
    assert row["responses"] == ["Initial answer", "Follow-up answer"]
    assert row["response"] == "Follow-up answer"
    assert row["thinking"] == "Need to revise."
    assert row["usage"]["totalTokens"] == 20
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant", "user", "assistant"]

    second_request = mock_urlopen.call_args_list[1].args[0]
    second_body = json.loads(second_request.data.decode("utf-8"))
    assert second_body["input"] == [
        {"role": "user", "content": "Build a todo app"},
        {"role": "assistant", "content": "Initial answer"},
        {"role": "user", "content": "Now add tests"},
    ]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"error": {"message": "Rate limit exceeded", "code": 429}}, "Rate limit exceeded"),
        ({"error": "temporarily unavailable"}, "temporarily unavailable"),
        ({"status": "failed", "error": {"message": "provider failed"}}, "provider failed"),
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "max_output_tokens"),
        ({"output": [{"type": "error", "message": "upstream timeout"}]}, "upstream timeout"),
        ({"choices": [{"error": {"message": "model overloaded"}}]}, "model overloaded"),
        ({"choices": [{"finish_reason": "content_filter", "message": {"role": "assistant", "content": "blocked"}}]}, "content_filter"),
    ],
)
def test_chat_runner_rejects_openai_compatible_error_payloads(tmp_path: Path, payload: dict[str, object], match: str):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openrouter", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("teich.runner.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match=match):
            runner.run_session("Hello", "chat-session")

    assert not (tmp_path / "output" / "chat-session.jsonl").exists()


def test_chat_runner_run_all_marks_api_error_as_failed_and_does_not_append_row(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openrouter", api_key="sk-test", wire_api="responses"),
        prompts=["Hello"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    updates = []
    payload = {"error": {"message": "Rate limit exceeded", "code": 429}}
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("teich.runner.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            runner.run_all(max_concurrency=1, progress_callback=updates.append)

    assert [update.status for update in updates] == ["queued", "running", "failed"]
    assert updates[-1].error and "Rate limit exceeded" in updates[-1].error
    destination = tmp_path / "output" / "chat.jsonl"
    assert not destination.exists()


def test_chat_runner_run_all_preserves_completed_rows_when_later_response_is_invalid_json(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openrouter", api_key="sk-test", wire_api="responses"),
        prompts=["Hello", "Bad JSON"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    updates = []
    good_payload = {
        "model": "gpt-4.1-mini",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]},
        ],
    }
    good_response = MagicMock()
    good_response.read.return_value = json.dumps(good_payload).encode("utf-8")
    good_response.__enter__.return_value = good_response
    good_response.__exit__.return_value = False
    bad_response = MagicMock()
    bad_response.read.return_value = b"<html>rate limited</html>"
    bad_response.__enter__.return_value = bad_response
    bad_response.__exit__.return_value = False

    with patch("teich.runner.urlopen", side_effect=[good_response, bad_response]):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            runner.run_all(max_concurrency=1, progress_callback=updates.append)

    destination = tmp_path / "output" / "chat.jsonl"
    assert destination.exists()
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in rows] == ["Hello"]
    assert [update.status for update in updates] == ["queued", "running", "completed", "queued", "running", "failed"]


def test_chat_runner_run_all_writes_one_dataset_file_with_all_rows(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["Hello", "Who are you?"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    updates = []

    def fake_completion(prompt: str) -> dict[str, object]:
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "content": f"Response to {prompt}", "thinking": None},
            ],
            "system": "You are a helpful assistant",
            "prompt": prompt,
            "thinking": None,
            "response": f"Response to {prompt}",
            "model": "gpt-4.1-mini",
            "provider": "openai",
            "usage": {"input": 1, "output": 2, "reasoning": 0, "totalTokens": 3},
            "metadata": {"trace_type": "chat", "model_provider": "openai", "model": "gpt-4.1-mini"},
        }

    with patch.object(runner, "_request_chat_completion", side_effect=fake_completion):
        results = runner.run_all(max_concurrency=2, progress_callback=updates.append)

    assert results == [tmp_path / "output" / "chat.jsonl"]
    assert sorted(path.name for path in (tmp_path / "output").glob("*.jsonl")) == ["chat.jsonl"]
    rows = [json.loads(line) for line in results[0].read_text(encoding="utf-8").splitlines()]
    assert sorted(row["prompt"] for row in rows) == ["Hello", "Who are you?"]
    assert [update.status for update in updates].count("queued") == 2
    assert [update.status for update in updates].count("completed") == 2
    assert all(update.trace_path == results[0] for update in updates if update.status == "completed")


def test_chat_runner_run_all_creates_and_appends_dataset_while_batch_is_running(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["fast", "blocked"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    release_blocked = threading.Event()

    def fake_completion(prompt: str) -> dict[str, object]:
        if prompt == "blocked":
            assert release_blocked.wait(timeout=2)
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "content": f"Response to {prompt}", "thinking": None},
            ],
            "prompt": prompt,
            "response": f"Response to {prompt}",
        }

    result_holder = {}
    with patch.object(runner, "_request_chat_completion", side_effect=fake_completion):
        thread = threading.Thread(
            target=lambda: result_holder.update(results=runner.run_all(max_concurrency=2))
        )
        thread.start()
        destination = tmp_path / "output" / "chat.jsonl"
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if destination.exists() and destination.read_text(encoding="utf-8").count("\n") == 1:
                break
            time.sleep(0.01)
        else:
            release_blocked.set()
            thread.join(timeout=2)
            pytest.fail("chat.jsonl did not receive the completed row while another prompt was still running")

        rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
        assert [row["prompt"] for row in rows] == ["fast"]
        release_blocked.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert result_holder["results"] == [destination]
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["prompt"] for row in rows) == ["blocked", "fast"]


def test_chat_runner_run_all_queues_prompts_lazily(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["Hello", "Who are you?", "What now?"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    updates = []
    update_thread_names = []
    started = []
    started_lock = threading.Lock()
    first_batch_started = threading.Event()
    release_first_batch = threading.Event()

    def record_update(update):
        update_thread_names.append(threading.current_thread().name)
        updates.append(update)

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, destination, progress_callback, append_lock=None):
        with started_lock:
            started.append(prompt_id)
            if len(started) == 2:
                first_batch_started.set()
        if prompt_id in {"prompt-1", "prompt-2"}:
            assert release_first_batch.wait(timeout=2)
        return prompt_index, {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                {"role": "user", "content": prompt_input.prompt, "thinking": None},
                {"role": "assistant", "content": f"Response to {prompt_input.prompt}", "thinking": None},
            ],
            "prompt": prompt_input.prompt,
            "response": f"Response to {prompt_input.prompt}",
        }

    with patch.object(runner, '_run_chat_prompt_task', side_effect=fake_task):
        result_holder = {}

        def run():
            result_holder["results"] = runner.run_all(max_concurrency=2, progress_callback=record_update)

        thread = threading.Thread(target=run)
        thread.start()
        assert first_batch_started.wait(timeout=2)
        time.sleep(0.05)
        assert [update.prompt_id for update in updates if update.status == "queued"] == ["prompt-1", "prompt-2"]
        release_first_batch.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert result_holder["results"] == [tmp_path / "output" / "chat.jsonl"]
    assert [update.prompt_id for update in updates if update.status == "queued"] == ["prompt-1", "prompt-2", "prompt-3"]
    queued_update_thread_names = [
        thread_name
        for update, thread_name in zip(updates, update_thread_names, strict=True)
        if update.status == "queued"
    ]
    assert all(thread_name.startswith("teich-chat-worker-") for thread_name in queued_update_thread_names)


def test_chat_runner_run_all_preserves_completed_rows_when_later_prompt_fails(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["Hello", "Who are you?"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, destination, progress_callback, append_lock=None):
        if prompt_id == "prompt-2":
            raise RuntimeError("boom")
        training_row = {
            "messages": [
                {"role": "user", "content": prompt_input.prompt, "thinking": None},
                {"role": "assistant", "content": "Hi", "thinking": None},
            ],
            "prompt": prompt_input.prompt,
            "response": "Hi",
        }
        if append_lock is not None:
            with append_lock:
                runner._append_chat_training_row(destination, training_row)
        return prompt_index, training_row

    with patch.object(runner, '_run_chat_prompt_task', side_effect=fake_task):
        with pytest.raises(RuntimeError, match="boom"):
            runner.run_all(max_concurrency=1)

    destination = tmp_path / "output" / "chat.jsonl"
    assert destination.exists()
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in rows] == ["Hello"]


def test_resume_detects_completed_chat_prompts(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chat.jsonl").write_text(
        json.dumps(
            {
                "prompt": "Hello",
                "response": "Hi",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_inputs = [PromptInput(prompt="Hello"), PromptInput(prompt="Who are you?")]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["Who are you?"]


def test_resume_detects_completed_structured_rows_without_top_level_prompt(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chat.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "response": "Hi",
            }
        )
        + "\n"
        + json.dumps({"metadata": {"note": "legacy sidecar row without messages"}})
        + "\n",
        encoding="utf-8",
    )
    prompt_inputs = [PromptInput(prompt="Hello"), PromptInput(prompt="Who are you?")]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["Who are you?"]


def test_resume_detects_completed_chat_follow_up_prompt_sets(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chat.jsonl").write_text(
        json.dumps(
            {
                "prompt": "Build app",
                "follow_up_prompts": ["Add tests"],
                "response": "Done",
                "messages": [
                    {"role": "user", "content": "Build app"},
                    {"role": "assistant", "content": "Built"},
                    {"role": "user", "content": "Add tests"},
                    {"role": "assistant", "content": "Done"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_inputs = [
        PromptInput(prompt="Build app"),
        PromptInput(prompt="Build app", follow_up_prompts=["Add tests"]),
        PromptInput(prompt="Build app", follow_up_prompts=["Add docs"]),
    ]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [(item.prompt, item.follow_up_prompts) for item in pending] == [
        ("Build app", []),
        ("Build app", ["Add docs"]),
    ]


def test_resume_detects_completed_agent_follow_up_prompt_sets(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "codex.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"codex-1"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build app"}]}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Built"}]}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Add tests"}]}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [
        PromptInput(prompt="Build app"),
        PromptInput(prompt="Build app", follow_up_prompts=["Add tests"]),
        PromptInput(prompt="Build app", follow_up_prompts=["Add docs"]),
    ]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [(item.prompt, item.follow_up_prompts) for item in pending] == [
        ("Build app", []),
        ("Build app", ["Add docs"]),
    ]


def test_resume_deduplicates_new_configured_prompts(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chat.jsonl").write_text(
        json.dumps(
            {
                "prompt": "Hello",
                "response": "Hi",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_inputs = [
        PromptInput(prompt="Hello"),
        PromptInput(prompt="New task"),
        PromptInput(prompt=" New task \n"),
        PromptInput(prompt="Another task"),
    ]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["New task", "Another task"]


def test_resume_detects_completed_codex_and_pi_traces(tmp_path: Path):
    output_dir = tmp_path / "output"
    recovered_dir = output_dir / "recovered-pi-sessions"
    recovered_dir.mkdir(parents=True)
    (output_dir / "codex.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"codex-1"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build app"}]}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}}\n',
        encoding="utf-8",
    )
    (recovered_dir / "pi.jsonl").write_text(
        '{"type":"session","id":"pi-1"}\n'
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Fix bug"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Fixed"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [
        PromptInput(prompt="Build app"),
        PromptInput(prompt="Fix bug"),
        PromptInput(prompt="Write tests"),
    ]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["Write tests"]


def test_resume_unwraps_teich_prompt_file_wrapper_from_completed_traces(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "pi.jsonl").write_text(
        '{"type":"session","id":"pi-1"}\n'
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"<file name=\\"/workspace/.teich-prompt.txt\\">\\nFix bug\\n</file>"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Fixed"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [PromptInput(prompt="Fix bug"), PromptInput(prompt="Write tests")]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["Write tests"]


def test_resume_ignores_partial_trace_directory(tmp_path: Path):
    output_dir = tmp_path / "output"
    partials_dir = output_dir / "partials"
    partials_dir.mkdir(parents=True)
    (partials_dir / "partial.jsonl").write_text(
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Fix bug"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Fixed"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [PromptInput(prompt="Fix bug")]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["Fix bug"]


def test_chat_runner_resume_appends_existing_chat_file(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["Who are you?"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    destination = tmp_path / "output" / "chat.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps({"prompt": "Hello", "response": "Hi", "messages": []}) + "\n", encoding="utf-8")

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, destination, progress_callback, append_lock=None):
        training_row = {"prompt": prompt_input.prompt, "response": "I am Teich", "messages": []}
        if append_lock is not None:
            with append_lock:
                runner._append_chat_training_row(destination, training_row)
        return prompt_index, training_row

    with patch.object(runner, '_run_chat_prompt_task', side_effect=fake_task):
        assert runner.run_all(max_concurrency=1, resume=True) == [destination]

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in rows] == ["Hello", "Who are you?"]


def test_chat_runner_resume_extends_existing_chat_row_with_follow_ups(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    destination = tmp_path / "output" / "chat.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "prompt": "Build app",
                "response": "Built",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                    {"role": "user", "content": "Build app", "thinking": None},
                    {"role": "assistant", "content": "Built", "thinking": "planned"},
                ],
                "usage": {"input": 3, "output": 4, "totalTokens": 7},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_turn(prompt: str, history: list[dict[str, str]] | None = None):
        calls.append((prompt, list(history or [])))
        if prompt == "Add tests":
            return "Tests added", None, {"input": 1, "output": 2, "totalTokens": 3}, "gpt-4.1-mini"
        if prompt == "Polish":
            return "Polished", "checked", {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "gpt-4.1-mini"
        raise AssertionError(f"unexpected prompt: {prompt}")

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests", "Polish"])
    with patch.object(runner, "_request_chat_turn", side_effect=fake_turn):
        assert runner.run_all(max_concurrency=1, prompt_inputs=[prompt_input], resume=True) == [destination]

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == "Build app"
    assert row["follow_up_prompts"] == ["Add tests", "Polish"]
    assert row["responses"] == ["Built", "Tests added", "Polished"]
    assert row["response"] == "Polished"
    assert row["thinking"] == "planned\n\nchecked"
    assert row["usage"] == {"input": 6, "output": 9, "reasoning": 0, "totalTokens": 15}
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [call[0] for call in calls] == ["Add tests", "Polish"]
    assert calls[0][1] == [
        {"role": "user", "content": "Build app"},
        {"role": "assistant", "content": "Built"},
    ]
    assert calls[1][1][-2:] == [
        {"role": "user", "content": "Add tests"},
        {"role": "assistant", "content": "Tests added"},
    ]


def test_chat_runner_resume_extends_existing_partial_follow_up_row(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    destination = tmp_path / "output" / "chat.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "prompt": "Build app",
                "follow_up_prompts": ["Add tests"],
                "response": "Tests added",
                "responses": ["Built", "Tests added"],
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                    {"role": "user", "content": "Build app", "thinking": None},
                    {"role": "assistant", "content": "Built", "thinking": None},
                    {"role": "user", "content": "Add tests", "thinking": None},
                    {"role": "assistant", "content": "Tests added", "thinking": None},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_turn(prompt: str, history: list[dict[str, str]] | None = None):
        calls.append((prompt, list(history or [])))
        return "Polished", None, {"input": 1, "output": 1, "totalTokens": 2}, "gpt-4.1-mini"

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests", "Polish"])
    with patch.object(runner, "_request_chat_turn", side_effect=fake_turn):
        assert runner.run_all(max_concurrency=1, prompt_inputs=[prompt_input], resume=True) == [destination]

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["follow_up_prompts"] == ["Add tests", "Polish"]
    assert rows[0]["responses"] == ["Built", "Tests added", "Polished"]
    assert [call[0] for call in calls] == ["Polish"]
    assert calls[0][1][-2:] == [
        {"role": "user", "content": "Add tests"},
        {"role": "assistant", "content": "Tests added"},
    ]


def test_chat_runner_resume_continues_queue_after_first_concurrency_window(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    prompt_inputs = [PromptInput(prompt=f"Prompt {index}") for index in range(5)]
    claimed = []

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, destination, progress_callback, append_lock=None):
        claimed.append(prompt_input.prompt)
        training_row = {
            "prompt": prompt_input.prompt,
            "response": "Done",
            "messages": [
                {"role": "user", "content": prompt_input.prompt},
                {"role": "assistant", "content": "Done"},
            ],
        }
        if append_lock is not None:
            with append_lock:
                runner._append_chat_training_row(destination, training_row)
        return prompt_index, training_row

    with patch.object(runner, "_run_chat_prompt_task", side_effect=fake_task):
        assert runner.run_all(max_concurrency=2, prompt_inputs=prompt_inputs, resume=True) == [
            tmp_path / "output" / "chat.jsonl"
        ]

    assert claimed == [item.prompt for item in prompt_inputs]


def test_chat_runner_stops_claiming_new_prompts_after_failure(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["one", "two", "three", "four"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    release = threading.Event()
    claimed = []
    claimed_lock = threading.Lock()

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, destination, progress_callback, append_lock=None):
        with claimed_lock:
            claimed.append(prompt_id)
        if prompt_id == "prompt-1":
            release.wait(timeout=2)
            return prompt_index, {"prompt": prompt_input.prompt, "messages": []}
        if prompt_id == "prompt-2":
            release.set()
            raise RuntimeError("boom")
        return prompt_index, {"prompt": prompt_input.prompt, "messages": []}

    with patch.object(runner, '_run_chat_prompt_task', side_effect=fake_task):
        with pytest.raises(RuntimeError, match="boom"):
            runner.run_all(max_concurrency=2)

    assert set(claimed) == {"prompt-1", "prompt-2"}


def test_chat_run_all_times_out_hung_worker_without_waiting_forever(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["hung"],
        output={"traces_dir": tmp_path / "output"},
        timeout_seconds=1,
    )
    runner = ChatRunner(config)
    started = threading.Event()
    release = threading.Event()

    def blocked_task(*args, **kwargs):
        started.set()
        release.wait(timeout=5)

    try:
        with patch.object(runner, '_run_chat_prompt_task', side_effect=blocked_task):
            start = time.monotonic()
            with pytest.raises(RuntimeError, match="timed out"):
                runner.run_all(max_concurrency=1)
            assert time.monotonic() - start < 3
    finally:
        release.set()


def test_chat_run_all_propagates_keyboard_interrupt_without_waiting_for_workers(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["Hello", "Who are you?"],
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    started = threading.Event()
    release = threading.Event()

    def blocked_task(prompt_id, prompt_index, total_prompts, prompt_input, destination, progress_callback, append_lock=None):
        started.set()
        release.wait(timeout=2)
        return prompt_index, {"prompt": prompt_input.prompt, "messages": []}

    def interrupt_after_worker_starts(*args, **kwargs):
        assert started.wait(timeout=1)
        raise KeyboardInterrupt

    try:
        with patch.object(runner, '_run_chat_prompt_task', side_effect=blocked_task):
            with patch("teich.runner.threading.Thread.join", side_effect=interrupt_after_worker_starts):
                start = time.monotonic()
                with pytest.raises(KeyboardInterrupt):
                    runner.run_all(max_concurrency=2)
                elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < 0.5


def test_pi_runner_init_uses_shared_runtime_image():
    with patch.object(PiRunner, '_ensure_image') as mock_ensure:
        runner = PiRunner(Config(agent={"provider": "pi"}))

    mock_ensure.assert_called_once()
    assert runner.image_name == "teich-runtime:v3"


def test_pi_runner_uses_synthetic_provider_name_for_custom_base_url():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                api=APIConfig(provider="openai", base_url="http://localhost:1234/v1"),
            )
        )

    assert runner._pi_provider_name() == "teich-openai"


def test_pi_runner_keeps_builtin_openrouter_provider_name_for_custom_base_url():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                api=APIConfig(provider="openrouter", base_url="https://openrouter.ai/api/v1"),
            )
        )

    assert runner._pi_provider_name() == "openrouter"


def test_pi_runner_keeps_provider_name_without_custom_base_url():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                api=APIConfig(provider="openai"),
            )
        )

    assert runner._pi_provider_name() == "openai"


def test_pi_runner_maps_wire_api_to_openai_completions():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                model=ModelConfig(model="gemma-4"),
                api=APIConfig(
                    provider="openai",
                    base_url="http://localhost:1234/v1",
                    wire_api="completions",
                ),
            )
        )

    assert runner._pi_provider_api() == "openai-completions"
    assert runner._pi_provider_settings() == {
        "baseUrl": "http://host.docker.internal:1234/v1",
        "api": "openai-completions",
        "authHeader": True,
        "apiKey": "local",
        "models": [
            {
                "id": "gemma-4",
                "maxTokens": 131072,
            }
        ],
    }


def test_pi_runner_preserves_builtin_openrouter_models_for_cost_tracking():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                model=ModelConfig(model="deepseek/deepseek-v4-pro"),
                api=APIConfig(
                    provider="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="sk-or-test",
                    wire_api="completions",
                ),
            )
        )

    assert runner._pi_provider_settings() == {
        "baseUrl": "https://openrouter.ai/api/v1",
        "api": "openai-completions",
        "authHeader": True,
        "modelOverrides": {
            "deepseek/deepseek-v4-pro": {
                "maxTokens": 131072,
            }
        },
        "apiKey": "sk-or-test",
    }
    assert runner._pi_models_config() == {
        "providers": {
            "openrouter": {
                "baseUrl": "https://openrouter.ai/api/v1",
                "api": "openai-completions",
                "authHeader": True,
                "modelOverrides": {
                    "deepseek/deepseek-v4-pro": {
                        "maxTokens": 131072,
                    }
                },
                "apiKey": "sk-or-test",
            }
        }
    }


def test_pi_runner_applies_optional_model_overrides_for_custom_provider():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                model=ModelConfig(
                    model="gemma-4",
                    pi_model_overrides={
                        "name": "Gemma 4",
                        "input": ["text", "image"],
                        "contextWindow": 262144,
                        "maxTokens": 32768,
                        "reasoning": True,
                    },
                ),
                api=APIConfig(
                    provider="openai",
                    base_url="http://localhost:1234/v1",
                    wire_api="completions",
                ),
            )
        )

    assert runner._pi_provider_settings() == {
        "baseUrl": "http://host.docker.internal:1234/v1",
        "api": "openai-completions",
        "authHeader": True,
        "apiKey": "local",
        "models": [
            {
                "id": "gemma-4",
                "name": "Gemma 4",
                "input": ["text", "image"],
                "contextWindow": 262144,
                "maxTokens": 32768,
                "reasoning": True,
            }
        ],
    }


def test_pi_runner_applies_optional_model_overrides_for_builtin_openrouter():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                model=ModelConfig(
                    model="minimax/minimax-m2.7",
                    pi_model_overrides={
                        "maxTokens": 32768,
                        "contextWindow": 196608,
                    },
                ),
                api=APIConfig(
                    provider="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="sk-or-test",
                    wire_api="completions",
                ),
            )
        )

    assert runner._pi_provider_settings() == {
        "baseUrl": "https://openrouter.ai/api/v1",
        "api": "openai-completions",
        "authHeader": True,
        "modelOverrides": {
            "minimax/minimax-m2.7": {
                "maxTokens": 32768,
                "contextWindow": 196608,
            }
        },
        "apiKey": "sk-or-test",
    }


def test_pi_runner_applies_default_max_tokens_override_for_builtin_openrouter():
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(
            Config(
                agent={"provider": "pi"},
                model=ModelConfig(model="google/gemma-4-26b-it"),
                api=APIConfig(
                    provider="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="sk-or-test",
                    wire_api="responses",
                ),
            )
        )

    assert runner._pi_provider_settings() == {
        "baseUrl": "https://openrouter.ai/api/v1",
        "api": "openai-responses",
        "authHeader": True,
        "modelOverrides": {
            "google/gemma-4-26b-it": {
                "maxTokens": 131072,
            }
        },
        "apiKey": "sk-or-test",
    }


def test_pi_runner_extracts_saved_session_file(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.jsonl"
    session_file.write_text('{"type":"session","id":"pi-session"}\n', encoding="utf-8")

    result = runner._extract_session_file(
        "session-id",
        session_dir,
        datetime.fromtimestamp(0, tz=timezone.utc),
    )

    assert result == tmp_path / "output" / "session.jsonl"
    assert result.read_text(encoding="utf-8") == '{"type":"session","id":"pi-session"}\n'


def test_pi_runner_strips_provider_from_exported_trace(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "model_change",
                        "provider": "teich-openrouter",
                        "modelId": "deepseek/deepseek-v4-pro",
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "provider": "teich-openrouter",
                            "model": "deepseek/deepseek-v4-pro",
                            "content": [{"type": "text", "text": "done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner._extract_session_file(
        "session-id",
        session_dir,
        datetime.fromtimestamp(0, tz=timezone.utc),
    )

    exported_events = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines() if line]
    assert exported_events[1] == {
        "type": "model_change",
        "modelId": "deepseek/deepseek-v4-pro",
    }
    assert exported_events[2] == {
        "type": "message",
        "message": {
            "role": "assistant",
            "model": "deepseek/deepseek-v4-pro",
            "content": [{"type": "text", "text": "done"}],
        },
    }


def test_pi_runner_injects_captured_system_prompt_into_exported_trace(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "id": "pi-session",
                        "timestamp": "2026-04-30T07:14:43.420Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "custom",
                        "id": "custom-1",
                        "parentId": None,
                        "timestamp": "2026-04-30T07:14:43.430Z",
                        "customType": "teich-system-prompt",
                        "data": {"systemPrompt": "You are Pi. Help with code."},
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "user-1",
                        "parentId": "custom-1",
                        "timestamp": "2026-04-30T07:14:43.483Z",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "who are you?"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner._extract_session_file(
        "session-id",
        session_dir,
        datetime.fromtimestamp(0, tz=timezone.utc),
    )

    exported_events = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines() if line]
    assert exported_events[0]["type"] == "session"
    assert exported_events[1] == {
        "type": "message",
        "id": exported_events[1]["id"],
        "parentId": None,
        "timestamp": "2026-04-30T07:14:43.430Z",
        "message": {
            "role": "developer",
            "content": [{"type": "text", "text": "You are Pi. Help with code."}],
        },
    }
    assert exported_events[2]["message"]["role"] == "user"
    assert all(event.get("type") != "custom" for event in exported_events)


def test_pi_runner_rejects_malformed_tool_call_trace(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-1",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {}},
                                {"type": "toolCall", "id": "", "name": "", "arguments": {"command": "ls -la"}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "tool-1",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": "call-1",
                            "toolName": "bash",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Validation failed for tool \"bash\":\n  - command: must have required properties command\n\nReceived arguments:\n{}",
                                }
                            ],
                            "isError": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "tool-2",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": "",
                            "toolName": "",
                            "content": [{"type": "text", "text": "Tool  not found"}],
                            "isError": True,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed tool calls/results"):
        runner._extract_session_file(
            "session-id",
            session_dir,
            datetime.fromtimestamp(0, tz=timezone.utc),
        )

    assert list((tmp_path / "output").glob("*.jsonl")) == []
