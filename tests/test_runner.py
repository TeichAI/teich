"""Tests for runner module."""

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import pytest

from teich.config import APIConfig, Config, MCPConfig, ModelConfig, PromptInput
from teich.runner import CodexRunner, PiRunner


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
        assert "Build a todo app" in cmd
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
        )
        assert command == [
            "docker",
            "run",
            "--rm",
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
            "Inspect the repo",
        ]


def test_pi_runner_resolves_pi_executable_from_path():
    assert PiRunner._resolve_pi_executable() == "@mariozechner/pi-coding-agent"


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
        "apiKey": "sk-or-test",
    }
    assert runner._pi_models_config() == {
        "providers": {
            "openrouter": {
                "baseUrl": "https://openrouter.ai/api/v1",
                "api": "openai-completions",
                "authHeader": True,
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
