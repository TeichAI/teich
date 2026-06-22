"""Tests for runner module."""

import concurrent.futures
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen
from unittest.mock import MagicMock, patch

import pytest

import teich.runner as runner_module
from teich.config import APIConfig, Config, MCPConfig, ModelConfig, PromptInput
from teich.runner import (
    ChatRunner,
    ClaudeCodeRunner,
    CLAUDE_OPENROUTER_PROXY_SCRIPT,
    CodexRunner,
    DockerRuntimeRunner,
    HermesRunner,
    PiRunner,
    RUNTIME_CONTAINER_USER,
    SessionProgressUpdate,
    pending_prompt_inputs_for_resume,
)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tcp_port(port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
                time.sleep(0.05)
    raise TimeoutError(f"port {port} did not open")


def _filesystem_preserves_chmod(tmp_path: Path) -> bool:
    probe = tmp_path / "chmod-probe"
    probe.write_text("", encoding="utf-8")
    try:
        probe.chmod(0o600)
    except OSError:
        return False
    return oct(probe.stat().st_mode & 0o777) == "0o600"


def _claude_home_from_command(command: list[str]) -> Path:
    for index, item in enumerate(command):
        if item == "-v" and index + 1 < len(command):
            mount = command[index + 1]
            suffix = ":/home/codex/.claude"
            if mount.endswith(suffix):
                return Path(mount[: -len(suffix)])
    raise AssertionError(f"Claude home mount not found in command: {command}")


def _hermes_home_from_command(command: list[str]) -> Path:
    for index, item in enumerate(command):
        if item == "-v" and index + 1 < len(command):
            mount = command[index + 1]
            suffix = ":/home/codex/.hermes"
            if mount.endswith(suffix):
                return Path(mount[: -len(suffix)])
    raise AssertionError(f"Hermes home mount not found in command: {command}")


def _write_fake_claude_native_session(
    command: list[str],
    *,
    prompt: str = "smoke",
    model: str = "claude-sonnet-4-6",
) -> None:
    home_dir = _claude_home_from_command(command)
    session_id = "native-claude-session"
    session_file = home_dir / "projects" / "-workspace" / f"{session_id}.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": session_id,
            "timestamp": "2026-05-13T00:00:00.000Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "done"}],
            },
            "uuid": "assistant-uuid",
            "parentUuid": "user-uuid",
            "sessionId": session_id,
            "timestamp": "2026-05-13T00:00:01.000Z",
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 1000,
            "result": "done",
            "session_id": session_id,
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        },
    ]
    session_file.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_fake_hermes_state_db(home_dir: Path, *, prompt: str = "Build app", answer: str = "done") -> None:
    home_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(home_dir / "state.db")
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL,
            message_count INTEGER,
            tool_call_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            has_finished INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        """
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, 'cli', 'codex-mini-latest', '{}', '', NULL, ?, 2, 0, 1, 1, 1)",
        ("hermes-session", 1_778_672_000),
    )
    connection.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        [
            ("hermes-session", "user", prompt, 1_778_672_000),
            ("hermes-session", "assistant", answer, 1_778_672_001),
        ],
    )
    connection.commit()
    connection.close()


def _write_fake_pi_native_session(path: Path, turns: list[tuple[str, str]]) -> None:
    rows: list[dict[str, object]] = [{"type": "session", "id": "pi-session"}]
    for index, (prompt, answer) in enumerate(turns, start=1):
        rows.extend(
            [
                {
                    "type": "message",
                    "id": f"user-{index}",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                },
                {
                    "type": "message",
                    "id": f"assistant-{index}",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                },
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


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


def test_tracked_container_cleans_up_when_start_fails():
    with patch.object(DockerRuntimeRunner, "_ensure_image"):
        runner = DockerRuntimeRunner(Config())

    with patch.object(runner, "_start_container", side_effect=RuntimeError("boom")), \
         patch.object(runner, "_remove_container") as mock_remove:
        with pytest.raises(RuntimeError, match="boom"):
            runner._start_tracked_container(["docker", "run"], "teich-broken-start")

        mock_remove.assert_called_once_with("teich-broken-start")
    assert "teich-broken-start" not in runner._active_containers


def test_copy_workspace_snapshot_ignores_dangling_symlinks(tmp_path: Path):
    workspace = tmp_path / "workspace"
    destination = tmp_path / "snapshot"
    bin_dir = workspace / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (bin_dir / "python").symlink_to(tmp_path / "missing-python")

    DockerRuntimeRunner._copy_workspace_snapshot(workspace, destination)

    assert (destination / "pyproject.toml").read_text(encoding="utf-8") == "[project]\nname = 'demo'\n"
    assert not (destination / ".venv" / "bin" / "python").exists()


def test_tracked_container_cleanup_unregisters_after_success():
    with patch.object(DockerRuntimeRunner, "_ensure_image"):
        runner = DockerRuntimeRunner(Config())

    with patch.object(runner, "_start_container") as mock_start, \
         patch.object(runner, "_remove_container") as mock_remove:
        runner._start_tracked_container(["docker", "run"], "teich-live")
        assert "teich-live" in runner._active_containers
        runner._cleanup_tracked_container("teich-live")

    mock_start.assert_called_once_with(["docker", "run"])
    mock_remove.assert_called_once_with("teich-live")
    assert "teich-live" not in runner._active_containers


def test_terminate_active_processes_removes_registered_persistent_containers():
    with patch.object(DockerRuntimeRunner, "_ensure_image"):
        runner = DockerRuntimeRunner(Config())

    runner._register_active_container("teich-stuck")

    with patch.object(runner, "_remove_container") as mock_remove:
        runner._terminate_active_processes()

    mock_remove.assert_called_once_with("teich-stuck")
    assert "teich-stuck" not in runner._active_containers


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
    assert oct(codex_home.stat().st_mode & 0o777) == "0o777"
    if _filesystem_preserves_chmod(tmp_path):
        assert oct(config_file.stat().st_mode & 0o777) == "0o666"


def test_codex_config_writes_service_tier(tmp_path: Path):
    """Fast mode is written to config.toml as service_tier when set."""
    config = Config(model=ModelConfig(model="gpt-5.5", service_tier="fast"))
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'service_tier = "fast"' in content


def test_codex_config_omits_service_tier_when_unset(tmp_path: Path):
    config = Config(model=ModelConfig(model="gpt-5.5"))
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "service_tier" not in content


def test_codex_config_writes_reasoning_summary(tmp_path: Path):
    """reasoning_summary is written to config.toml as model_reasoning_summary when set."""
    config = Config(model=ModelConfig(model="gpt-5.5", reasoning_summary="detailed"))
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'model_reasoning_summary = "detailed"' in content


def test_codex_config_omits_reasoning_summary_when_unset(tmp_path: Path):
    config = Config(model=ModelConfig(model="gpt-5.5"))
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "model_reasoning_summary" not in content


def _codex_host_auth_config(tmp_path: Path) -> tuple[Config, Path]:
    host_auth = tmp_path / "host" / "auth.json"
    host_auth.parent.mkdir(parents=True, exist_ok=True)
    host_auth.write_text('{"tokens":{"refresh_token":"R0"}}', encoding="utf-8")
    config = Config(
        agent={
            "provider": "codex",
            "codex": {
                "use_host_auth": True,
                "host_auth_file": str(host_auth),
                "auth_dir": str(tmp_path / "project" / ".teich" / "codex-auth"),
            },
        },
    )
    return config, host_auth


def test_codex_prepare_shared_host_auth_seeds_from_host(tmp_path: Path):
    config, host_auth = _codex_host_auth_config(tmp_path)
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    shared = runner._prepare_shared_host_auth()
    assert shared == (tmp_path / "project" / ".teich" / "codex-auth" / "auth.json")
    assert shared.read_text(encoding="utf-8") == '{"tokens":{"refresh_token":"R0"}}'
    if _filesystem_preserves_chmod(tmp_path):
        assert oct(shared.stat().st_mode & 0o777) == "0o666"


def test_codex_prepare_shared_host_auth_gitignores_credentials(tmp_path: Path):
    """The credential snapshot dir is gitignored by default, not just by docs."""
    config, _ = _codex_host_auth_config(tmp_path)
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    shared = runner._prepare_shared_host_auth()
    gitignore = shared.parent / ".gitignore"
    assert gitignore.exists()
    assert "*" in gitignore.read_text(encoding="utf-8").split()


def test_codex_prepare_shared_host_auth_preserves_rotated_token(tmp_path: Path):
    """A refreshed (rotated) shared token must not be clobbered by the stale host file."""
    config, host_auth = _codex_host_auth_config(tmp_path)
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    shared = runner._prepare_shared_host_auth()
    # Codex refreshes in place: shared advances to R1, newer than the host file.
    shared.write_text('{"tokens":{"refresh_token":"R1"}}', encoding="utf-8")
    os.utime(shared, (time.time() + 10, time.time() + 10))
    again = runner._prepare_shared_host_auth()
    assert again.read_text(encoding="utf-8") == '{"tokens":{"refresh_token":"R1"}}'


def test_codex_prepare_shared_host_auth_reseeds_when_host_newer(tmp_path: Path):
    """A fresh host login (newer mtime) re-seeds the shared snapshot."""
    config, host_auth = _codex_host_auth_config(tmp_path)
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    runner._prepare_shared_host_auth()
    host_auth.write_text('{"tokens":{"refresh_token":"R_new"}}', encoding="utf-8")
    os.utime(host_auth, (time.time() + 10, time.time() + 10))
    shared = runner._prepare_shared_host_auth()
    assert shared.read_text(encoding="utf-8") == '{"tokens":{"refresh_token":"R_new"}}'


def test_codex_prepare_shared_host_auth_errors_when_missing(tmp_path: Path):
    config = Config(
        agent={
            "provider": "codex",
            "codex": {"use_host_auth": True, "host_auth_file": str(tmp_path / "missing.json")},
        },
    )
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    with pytest.raises(RuntimeError, match="codex login"):
        runner._prepare_shared_host_auth()


def _codex_host_auth_config_with_tokens(tmp_path: Path) -> Config:
    """Host-auth config whose snapshot is a complete ChatGPT login (the broker
    validates that both an access and a refresh token are present)."""
    host_auth = tmp_path / "host" / "auth.json"
    host_auth.parent.mkdir(parents=True, exist_ok=True)
    host_auth.write_text(
        '{"tokens":{"access_token":"A0","refresh_token":"R0","account_id":"acct"}}',
        encoding="utf-8",
    )
    return Config(
        agent={
            "provider": "codex",
            "codex": {
                "use_host_auth": True,
                "host_auth_file": str(host_auth),
                "auth_dir": str(tmp_path / "project" / ".teich" / "codex-auth"),
            },
        },
    )


def test_codex_command_uses_broker_and_suppresses_api_key(tmp_path: Path, monkeypatch):
    """With host-auth on, point Codex at the broker (refresh override + host-gateway),
    mount no auth.json, and pass no API key env (even an ambient one)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-should-not-leak")
    config = _codex_host_auth_config_with_tokens(tmp_path)
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    try:
        cmd = runner._build_codex_command(
            "Build app",
            workspace=tmp_path / "ws",
            codex_home=tmp_path / "ch",
            container_name="teich-codex-x",
        )
        override = next(
            part for part in cmd if part.startswith("CODEX_REFRESH_TOKEN_URL_OVERRIDE=")
        )
        assert override.startswith("CODEX_REFRESH_TOKEN_URL_OVERRIDE=http://host.docker.internal:")
        assert override.endswith("/oauth/token")
        assert "host.docker.internal:host-gateway" in cmd
        # No auth.json bind-mount, and no API key env (broker owns the real token).
        assert not any(":/home/codex/.codex/auth.json" in part for part in cmd)
        assert not any(part.startswith("OPENAI_API_KEY=") for part in cmd)
    finally:
        if runner._broker is not None:
            runner._broker.stop()


def test_codex_write_seeded_auth_hides_real_refresh_token(tmp_path: Path):
    """Each container's seeded auth.json carries real tokens but a secret refresh token."""
    config = _codex_host_auth_config_with_tokens(tmp_path)
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    try:
        broker = runner._ensure_broker()
        assert broker is not None
        codex_home = tmp_path / "ch"
        codex_home.mkdir()
        runner._write_seeded_codex_auth(codex_home, broker)
        seeded = json.loads((codex_home / "auth.json").read_text(encoding="utf-8"))
        assert seeded["tokens"]["access_token"] == "A0"
        assert seeded["tokens"]["refresh_token"] == broker.secret
        assert seeded["tokens"]["refresh_token"] != "R0"
    finally:
        if runner._broker is not None:
            runner._broker.stop()


def test_codex_command_without_host_auth_still_passes_api_key(tmp_path: Path):
    config = Config(model=ModelConfig(model="o4-mini"), openai_api_key="sk-test123")
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(config)
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "OPENAI_API_KEY=sk-test123" in cmd
    assert not any(":/home/codex/.codex/auth.json" in part for part in cmd)


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


def test_codex_run_session_salvages_complete_trace_after_nonzero_exit(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    trace_path = tmp_path / "output" / "codex.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "codex-salvage"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Build app"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    exit_error = subprocess.CalledProcessError(1, ["codex"], output="", stderr="exited after writing trace")

    with patch.object(runner, "_run_process", side_effect=exit_error), \
         patch.object(runner, "_extract_session_file", return_value=trace_path) as mock_extract, \
         patch.object(runner, "_copy_workspace_snapshot") as mock_copy_snapshot:
        result = runner.run_session("Build app", "codex-salvage")

    assert result == trace_path
    mock_extract.assert_called_once()
    mock_copy_snapshot.assert_called_once()


def test_codex_run_session_retries_clean_exit_with_incomplete_trace(tmp_path: Path):
    config = Config(
        output={
            "traces_dir": tmp_path / "output",
            "sandbox_dir": tmp_path / "sandbox",
            "failures_dir": tmp_path / "failures",
        }
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    commands: list[list[str]] = []

    def mounted_codex_home(command: list[str]) -> Path:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        match = next(mount for mount in mounts if mount.endswith(":/home/codex/.codex"))
        return Path(match.rsplit(":", maxsplit=1)[0])

    def write_trace(command: list[str], complete: bool) -> None:
        trace_path = mounted_codex_home(command) / "sessions" / "codex-retry.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"type": "session_meta", "payload": {"id": "codex-retry", "model_provider": "openrouter"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Build app"}],
                },
            },
        ]
        if complete:
            rows.append(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done"}],
                    },
                }
            )
        else:
            rows.extend(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": "{",
                            "call_id": "call-bad",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-bad",
                            "output": "failed to parse function arguments: EOF while parsing an object at line 1 column 1",
                        },
                    },
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "task_complete",
                            "last_agent_message": None,
                        },
                    },
                ]
            )
        trace_path.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )

    def run_then_retry(command, *args, **kwargs) -> None:
        commands.append(command)
        write_trace(command, complete=len(commands) > 1)

    with patch.object(runner, "_run_process", side_effect=run_then_retry) as mock_run_process, \
         patch.object(runner, "_copy_workspace_snapshot"):
        result = runner.run_session("Build app", "codex-retry")

    assert result == tmp_path / "output" / "codex-retry.jsonl"
    assert mock_run_process.call_count == 2
    assert "resume --last" not in " ".join(commands[0])
    assert "resume --last" in " ".join(commands[1])
    assert not list((tmp_path / "failures").glob("*.jsonl"))


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
        command = args[0]
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        session_mount = next(mount for mount in mounts if mount.endswith(":/home/codex/pi-sessions"))
        session_dir = Path(session_mount.rsplit(":", maxsplit=1)[0])
        assert captured_workspace is not None
        assert (captured_workspace / ".teich-prompt.txt").read_text(encoding="utf-8") == long_prompt
        _write_fake_pi_native_session(session_dir / "trace.jsonl", [(long_prompt, "Done")])

    with patch.object(pi_runner, '_run_process', side_effect=assert_pi_prompt_file_before_cleanup) as mock_pi_run_process, \
         patch.object(pi_runner, '_copy_workspace_snapshot'), \
         patch.object(pi_runner, '_write_pi_agent_settings'), \
         patch.object(pi_runner, '_write_pi_project_settings', side_effect=capture_project_settings):
        pi_runner.run_session(long_prompt, "pi-session")

    pi_command = mock_pi_run_process.call_args.args[0]
    assert long_prompt not in pi_command
    assert any("$(cat /workspace/.teich-prompt.txt)" in part for part in pi_command)


def test_docker_agent_commands_run_as_runtime_container_user(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with patch.object(CodexRunner, "_ensure_image"):
        codex_runner = CodexRunner(Config())
    codex_run_command, _proxy_target = codex_runner._build_codex_docker_base_command(
        workspace,
        tmp_path / "codex-home",
        "teich-codex-root",
    )
    codex_exec_command = codex_runner._build_codex_exec_command("teich-codex-root")

    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        claude_runner = ClaudeCodeRunner(Config(agent={"provider": "claude-code"}))
    claude_run_command = claude_runner._build_external_docker_base_command(
        workspace,
        tmp_path / "claude-home",
        "teich-claude-root",
    )
    claude_exec_command = claude_runner._build_external_exec_command("teich-claude-root")

    with patch.object(HermesRunner, "_ensure_image"):
        hermes_runner = HermesRunner(Config(agent={"provider": "hermes"}))
    hermes_run_command = hermes_runner._build_external_docker_base_command(
        workspace,
        tmp_path / "hermes-home",
        "teich-hermes-root",
    )
    hermes_exec_command = hermes_runner._build_external_exec_command("teich-hermes-root")

    with patch.object(PiRunner, "_ensure_image"), patch.object(PiRunner, "_resolve_pi_executable", return_value="pi"):
        pi_runner = PiRunner(Config(agent={"provider": "pi"}))
        pi_run_command = pi_runner._build_pi_command(
            "Inspect",
            workspace,
            tmp_path / "pi-agent",
            tmp_path / "pi-sessions",
            "teich-pi-root",
        )
        pi_exec_command = pi_runner._build_pi_exec_command("teich-pi-root")

    for command in [
        codex_run_command,
        codex_exec_command,
        claude_run_command,
        claude_exec_command,
        hermes_run_command,
        hermes_exec_command,
        pi_run_command,
        pi_exec_command,
    ]:
        assert "--user" in command
        assert command[command.index("--user") + 1] == RUNTIME_CONTAINER_USER


def test_external_runner_chmods_workspace_before_snapshot():
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(Config(agent={"provider": "hermes"}))

    command = runner._wrap_external_shell_command("echo done")

    assert "chmod -R a+rwX /home/codex/.hermes /workspace" in command


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

    def fake_run(command: list[str], _container_name: str | None, **_kwargs: object) -> tuple[str, str]:
        _write_fake_claude_native_session(command, prompt=long_prompt)
        return '{"type":"result","result":"done"}\n', ""

    with patch.object(runner, "_run_native_process_with_progress", side_effect=fake_run) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        trace_path = runner.run_session(long_prompt, "claude-session")

    command = mock_run.call_args.args[0]
    command_text = " ".join(command)
    assert long_prompt not in command
    assert "claude" in command_text
    assert "--output-format stream-json" in command_text
    assert "--permission-mode bypassPermissions" in command_text
    assert "< /workspace/.teich-prompt.txt" in command_text
    assert "chmod -R a+rwX /home/codex/.claude" in command_text
    assert "ANTHROPIC_API_KEY=sk-ant-test" in command
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "user"
    assert rows[0]["message"]["content"] == long_prompt
    assert rows[1]["type"] == "assistant"
    assert rows[2]["type"] == "result"
    assert "external_session_meta" not in {row["type"] for row in rows}


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

    def fake_run(command: list[str], _container_name: str | None, **_kwargs: object) -> tuple[str, str]:
        _write_fake_claude_native_session(command, prompt="smoke", model="minimax/minimax-m2.5:free")
        return '{"type":"result","result":"done"}\n', ""

    with patch.object(runner, "_run_native_process_with_progress", side_effect=fake_run) as mock_run, \
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
    assert "sleep 1" not in command_text
    assert "/dev/tcp/127.0.0.1/17891" in command_text
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "user"
    assert rows[0]["message"]["content"] == "smoke"
    assert rows[1]["message"]["model"] == "minimax/minimax-m2.5:free"


def test_claude_openrouter_proxy_strips_decoded_compression_headers(tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("node is required for the proxy smoke test")

    upstream_port = _free_tcp_port()
    proxy_port = _free_tcp_port()
    response_payload = b'{"ok":true}\n'
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            seen["accept_encoding"] = self.headers.get("accept-encoding")
            seen["authorization"] = self.headers.get("authorization")
            seen["openrouter_cache"] = self.headers.get("x-openrouter-cache")
            seen["path"] = self.path
            seen["body"] = json.loads(body.decode("utf-8"))
            compressed = gzip.compress(response_payload)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-encoding", "gzip")
            self.send_header("content-length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", upstream_port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    proxy_script = tmp_path / "claude_openrouter_proxy.js"
    proxy_script.write_text(CLAUDE_OPENROUTER_PROXY_SCRIPT + "\n", encoding="utf-8")
    process = subprocess.Popen(
        ["node", str(proxy_script)],
        env={
            **os.environ,
            "TEICH_CLAUDE_PROXY_TARGET": f"http://127.0.0.1:{upstream_port}/v1",
            "TEICH_CLAUDE_PROXY_TARGET_MODEL": "minimax/minimax-m3",
            "TEICH_CLAUDE_PROXY_PORT": str(proxy_port),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_tcp_port(proxy_port)
        request = Request(
            f"http://127.0.0.1:{proxy_port}/v1/messages?beta=true",
            data=json.dumps(
                {
                    "model": "claude-sonnet-4-6",
                    "thinking": {"type": "enabled"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "cache me",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": "sk-test",
                "accept-encoding": "gzip, deflate, br",
                "x-openrouter-cache": "true",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            body = response.read()

        assert body == response_payload
        assert "content-encoding" not in headers
        assert "content-length" not in headers
        assert seen["accept_encoding"] == "gzip, deflate, br"
        assert seen["authorization"] == "Bearer sk-test"
        assert seen["openrouter_cache"] == "true"
        assert seen["path"] == "/v1/messages?beta=true"
        assert seen["body"] == {
            "model": "minimax/minimax-m3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "cache me",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        server.shutdown()
        server.server_close()


def test_claude_openrouter_proxy_handles_requests_concurrently(tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("node is required for the proxy smoke test")

    upstream_port = _free_tcp_port()
    proxy_port = _free_tcp_port()
    request_count = 5
    response_delay = 0.35
    stats = {"inflight": 0, "max_inflight": 0, "requests": 0}
    stats_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            if length:
                self.rfile.read(length)
            with stats_lock:
                stats["inflight"] += 1
                stats["requests"] += 1
                stats["max_inflight"] = max(stats["max_inflight"], stats["inflight"])
            try:
                time.sleep(response_delay)
                payload = json.dumps({"ok": True, "path": self.path}).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                with stats_lock:
                    stats["inflight"] -= 1

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", upstream_port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    proxy_script = tmp_path / "claude_openrouter_proxy.js"
    proxy_script.write_text(CLAUDE_OPENROUTER_PROXY_SCRIPT + "\n", encoding="utf-8")
    process = subprocess.Popen(
        ["node", str(proxy_script)],
        env={
            **os.environ,
            "TEICH_CLAUDE_PROXY_TARGET": f"http://127.0.0.1:{upstream_port}/v1",
            "TEICH_CLAUDE_PROXY_TARGET_MODEL": "minimax/minimax-m3",
            "TEICH_CLAUDE_PROXY_PORT": str(proxy_port),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def send_request(index: int) -> bytes:
        request = Request(
            f"http://127.0.0.1:{proxy_port}/v1/messages?request={index}",
            data=json.dumps({"model": "claude-sonnet-4-6"}).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": "sk-test"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return response.read()

    try:
        _wait_for_tcp_port(proxy_port)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as executor:
            bodies = list(executor.map(send_request, range(request_count)))
        elapsed = time.perf_counter() - started

        assert len(bodies) == request_count
        assert stats["requests"] == request_count
        assert stats["max_inflight"] >= 2
        assert elapsed < response_delay * request_count
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        server.shutdown()
        server.server_close()


def test_claude_code_native_process_emits_live_trace_progress(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        api=APIConfig(provider="anthropic", api_key="none"),
        model=ModelConfig(model="claude-opus-4-6"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
        timeout_seconds=5,
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    home_dir = tmp_path / "claude-home"
    native_trace = home_dir / "projects" / "-workspace" / "live.jsonl"
    trace_text = (
        json.dumps(
            {
                "type": "user",
                "sessionId": "claude-live",
                "message": {"role": "user", "content": "Build app"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "sessionId": "claude-live",
                "message": {
                    "role": "assistant",
                    "model": "claude-opus-4-6",
                    "content": [{"type": "text", "text": "Working on it."}],
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                },
            }
        )
        + "\n"
    )
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib, sys, time;"
            "p=pathlib.Path(sys.argv[1]);"
            "p.parent.mkdir(parents=True, exist_ok=True);"
            "time.sleep(0.1);"
            "p.write_text(sys.argv[2], encoding='utf-8');"
            "time.sleep(0.6)"
        ),
        str(native_trace),
        trace_text,
    ]
    started_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    updates: list[SessionProgressUpdate] = []
    progress_base = SessionProgressUpdate(
        prompt_id="prompt-1",
        prompt_index=1,
        total_prompts=1,
        prompt="Build app",
        prompt_preview="Build app",
        status="running",
        session_id="claude-live",
        started_at=started_at,
    )

    runner._run_native_process_with_progress(
        command,
        None,
        session_id="claude-live",
        home_dir=home_dir,
        existing_sessions=set(),
        started_at=started_at,
        progress_callback=updates.append,
        progress_base=progress_base,
    )

    trace_updates = [update for update in updates if update.trace_path == native_trace]
    assert trace_updates
    assert any(update.details == "Claude trace: live.jsonl" for update in trace_updates)


def test_claude_code_runner_uses_proxy_for_custom_non_claude_model(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        api=APIConfig(
            provider="openai",
            api_key="none",
            base_url="https://lm.gptbox.dev/v1",
        ),
        model=ModelConfig(model="Opus-Agent", approval_policy="never"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    def write_complete_trace(_command, _container_name, **kwargs) -> None:
        home_dir = kwargs["home_dir"]
        trace_file = home_dir / "projects" / "-workspace" / "trace.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "claude-custom-session",
                            "message": {"role": "user", "content": "smoke"},
                        },
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "claude-custom-session",
                            "message": {
                                "role": "assistant",
                                "model": "claude-sonnet-4-6",
                                "content": [{"type": "text", "text": "done"}],
                            },
                        },
                        separators=(",", ":"),
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    with patch.object(runner, "_run_native_process_with_progress", side_effect=write_complete_trace) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        runner.run_session("smoke", "claude-custom-session")

    command = mock_run.call_args.args[0]
    command_text = " ".join(command)
    assert "ANTHROPIC_AUTH_TOKEN=none" in command
    assert "ANTHROPIC_API_KEY=" in command
    assert "OPENAI_API_KEY=none" in command
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:17891" in command
    assert "TEICH_CLAUDE_PROXY_TARGET=https://lm.gptbox.dev/v1" in command
    assert "TEICH_CLAUDE_PROXY_TARGET_MODEL=Opus-Agent" in command
    assert "--model claude-sonnet-4-6" in command_text
    assert "--model Opus-Agent" not in command_text


def test_claude_code_run_session_salvages_complete_trace_after_nonzero_exit(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        api=APIConfig(provider="anthropic", api_key="sk-ant-test"),
        model=ModelConfig(model="claude-sonnet-4-6", approval_policy="never"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    exit_error = subprocess.CalledProcessError(1, ["claude"], output="", stderr="exited after writing trace")

    def fail_after_writing_complete_trace(_command, _container_name, **kwargs) -> None:
        home_dir = kwargs["home_dir"]
        trace_path = home_dir / "projects" / "-workspace" / "claude.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "claude-salvage",
                            "message": {"role": "user", "content": "Build app"},
                        },
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "claude-salvage",
                            "message": {
                                "role": "assistant",
                                "model": "claude-sonnet-4-6",
                                "content": [{"type": "text", "text": "Done"}],
                            },
                        },
                        separators=(",", ":"),
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raise exit_error

    with patch.object(runner, "_run_native_process_with_progress", side_effect=fail_after_writing_complete_trace) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot") as mock_copy_snapshot:
        result = runner.run_session("Build app", "claude-salvage")

    assert result == tmp_path / "output" / "claude.jsonl"
    mock_run.assert_called_once()
    mock_copy_snapshot.assert_called_once()


def test_claude_code_run_session_retries_provider_error_trace_before_exporting(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        api=APIConfig(provider="anthropic", api_key="sk-ant-test"),
        model=ModelConfig(model="claude-sonnet-4-6", approval_policy="never"),
        output={
            "traces_dir": tmp_path / "output",
            "sandbox_dir": tmp_path / "sandbox",
            "failures_dir": tmp_path / "failures",
        },
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    commands: list[list[str]] = []

    def write_native_trace(home_dir: Path, *, complete: bool) -> None:
        trace_path = home_dir / "projects" / "-workspace" / "claude-retry.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "type": "user",
                "sessionId": "claude-retry",
                "uuid": "user-1",
                "message": {"role": "user", "content": "Build app"},
            }
        ]
        if complete:
            rows.append(
                {
                    "type": "assistant",
                    "sessionId": "claude-retry",
                    "uuid": "assistant-2",
                    "parentUuid": "user-1",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [{"type": "text", "text": "Recovered and finished."}],
                    },
                }
            )
        else:
            rows.append(
                {
                    "type": "assistant",
                    "sessionId": "claude-retry",
                    "uuid": "assistant-error",
                    "parentUuid": "user-1",
                    "isApiErrorMessage": True,
                    "error": "api_error",
                    "message": {
                        "role": "assistant",
                        "model": "<synthetic>",
                        "content": [
                            {
                                "type": "text",
                                "text": 'API Error: ZlibError fetching "http://127.0.0.1:17891/v1/messages?beta=true".',
                            }
                        ],
                    },
                }
            )
        trace_path.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )

    def fail_then_recover(command, _container_name, **kwargs) -> None:
        commands.append(command)
        write_native_trace(kwargs["home_dir"], complete=len(commands) > 1)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(
                1,
                command,
                output="",
                stderr='API Error: ZlibError fetching "http://127.0.0.1:17891/v1/messages?beta=true".',
            )

    with patch.object(runner, "_run_native_process_with_progress", side_effect=fail_then_recover) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        result = runner.run_session("Build app", "claude-retry")

    assert result == tmp_path / "output" / "claude-retry.jsonl"
    assert mock_run.call_count == 2
    assert "--continue" not in " ".join(commands[0])
    assert "--resume claude-retry" in " ".join(commands[1])
    assert "--continue" not in " ".join(commands[1])
    assert not list((tmp_path / "failures").glob("*.jsonl"))


def test_claude_code_extract_makes_native_session_tree_readable(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    home_dir = tmp_path / "home"
    session_file = home_dir / "projects" / "-workspace" / "native-session.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                "sessionId": "native-session",
                "timestamp": "2026-05-13T00:00:00.000Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session_file.chmod(0o600)

    result = runner._extract_native_session_file(
        "native-session",
        home_dir,
        set(),
        datetime.fromtimestamp(0, tz=timezone.utc),
    )

    assert result.read_text(encoding="utf-8").strip()
    if _filesystem_preserves_chmod(tmp_path):
        assert oct(session_file.stat().st_mode & 0o777) == "0o666"


def test_claude_code_extract_orders_split_reasoning_before_output(tmp_path: Path):
    config = Config(
        agent={"provider": "claude-code"},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)

    home_dir = tmp_path / "home"
    session_file = home_dir / "projects" / "-workspace" / "native-session.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "I'll edit it."}],
                        },
                        "sessionId": "native-session",
                        "timestamp": "2026-05-13T00:00:02.000Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{"type": "thinking", "thinking": "Need a small edit."}],
                        },
                        "sessionId": "native-session",
                        "timestamp": "2026-05-13T00:00:03.000Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": {}}],
                        },
                        "sessionId": "native-session",
                        "timestamp": "2026-05-13T00:00:04.000Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{"type": "redacted_thinking", "data": "opaque"}],
                        },
                        "sessionId": "native-session",
                        "timestamp": "2026-05-13T00:00:05.000Z",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner._extract_native_session_file(
        "native-session",
        home_dir,
        set(),
        datetime.fromtimestamp(0, tz=timezone.utc),
    )

    rows = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["message"]["content"][0]["type"] == "thinking"
    assert rows[0]["timestamp"] == "2026-05-13T00:00:02.000Z"
    assert rows[1]["message"]["content"][0]["type"] == "redacted_thinking"
    assert rows[1]["timestamp"] == "2026-05-13T00:00:03.000Z"
    assert rows[2]["message"]["content"][0]["type"] == "text"
    assert rows[2]["timestamp"] == "2026-05-13T00:00:04.000Z"
    assert rows[3]["message"]["content"][0]["type"] == "tool_use"
    assert rows[3]["timestamp"] == "2026-05-13T00:00:05.000Z"


def test_claude_code_runner_uses_resume_for_followups(tmp_path: Path):
    config = Config(
        agent={"provider": "claude"},
        model=ModelConfig(model="claude-sonnet-4-6"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(config)
    native_home: Path | None = None

    def fake_start(command: list[str]) -> None:
        nonlocal native_home
        native_home = _claude_home_from_command(command)

    calls: list[list[str]] = []

    def fake_run(command: list[str], _container_name: str | None, **_kwargs: object) -> tuple[str, str]:
        calls.append(command)
        home_dir = native_home or _kwargs["home_dir"]
        session_file = home_dir / "projects" / "-workspace" / "native-claude-session.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "type": "user",
                "message": {"role": "user", "content": "Build app"},
                "uuid": "user-uuid",
                "parentUuid": None,
                "sessionId": "native-claude-session",
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "done"}],
                },
                "uuid": "assistant-uuid",
                "parentUuid": "user-uuid",
                "sessionId": "native-claude-session",
            },
        ]
        if len(calls) > 1:
            rows.extend(
                [
                    {
                        "type": "user",
                        "message": {"role": "user", "content": "Add tests"},
                        "uuid": "user-followup-uuid",
                        "parentUuid": "assistant-uuid",
                        "sessionId": "native-claude-session",
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "model": "claude-sonnet-4-6",
                            "content": [{"type": "text", "text": "added tests"}],
                        },
                        "uuid": "assistant-followup-uuid",
                        "parentUuid": "user-followup-uuid",
                        "sessionId": "native-claude-session",
                    },
                ]
            )
        session_file.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )
        return '{"type":"result","result":"done"}\n', ""

    with patch.object(runner, "_start_container", side_effect=fake_start) as mock_start, \
         patch.object(runner, "_run_native_process_with_progress", side_effect=fake_run) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"), \
         patch.object(runner, "_remove_container") as mock_remove:
        runner.run_session(prompt_input.prompt, "claude-followups", prompt_input=prompt_input)

    mock_start.assert_called_once()
    assert mock_run.call_count == 2
    first_command = " ".join(mock_run.call_args_list[0].args[0])
    second_command = " ".join(mock_run.call_args_list[1].args[0])
    assert "--continue" not in first_command
    assert "--resume native-claude-session" in second_command
    assert "--continue" not in second_command
    mock_remove.assert_called_once_with("teich-claude-claude-followups")
    assert "teich-claude-claude-followups" not in runner._active_containers


def test_hermes_runner_uses_chat_query_and_prompt_file(tmp_path: Path):
    long_prompt = "x" * 40000
    config = Config(
        agent={"provider": "hermes"},
        api=APIConfig(provider="openrouter", api_key="sk-or-test"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)

    def fake_run(command: list[str], _container_name: str | None):
        _write_fake_hermes_state_db(_hermes_home_from_command(command), prompt=long_prompt, answer="done")
        return "session_id: hermes-session\n", ""

    with patch.object(runner, "_run_external_process", side_effect=fake_run) as mock_run, \
         patch.object(runner, "_copy_workspace_snapshot"):
        trace_path = runner.run_session(long_prompt, "hermes-session")

    command = mock_run.call_args.args[0]
    command_text = " ".join(command)
    assert long_prompt not in command
    assert "hermes chat --provider openrouter" in command_text
    assert "--model codex-mini-latest" in command_text
    assert "--toolsets safe,terminal,file,skills,memory,session_search,delegation" in command_text
    assert "--ignore-user-config" in command_text
    assert '--source teich -q "$(cat /workspace/.teich-prompt.txt)"' in command_text
    assert "OPENROUTER_API_KEY=sk-or-test" in command
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert trace_path.name == "sessions.jsonl"
    assert rows[0]["id"] == "hermes-session"
    assert rows[0]["source"] == "cli"
    assert rows[0]["hermes_source"] == "cli"
    assert rows[0]["model_provider"] == "openrouter"
    assert rows[0]["toolsets"] == ["safe", "terminal", "file", "skills", "memory", "session_search", "delegation"]
    tool_names = {tool["function"]["name"] for tool in rows[0]["tools"]}
    assert {"delegate_task", "terminal", "read_file", "write_file"}.issubset(tool_names)
    assert rows[0]["messages"][0]["role"] == "user"
    assert rows[0]["messages"][0]["content"] == long_prompt
    assert rows[0]["messages"][1]["role"] == "assistant"
    assert rows[0]["messages"][1]["content"] == "done"


def test_hermes_runner_writes_custom_endpoint_runtime_config(tmp_path: Path):
    config = Config(
        agent={"provider": "hermes"},
        api=APIConfig(provider="openai", base_url="https://lm.gptbox.dev/v1", api_key="none"),
        model=ModelConfig(model="Opus-Agent", context_length=131072),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    runner._write_hermes_runtime_config(home_dir)

    command = runner._build_shell_command()
    hermes_config = json.loads((home_dir / "config.yaml").read_text(encoding="utf-8"))

    assert "hermes chat --provider custom" in command
    assert hermes_config["model"] == {
        "default": "Opus-Agent",
        "provider": "custom",
        "base_url": "https://lm.gptbox.dev/v1",
        "api_mode": "chat_completions",
        "context_length": 131072,
    }
    assert hermes_config["custom_providers"] == [
        {
            "name": "teich-custom",
            "base_url": "https://lm.gptbox.dev/v1",
            "model": "Opus-Agent",
            "api_mode": "chat_completions",
            "models": {"Opus-Agent": {"context_length": 131072}},
        }
    ]


_DEV_INSTRUCTIONS = "Think out loud and explain your reasoning."


def _dev_instructions_config(provider: str, tmp_path: Path) -> Config:
    return Config(
        agent={"provider": provider},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
        developer_instructions=_DEV_INSTRUCTIONS,
    )


def _plain_config(provider: str, tmp_path: Path) -> Config:
    return Config(
        agent={"provider": provider},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )


def test_claude_appends_developer_instructions_as_system_prompt(tmp_path: Path):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_dev_instructions_config("claude-code", tmp_path))
    command = runner._build_shell_command()
    assert "--append-system-prompt" in command
    assert _DEV_INSTRUCTIONS in command


def test_claude_omits_system_prompt_without_developer_instructions(tmp_path: Path):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_plain_config("claude-code", tmp_path))
    assert "--append-system-prompt" not in runner._build_shell_command()


def test_pi_appends_developer_instructions_as_system_prompt(tmp_path: Path):
    with patch.object(PiRunner, "_ensure_image"):
        runner = PiRunner(_dev_instructions_config("pi", tmp_path))
    joined = " ".join(runner._build_pi_agent_command())
    assert "--append-system-prompt" in joined
    assert _DEV_INSTRUCTIONS in joined


def test_pi_omits_system_prompt_without_developer_instructions(tmp_path: Path):
    with patch.object(PiRunner, "_ensure_image"):
        runner = PiRunner(_plain_config("pi", tmp_path))
    assert "--append-system-prompt" not in " ".join(runner._build_pi_agent_command())


def test_hermes_writes_developer_instructions_to_agents_md(tmp_path: Path):
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(_dev_instructions_config("hermes", tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    runner._write_agents_md(workspace)
    agents_md = workspace / "AGENTS.md"
    assert agents_md.exists()
    assert _DEV_INSTRUCTIONS in agents_md.read_text(encoding="utf-8")


def test_hermes_appends_to_existing_agents_md(tmp_path: Path):
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(_dev_instructions_config("hermes", tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("# Existing project rules\n", encoding="utf-8")
    runner._write_agents_md(workspace)
    content = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "# Existing project rules" in content
    assert _DEV_INSTRUCTIONS in content


def test_hermes_no_agents_md_without_developer_instructions(tmp_path: Path):
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(_plain_config("hermes", tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    runner._write_agents_md(workspace)
    assert not (workspace / "AGENTS.md").exists()


def test_codex_writes_developer_instructions_to_config(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_dev_instructions_config("codex", tmp_path))
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert f'developer_instructions = "{_DEV_INSTRUCTIONS}"' in content


def test_external_runner_decodes_subprocess_output_as_utf8(tmp_path: Path):
    config = Config(
        agent={"provider": "hermes"},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)

    process = MagicMock()
    process.communicate.return_value = ("ok", "")
    process.returncode = 0
    with patch("teich.runner.subprocess.Popen", return_value=process) as mock_popen:
        stdout, stderr = runner._run_external_process(["docker", "run", "image"], None)

    assert (stdout, stderr) == ("ok", "")
    mock_popen.assert_called_once()
    assert mock_popen.call_args.kwargs["encoding"] == "utf-8"
    assert mock_popen.call_args.kwargs["errors"] == "replace"


def test_hermes_runner_exports_delegated_sessions_to_single_sessions_jsonl(tmp_path: Path):
    config = Config(
        agent={"provider": "hermes"},
        api=APIConfig(provider="openrouter"),
        model=ModelConfig(model="minimax/minimax-m2.5:free"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    state_db = home_dir / "state.db"
    connection = sqlite3.connect(state_db)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL,
            updated_at REAL,
            last_message_at REAL,
            message_count INTEGER,
            tool_call_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            total_cost REAL,
            has_finished INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT
        );
        """
    )
    connection.executemany(
        """
        INSERT INTO sessions VALUES (
            ?, 'cli', ?, '{}', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1
        )
        """,
        [
            ("parent-session", "minimax/minimax-m2.5:free", None, 1_778_672_000, 1_778_672_001, 1_778_672_001, 3, 1, 10, 4, 14, 0.0),
            ("child-session", "minimax/minimax-m2.5:free", "parent-session", 1_778_672_002, 1_778_672_003, 1_778_672_003, 2, 0, 5, 3, 8, 0.0),
        ],
    )
    connection.executemany(
        """
        INSERT INTO messages (
            session_id, role, content, tool_call_id, tool_calls, tool_name,
            timestamp, token_count, finish_reason, reasoning, reasoning_content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("parent-session", "user", "delegate this", None, None, None, 1_778_672_000, None, None, None, None),
            (
                "parent-session",
                "assistant",
                "",
                None,
                json.dumps(
                    [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "delegate_task", "arguments": {"prompt": "sub task"}},
                        }
                    ]
                ),
                None,
                1_778_672_001,
                None,
                None,
                None,
                None,
            ),
            ("child-session", "user", "sub task", None, None, None, 1_778_672_002, None, None, None, None),
            ("child-session", "assistant", "subagent smoke ok", None, None, None, 1_778_672_003, None, None, None, None),
        ],
    )
    connection.commit()
    connection.close()

    exported = runner._export_hermes_state_sessions(home_dir, tmp_path / "workspace")

    assert set(exported) == {"parent-session", "child-session"}
    assert exported["parent-session"].name == "sessions.jsonl"
    assert exported["child-session"] == exported["parent-session"]
    rows = [json.loads(line) for line in exported["parent-session"].read_text(encoding="utf-8").splitlines()]
    rows_by_id = {row["id"]: row for row in rows}
    assert rows_by_id["parent-session"]["source"] == "cli"
    assert rows_by_id["parent-session"]["hermes_source"] == "cli"
    assert rows_by_id["parent-session"]["parent_session_id"] is None
    assert rows_by_id["child-session"]["parent_session_id"] == "parent-session"
    assert rows_by_id["child-session"]["messages"][0]["role"] == "user"
    assert rows_by_id["child-session"]["messages"][0]["content"] == "sub task"
    assert rows_by_id["child-session"]["messages"][1]["role"] == "assistant"
    assert rows_by_id["child-session"]["messages"][1]["content"] == "subagent smoke ok"
    assert rows_by_id["parent-session"]["messages"][1]["tool_calls"][0]["function"]["name"] == "delegate_task"


def test_hermes_runner_exports_current_state_db_fields_as_native_rows(tmp_path: Path):
    config = Config(
        agent={"provider": "hermes"},
        api=APIConfig(provider="openrouter"),
        model=ModelConfig(model="minimax/minimax-m2.5:free"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    connection = sqlite3.connect(home_dir / "state.db")
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            codex_message_items TEXT
        );
        """
    )
    connection.execute(
        """
        INSERT INTO sessions (
            id, source, model, model_config, system_prompt, parent_session_id,
            started_at, ended_at, end_reason, message_count, tool_call_count,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, billing_provider, billing_base_url, billing_mode,
            estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
            api_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "current-session",
            "cli",
            "minimax/minimax-m2.5:free",
            "{}",
            "System for Hermes",
            None,
            1_778_672_000,
            1_778_672_003,
            "completed",
            2,
            1,
            10,
            4,
            2,
            1,
            3,
            "openrouter",
            "https://openrouter.ai/api/v1",
            None,
            0.001,
            None,
            "estimated",
            "provider",
            1,
        ),
    )
    connection.executemany(
        """
        INSERT INTO messages (
            session_id, role, content, tool_call_id, tool_calls, tool_name,
            timestamp, token_count, finish_reason, reasoning, reasoning_content,
            reasoning_details, codex_reasoning_items, codex_message_items
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("current-session", "user", "inspect", None, None, None, 1_778_672_000, None, None, None, None, None, None, None),
            (
                "current-session",
                "assistant",
                "",
                None,
                json.dumps([{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "README.md"}}}]),
                None,
                1_778_672_001,
                4,
                "tool_calls",
                None,
                "Need to inspect the README.",
                json.dumps([{"type": "reasoning_text", "text": "Need context."}]),
                json.dumps([{"type": "reasoning", "summary": []}]),
                json.dumps([{"type": "message", "content": []}]),
            ),
        ],
    )
    connection.commit()
    connection.close()

    exported = runner._export_hermes_state_sessions(home_dir, tmp_path / "workspace")
    rows = [json.loads(line) for line in exported["current-session"].read_text(encoding="utf-8").splitlines()]
    row = rows[0]

    assert len(rows) == 1
    assert exported["current-session"].name == "sessions.jsonl"
    assert row["id"] == "current-session"
    assert row["source"] == "cli"
    assert row["hermes_source"] == "cli"
    assert row["teich_export_status"] == "completed"
    assert row["teich_partial"] is False
    assert row["total_tokens"] == 14
    assert row["estimated_cost_usd"] == 0.001
    assert row["ended_at"] == 1_778_672_003
    assert row["system_prompt"] == "System for Hermes"
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][0]["content"] == "inspect"
    assert row["messages"][1]["role"] == "assistant"
    assert row["messages"][1]["reasoning_content"] == "Need to inspect the README."
    assert row["messages"][1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"path": "README.md"}


def _write_minimal_hermes_delegation_state_db(home_dir: Path) -> None:
    connection = sqlite3.connect(home_dir / "state.db")
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL,
            message_count INTEGER,
            tool_call_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            has_finished INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        """
    )
    connection.executemany(
        "INSERT INTO sessions VALUES (?, 'cli', 'minimax/minimax-m2.5:free', '{}', '', ?, ?, ?, ?, ?, ?, 0)",
        [
            ("parent-session", None, 1_778_672_000, 2, 1, 10, 4),
            ("child-session", "parent-session", 1_778_672_001, 2, 0, 5, 3),
        ],
    )
    connection.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        [
            ("parent-session", "user", "delegate this", 1_778_672_000),
            ("parent-session", "assistant", "parent partial", 1_778_672_001),
            ("child-session", "user", "sub task", 1_778_672_002),
            ("child-session", "assistant", "subagent smoke ok", 1_778_672_003),
        ],
    )
    connection.commit()
    connection.close()


def test_hermes_runner_salvages_complete_state_db_after_nonzero_exit(tmp_path: Path):
    config = Config(
        agent={"provider": "hermes"},
        api=APIConfig(provider="openrouter"),
        model=ModelConfig(model="minimax/minimax-m2.5:free"),
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(config)

    workspace_root = tmp_path / "workspace-root"
    workspace = workspace_root
    home_dir = tmp_path / "hermes-home"
    workspace.mkdir()
    home_dir.mkdir()

    def fail_after_writing_state_db(*args, **kwargs):
        _write_minimal_hermes_delegation_state_db(home_dir)
        raise subprocess.CalledProcessError(1, ["hermes"], output="", stderr="provider failed")

    with patch.object(runner, "_prepare_workspace", return_value=(workspace_root, workspace)), \
         patch("teich.runner.tempfile.mkdtemp", return_value=str(home_dir)), \
         patch.object(runner, "_run_external_process", side_effect=fail_after_writing_state_db):
        result = runner.run_session("delegate this", "hermes-session")

    assert result == tmp_path / "output" / "sessions.jsonl"
    rows = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == "parent-session"
    assert [row["content"] for row in rows[0]["messages"]] == [
        "delegate this",
        "parent partial",
    ]


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
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_codex_run_session_moves_partial_trace_to_failures_on_failure(tmp_path: Path):
    config = Config(
        output={
            "traces_dir": tmp_path / "output",
            "sandbox_dir": tmp_path / "sandbox",
            "failures_dir": tmp_path / "failures",
        }
    )

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

    assert not list((tmp_path / "output").rglob("*.jsonl"))
    failed_traces = list((tmp_path / "failures").glob("*.jsonl"))
    assert len(failed_traces) == 1
    assert failed_traces[0].name == "partial.jsonl"


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

    def record_process(command, *args, **kwargs) -> None:
        assert command[:3] == ["docker", "exec", "-i"]
        assert "--user" in command
        assert command[command.index("--user") + 1] == RUNTIME_CONTAINER_USER
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
    assert "teich-codex-test-session" not in runner._active_containers


def test_codex_run_session_cleans_persistent_container_after_followup_failure(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])

    with patch.object(runner, "_start_container"), \
         patch.object(runner, "_run_process", side_effect=RuntimeError("turn failed")), \
         patch.object(runner, "_remove_container") as mock_remove_container:
        with pytest.raises(RuntimeError, match="turn failed"):
            runner.run_session("Build app", "failure-session", prompt_input=prompt_input)

    mock_remove_container.assert_called_once_with("teich-codex-failure-session")
    assert "teich-codex-failure-session" not in runner._active_containers


def test_pi_run_session_runs_follow_up_prompts_by_continuing_session(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    captured_workspace = None
    prompt_file_values = []
    commands = []
    run_process_session_dirs = []
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
        assert command[:2] == ["docker", "exec"]
        assert "-i" not in command
        assert "--user" in command
        assert command[command.index("--user") + 1] == RUNTIME_CONTAINER_USER
        assert "-w" in command
        assert command[command.index("-w") + 1] == "/workspace"
        assert "teich-pi-pi-session" in command
        run_process_session_dirs.append(args[2])
        assert container_workspace is not None
        commands.append(command)
        prompt_file_values.append((captured_workspace / ".teich-prompt.txt").read_text(encoding="utf-8"))
        assert container_session_dir is not None
        trace_path = container_session_dir / "session.jsonl"
        if prompt_file_values[-1] == "Build app":
            (container_workspace / "created-by-first-turn.txt").write_text("persisted", encoding="utf-8")
            _write_fake_pi_native_session(trace_path, [("Build app", "Built")])
        else:
            assert (container_workspace / "created-by-first-turn.txt").read_text(encoding="utf-8") == "persisted"
            _write_fake_pi_native_session(trace_path, [("Build app", "Built"), ("Add tests", "Done")])

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])

    with patch.object(runner, "_start_container", side_effect=start_container) as mock_start_container, \
         patch.object(runner, "_write_pi_project_settings", side_effect=capture_project_settings), \
         patch.object(runner, "_run_process", side_effect=record_prompt_file), \
         patch.object(runner, "_remove_container") as mock_remove_container, \
         patch.object(runner, "_copy_workspace_snapshot"):
        result = runner.run_session("Build app", "pi-session", prompt_input=prompt_input)

    assert result == tmp_path / "output" / "session.jsonl"
    assert prompt_file_values == ["Build app", "Add tests"]
    assert run_process_session_dirs == [None, None]
    assert "--continue" not in " ".join(commands[0])
    assert "--continue" in " ".join(commands[1])
    assert mock_start_container.call_count == 1
    assert container_session_dir is not None
    assert mock_remove_container.call_args.args == ("teich-pi-pi-session",)
    assert "teich-pi-pi-session" not in runner._active_containers


def test_pi_run_session_cleans_persistent_container_after_followup_failure(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])

    with patch.object(runner, "_start_container"), \
         patch.object(runner, "_run_process", side_effect=RuntimeError("turn failed")), \
         patch.object(runner, "_remove_container") as mock_remove_container, \
         patch.object(runner, "_write_pi_project_settings"):
        with pytest.raises(RuntimeError, match="turn failed"):
            runner.run_session("Build app", "pi-failure", prompt_input=prompt_input)

    mock_remove_container.assert_called_once_with("teich-pi-pi-failure")
    assert "teich-pi-pi-failure" not in runner._active_containers


def test_pi_run_session_retries_followup_without_removing_persistent_container(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    container_session_dir = None
    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])

    def mounted_path(command: list[str], target: str) -> Path:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        match = next(mount for mount in mounts if mount.endswith(f":{target}"))
        return Path(match.rsplit(":", maxsplit=1)[0])

    def start_container(command: list[str]) -> None:
        nonlocal container_session_dir
        container_session_dir = mounted_path(command, "/home/codex/pi-sessions")

    def write_trace(turns: list[tuple[str, str]]) -> None:
        assert container_session_dir is not None
        _write_fake_pi_native_session(container_session_dir / "session.jsonl", turns)

    calls = 0

    def run_turn(command, *args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            write_trace([("Build app", "Built")])
            return
        if calls == 2:
            assert kwargs["remove_container_on_error"] is False
            assert "--continue" in " ".join(command)
            write_trace([("Build app", "Built")])
            raise subprocess.CalledProcessError(1, command, output="", stderr="provider stopped early")
        write_trace([("Build app", "Built"), ("Add tests", "Done")])

    with patch.object(runner, "_start_container", side_effect=start_container), \
         patch.object(runner, "_run_process", side_effect=run_turn) as mock_run_process, \
         patch.object(runner, "_remove_container") as mock_remove_container, \
         patch.object(runner, "_write_pi_project_settings"), \
         patch.object(runner, "_copy_workspace_snapshot"):
        result = runner.run_session("Build app", "pi-followup-retry", prompt_input=prompt_input)

    assert result == tmp_path / "output" / "session.jsonl"
    assert mock_run_process.call_count == 3
    mock_remove_container.assert_called_once_with("teich-pi-pi-followup-retry")


def test_pi_run_session_salvages_complete_trace_after_nonzero_exit(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    exit_error = subprocess.CalledProcessError(
        1,
        ["pi"],
        output="",
        stderr="provider reported a transient error after writing trace",
    )

    def mounted_path(command: list[str], target: str) -> Path:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        match = next(mount for mount in mounts if mount.endswith(f":{target}"))
        return Path(match.rsplit(":", maxsplit=1)[0])

    def fail_after_writing_complete_trace(command, *args, **kwargs) -> None:
        session_dir = mounted_path(command, "/home/codex/pi-sessions")
        _write_fake_pi_native_session(session_dir / "pi.jsonl", [("Build app", "Done")])
        raise exit_error

    with patch.object(runner, "_run_process", side_effect=fail_after_writing_complete_trace) as mock_run_process, \
         patch.object(runner, "_copy_workspace_snapshot") as mock_copy_snapshot, \
         patch.object(runner, "_write_pi_project_settings"):
        result = runner.run_session("Build app", "pi-salvage")

    assert result == tmp_path / "output" / "pi.jsonl"
    mock_run_process.assert_called_once()
    mock_copy_snapshot.assert_called_once()


def test_pi_run_session_retries_incomplete_tool_turn_before_exporting(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    commands: list[list[str]] = []

    def mounted_path(command: list[str], target: str) -> Path:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        match = next(mount for mount in mounts if mount.endswith(f":{target}"))
        return Path(match.rsplit(":", maxsplit=1)[0])

    def write_incomplete_tool_trace(path: Path) -> None:
        rows = [
            {"type": "session", "id": "pi-session"},
            {
                "type": "message",
                "id": "user-1",
                "message": {"role": "user", "content": [{"type": "text", "text": "Build app"}]},
            },
            {
                "type": "message",
                "id": "assistant-1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {"command": "ls"}}],
                },
            },
            {
                "type": "message",
                "id": "tool-1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )

    def run_then_retry(command, *args, **kwargs) -> None:
        commands.append(command)
        session_dir = mounted_path(command, "/home/codex/pi-sessions")
        trace_path = session_dir / "pi.jsonl"
        if len(commands) == 1:
            write_incomplete_tool_trace(trace_path)
            raise subprocess.CalledProcessError(1, command, output="", stderr="provider stopped early")
        _write_fake_pi_native_session(trace_path, [("Build app", "Recovered")])

    with patch.object(runner, "_run_process", side_effect=run_then_retry) as mock_run_process, \
         patch.object(runner, "_copy_workspace_snapshot"), \
         patch.object(runner, "_write_pi_project_settings"):
        result = runner.run_session("Build app", "pi-retry")

    assert result == tmp_path / "output" / "pi.jsonl"
    assert mock_run_process.call_count == 2
    assert "--continue" not in " ".join(commands[0])
    assert "--continue" in " ".join(commands[1])
    assert not list((tmp_path / "failures").glob("*.jsonl"))


def test_pi_run_session_quarantines_nonzero_exit_before_followup_turn(tmp_path: Path):
    config = Config(
        agent={"provider": "pi"},
        output={
            "traces_dir": tmp_path / "output",
            "sandbox_dir": tmp_path / "sandbox",
            "failures_dir": tmp_path / "failures",
        },
    )

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests"])
    exit_error = subprocess.CalledProcessError(1, ["pi"], output="", stderr="first turn failed")

    with patch.object(runner, "_start_container"), \
         patch.object(runner, "_run_process", side_effect=exit_error) as mock_run_process, \
         patch.object(runner, "_remove_container") as mock_remove_container, \
         patch.object(runner, "_write_pi_project_settings"):
        with pytest.raises(RuntimeError, match="first turn failed"):
            runner.run_session("Build app", "pi-followup-nonzero", prompt_input=prompt_input)

    assert mock_run_process.call_count == runner_module.AGENT_TURN_RETRY_LIMIT + 1
    mock_remove_container.assert_called_once_with("teich-pi-pi-followup-nonzero")


def test_pi_run_session_keeps_nonzero_failure_when_final_trace_is_invalid(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"})

    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    exit_error = subprocess.CalledProcessError(1, ["pi"], output="", stderr="provider failed")

    with patch.object(runner, "_run_process", side_effect=exit_error), \
         patch.object(runner, "_extract_session_file", side_effect=RuntimeError("terminal provider error")), \
         patch.object(runner, "_copy_workspace_snapshot") as mock_copy_snapshot, \
         patch.object(runner, "_write_pi_project_settings"):
        with pytest.raises(RuntimeError, match="provider failed"):
            runner.run_session("Build app", "pi-invalid-trace")

    mock_copy_snapshot.assert_not_called()


def test_pi_runner_preserves_tool_validation_failures_as_trace_events(tmp_path: Path):
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
                        "id": "assistant-2",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "I will correct the command."}],
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

    exported = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    assert exported[2]["message"]["role"] == "toolResult"
    assert exported[2]["message"]["isError"] is True
    assert "Validation failed for tool" in exported[2]["message"]["content"][0]["text"]
    assert exported[3]["message"]["content"][0]["text"] == "I will correct the command."


def test_pi_stdout_progress_reports_model_start_and_usage(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    metrics, details = runner._stdout_progress_update(
        [
            {"type": "session", "id": "pi-session"},
            {"type": "message_end", "message": {"role": "user", "content": []}},
            {
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "provider": "openrouter",
                    "model": "qwen/qwen3.7-max",
                    "usage": {"input": 12, "output": 0, "totalTokens": 12, "cost": {"total": 0.001}},
                },
            },
        ]
    )

    assert details == "pi model request started"
    assert metrics is not None
    assert metrics.provider == "openrouter"
    assert metrics.model == "qwen/qwen3.7-max"
    assert metrics.total_tokens == 12
    assert metrics.total_cost == 0.001


def test_pi_monitor_reports_stdout_progress_before_session_file(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"}, timeout_seconds=5)
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    class FakeProcess:
        args = ["pi"]

        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def kill(self) -> None:
            raise AssertionError("process should not be killed")

        def wait(self) -> int:
            return 0

    stdout_path = tmp_path / "stdout.jsonl"
    stderr_path = tmp_path / "stderr.txt"
    stdout_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"type": "session", "id": "pi-session"},
                {"type": "message_end", "message": {"role": "user", "content": []}},
                {
                    "type": "message_start",
                    "message": {
                        "role": "assistant",
                        "provider": "openai",
                        "model": "Opus-Agent",
                        "usage": {"input": 42, "output": 0, "totalTokens": 42},
                    },
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    updates: list[runner_module.SessionProgressUpdate] = []
    progress_base = runner_module.SessionProgressUpdate(
        prompt_id="prompt-1",
        prompt_index=1,
        total_prompts=1,
        prompt="Build app",
        prompt_preview="Build app",
        status="running",
        session_id="pi-session",
        started_at=datetime.now(timezone.utc),
    )

    with stdout_path.open("r+", encoding="utf-8") as stdout_handle, stderr_path.open("r+", encoding="utf-8") as stderr_handle:
        runner._monitor_process(
            FakeProcess(),  # type: ignore[arg-type]
            "pi-session",
            datetime.now(timezone.utc),
            None,
            updates.append,
            progress_base,
            stdout_handle,
            stderr_handle,
        )

    assert updates
    assert updates[-1].details == "pi model request started"
    assert updates[-1].metrics is not None
    assert updates[-1].metrics.model == "Opus-Agent"
    assert updates[-1].metrics.total_tokens == 42


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
    assert completed[0].metrics.has_token_usage is True
    assert completed[0].metrics.has_cost is True
    assert completed[0].metrics.total_tokens == 17
    assert completed[0].metrics.total_cost == 0.25


def test_run_all_does_not_block_queue_on_provider_stats_lookup(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2"], max_concurrency=1, output={"traces_dir": tmp_path / "output"})

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    claimed: list[str] = []

    def fake_run_session(
        prompt: str,
        session_id: str | None = None,
        progress_callback=None,
        progress_base=None,
        prompt_input=None,
    ) -> Path:
        claimed.append(prompt)
        trace_file = tmp_path / f"{prompt.replace(' ', '_').lower()}.jsonl"
        trace_file.write_text(
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"'
            + prompt
            + '"}]}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}}\n',
            encoding="utf-8",
        )
        return trace_file

    with patch.object(runner, "run_session", side_effect=fake_run_session), \
         patch.object(runner, "_summarize_trace_file_with_provider_stats", side_effect=AssertionError("provider stats should not block queue")):
        results = runner.run_all(max_concurrency=1)

    assert claimed == ["Prompt 1", "Prompt 2"]
    assert [path.name for path in results] == ["prompt_1.jsonl", "prompt_2.jsonl"]


def test_run_all_prequeues_all_prompts_before_workers_free_up(tmp_path: Path):
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
        assert [update.prompt_id for update in updates if update.status == "queued"] == [
            "prompt-1",
            "prompt-2",
            "prompt-3",
        ]
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
    assert all(thread_name == thread.name for thread_name in queued_update_thread_names)


def test_run_all_starts_next_queued_prompt_when_one_worker_finishes(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2", "Prompt 3"], max_concurrency=2)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    started: list[str] = []
    started_lock = threading.Lock()
    first_two_started = threading.Event()
    third_started = threading.Event()
    release_prompt_1 = threading.Event()
    release_prompt_2 = threading.Event()
    release_prompt_3 = threading.Event()

    def fake_task(prompt_id, prompt_index, total_prompts, prompt_input, progress_callback):
        with started_lock:
            started.append(prompt_id)
            if len(started) == 2:
                first_two_started.set()
            if prompt_id == "prompt-3":
                third_started.set()
        if prompt_id == "prompt-1":
            assert release_prompt_1.wait(timeout=2)
        elif prompt_id == "prompt-2":
            assert release_prompt_2.wait(timeout=2)
        else:
            assert release_prompt_3.wait(timeout=2)
        return tmp_path / f"{prompt_id}.jsonl"

    with patch.object(runner, '_run_prompt_task', side_effect=fake_task):
        result_holder = {}

        def run():
            result_holder["results"] = runner.run_all(max_concurrency=2)

        thread = threading.Thread(target=run)
        thread.start()
        assert first_two_started.wait(timeout=2)
        assert started == ["prompt-1", "prompt-2"]
        release_prompt_1.set()
        assert third_started.wait(timeout=2)
        assert started == ["prompt-1", "prompt-2", "prompt-3"]
        assert thread.is_alive()
        release_prompt_2.set()
        release_prompt_3.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert [path.name for path in result_holder["results"]] == ["prompt-1.jsonl", "prompt-2.jsonl", "prompt-3.jsonl"]


def test_run_all_continues_claiming_agent_prompts_after_failure(tmp_path: Path):
    config = Config(prompts=["Prompt 1", "Prompt 2", "Prompt 3"], max_concurrency=1)

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    trace_paths = {"Prompt 1": tmp_path / "first.jsonl", "Prompt 3": tmp_path / "third.jsonl"}

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
    assert trace_paths["Prompt 3"].exists()
    assert json.loads(trace_paths["Prompt 1"].read_text(encoding="utf-8")) == {"prompt": "Prompt 1"}
    assert json.loads(trace_paths["Prompt 3"].read_text(encoding="utf-8")) == {"prompt": "Prompt 3"}


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
    assert metrics.has_token_usage is True
    assert metrics.has_cost is True


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
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
                },
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
    assert metrics.cache_read_tokens == 2
    assert metrics.cache_write_tokens == 1
    assert metrics.total_tokens == 7
    assert metrics.est_total_tokens == 7
    assert metrics.has_token_usage is True
    assert metrics.has_cost is False


def test_summarize_trace_file_keeps_missing_provider_usage_unknown(tmp_path: Path):
    trace_file = tmp_path / "claude-trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "type": "external_session_meta",
                "payload": {
                    "model_provider": "openrouter",
                    "model": "minimax/minimax-m2.5:free",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "openrouter"
    assert metrics.model == "minimax/minimax-m2.5:free"
    assert metrics.total_tokens == 0
    assert metrics.total_cost == 0.0
    assert metrics.has_token_usage is False
    assert metrics.has_cost is False


def test_summarize_trace_file_reads_external_session_cache_metrics(tmp_path: Path):
    trace_file = tmp_path / "external-trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "type": "external_session_meta",
                "payload": {
                    "model_provider": "openrouter",
                    "model": "minimax/minimax-m3",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "reasoning_tokens": 2,
                    "cache_read_tokens": 7,
                    "cache_write_tokens": 3,
                    "total_tokens": 27,
                    "total_cost": 0.004,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "openrouter"
    assert metrics.model == "minimax/minimax-m3"
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 5
    assert metrics.reasoning_tokens == 2
    assert metrics.cache_read_tokens == 7
    assert metrics.cache_write_tokens == 3
    assert metrics.total_tokens == 27
    assert metrics.total_cost == 0.004
    assert metrics.has_token_usage is True
    assert metrics.has_cost is True


def test_summarize_trace_file_reads_hermes_session_meta(tmp_path: Path):
    trace_file = tmp_path / "hermes-trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "type": "hermes_session_meta",
                "payload": {
                    "model_provider": "openrouter",
                    "model": "minimax/minimax-m2.5:free",
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_tokens": 2,
                    "reasoning_tokens": 1,
                    "total_tokens": 17,
                    "estimated_cost_usd": 0.001,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "openrouter"
    assert metrics.model == "minimax/minimax-m2.5:free"
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 4
    assert metrics.cache_read_tokens == 2
    assert metrics.reasoning_tokens == 1
    assert metrics.total_tokens == 17
    assert metrics.total_cost == 0.001
    assert metrics.has_token_usage is True
    assert metrics.has_cost is True


def test_summarize_trace_file_reads_external_hermes_session_meta(tmp_path: Path):
    trace_file = tmp_path / "hermes-trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "type": "external_session_meta",
                "payload": {
                    "source": "hermes-agent",
                    "hermes_source": "cli",
                    "model_provider": "openrouter",
                    "model": "minimax/minimax-m2.5:free",
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_tokens": 2,
                    "reasoning_tokens": 1,
                    "total_tokens": 17,
                    "estimated_cost_usd": 0.001,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "openrouter"
    assert metrics.model == "minimax/minimax-m2.5:free"
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 4
    assert metrics.cache_read_tokens == 2
    assert metrics.reasoning_tokens == 1
    assert metrics.total_tokens == 17
    assert metrics.total_cost == 0.001
    assert metrics.has_token_usage is True
    assert metrics.has_cost is True


def test_summarize_trace_file_reads_hermes_export_session(tmp_path: Path):
    trace_file = tmp_path / "hermes-trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "id": "session-1",
                "task": "hello",
                "traces": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hi"},
                ],
                "tools": [],
                "metadata": {
                    "source": "cli",
                    "model_provider": "custom",
                    "model": "Opus-Agent",
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_tokens": 3,
                    "reasoning_tokens": 2,
                    "total_tokens": 23,
                    "estimated_cost_usd": 0.002,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = CodexRunner._summarize_trace_file(trace_file)

    assert metrics.provider == "custom"
    assert metrics.model == "Opus-Agent"
    assert metrics.input_tokens == 11
    assert metrics.output_tokens == 7
    assert metrics.cache_read_tokens == 3
    assert metrics.reasoning_tokens == 2
    assert metrics.total_tokens == 23
    assert metrics.total_cost == 0.002
    assert metrics.has_token_usage is True
    assert metrics.has_cost is True


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


def test_monitor_process_keeps_live_pi_tool_call_corruption_running(tmp_path: Path):
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

    process.poll.side_effect = [None, 0]

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

    process.kill.assert_not_called()
    process.wait.assert_not_called()


def test_monitor_process_does_not_kill_live_pi_provider_error(tmp_path: Path):
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
                        "id": "assistant-error",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "stopReason": "error",
                            "errorMessage": "OpenRouter upstream request failed.",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    process = MagicMock()
    process.poll.side_effect = [None, 0]
    process.args = ["docker", "run"]
    stdout_handle = io.StringIO()
    stderr_handle = io.StringIO()

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

    process.kill.assert_not_called()
    process.wait.assert_not_called()


def test_monitor_process_does_not_kill_live_pi_user_only_trace(tmp_path: Path):
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
                        "id": "user-1",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Build app"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    process = MagicMock()
    process.poll.side_effect = [None, 0]
    process.args = ["docker", "run"]
    stdout_handle = io.StringIO()
    stderr_handle = io.StringIO()

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

    process.kill.assert_not_called()
    process.wait.assert_not_called()


def test_pi_trace_with_model_error_is_copied_as_native_trace(tmp_path: Path):
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

    runner._copy_normalized_session_file(source, destination)

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_pi_trace_with_user_only_turns_is_copied_as_native_trace(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    source = tmp_path / "user-only-session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "user-1",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Build app"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "user-2",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Add a follow up"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "output" / "user-only-session.jsonl"
    destination.parent.mkdir(parents=True)

    runner._copy_normalized_session_file(source, destination)

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_pi_trace_with_terminal_length_stop_is_copied_as_native_trace(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    source = tmp_path / "length-session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "user-1",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Build app"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-1",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Partial answer"}],
                            "stopReason": "length",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "output" / "length-session.jsonl"
    destination.parent.mkdir(parents=True)

    runner._copy_normalized_session_file(source, destination)

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_pi_trace_preserves_recovered_provider_error(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    source = tmp_path / "recovered-session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-error",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "api": "openai-completions",
                            "model": "qwen/qwen3.7-max",
                            "usage": {"input": 0, "output": 0, "totalTokens": 0},
                            "stopReason": "error",
                            "errorMessage": "OpenRouter upstream request failed.",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-recovered",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Recovered and continuing."}],
                            "model": "qwen/qwen3.7-max",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "output" / "recovered-session.jsonl"
    destination.parent.mkdir(parents=True)

    runner._copy_normalized_session_file(source, destination)

    exported = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert exported[1]["message"]["errorMessage"] == "OpenRouter upstream request failed."
    assert exported[2]["message"]["content"][0]["text"] == "Recovered and continuing."


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
        assert 'local_provider_proxy.js' not in cmd_str
        assert 'TEICH_LOCAL_PROVIDER_TARGET=https://lm.gptbox.dev/v1' not in cmd_str


def test_codex_custom_endpoint_uses_placeholder_key_when_configured_as_none():
    config = Config(
        model=ModelConfig(model="Opus-Agent"),
        api=APIConfig(
            provider="openai",
            base_url="https://lm.gptbox.dev/v1",
            api_key="none",
        ),
    )

    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    command, _proxy_target = runner._build_codex_docker_base_command(
        Path("/workspace"),
        Path("/tmp/codex-home"),
        "teich-codex-test",
    )

    assert "OPENAI_API_KEY=none" in command


def test_codex_copies_metadata_only_trace_without_export_validation(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "dest.jsonl"
    source.write_text(
        json.dumps({"timestamp": "2026-05-22T00:00:00Z", "type": "session_meta", "payload": {"id": "s"}})
        + "\n"
        + json.dumps({"timestamp": "2026-05-22T00:00:00Z", "type": "event_msg", "payload": {"type": "task_complete"}})
        + "\n",
        encoding="utf-8",
    )

    CodexRunner._copy_normalized_session_file(source, destination)

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_codex_finalizer_appends_configured_tool_schemas(tmp_path: Path):
    config = Config(output={"traces_dir": tmp_path / "output"})
    with patch.object(CodexRunner, '_ensure_image'):
        runner = CodexRunner(config)

    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "codex-session"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Say hi"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Hi."}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tool = {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run commands.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        },
    }

    with patch("teich.runner.snapshot_configured_tools", return_value=[tool]):
        runner._finalize_trace_export(trace_file)

    rows = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    schema_rows = [
        row for row in rows
        if row.get("type") == "response_item" and row.get("payload", {}).get("type") == "tool_schema"
    ]
    assert len(schema_rows) == 1
    assert schema_rows[0]["payload"]["name"] == "exec_command"
    assert schema_rows[0]["payload"]["schema"]["parameters"]["required"] == ["cmd"]


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

    def write_completed_trace(command, *args, **kwargs) -> None:
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "-v"]
        session_mount = next(mount for mount in mounts if mount.endswith(":/home/codex/pi-sessions"))
        session_dir = Path(session_mount.rsplit(":", maxsplit=1)[0])
        _write_fake_pi_native_session(session_dir / "trace-1.jsonl", [("Fix the issue", "Done")])

    prompt_input = PromptInput(github_repo="armand0e/perplexica-mcp", prompt="Fix the issue")

    with patch.object(runner, '_clone_github_repo', side_effect=capture_clone) as mock_clone, \
         patch.object(runner, '_run_process', side_effect=write_completed_trace), \
         patch.object(runner, '_copy_workspace_snapshot'), \
         patch.object(runner, '_write_pi_agent_settings'), \
         patch.object(runner, '_write_pi_project_settings', side_effect=capture_project_settings):
        runner.run_session('Fix the issue', 'test-session', prompt_input=prompt_input)

    assert mock_clone.called
    assert cloned_destination is not None
    assert cloned_destination.name == "perplexica-mcp"
    assert captured_workspace == cloned_destination


def test_copy_normalized_session_file_preserves_codex_native_reasoning_event(tmp_path: Path):
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

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["payload"]["summary"] == [
        {"type": "summary_text", "text": "Use a direct factorial implementation."}
    ]
    assert rows[0]["payload"]["content"] == [
        {"type": "reasoning_text", "text": "Use a direct factorial implementation."}
    ]
    assert rows[1]["payload"]["type"] == "message"


def test_copy_normalized_session_file_orders_codex_late_reasoning_before_tool_call(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-13T00:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "call_id": "call_1",
                            "arguments": '{"cmd":"touch app.py"}',
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-13T00:00:03.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "reasoning",
                            "summary": [],
                            "content": [{"type": "reasoning_text", "text": "Create the file first."}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    CodexRunner._copy_normalized_session_file(source, destination)

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["payload"]["type"] == "reasoning"
    assert rows[0]["timestamp"] == "2026-05-13T00:00:02.000Z"
    assert rows[0]["payload"]["summary"] == [{"type": "summary_text", "text": "Create the file first."}]
    assert rows[1]["payload"]["type"] == "function_call"
    assert rows[1]["timestamp"] == "2026-05-13T00:00:03.000Z"
    assert rows[1]["payload"]["call_id"] == "call_1"


def test_copy_normalized_session_file_orders_codex_reasoning_before_text(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-13T00:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "I'll create it."}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-13T00:00:03.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "reasoning",
                            "summary": [{"text": "Create the file first."}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    CodexRunner._copy_normalized_session_file(source, destination)

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["payload"]["type"] == "reasoning"
    assert rows[0]["timestamp"] == "2026-05-13T00:00:02.000Z"
    assert rows[1]["payload"]["type"] == "message"
    assert rows[1]["timestamp"] == "2026-05-13T00:00:03.000Z"


def test_copy_normalized_session_file_preserves_codex_native_prompt_file_user_message(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": '<file name="/workspace/.teich-prompt.txt">\nCan you build me a dependency map for a software system?\n</file>',
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with source.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                    },
                }
            )
            + "\n"
        )

    CodexRunner._copy_normalized_session_file(source, destination)

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_copy_normalized_session_file_preserves_codex_user_only_trace(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Build app"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    CodexRunner._copy_normalized_session_file(source, destination)

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_copy_normalized_session_file_preserves_codex_native_custom_tool_events(tmp_path: Path):
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

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


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
    models = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))

    assert settings == {
        "defaultProvider": "teich-anthropic",
        "defaultModel": "claude-sonnet-4-20250514",
        "defaultThinkingLevel": "high",
    }
    assert not (tmp_path / "extensions").exists()
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

    with patch.object(PiRunner, "_resolve_pi_executable", return_value="pi"):
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
            RUNTIME_CONTAINER_USER,
            "-e",
            "HOME=/home/codex",
            "-e",
            "PI_CODING_AGENT_DIR=/home/codex/.pi/agent",
            "-e",
            "PI_OFFLINE=1",
            "-v",
            f"{tmp_path / 'workspace'}:/workspace",
            "-v",
            f"{tmp_path}:/home/codex/.pi/agent",
            "-v",
            f"{tmp_path / 'sessions'}:/home/codex/pi-sessions",
            "-w",
            "/workspace",
            "teich-runtime:v3",
            "sh",
            "-lc",
            "exec pi --mode json --session-dir "
            "/home/codex/pi-sessions --provider teich-anthropic --model "
            "claude-sonnet-4-20250514 --thinking high --print "
            '"$(cat /workspace/.teich-prompt.txt)"',
        ]


def test_pi_runner_resolves_pi_executable_from_path():
    assert PiRunner._resolve_pi_executable() == "pi"


def test_pi_run_session_moves_partial_trace_to_failures_on_failure(tmp_path: Path):
    config = Config(
        agent={"provider": "pi"},
        output={
            "traces_dir": tmp_path / "output",
            "sandbox_dir": tmp_path / "sandbox",
            "failures_dir": tmp_path / "failures",
        },
    )

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

    assert not list((tmp_path / "output").rglob("*.jsonl"))
    failed_traces = list((tmp_path / "failures").glob("*.jsonl"))
    assert len(failed_traces) == 1
    assert failed_traces[0].name == "partial.jsonl"


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
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert "system" not in row
    assert row["messages"][1]["thinking"] == "I should greet the user."
    assert row["usage"]["totalTokens"] == 7
    request = mock_urlopen.call_args.args[0]
    assert request.full_url == "https://api.openai.com/v1/responses"
    assert request.headers["User-agent"] == "teich"
    body = json.loads(request.data.decode("utf-8"))
    assert "instructions" not in body


def test_chat_runner_normalizes_openrouter_chat_completion_reasoning_usage():
    usage = ChatRunner._normalize_usage(
        {
            "prompt_tokens": 263,
            "completion_tokens": 80,
            "total_tokens": 343,
            "completion_tokens_details": {"reasoning_tokens": 67},
            "cost": 0.000032548824,
        }
    )

    assert usage == {
        "input": 263,
        "output": 80,
        "reasoning": 67,
        "totalTokens": 343,
        "cost": {"total": 0.000032548824},
    }


def test_chat_runner_uses_prompt_level_system_prompt(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    payload = {
        "model": "gpt-4.1-mini",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Brief answer"}]},
        ],
    }
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("teich.runner.urlopen", return_value=response) as mock_urlopen:
        result = runner.run_session(
            "Hello",
            "chat-session",
            prompt_input=PromptInput(prompt="Hello", system="Be concise."),
        )

    row = json.loads(result.read_text(encoding="utf-8").strip())
    assert row["system"] == "Be concise."
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    request = mock_urlopen.call_args.args[0]
    body = json.loads(request.data.decode("utf-8"))
    assert body["instructions"] == "Be concise."


def test_chat_runner_prefers_openrouter_generation_stats_usage(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="minimax/minimax-m2.5:free"),
        api=APIConfig(
            provider="openrouter",
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            wire_api="chat_completions",
        ),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    completion_payload = {
        "id": "gen-openrouter-123",
        "model": "minimax/minimax-m2.5:free",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    stats_payload = {
        "data": {
            "id": "gen-openrouter-123",
            "model": "minimax/minimax-m2.5:free",
            "provider_name": "MiniMax",
            "native_tokens_prompt": 12,
            "native_tokens_completion": 8,
            "native_tokens_reasoning": 3,
            "native_tokens_cached": 4,
            "total_cost": 0.00042,
        }
    }
    completion_response = MagicMock()
    completion_response.read.return_value = json.dumps(completion_payload).encode("utf-8")
    completion_response.__enter__.return_value = completion_response
    completion_response.__exit__.return_value = False
    stats_response = MagicMock()
    stats_response.read.return_value = json.dumps(stats_payload).encode("utf-8")
    stats_response.__enter__.return_value = stats_response
    stats_response.__exit__.return_value = False

    with patch("teich.runner.urlopen", side_effect=[completion_response, stats_response]) as mock_urlopen:
        result = runner.run_session("Hello", "chat-session")

    row = json.loads(result.read_text(encoding="utf-8").strip())
    assert row["usage"] == {
        "input": 12,
        "output": 8,
        "reasoning": 3,
        "cacheRead": 4,
        "totalTokens": 23,
        "cost": {"total": 0.00042},
        "generation_ids": ["gen-openrouter-123"],
    }
    metrics = runner._metrics_from_training_row(row)
    assert metrics.total_tokens == 23
    assert metrics.total_cost == 0.00042
    assert metrics.has_token_usage is True
    assert metrics.has_cost is True
    stats_request = mock_urlopen.call_args_list[1].args[0]
    assert mock_urlopen.call_args_list[1].kwargs["timeout"] == 3
    assert stats_request.full_url == "https://openrouter.ai/api/v1/generation?id=gen-openrouter-123"


def test_chat_usage_reads_openrouter_prompt_cache_details():
    usage = ChatRunner._normalize_usage(
        {
            "prompt_tokens": 10339,
            "completion_tokens": 60,
            "total_tokens": 10399,
            "prompt_tokens_details": {
                "cached_tokens": 10318,
                "cache_write_tokens": 21,
            },
        }
    )

    assert usage == {
        "input": 10339,
        "output": 60,
        "reasoning": 0,
        "totalTokens": 10399,
        "cacheRead": 10318,
        "cacheWrite": 21,
    }


def test_openrouter_generation_stats_reads_prompt_cache_details():
    usage = DockerRuntimeRunner._openrouter_usage_from_generation_data(
        {
            "id": "gen-openrouter-cache",
            "model": "qwen/qwen3-coder-plus",
            "provider_name": "Qwen",
            "native_tokens_prompt": 10339,
            "native_tokens_completion": 60,
            "prompt_tokens_details": {
                "cached_tokens": 10318,
                "cache_write_tokens": 21,
            },
            "total_cost": 0.001,
        }
    )

    assert usage == {
        "input": 10339,
        "output": 60,
        "reasoning": 0,
        "cacheRead": 10318,
        "totalTokens": 10399,
        "cacheWrite": 21,
        "cost": {"total": 0.001},
        "generation_id": "gen-openrouter-cache",
        "provider_name": "Qwen",
    }


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
    assert [message["role"] for message in row["messages"]] == ["user", "assistant", "user", "assistant"]

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


def test_chat_runner_retries_transient_provider_error_before_marking_failed(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openrouter", api_key="sk-test", wire_api="responses"),
        output={"traces_dir": tmp_path / "output"},
    )
    runner = ChatRunner(config)
    success = MagicMock()
    success.read.return_value = json.dumps(
        {
            "model": "gpt-4.1-mini",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    ).encode("utf-8")
    success.__enter__.return_value = success
    success.__exit__.return_value = False
    transient_error = runner_module.HTTPError(
        url="https://api.example.test/v1/responses",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"rate limited"}}'),
    )

    with patch("teich.runner.urlopen", side_effect=[transient_error, success]) as mock_urlopen, \
         patch("teich.runner.time.sleep") as mock_sleep:
        result = runner.run_session("Hello", "chat-retry")

    assert mock_urlopen.call_count == 2
    mock_sleep.assert_called_once()
    row = json.loads(result.read_text(encoding="utf-8"))
    assert row["response"] == "ok"


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
    assert [update.status for update in updates] == ["queued", "queued", "running", "completed", "running", "failed"]


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

    def fake_completion(prompt_input: PromptInput) -> dict[str, object]:
        prompt = prompt_input.prompt
        return {
            "messages": [
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "content": f"Response to {prompt}", "thinking": None},
            ],
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

    def fake_completion(prompt_input: PromptInput) -> dict[str, object]:
        prompt = prompt_input.prompt
        if prompt == "blocked":
            assert release_blocked.wait(timeout=2)
        return {
            "messages": [
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


def test_chat_runner_run_all_prequeues_all_prompts(tmp_path: Path):
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
        assert [update.prompt_id for update in updates if update.status == "queued"] == [
            "prompt-1",
            "prompt-2",
            "prompt-3",
        ]
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
    assert all(thread_name == thread.name for thread_name in queued_update_thread_names)


def test_chat_runner_run_all_preserves_completed_rows_when_later_prompt_fails(tmp_path: Path):
    config = Config(
        agent={"provider": "chat"},
        model=ModelConfig(model="gpt-4.1-mini"),
        api=APIConfig(provider="openai", api_key="sk-test", wire_api="responses"),
        prompts=["Hello", "Who are you?", "Continue anyway"],
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
    assert [row["prompt"] for row in rows] == ["Hello", "Continue anyway"]


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


def test_resume_treats_prompt_level_system_as_part_of_chat_completion_key(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chat.jsonl").write_text(
        json.dumps(
            {
                "system": "Be concise.",
                "prompt": "Hello",
                "response": "Hi",
                "messages": [
                    {"role": "system", "content": "Be concise."},
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
        PromptInput(prompt="Hello", system="Be concise."),
        PromptInput(prompt="Hello", system="Be thorough."),
    ]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert pending == []


def test_resume_matches_completed_agent_trace_when_configured_prompt_has_system(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "pi.jsonl").write_text(
        '{"type":"session","id":"pi-1"}\n'
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Build app"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Built"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [
        PromptInput(
            prompt="Build app",
            system="Use a specific system prompt that does not appear in native agent traces.",
        )
    ]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert pending == []


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


def test_resume_regenerates_agent_trace_that_ends_on_provider_error(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "claude-error.jsonl").write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "Build app"},
                "sessionId": "claude-error-session",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "isApiErrorMessage": True,
                "error": "api_error",
                "message": {
                    "role": "assistant",
                    "model": "<synthetic>",
                    "content": [
                        {
                            "type": "text",
                            "text": 'API Error: ZlibError fetching "http://127.0.0.1:17891/v1/messages?beta=true".',
                        }
                    ],
                },
                "sessionId": "claude-error-session",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pending = pending_prompt_inputs_for_resume([PromptInput(prompt="Build app")], output_dir)

    assert [item.prompt for item in pending] == ["Build app"]


def test_completed_turns_rejects_trace_that_ends_on_tool_result():
    incomplete = {
        "messages": [
            {"role": "user", "content": "Build app"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]
    }
    complete = {
        "messages": [
            *incomplete["messages"],
            {"role": "assistant", "content": "Done"},
        ]
    }

    assert not runner_module._training_example_completed_turns(incomplete, ["Build app"])
    assert runner_module._training_example_completed_turns(complete, ["Build app"])


def test_completed_turns_rejects_trace_that_ends_on_provider_error():
    trace = {
        "messages": [
            {"role": "user", "content": "Build app"},
            {"role": "assistant", "content": "I created the app."},
            {
                "role": "assistant",
                "content": 'API Error: ZlibError fetching "http://127.0.0.1:17891/v1/messages?beta=true".',
                "teich_provider_error": True,
            },
        ]
    }
    recovered = {
        "messages": [
            *trace["messages"],
            {"role": "assistant", "content": "Recovered and finished."},
        ]
    }

    assert not runner_module._training_example_completed_turns(trace, ["Build app"])
    assert runner_module._training_example_completed_turns(recovered, ["Build app"])


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


def test_resume_ignores_non_data_trace_directories(tmp_path: Path):
    output_dir = tmp_path / "output"
    partials_dir = output_dir / "partials"
    failures_dir = output_dir / "failures"
    partials_dir.mkdir(parents=True)
    failures_dir.mkdir(parents=True)
    (partials_dir / "partial.jsonl").write_text(
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Fix bug"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Fixed"}]}}\n',
        encoding="utf-8",
    )
    (failures_dir / "failed.jsonl").write_text(
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Build app"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Built"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [PromptInput(prompt="Fix bug"), PromptInput(prompt="Build app")]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir)

    assert [item.prompt for item in pending] == ["Fix bug", "Build app"]


def test_resume_ignores_configured_failures_dir_inside_output(tmp_path: Path):
    output_dir = tmp_path / "output"
    failed_dir = output_dir / "failed-traces"
    failed_dir.mkdir(parents=True)
    (failed_dir / "failed.jsonl").write_text(
        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Build app"}]}}\n'
        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Built"}]}}\n',
        encoding="utf-8",
    )
    prompt_inputs = [PromptInput(prompt="Build app")]

    pending = pending_prompt_inputs_for_resume(prompt_inputs, output_dir, excluded_dirs=[failed_dir])

    assert [item.prompt for item in pending] == ["Build app"]


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


def test_chat_runner_resume_regenerates_incomplete_follow_up_row(tmp_path: Path):
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
        if prompt == "Build app":
            return "Rebuilt", "replanned", {"input": 3, "output": 5, "totalTokens": 8}, "gpt-4.1-mini"
        if prompt == "Add tests":
            return "Tests added", None, {"input": 1, "output": 2, "totalTokens": 3}, "gpt-4.1-mini"
        if prompt == "Polish":
            return "Polished", "checked", {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "gpt-4.1-mini"
        raise AssertionError(f"unexpected prompt: {prompt}")

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests", "Polish"])
    with patch.object(runner, "_request_chat_turn", side_effect=fake_turn):
        assert runner.run_all(max_concurrency=1, prompt_inputs=[prompt_input], resume=True) == [destination]

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["response"] == "Built"
    row = rows[1]
    assert row["prompt"] == "Build app"
    assert row["follow_up_prompts"] == ["Add tests", "Polish"]
    assert row["responses"] == ["Rebuilt", "Tests added", "Polished"]
    assert row["response"] == "Polished"
    assert row["thinking"] == "replanned\n\nchecked"
    assert row["usage"] == {"input": 6, "output": 10, "reasoning": 0, "totalTokens": 16}
    assert [message["role"] for message in row["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [call[0] for call in calls] == ["Build app", "Add tests", "Polish"]
    assert calls[0][1] == []
    assert calls[1][1][-2:] == [
        {"role": "user", "content": "Build app"},
        {"role": "assistant", "content": "Rebuilt"},
    ]
    assert calls[2][1][-2:] == [
        {"role": "user", "content": "Add tests"},
        {"role": "assistant", "content": "Tests added"},
    ]


def test_chat_runner_resume_regenerates_existing_partial_follow_up_row(tmp_path: Path):
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
        if prompt == "Build app":
            return "Rebuilt", None, {"input": 1, "output": 1, "totalTokens": 2}, "gpt-4.1-mini"
        if prompt == "Add tests":
            return "Retested", None, {"input": 1, "output": 1, "totalTokens": 2}, "gpt-4.1-mini"
        if prompt == "Polish":
            return "Polished", None, {"input": 1, "output": 1, "totalTokens": 2}, "gpt-4.1-mini"
        raise AssertionError(f"unexpected prompt: {prompt}")

    prompt_input = PromptInput(prompt="Build app", follow_up_prompts=["Add tests", "Polish"])
    with patch.object(runner, "_request_chat_turn", side_effect=fake_turn):
        assert runner.run_all(max_concurrency=1, prompt_inputs=[prompt_input], resume=True) == [destination]

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["follow_up_prompts"] == ["Add tests"]
    assert rows[1]["follow_up_prompts"] == ["Add tests", "Polish"]
    assert rows[1]["responses"] == ["Rebuilt", "Retested", "Polished"]
    assert [call[0] for call in calls] == ["Build app", "Add tests", "Polish"]
    assert calls[0][1] == []
    assert calls[1][1][-2:] == [
        {"role": "user", "content": "Build app"},
        {"role": "assistant", "content": "Rebuilt"},
    ]
    assert calls[2][1][-2:] == [
        {"role": "user", "content": "Add tests"},
        {"role": "assistant", "content": "Retested"},
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


def test_chat_runner_continues_claiming_new_prompts_after_failure(tmp_path: Path):
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

    assert set(claimed) == {"prompt-1", "prompt-2", "prompt-3", "prompt-4"}


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


def test_pi_runner_uses_completions_for_builtin_openrouter_even_when_config_says_responses():
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
        "api": "openai-completions",
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


def test_pi_runner_preserves_provider_in_native_exported_trace(tmp_path: Path):
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

    assert result.read_text(encoding="utf-8") == session_file.read_text(encoding="utf-8")


def test_pi_runner_preserves_prompt_file_user_message_in_native_exported_trace(tmp_path: Path):
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
                        "id": "44931461",
                        "parentId": "02980b33",
                        "timestamp": "2026-05-22T00:28:39.346Z",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": '<file name="/workspace/.teich-prompt.txt">\nCan you build me a dependency map for a software system?\n</file>',
                                }
                            ],
                            "timestamp": 1_779_409_719_345,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-1",
                        "parentId": "44931461",
                        "timestamp": "2026-05-22T00:28:40.346Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Done."}],
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

    assert result.read_text(encoding="utf-8") == session_file.read_text(encoding="utf-8")


def test_pi_runner_copies_native_trace_without_system_prompt_injection(tmp_path: Path):
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
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-1",
                        "parentId": "user-1",
                        "timestamp": "2026-04-30T07:14:44.483Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "I am Pi."}],
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

    assert result.read_text(encoding="utf-8") == session_file.read_text(encoding="utf-8")


def test_pi_runner_appends_prompt_level_system_metadata_at_end(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    trace_file = tmp_path / "trace.jsonl"
    native_lines = [
        json.dumps({"type": "session", "id": "pi-session"}),
        json.dumps(
            {
                "type": "message",
                "id": "user-1",
                "message": {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            }
        ),
        json.dumps(
            {
                "type": "message",
                "id": "assistant-1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            }
        ),
    ]
    trace_file.write_text("\n".join(native_lines) + "\n", encoding="utf-8")

    runner._append_pi_system_prompt_metadata(
        trace_file,
        PromptInput(prompt="Hello", system="Use the prompt-level system."),
    )

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert lines[:-1] == native_lines
    metadata_event = json.loads(lines[-1])
    assert metadata_event["type"] == "custom"
    assert metadata_event["customType"] == "teich-system-prompt"
    assert metadata_event["data"] == {
        "systemPrompt": "Use the prompt-level system.",
        "source": "teich",
    }


def test_pi_runner_finalizer_appends_available_tools_metadata(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "user-1",
                        "message": {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "assistant-1",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run shell commands.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }

    with patch("teich.runner.snapshot_configured_tools", return_value=[tool]):
        runner._finalize_trace_export(trace_file)

    rows = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    metadata_event = rows[-1]
    assert metadata_event["type"] == "custom"
    assert metadata_event["customType"] == "teich-available-tools"
    assert metadata_event["data"]["source"] == "teich"
    assert metadata_event["data"]["tools"][0]["function"]["name"] == "bash"


def test_pi_runner_does_not_duplicate_prompt_level_system_metadata(tmp_path: Path):
    config = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    with patch.object(PiRunner, '_ensure_image'):
        runner = PiRunner(config)

    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-session"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "system-1",
                        "message": {
                            "role": "developer",
                            "content": [{"type": "text", "text": "Use the prompt-level system."}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "user-1",
                        "message": {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = trace_file.read_text(encoding="utf-8")

    runner._append_pi_system_prompt_metadata(
        trace_file,
        PromptInput(prompt="Hello", system="Use the prompt-level system."),
    )

    assert trace_file.read_text(encoding="utf-8") == before


def test_pi_runner_preserves_malformed_tool_call_trace(tmp_path: Path):
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

    output_path = runner._extract_session_file(
        "session-id",
        session_dir,
        datetime.fromtimestamp(0, tz=timezone.utc),
    )

    assert output_path.exists()
    exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exported[1]["message"]["content"][1]["type"] == "toolCall"
    assert exported[1]["message"]["content"][1]["id"] == ""
    assert exported[2]["message"]["role"] == "toolResult"
    assert "Validation failed for tool" in exported[2]["message"]["content"][0]["text"]
