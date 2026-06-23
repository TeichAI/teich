"""Tests for the shared Langfuse config + Claude Code tracing wiring.

Hermetic: no Docker, network, or real Langfuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from teich.config import Config, LangfuseConfig, ModelConfig
from teich.runner import ClaudeCodeRunner


# -- shared config -----------------------------------------------------------

def test_langfuse_disabled_by_default():
    assert Config().agent.langfuse.enabled is False


def test_langfuse_ok_with_all_credentials():
    cfg = LangfuseConfig(enabled=True, public_key="pk", secret_key="sk", base_url="https://x")
    assert cfg.enabled and cfg.public_key == "pk"


@pytest.mark.parametrize("missing", ["public_key", "secret_key", "base_url"])
def test_langfuse_requires_each_credential(missing: str):
    kwargs = {"public_key": "pk", "secret_key": "sk", "base_url": "https://x"}
    del kwargs[missing]
    with pytest.raises(ValueError, match=missing):
        LangfuseConfig(enabled=True, **kwargs)


@pytest.mark.parametrize("blank", ["public_key", "secret_key", "base_url"])
def test_langfuse_rejects_blank_credential(blank: str):
    kwargs = {"public_key": "pk", "secret_key": "sk", "base_url": "https://x", blank: "   "}
    with pytest.raises(ValueError, match=blank):
        LangfuseConfig(enabled=True, **kwargs)


# -- Claude Code env items ---------------------------------------------------

def _claude_langfuse_config(base_url: str = "https://langfuse.example.com") -> Config:
    return Config(
        model=ModelConfig(model="claude-sonnet-4-6"),
        agent={
            "provider": "claude-code",
            "langfuse": {
                "enabled": True,
                "public_key": "pk-lf-1",
                "secret_key": "sk-lf-2",
                "base_url": base_url,
            },
        },
    )


def test_claude_langfuse_uses_tracing_runtime_image():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_claude_langfuse_config())
    assert runner.image_name == "teich-runtime:v3-langfuse"
    assert runner._runtime_build_args() == ["--build-arg", "TEICH_INSTALL_LANGFUSE=1"]


def test_claude_without_langfuse_uses_standard_runtime_image():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(Config(model=ModelConfig(model="claude-sonnet-4-6")))
    assert runner.image_name == "teich-runtime:v3"
    assert runner._runtime_build_args() == []


def test_claude_langfuse_env_items_when_enabled():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_claude_langfuse_config())
    items = dict(runner._langfuse_env_items())
    assert items["TRACE_TO_LANGFUSE"] == "true"
    assert items["LANGFUSE_PUBLIC_KEY"] == "pk-lf-1"
    assert items["LANGFUSE_SECRET_KEY"] == "sk-lf-2"
    assert items["LANGFUSE_BASE_URL"] == "https://langfuse.example.com"


def test_claude_langfuse_env_items_empty_when_disabled():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(Config(model=ModelConfig(model="claude-sonnet-4-6")))
    assert runner._langfuse_env_items() == []


# -- Claude settings.json hook -----------------------------------------------

def test_claude_prepare_home_writes_stop_hook(tmp_path: Path):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_claude_langfuse_config())
    home = tmp_path / "home"
    home.mkdir()
    runner._prepare_agent_home(home)
    settings = json.loads((home / "settings.json").read_text())
    for event in ("Stop", "SessionEnd"):
        cmd = settings["hooks"][event][0]["hooks"][0]["command"]
        # Must use the venv python by absolute path (claude sanitizes PATH for hooks).
        assert cmd.startswith("/opt/venv/bin/python3 ")
        assert cmd.endswith("langfuse_hook.py")


def test_claude_prepare_home_noop_when_disabled(tmp_path: Path):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(Config(model=ModelConfig(model="claude-sonnet-4-6")))
    home = tmp_path / "home"
    home.mkdir()
    runner._prepare_agent_home(home)
    assert not (home / "settings.json").exists()


# -- host-local base_url rewriting -------------------------------------------

@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_claude_langfuse_base_url_rewrites_host_local(host: str):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_claude_langfuse_config(base_url=f"http://{host}:3000"))
    items = dict(runner._langfuse_env_items())
    assert items["LANGFUSE_BASE_URL"] == "http://host.docker.internal:3000"
    assert runner._langfuse_is_host_local() is True


def test_cloud_base_url_is_not_rewritten():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_claude_langfuse_config())
    items = dict(runner._langfuse_env_items())
    assert items["LANGFUSE_BASE_URL"] == "https://langfuse.example.com"
    assert runner._langfuse_is_host_local() is False
