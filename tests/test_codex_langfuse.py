"""Tests for Codex -> Langfuse tracing wiring.

These tests are hermetic: they never touch Docker, the network, or a real
Langfuse instance. They cover the config.toml blocks Teich writes to enable the
plugin, the Langfuse env vars passed to the container, the exec hook-trust flag,
and the per-session install of the (image-baked) plugin tree into CODEX_HOME.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from teich.config import Config, ModelConfig
from teich.runner import CodexRunner


def _langfuse_config(**overrides) -> Config:
    langfuse = {
        "enabled": True,
        "public_key": "pk-lf-123",
        "secret_key": "sk-lf-456",
        "base_url": "https://langfuse.example.com",
    }
    langfuse.update(overrides)
    return Config(
        model=ModelConfig(model="gpt-5.5"),
        agent={"provider": "codex", "langfuse": langfuse},
    )


# -- runtime image selection -------------------------------------------------

def test_codex_langfuse_uses_tracing_runtime_image():
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    assert runner.image_name == "teich-runtime:v3-langfuse"
    assert runner._runtime_build_args() == ["--build-arg", "TEICH_INSTALL_LANGFUSE=1"]


def test_codex_without_langfuse_uses_standard_runtime_image():
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    assert runner.image_name == "teich-runtime:v3"
    assert runner._runtime_build_args() == []


# -- config.toml blocks ------------------------------------------------------

def test_codex_config_writes_langfuse_blocks_when_enabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "[features]" in content
    assert "plugin_hooks = true" in content
    assert '[plugins."tracing@codex-observability-plugin"]' in content
    assert "enabled = true" in content


def test_codex_config_omits_langfuse_blocks_when_disabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "plugin_hooks" not in content
    assert "tracing@codex-observability-plugin" not in content


# -- container env vars ------------------------------------------------------

def test_codex_command_passes_langfuse_env_when_enabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "TRACE_TO_LANGFUSE=true" in cmd
    assert "LANGFUSE_PUBLIC_KEY=pk-lf-123" in cmd
    assert "LANGFUSE_SECRET_KEY=sk-lf-456" in cmd
    assert "LANGFUSE_BASE_URL=https://langfuse.example.com" in cmd


def test_codex_command_omits_langfuse_env_when_disabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert not any(part.startswith("TRACE_TO_LANGFUSE=") for part in cmd)
    assert not any(part.startswith("LANGFUSE_") for part in cmd)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_codex_command_rewrites_host_local_langfuse_base_url(tmp_path: Path, host: str):
    cfg = _langfuse_config(base_url=f"http://{host}:3000")
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(cfg)
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "LANGFUSE_BASE_URL=http://host.docker.internal:3000" in cmd
    assert "host.docker.internal:host-gateway" in cmd


def test_codex_command_adds_host_gateway_for_host_local_langfuse(tmp_path: Path):
    cfg = _langfuse_config(base_url="http://host.docker.internal:3000")
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(cfg)
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "host.docker.internal:host-gateway" in cmd


# -- hook-trust bypass (exec) ------------------------------------------------

def test_codex_agent_command_bypasses_hook_trust_when_enabled():
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    cmd = runner._build_codex_agent_command()
    assert "exec" in cmd
    assert "--dangerously-bypass-hook-trust" in cmd


def test_codex_agent_command_no_hook_trust_bypass_when_disabled():
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    cmd = runner._build_codex_agent_command()
    assert "--dangerously-bypass-hook-trust" not in cmd


# -- per-session plugin install ---------------------------------------------

def test_install_codex_langfuse_plugin_copies_tree(tmp_path: Path):
    # Fake the image-baked cache so the test never touches Docker.
    cache = tmp_path / "cache"
    leaf = cache / "plugins" / "cache" / "codex-observability-plugin" / "tracing" / "0.1.0" / "dist"
    leaf.mkdir(parents=True)
    (leaf / "index.mjs").write_text("// bundle", encoding="utf-8")

    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()

    with patch.object(runner, "_ensure_langfuse_plugin_cache", return_value=cache):
        runner._install_codex_langfuse_plugin(codex_home)

    installed = (
        codex_home / "plugins" / "cache" / "codex-observability-plugin"
        / "tracing" / "0.1.0" / "dist" / "index.mjs"
    )
    assert installed.exists()
    assert installed.read_text(encoding="utf-8") == "// bundle"


def test_install_codex_langfuse_plugin_noop_when_disabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    # Disabled -> cache is None -> nothing copied, no error.
    runner._install_codex_langfuse_plugin(codex_home)
    assert not (codex_home / "plugins").exists()
