"""Tests for config module."""

import tempfile
from pathlib import Path

import pytest

from teich.config import Config, MCPConfig, ModelConfig


def test_default_config():
    """Test default configuration values."""
    config = Config()
    assert config.model.model == "codex-mini-latest"
    assert config.model.approval_mode == "none"
    assert config.output.traces_dir == Path("./output")
    assert config.output.sandbox_dir == Path("./sandbox")
    assert config.max_concurrency == 1
    assert config.timeout_seconds == 600
    assert config.mcp_servers == []
    assert config.prompts == []


def test_config_from_yaml(tmp_path: Path, monkeypatch):
    """Test loading config from YAML."""
    # Clear env vars that would override YAML values
    monkeypatch.delenv("TEICH_MODEL", raising=False)
    monkeypatch.delenv("TEICH_BASE_URL", raising=False)
    monkeypatch.delenv("TEICH_API_KEY", raising=False)
    monkeypatch.delenv("TEICH_PROVIDER", raising=False)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  model: o4-mini
  approval_mode: suggest

mcp_servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

prompts:
  - Build a todo app
  - Create a python script

output:
  traces_dir: ./traces
  sandbox_dir: ./sandboxes
  pretty_name: "Test Traces"
  tags:
    - test
    - traces

max_concurrency: 3
timeout_seconds: 300
openai_api_key: sk-test123
""")

    config = Config.from_yaml(config_file)

    assert config.model.model == "o4-mini"
    assert config.model.approval_mode == "suggest"
    assert config.model.approval_policy == "on-request"
    assert len(config.mcp_servers) == 1
    assert config.mcp_servers[0].name == "filesystem"
    assert config.mcp_servers[0].command == "npx"
    assert config.output.traces_dir == Path("./traces")
    assert config.output.sandbox_dir == Path("./sandboxes")
    assert config.output.pretty_name == "Test Traces"
    assert config.output.tags == ["test", "traces"]
    assert config.max_concurrency == 3
    assert config.timeout_seconds == 300
    assert config.openai_api_key == "sk-test123"
    assert config.prompts == ["Build a todo app", "Create a python script"]


def test_config_prompts_file(tmp_path: Path):
    """Test loading structured prompts from CSV file."""
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        "image,github_repo,prompt\n"
        'None,None,"Build a todo app"\n'
        'None,armand0e/perplexica-mcp,"Improve the search flow"\n',
        encoding="utf-8",
    )

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
prompts_file: {prompts_file}
prompts:
  - Extra prompt
""")

    config = Config.from_yaml(config_file)
    prompts = config.get_prompts()
    prompt_inputs = config.get_prompt_inputs()

    assert len(prompts) == 3
    assert "Extra prompt" in prompts
    assert "Build a todo app" in prompts
    assert "Improve the search flow" in prompts
    assert prompt_inputs[2].github_repo == "armand0e/perplexica-mcp"
    assert prompt_inputs[2].image is None


def test_config_prompts_file_resolves_relative_to_yaml(tmp_path: Path):
    """Test loading prompts_file relative to the config file location."""
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        "image,github_repo,prompt\n"
        'None,None,"Build a dashboard"\n',
        encoding="utf-8",
    )

    config_dir = tmp_path / "nested"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text("prompts_file: ../prompts.csv\n", encoding="utf-8")

    config = Config.from_yaml(config_file)

    assert config.prompts_file == prompts_file.resolve()
    assert config.get_prompts() == ["Build a dashboard"]


def test_config_prompts_file_rejects_invalid_github_repo(tmp_path: Path):
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        "image,github_repo,prompt\n"
        'None,not a repo,"Build a dashboard"\n',
        encoding="utf-8",
    )
    config = Config(prompts_file=prompts_file)

    with pytest.raises(ValueError, match="github_repo must be in owner/repo format"):
        config.get_prompt_inputs()


def test_config_prompts_file_rejects_non_none_images(tmp_path: Path):
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        "image,github_repo,prompt\n"
        'diagram.png,None,"Build a dashboard"\n',
        encoding="utf-8",
    )
    config = Config(prompts_file=prompts_file)

    with pytest.raises(ValueError, match="Prompt image inputs are not supported yet"):
        config.get_prompt_inputs()


def test_config_missing_prompts_file(tmp_path: Path):
    """Test that missing prompts file raises validation error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
prompts_file: /nonexistent/file.txt
""")

    with pytest.raises(ValueError, match="Prompts file not found"):
        Config.from_yaml(config_file)


def test_config_rejects_api_model_key(tmp_path: Path):
    """Test that misplaced api.model fails with a clear error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  model: gemma-4
api:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key: sk-test
  model: deepseek/deepseek-v4-pro
""")

    with pytest.raises(ValueError, match="Unsupported config key 'api.model'"):
        Config.from_yaml(config_file)


def test_mcp_config():
    """Test MCP server configuration."""
    mcp = MCPConfig(
        name="test-server",
        command="python",
        args=["-m", "mcp_server"],
        env={"KEY": "value"}
    )
    assert mcp.name == "test-server"
    assert mcp.command == "python"
    assert mcp.args == ["-m", "mcp_server"]
    assert mcp.env == {"KEY": "value"}
