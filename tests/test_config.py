"""Tests for config module."""

import json
from pathlib import Path

import pytest

from teich.config import Config, MCPConfig, ModelConfig


def test_default_config():
    """Test default configuration values."""
    config = Config()
    assert config.model.model == "codex-mini-latest"
    assert config.model.approval_mode == "none"
    assert config.model.pi_model_overrides == {"maxTokens": 131072}
    assert config.output.traces_dir == Path("./output")
    assert config.output.sandbox_dir == Path("./sandbox")
    assert config.output.failures_dir == Path("./failures")
    assert config.max_concurrency == 1
    assert config.timeout_seconds == 600
    assert config.mcp_servers == []
    assert config.prompts == []
    assert config.get_dataset_tags() == [
        "agent-traces",
        "format:agent-traces",
        "codex",
        "distillation",
        "codex-mini-latest",
        "teich",
    ]


def test_config_from_yaml(tmp_path: Path, monkeypatch):
    """Test loading config from YAML."""
    # Clear env vars that would override YAML values
    monkeypatch.delenv("TEICH_MODEL", raising=False)
    monkeypatch.delenv("TEICH_BASE_URL", raising=False)
    monkeypatch.delenv("TEICH_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
  failures_dir: ./failed-traces
  pretty_name: "Test Traces"

publish:
  repo_id: armand0e/test-dataset
  hf_token: hf-test123
  private: true

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
    assert config.output.failures_dir == Path("./failed-traces")
    assert config.output.pretty_name == "Test Traces"
    assert config.publish.repo_id == "armand0e/test-dataset"
    assert config.publish.hf_token == "hf-test123"
    assert config.publish.private is True
    assert config.max_concurrency == 3
    assert config.timeout_seconds == 300
    assert config.openai_api_key == "sk-test123"
    assert config.prompts == ["Build a todo app", "Create a python script"]


def test_openrouter_api_key_env_alias(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEICH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
api:
  provider: openrouter
prompts:
  - Hello
""")

    config = Config.from_yaml(config_file)

    assert config.api.api_key == "sk-or-v1-test"
    assert Config(api={"provider": "openrouter"}).get_api_key() == "sk-or-v1-test"


def test_placeholder_api_keys_are_treated_as_absent(monkeypatch):
    monkeypatch.delenv("TEICH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert Config(api={"api_key": "none"}).get_api_key() is None
    assert Config(api={"api_key": " null "}).get_api_key() is None
    assert Config(openai_api_key="local").get_api_key() is None


def test_openrouter_api_key_env_alias_does_not_override_explicit_config(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEICH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
api:
  provider: openrouter
  api_key: sk-or-v1-config
prompts:
  - Hello
""")

    config = Config.from_yaml(config_file)

    assert config.api.api_key == "sk-or-v1-config"


def test_teich_api_key_env_still_overrides_explicit_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEICH_API_KEY", "sk-teich-env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
api:
  provider: openrouter
  api_key: sk-or-v1-config
prompts:
  - Hello
""")

    config = Config.from_yaml(config_file)

    assert config.api.api_key == "sk-teich-env"


def test_config_generates_chat_dataset_tags():
    config = Config(agent={"provider": "chat"}, model=ModelConfig(model="gpt-4.1-mini"))

    assert config.get_dataset_tags() == ["conversational", "distillation", "teich", "gpt-4.1-mini"]


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


def test_config_prompts_file_supports_multiline_csv_prompts(tmp_path: Path):
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        '"image","github_repo","prompt"\n'
        '"None","None","Premise:\n'
        '""Apparently Thorn thought the same thing.""\n'
        'Available choices:\n'
        ' - yes\n'
        ' - no"\n'
        '"None","None","Solve 1148583*a = 1148360*a - 5352 for a.\n'
        'Solve this problem."\n',
        encoding="utf-8",
    )

    config = Config(prompts_file=prompts_file)
    prompts = config.get_prompts()

    assert len(prompts) == 2
    assert prompts[0] == (
        'Premise:\n'
        '"Apparently Thorn thought the same thing."\n'
        'Available choices:\n'
        ' - yes\n'
        ' - no'
    )
    assert prompts[1] == "Solve 1148583*a = 1148360*a - 5352 for a.\nSolve this problem."


def test_config_prompts_file_rejects_csv_rows_with_extra_columns(tmp_path: Path):
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text(
        "prompt,github_repo\n"
        "Build a dashboard,with an unquoted comma,None\n",
        encoding="utf-8",
    )
    config = Config(prompts_file=prompts_file)

    with pytest.raises(ValueError, match="more columns than the header"):
        config.get_prompt_inputs()


def test_config_prompts_file_supports_jsonl_prompts(tmp_path: Path):
    prompts_file = tmp_path / "prompts.jsonl"
    rows = [
        {
            "system": "Answer as a senior frontend reviewer.",
            "prompt": "Premise:\nUse a safer long prompt format.\nAvailable choices:\n - yes\n - no",
            "github_repo": "armand0e/perplexica-mcp",
            "follow_up_prompts": ["Now add tests.", "Now update the README."],
        },
        {"prompt": "Build a todo app"},
    ]
    prompts_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    config = Config(prompts_file=prompts_file)
    prompt_inputs = config.get_prompt_inputs()

    assert len(prompt_inputs) == 2
    assert prompt_inputs[0].prompt == rows[0]["prompt"]
    assert prompt_inputs[0].system == "Answer as a senior frontend reviewer."
    assert prompt_inputs[0].github_repo == "armand0e/perplexica-mcp"
    assert prompt_inputs[0].follow_up_prompts == ["Now add tests.", "Now update the README."]
    assert prompt_inputs[1].prompt == "Build a todo app"


def test_config_inline_prompts_support_structured_follow_up_prompts():
    config = Config(
        prompts=[
            {
                "system": "Keep edits minimal.",
                "prompt": "Build a todo app",
                "follow_up_prompts": ["Add keyboard shortcuts", "Polish the empty state"],
            }
        ]
    )

    prompt_inputs = config.get_prompt_inputs()

    assert len(prompt_inputs) == 1
    assert prompt_inputs[0].prompt == "Build a todo app"
    assert prompt_inputs[0].system == "Keep edits minimal."
    assert prompt_inputs[0].follow_up_prompts == ["Add keyboard shortcuts", "Polish the empty state"]


def test_config_rejects_non_list_follow_up_prompts(tmp_path: Path):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        json.dumps({"prompt": "Build a todo app", "follow_up_prompts": "Add tests"}) + "\n",
        encoding="utf-8",
    )
    config = Config(prompts_file=prompts_file)

    with pytest.raises(ValueError, match="follow_up_prompts must be a list"):
        config.get_prompt_inputs()


def test_config_prompts_file_supports_json_prompt_list(tmp_path: Path):
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(
        json.dumps(
            {
                "prompts": [
                    {"prompt": "Review this code:\n```python\nprint('hello')\n```"},
                    "Create a landing page",
                ]
            }
        ),
        encoding="utf-8",
    )

    config = Config(prompts_file=prompts_file)

    assert config.get_prompts() == [
        "Review this code:\n```python\nprint('hello')\n```",
        "Create a landing page",
    ]


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


def test_model_service_tier_defaults_to_none():
    """Fast mode is opt-in: service_tier is unset by default."""
    assert Config().model.service_tier is None


def test_model_service_tier_is_free_string_passthrough():
    """service_tier accepts arbitrary tiers (fast/flex/priority) without an enum."""
    assert ModelConfig(service_tier="fast").service_tier == "fast"
    assert ModelConfig(service_tier="flex").service_tier == "flex"


def test_model_reasoning_summary_defaults_to_none():
    """Reasoning summaries use Codex's default unless explicitly set."""
    assert Config().model.reasoning_summary is None


def test_model_reasoning_summary_is_free_string_passthrough():
    """reasoning_summary accepts auto/concise/detailed/none without an enum."""
    assert ModelConfig(reasoning_summary="detailed").reasoning_summary == "detailed"
    assert ModelConfig(reasoning_summary="concise").reasoning_summary == "concise"


def test_codex_auth_config_defaults():
    """Codex host-auth is off by default with a project-local auth dir."""
    codex = Config().agent.codex
    assert codex.use_host_auth is False
    assert codex.host_auth_file is None
    assert codex.auth_dir == Path("./.teich/codex-auth")


def test_codex_auth_config_from_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEICH_MODEL", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
agent:
  provider: codex
  codex:
    use_host_auth: true
    host_auth_file: ~/.codex/auth.json
    auth_dir: ./.teich/codex-auth
model:
  model: gpt-5.5
  service_tier: fast
"""
    )
    config = Config.from_yaml(config_file)
    assert config.agent.codex.use_host_auth is True
    assert config.agent.codex.host_auth_file == Path("~/.codex/auth.json")
    assert config.agent.codex.auth_dir == Path("./.teich/codex-auth")
    assert config.model.service_tier == "fast"


def test_codex_auth_dir_under_output_rejected_when_enabled():
    """The auth snapshot must never live under uploaded output dirs."""
    with pytest.raises(ValueError, match="auth_dir"):
        Config(
            agent={"provider": "codex", "codex": {"use_host_auth": True, "auth_dir": "./output/secrets"}},
            output={"traces_dir": "./output"},
        )


def test_codex_auth_dir_under_output_allowed_when_disabled():
    """When host-auth is off the snapshot dir is never used, so don't over-validate."""
    config = Config(
        agent={"provider": "codex", "codex": {"use_host_auth": False, "auth_dir": "./output/secrets"}},
        output={"traces_dir": "./output"},
    )
    assert config.agent.codex.auth_dir == Path("./output/secrets")


def test_codex_host_auth_source_prefers_explicit_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    override = tmp_path / "creds" / "auth.json"
    config = Config(agent={"provider": "codex", "codex": {"host_auth_file": str(override)}})
    assert config.get_codex_host_auth_source() == override


def test_codex_host_auth_source_uses_codex_home_env(monkeypatch):
    monkeypatch.setenv("CODEX_HOME", "/custom/codex")
    config = Config(agent={"provider": "codex"})
    assert config.get_codex_host_auth_source() == Path("/custom/codex/auth.json")


def test_codex_host_auth_source_defaults_to_home(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    config = Config(agent={"provider": "codex"})
    assert config.get_codex_host_auth_source() == Path.home() / ".codex" / "auth.json"


def test_claude_config_defaults():
    """No token, no passthroughs: everything under agent.claude is opt-in."""
    claude = Config().agent.claude
    assert claude.oauth_token is None
    assert claude.fallback_model is None
    assert claude.always_thinking is None
    assert claude.max_thinking_tokens is None


def test_claude_oauth_token_prefers_explicit_config(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    config = Config(agent={"claude": {"oauth_token": "config-token"}})
    assert config.get_claude_oauth_token() == "config-token"


def test_claude_oauth_token_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    assert Config().get_claude_oauth_token() == "env-token"


def test_claude_oauth_token_teich_env_wins_over_claude_env(monkeypatch):
    monkeypatch.setenv("TEICH_CLAUDE_OAUTH_TOKEN", "teich-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-token")
    assert Config().get_claude_oauth_token() == "teich-token"


def test_claude_oauth_token_absent_when_unset(monkeypatch):
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert Config().get_claude_oauth_token() is None


def test_claude_oauth_token_placeholder_is_treated_as_absent(monkeypatch):
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    config = Config(
        agent={"provider": "claude-code", "claude": {"oauth_token": " dummy "}},
        api={"provider": "anthropic", "api_key": "sk-ant-test"},
    )

    assert config.get_claude_oauth_token() is None
    assert config.get_claude_oauth_token_source() is None
    assert config.claude_host_auth_active() is False
    assert config.get_api_key() == "sk-ant-test"


def test_claude_oauth_token_placeholder_allows_base_url(monkeypatch):
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    config = Config(
        agent={"provider": "claude-code", "claude": {"oauth_token": "none"}},
        api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
    )

    assert config.get_claude_oauth_token() is None


def test_claude_oauth_token_source_names_the_resolving_source(monkeypatch):
    """The CLI notice reports the source; it must track the getter's resolution order."""
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert Config().get_claude_oauth_token_source() is None
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    assert Config().get_claude_oauth_token_source() == "CLAUDE_CODE_OAUTH_TOKEN"
    monkeypatch.setenv("TEICH_CLAUDE_OAUTH_TOKEN", "teich-token")
    assert Config().get_claude_oauth_token_source() == "TEICH_CLAUDE_OAUTH_TOKEN"
    config = Config(agent={"claude": {"oauth_token": "config-token"}})
    assert config.get_claude_oauth_token_source() == "agent.claude.oauth_token"


def test_claude_host_auth_active_with_token(monkeypatch):
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    assert Config(agent={"provider": "claude-code"}).claude_host_auth_active() is True


def test_claude_host_auth_inactive_without_token(monkeypatch):
    monkeypatch.delenv("TEICH_CLAUDE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert Config(agent={"provider": "claude-code"}).claude_host_auth_active() is False


def test_claude_host_auth_inactive_for_other_providers(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    assert Config(agent={"provider": "codex"}).claude_host_auth_active() is False


def test_claude_ambient_token_yields_to_base_url(monkeypatch):
    """An ambient env token must not override an explicit base_url config (no error either)."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    config = Config(
        agent={"provider": "claude-code"},
        api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
    )
    assert config.claude_host_auth_active() is False


def test_claude_configured_token_rejects_base_url():
    """An explicit config token + base_url is a contradiction, not a silent fallback."""
    with pytest.raises(ValueError, match="oauth_token"):
        Config(
            agent={"provider": "claude-code", "claude": {"oauth_token": "config-token"}},
            api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
        )


def test_claude_configured_token_base_url_allowed_for_other_providers():
    """A leftover agent.claude block must not break codex + base_url configs."""
    config = Config(
        agent={"provider": "codex", "claude": {"oauth_token": "config-token"}},
        api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
    )
    assert config.agent.claude.oauth_token == "config-token"


def test_claude_fallback_model_defaults_to_none():
    assert Config().agent.claude.fallback_model is None
    assert Config().get_claude_fallback_model() is None


def test_claude_fallback_model_accepts_string_and_list():
    def cfg(fallback: object) -> Config:
        return Config(agent={"claude": {"fallback_model": fallback}})

    assert cfg("sonnet").get_claude_fallback_model() == "sonnet"
    assert cfg("sonnet, haiku").get_claude_fallback_model() == "sonnet,haiku"
    assert cfg(["sonnet", "haiku"]).get_claude_fallback_model() == "sonnet,haiku"


def test_claude_fallback_model_blank_entries_are_dropped():
    assert Config(agent={"claude": {"fallback_model": [" ", ""]}}).get_claude_fallback_model() is None
    assert Config(agent={"claude": {"fallback_model": ",,"}}).get_claude_fallback_model() is None


def test_claude_max_thinking_tokens_rejects_negative():
    with pytest.raises(ValueError):
        Config(agent={"claude": {"max_thinking_tokens": -1}})


def test_claude_settings_from_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEICH_MODEL", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
agent:
  provider: claude-code
  claude:
    oauth_token: sk-ant-oat01-test
    fallback_model: [sonnet, haiku]
    always_thinking: true
    max_thinking_tokens: 31999
model:
  model: claude-opus-4-8
  reasoning_effort: xhigh
"""
    )
    config = Config.from_yaml(config_file)
    assert config.claude_host_auth_active() is True
    assert config.model.reasoning_effort == "xhigh"
    assert config.get_claude_fallback_model() == "sonnet,haiku"
    assert config.agent.claude.always_thinking is True
    assert config.agent.claude.max_thinking_tokens == 31999


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
