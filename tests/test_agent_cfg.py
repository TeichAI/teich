"""Tests for the shared agent-config helpers (used by both prompt mode and bench)."""

from teich import agent_cfg
from teich.config import Config


def test_provider_env_key_covers_all_providers():
    # The full provider map — previously bench only knew openai/openrouter/anthropic.
    assert agent_cfg.provider_env_key("zai") == "GLM_API_KEY"
    assert agent_cfg.provider_env_key("z-ai") == "GLM_API_KEY"
    assert agent_cfg.provider_env_key("deepseek") == "DEEPSEEK_API_KEY"
    assert agent_cfg.provider_env_key("xai") == "XAI_API_KEY"
    assert agent_cfg.provider_env_key("grok") == "XAI_API_KEY"
    assert agent_cfg.provider_env_key("google") == "GOOGLE_API_KEY"
    assert agent_cfg.provider_env_key("openrouter") == "OPENROUTER_API_KEY"
    assert agent_cfg.provider_env_key("anthropic") == "ANTHROPIC_API_KEY"
    assert agent_cfg.provider_env_key("mystery") == "MYSTERY_API_KEY"  # fallback


def test_container_base_url_rewrites_host_local():
    assert agent_cfg.container_base_url("http://localhost:8080/v1") == "http://host.docker.internal:8080/v1"
    assert agent_cfg.container_base_url("http://127.0.0.1:1234") == "http://host.docker.internal:1234"
    assert agent_cfg.container_base_url("https://api.z.ai/api/paas/v4") == "https://api.z.ai/api/paas/v4"
    assert agent_cfg.container_base_url(None) is None


def test_bench_auth_env_sets_provider_specific_key_and_rewrites_localhost(monkeypatch):
    for var in ("TEICH_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = Config(
        agent={"provider": "pi"},
        api={"provider": "zai", "base_url": "http://localhost:9000/v1", "api_key": "glm-key"},
    )
    env = agent_cfg.bench_auth_env(cfg)
    assert env["GLM_API_KEY"] == "glm-key"  # the correct var for zai (was dropped before)
    assert env["TEICH_API_KEY"] == "glm-key"
    assert env["OPENAI_API_KEY"] == "glm-key"  # compat var for OpenAI-family CLIs
    assert env["OPENAI_BASE_URL"] == "http://host.docker.internal:9000/v1"  # host-local rewritten


def test_pi_prefixed_model_only_prefixes_pi():
    pi = Config(agent={"provider": "pi"}, api={"provider": "openrouter"}, model={"model": "z-ai/glm-5.2"})
    assert agent_cfg.pi_prefixed_model(pi) == "openrouter/z-ai/glm-5.2"
    codex = Config(agent={"provider": "codex"}, api={"provider": "openai"}, model={"model": "gpt-5"})
    assert agent_cfg.pi_prefixed_model(codex) == "gpt-5"  # non-pi agents get the model unchanged
