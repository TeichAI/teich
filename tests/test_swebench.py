"""Unit tests for the swe-bench bench backend (the Docker-free surface).

The agent-run + grading seams need real Docker + the swebench package and are covered by an
opt-in integration pass; here we test dataset loading, the Dockerfile render, the per-provider
agent recipes, auth env, reward mapping, registration, and the install-hint guard.
"""

from __future__ import annotations

import sys
import types

import pytest

from teich.bench.backends import get_backend
from teich.bench.backends.swebench import (
    SweBenchBackend,
    _agent_layer,
    _auth_env,
    _load_instances,
    _model_name,
    _rewards_from_report,
    _run_command,
    render_agent_dockerfile,
)
from teich.config import BenchSource, Config


def _cfg(provider="pi", api=None):
    return Config(agent={"provider": provider}, api=api or {})


# --------------------------------------------------------------------------- registration


def test_get_backend_returns_swebench():
    backend = get_backend("swe-bench")
    assert isinstance(backend, SweBenchBackend)
    assert backend.type == "swe-bench"


def test_require_hint_without_swebench():
    try:
        import swebench  # noqa: F401

        pytest.skip("swebench is installed; the missing-extra path can't be exercised")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match=r"teich\[swe\]"):
        SweBenchBackend().require()


# --------------------------------------------------------------------------- dataset


def _inject_fake_swebench(monkeypatch, loader):
    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    utils = types.ModuleType("swebench.harness.utils")
    utils.load_swebench_dataset = loader
    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.utils", utils)


def test_load_instances_passes_split_and_ids(monkeypatch):
    seen = {}

    def loader(name, split, instance_ids):
        seen.update(name=name, split=split, instance_ids=instance_ids)
        return [{"instance_id": "django__django-1", "problem_statement": "fix it"}]

    _inject_fake_swebench(monkeypatch, loader)
    source = BenchSource(
        type="swe-bench", source="SWE-bench/SWE-bench_Lite", split="dev", instances=["django__django-1"]
    )
    out = _load_instances(source)

    assert out == [{"instance_id": "django__django-1", "problem_statement": "fix it"}]
    assert seen == {"name": "SWE-bench/SWE-bench_Lite", "split": "dev", "instance_ids": ["django__django-1"]}


def test_load_instances_defaults_split_to_test(monkeypatch):
    seen = {}

    def loader(name, split, instance_ids):
        seen.update(split=split, instance_ids=instance_ids)
        return []

    _inject_fake_swebench(monkeypatch, loader)
    _load_instances(BenchSource(type="swe-bench", source="x.jsonl"))
    assert seen == {"split": "test", "instance_ids": None}


def test_load_instances_wraps_errors(monkeypatch):
    def loader(name, split, instance_ids):
        raise KeyError("nope")

    _inject_fake_swebench(monkeypatch, loader)
    with pytest.raises(RuntimeError, match="Failed to load swe-bench dataset"):
        _load_instances(BenchSource(type="swe-bench", source="bad"))


def test_tasks_maps_instances_to_bench_tasks(monkeypatch):
    import teich.bench.backends.swebench as mod

    instances = [{"instance_id": "a__b-1"}, {"instance_id": "a__b-2"}]
    monkeypatch.setattr(mod, "_load_instances", lambda source: instances)
    tasks = SweBenchBackend().tasks(_cfg(), BenchSource(type="swe-bench", source="ds"))

    assert [t.id for t in tasks] == ["a__b-1", "a__b-2"]
    assert tasks[0].raw is instances[0]


def test_tasks_rejects_langfuse_enabled():
    # swe-bench doesn't wire Langfuse into the agent container, so it fails loudly rather than
    # silently no-op. The guard fires before any dataset/Docker work, so no mocks are needed.
    cfg = Config(agent={
        "provider": "pi",
        "langfuse": {"enabled": True, "public_key": "pk", "secret_key": "sk", "base_url": "https://lf"},
    })
    with pytest.raises(RuntimeError, match="does not support Langfuse"):
        SweBenchBackend().tasks(cfg, BenchSource(type="swe-bench", source="ds"))


# --------------------------------------------------------------------------- agent layer


def test_agent_layer_for_known_providers():
    assert _agent_layer(_cfg("pi")).provider == "pi"
    assert _agent_layer(_cfg("codex")).provider == "codex"
    # claude alias maps to claude-code (needs an anthropic project, see the guard test below)
    assert _agent_layer(_cfg("claude", api={"provider": "anthropic"})).provider == "claude-code"


def test_agent_layer_rejects_unsupported_provider():
    with pytest.raises(RuntimeError, match="does not support agent provider 'chat'"):
        _agent_layer(_cfg("chat"))


def test_agent_layer_rejects_claude_code_without_anthropic_provider():
    # claude-code runs the Anthropic `claude` CLI; a non-anthropic project has no usable Claude
    # creds in the container, so fail fast rather than emit empty traces.
    with pytest.raises(RuntimeError, match="needs an anthropic-provider config"):
        _agent_layer(_cfg("claude", api={"provider": "openrouter", "api_key": "sk-or"}))
    assert _agent_layer(_cfg("claude", api={"provider": "anthropic", "api_key": "sk-ant"})).provider == "claude-code"


def test_auth_env_openrouter_sets_both_keys():
    cfg = _cfg("pi", api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-or"})
    env = _auth_env(cfg)
    assert env["OPENAI_API_KEY"] == "sk-or"
    assert env["OPENROUTER_API_KEY"] == "sk-or"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_auth_env_openai_only_sets_openai_key():
    cfg = _cfg("codex", api={"provider": "openai", "api_key": "sk-oai"})
    env = _auth_env(cfg)
    assert env["OPENAI_API_KEY"] == "sk-oai"
    assert "OPENROUTER_API_KEY" not in env


def test_auth_env_anthropic_sets_anthropic_key_only():
    # The anthropic key must not be shadowed under OPENAI_API_KEY (would break OpenAI agents).
    cfg = _cfg("claude", api={"provider": "anthropic", "api_key": "sk-ant"})
    env = _auth_env(cfg)
    assert env["ANTHROPIC_API_KEY"] == "sk-ant"
    assert "OPENAI_API_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env


# --------------------------------------------------------------------------- run command (model)


def test_run_command_pins_configured_model():
    cfg = Config(agent={"provider": "codex"}, model={"model": "gpt-5-codex"},
                 api={"provider": "openai", "api_key": "sk"})
    assert _run_command(cfg, _agent_layer(cfg)).endswith("--model gpt-5-codex")


def test_run_command_pi_prefixes_provider():
    cfg = Config(agent={"provider": "pi"}, model={"model": "z-ai/glm-5.2"},
                 api={"provider": "openrouter"})
    assert _run_command(cfg, _agent_layer(cfg)).endswith("--model openrouter/z-ai/glm-5.2")


def test_run_command_no_model_is_unchanged():
    # model.model defaults to a non-empty value, so the no-flag path needs an explicit empty model.
    cfg = Config(agent={"provider": "codex"}, model={"model": ""})
    layer = _agent_layer(cfg)
    assert _model_name(cfg) == ""
    assert _run_command(cfg, layer) == layer.run


def test_ensure_instance_image_pull_bounds_timeout(monkeypatch):
    # The remote-image pull must be bounded by the run timeout, not block forever.
    import teich.bench.backends.swebench as mod

    calls = []

    class _CP:
        stdout = ""  # empty -> the image isn't present locally -> take the pull path

    def fake_docker(args, *, timeout=None, check=True):
        calls.append((args, timeout))
        return _CP()

    monkeypatch.setattr(mod, "_docker", fake_docker)
    spec = types.SimpleNamespace(instance_image_key="swebench/x:latest")
    mod._ensure_instance_image(spec, namespace="swebench", timeout=42)
    pull_timeout = next(t for a, t in calls if a[0] == "pull")
    assert pull_timeout == 42


# --------------------------------------------------------------------------- dockerfile render


def test_render_dockerfile_from_instance_image_with_agent():
    df = render_agent_dockerfile("swebench/sweb.eval.x86_64.django__django-1:latest", _agent_layer(_cfg("pi")))
    lines = df.splitlines()
    # The buildkit syntax directive must be the very first line, FROM the next directive.
    assert lines[0] == "# syntax=docker/dockerfile:1"
    directives = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    assert directives[0] == "FROM swebench/sweb.eval.x86_64.django__django-1:latest"
    assert "npm install -g @mariozechner/pi-coding-agent" in df
    assert "WORKDIR /testbed" in df
    assert "deb.nodesource.com" in df  # node runtime layer
    assert "Langfuse" not in df  # off by default


def test_render_dockerfile_includes_langfuse_block_when_enabled():
    layer = _agent_layer(_cfg("codex"))
    df = render_agent_dockerfile("base:img", layer, langfuse=True, langfuse_install="RUN echo lf")
    assert "Langfuse" in df
    assert "RUN echo lf" in df


# --------------------------------------------------------------------------- rewards


def test_rewards_resolved():
    rewards = _rewards_from_report(
        {
            "resolved": True,
            "patch_successfully_applied": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["t1", "t2"], "failure": []},
                "PASS_TO_PASS": {"success": ["t3"], "failure": ["t4"]},
            },
        }
    )
    assert rewards["reward"] == 1.0
    assert rewards["resolved"] == 1.0
    assert rewards["patch_applied"] == 1.0
    assert rewards["fail_to_pass"] == 1.0
    assert rewards["pass_to_pass"] == 0.5


def test_rewards_unresolved_routes_to_failed():
    from teich.bench.backends import base

    rewards = _rewards_from_report({"resolved": False, "patch_successfully_applied": False})
    assert rewards["reward"] == 0.0
    assert base.route_split(base.primary_score(rewards)) == "failed"


def test_rewards_resolved_routes_to_passed():
    from teich.bench.backends import base

    rewards = _rewards_from_report({"resolved": True, "patch_successfully_applied": True})
    assert base.route_split(base.primary_score(rewards)) == "passed"
