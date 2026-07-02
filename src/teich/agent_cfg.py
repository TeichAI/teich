"""Shared, cfg-only agent-configuration helpers.

Prompt mode (``runner.py``) and bench mode (``bench/backends/*``) both need to turn a ``Config``
into the credentials / model name / base URL an agent CLI consumes. This module is the single
source of truth for the *pure* parts (no ``self``, no Docker), so bench stops re-deriving a
thinner, diverging copy (which previously only knew openai/openrouter/anthropic and dropped the
localhost -> host.docker.internal rewrite).

Note: agent-specific *config files* (codex ``config.toml``, pi ``models.json``/``settings.json``)
are threaded separately by each backend; this module owns the shared env/model/url derivation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from .config import Config

# api.provider -> the ENV var its CLI reads for the key. Mirrors runner.ExternalCliRunner
# (the full map, incl. zai->GLM_API_KEY, deepseek/xai/google/...); unknown providers fall back
# to ``<PROVIDER>_API_KEY``.
_PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "claude_code": "ANTHROPIC_API_KEY",
    "claude-code": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nous": "NOUS_API_KEY",
    "nous_portal": "NOUS_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "grok": "XAI_API_KEY",
    "zai": "GLM_API_KEY",
    "z_ai": "GLM_API_KEY",
    "glm": "GLM_API_KEY",
}

# api.wire_api values that mean "chat completions" (vs the OpenAI Responses API default).
CHAT_WIRE_APIS = {"completions", "chat_completions", "chat-completions", "openai-completions"}


def provider_env_key(provider: str) -> str:
    """The ENV var name an agent reads for ``provider``'s API key."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", provider.strip().lower())
    return _PROVIDER_ENV_KEYS.get(normalized, f"{normalized.upper() or 'TEICH'}_API_KEY")


def container_base_url(base_url: str | None) -> str | None:
    """Rewrite a host-local base URL so a container can reach it (localhost -> host.docker.internal)."""
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    if hostname not in {"localhost", "127.0.0.1"}:
        return base_url
    netloc = parsed.netloc.replace(hostname, "host.docker.internal", 1)
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def pi_prefixed_model(cfg: Config) -> str:
    """Model for a bench agent; pi picks its provider from a ``<provider>/<model>`` prefix.

    ``model: z-ai/glm-5.2`` + ``api.provider: openrouter`` -> ``openrouter/z-ai/glm-5.2``. Only the
    pi agent needs the prefix; other agents get the model unchanged.
    """
    model = cfg.get_effective_model().strip()
    if not model:
        return model
    provider = (cfg.api.provider or "").strip()
    if cfg.get_agent_provider() == "pi" and provider and not model.startswith(f"{provider}/"):
        return f"{provider}/{model}"
    return model


def bench_auth_env(cfg: Config) -> dict[str, str]:
    """Credential + base-URL env for a bench agent container.

    Sets the provider-specific key from the full provider map (so ``provider: zai`` -> ``GLM_API_KEY``,
    ``deepseek`` -> ``DEEPSEEK_API_KEY``, ...) plus the OpenAI/OpenRouter/Anthropic compat vars the
    CLIs actually read, and routes ``base_url`` through :func:`container_base_url` so a host-local
    endpoint is reachable from inside the container.
    """
    env: dict[str, str] = {}
    provider = cfg.api.provider.strip().lower()
    api_key = cfg.get_api_key()
    if api_key:
        env["TEICH_API_KEY"] = api_key
        env[provider_env_key(provider)] = api_key  # the correct var for zai/deepseek/xai/...
        if provider == "anthropic":
            env["ANTHROPIC_API_KEY"] = api_key
        else:
            # codex/claude-code read OPENAI_API_KEY against the configured base_url; pi/hermes on
            # OpenRouter also read OPENROUTER_API_KEY.
            env["OPENAI_API_KEY"] = api_key
            if provider == "openrouter":
                env["OPENROUTER_API_KEY"] = api_key
    base_url = container_base_url(cfg.get_base_url())
    if base_url:
        env["TEICH_BASE_URL"] = base_url
        if provider == "anthropic":
            env["ANTHROPIC_BASE_URL"] = base_url
        else:
            env["OPENAI_BASE_URL"] = base_url
    return env
