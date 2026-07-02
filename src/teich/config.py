"""Configuration management for teich."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


GITHUB_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PLACEHOLDER_API_KEYS = {"", "none", "null", "dummy", "placeholder", "example", "local"}


def _api_key_env_aliases(provider: str | None) -> list[str]:
    normalized = provider.strip().lower() if isinstance(provider, str) else ""
    aliases = ["TEICH_API_KEY"]
    if normalized == "openrouter":
        aliases.append("OPENROUTER_API_KEY")
    if normalized == "openai":
        aliases.append("OPENAI_API_KEY")
    aliases.append("OPENAI_API_KEY")
    return list(dict.fromkeys(aliases))


def _get_env_alias(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and _normalize_api_key(value):
            return _normalize_api_key(value)
    return None


def _normalize_api_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in PLACEHOLDER_API_KEYS:
        return None
    return normalized


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


class MCPConfig(BaseModel):
    """MCP server configuration."""
    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    env_vars: list[str] = Field(default_factory=list)
    cwd: str | None = None
    url: str | None = None
    bearer_token_env_var: str | None = None
    http_headers: dict[str, str] = Field(default_factory=dict)
    env_http_headers: dict[str, str] = Field(default_factory=dict)
    startup_timeout_sec: int | None = None
    tool_timeout_sec: int | None = None
    enabled: bool = True
    required: bool = False
    enabled_tools: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transport(self) -> MCPConfig:
        if not self.command and not self.url:
            raise ValueError(
                f"MCP server '{self.name}' must define either command or url"
            )
        if self.command and self.url:
            raise ValueError(
                f"MCP server '{self.name}' cannot define both command and url"
            )
        return self


class APIConfig(BaseModel):
    """API configuration for OpenAI-compatible endpoints."""
    provider: str = "openai"  # openai, openrouter, azure, etc.
    base_url: str | None = None  # e.g., https://openrouter.ai/api/v1
    api_key: str | None = None  # Override global api_key for this provider
    wire_api: str = "responses"


class LangfuseConfig(BaseModel):
    """Langfuse tracing credentials, set under ``agent.langfuse``.

    When enabled, Teich wires each agent's Langfuse integration (the Codex
    plugin, the Claude Code Stop hook) and passes these credentials into the
    container. Tracing is side-channel only: it reads transcripts, fails open,
    and does not change agent behavior or Teich's output files.
    """
    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None

    @model_validator(mode="after")
    def require_credentials_when_enabled(self) -> LangfuseConfig:
        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("public_key", self.public_key),
                    ("secret_key", self.secret_key),
                    ("base_url", self.base_url),
                )
                if not (value and value.strip())
            ]
            if missing:
                raise ValueError(
                    "langfuse requires " + ", ".join(missing) + " when enabled"
                )
        return self


class CodexAuthConfig(BaseModel):
    """Codex ChatGPT-subscription auth handling.

    When ``use_host_auth`` is enabled, Teich seeds an ``auth.json`` snapshot
    under ``auth_dir`` from the host's Codex login and runs a single in-process
    token broker that owns the rotating OAuth refresh token. Each Codex
    container gets its own seeded copy (with the refresh token replaced by a
    per-run secret) and is pointed at the broker via
    ``CODEX_REFRESH_TOKEN_URL_OVERRIDE``, so the broker is the sole caller of the
    real refresh endpoint and concurrent containers cannot invalidate one
    another. The broker reads and writes only the ``auth_dir`` copy; it never
    touches the host ``~/.codex/auth.json``.
    """
    use_host_auth: bool = False
    host_auth_file: Path | None = None
    auth_dir: Path = Field(default=Path("./.teich/codex-auth"))
    # Port for the host-side token broker. 0 = pick an ephemeral free port.
    broker_port: int = Field(default=0, ge=0, le=65535)


class AgentConfig(BaseModel):
    """Agent runtime selection."""
    provider: str = "codex"
    # Langfuse tracing, applied to every agent that supports it (Codex, Claude).
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    codex: CodexAuthConfig = Field(default_factory=CodexAuthConfig)


class ModelConfig(BaseModel):
    """Model configuration."""
    model: str = "codex-mini-latest"
    approval_policy: str = "never"
    sandbox: str = "danger-full-access"
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    service_tier: str | None = None
    context_length: int | None = None
    approval_mode: str | None = "none"
    pi_model_overrides: dict[str, object] = Field(default_factory=lambda: {"maxTokens": 131072})

    @model_validator(mode="after")
    def normalize_legacy_approval_mode(self) -> ModelConfig:
        if self.approval_mode:
            self.approval_policy = {
                "none": "never",
                "suggest": "on-request",
                "auto-edit": "on-request",
                "full-auto": "on-request",
            }.get(self.approval_mode, self.approval_policy)
        return self


class OutputConfig(BaseModel):
    """Output configuration."""
    traces_dir: Path = Field(default=Path("./output"))
    sandbox_dir: Path = Field(default=Path("./sandbox"))
    failures_dir: Path = Field(default=Path("./failures"))
    # Harbor bench intermediates (trials/sources/sessions). None -> a sibling ``bench``
    # dir next to traces_dir, parallel to sandbox/failures (never inside the dataset).
    bench_dir: Path | None = None
    # Remove each task's per-task Docker image after the bench run (harbor's ``hb__<task>``
    # image or the pulled prebuilt task image, the swe-bench agent layer) so they don't
    # accumulate and fill the disk. Set True to keep them (e.g. for debugging, or to avoid
    # rebuilds/re-pulls across runs).
    keep_bench_images: bool = False
    pretty_name: str = "Agentic Training Traces"
    # Dataset-card customization (all optional; the default card is unchanged when unset).
    # ``license`` adds an SPDX id to the card frontmatter; ``card_extra`` merges arbitrary
    # extra frontmatter keys; ``readme_template`` points at a user Jinja2 template that
    # replaces the shipped default (rendered with the same context).
    license: str | None = None
    card_extra: dict[str, Any] = Field(default_factory=dict)
    readme_template: Path | None = None

    @field_validator("readme_template")
    @classmethod
    def validate_readme_template(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"output.readme_template not found: {path}")
        return path


class PublishConfig(BaseModel):
    """Publishing configuration."""
    repo_id: str | None = None
    hf_token: str | None = None
    private: bool = False

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not GITHUB_REPO_ID_PATTERN.fullmatch(normalized):
            raise ValueError("publish.repo_id must be in owner/repo format, e.g. armand0e/my-dataset")
        return normalized


class PromptInput(BaseModel):
    """Structured prompt input row."""
    image: str | None = None
    github_repo: str | None = None
    system: str | None = None
    prompt: str
    follow_up_prompts: list[str] = Field(default_factory=list)

    @staticmethod
    def _normalize_optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = value if isinstance(value, str) else str(value)
        normalized = text.strip()
        if not normalized or normalized.lower() == "none":
            return None
        return normalized

    @field_validator("image", "github_repo", "system", mode="before")
    @classmethod
    def normalize_optional_fields(cls, value: object) -> str | None:
        return cls._normalize_optional_text(value)

    @field_validator("prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value: object) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else str(value)
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value:
            raise ValueError("Prompt cannot be empty")
        return value

    @field_validator("follow_up_prompts", mode="before")
    @classmethod
    def normalize_follow_up_prompts(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("follow_up_prompts must be a list of prompt strings")
        prompts: list[str] = []
        for index, item in enumerate(value, start=1):
            text = item if isinstance(item, str) else str(item)
            normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not normalized:
                raise ValueError(f"follow_up_prompts entry {index} cannot be empty")
            prompts.append(normalized)
        return prompts

    @field_validator("github_repo")
    @classmethod
    def validate_github_repo(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not GITHUB_REPO_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "github_repo must be in owner/repo format, e.g. armand0e/perplexica-mcp"
            )
        return value

    def turn_prompts(self) -> list[str]:
        """Return the initial prompt plus any configured follow-up prompts."""
        return [self.prompt, *self.follow_up_prompts]


class BenchSource(BaseModel):
    """One benchmark source for ``generate --mode bench``.

    ``type`` selects the backend (``harbor`` | ``swe-bench``). ``source`` is the dataset
    spec for that backend: harbor → a local task dir, a registry spec (``name@version`` or
    ``org/name@ref``), or (with ``repo``) a dataset name in a git/HF registry; swe-bench →
    a Hugging Face dataset id (or a local json/jsonl path). ``version``/``repo`` are harbor
    knobs; ``split``/``instances`` select swe-bench rows; ``backend`` is harbor's
    EnvironmentType. Remote data is fetched into ``<output.bench_dir>/`` and run locally.
    """
    type: str = "harbor"
    source: str
    repo: str | None = None
    version: str | None = None
    split: str | None = None
    instances: list[str] | None = None
    backend: str = "docker"


class BenchConfig(BaseModel):
    """Bench-mode settings: an array of benchmark sources, each run by its backend."""
    sources: list[BenchSource] = Field(default_factory=list)


class Config(BaseModel):
    """Main configuration."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mcp_servers: list[MCPConfig] = Field(default_factory=list)
    prompts: list[str | dict[str, Any]] = Field(default_factory=list)
    prompts_file: Path | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)
    max_concurrency: int = Field(default=1, ge=1)
    timeout_seconds: int = 600
    openai_api_key: str | None = None
    developer_instructions: str | None = None

    @field_validator("prompts_file")
    @classmethod
    def validate_prompts_file(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"Prompts file not found: {v}")
        return v

    @model_validator(mode="after")
    def validate_codex_auth_dir(self) -> Config:
        """Keep the Codex auth snapshot out of uploaded output directories."""
        codex = self.agent.codex
        if not codex.use_host_auth:
            return self
        auth_dir = codex.auth_dir.resolve()
        for label, output_dir in (
            ("traces_dir", self.output.traces_dir),
            ("sandbox_dir", self.output.sandbox_dir),
            ("failures_dir", self.output.failures_dir),
        ):
            if auth_dir.is_relative_to(output_dir.resolve()):
                raise ValueError(
                    f"agent.codex.auth_dir ({codex.auth_dir}) must not be inside "
                    f"output.{label} ({output_dir}); it holds your Codex credentials "
                    "and output directories are uploaded to Hugging Face."
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load config from YAML file.

        Environment variables override config file values (for testing):
            TEICH_MODEL - Override model ID
            TEICH_BASE_URL - Override API base URL
            TEICH_API_KEY - Override API key
            OPENROUTER_API_KEY - API key fallback for OpenRouter configs
            OPENAI_API_KEY - API key fallback for OpenAI configs and legacy configs
            TEICH_PROVIDER - Override provider name
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        api_config = data.get("api")
        if isinstance(api_config, dict) and "model" in api_config:
            raise ValueError(
                "Unsupported config key 'api.model'. Set the model under 'model.model' instead."
            )
        prompts_file = data.get("prompts_file")
        if isinstance(prompts_file, str) and prompts_file.strip():
            if prompts_file_path := data.get("prompts_file"):
                if not Path(prompts_file_path).is_absolute():
                    data["prompts_file"] = (path.parent / prompts_file_path).resolve()

        # Resolve a relative output.readme_template against the config file, like prompts_file
        # above — otherwise it's validated against the process CWD and `teich -c project/x.yaml`
        # from elsewhere fails even when the template sits next to the config.
        output_section = data.get("output")
        if isinstance(output_section, dict):
            template = output_section.get("readme_template")
            if isinstance(template, str) and template.strip():
                template_path = Path(template).expanduser()
                if not template_path.is_absolute():
                    output_section["readme_template"] = (path.parent / template_path).resolve()

        # Resolve a relative *local* bench source (./tasks, ../x, ~/x) against the config dir, like
        # prompts_file — otherwise it's tried against the process CWD and falls through to the
        # remote-download path. A registry spec (name@version, org/name) never starts with those.
        bench_section = data.get("bench")
        if isinstance(bench_section, dict):
            for bench_source in bench_section.get("sources") or []:
                if not isinstance(bench_source, dict):
                    continue
                spec = bench_source.get("source")
                if isinstance(spec, str) and spec.strip():
                    spec_str = spec.strip()
                    candidate = path.parent / Path(spec_str).expanduser()
                    # Rewrite an explicit ./ ../ ~ path, or a bare relative that actually exists
                    # beside the config (data/tasks, instances.jsonl). A registry/HF spec
                    # (name@version, org/name) has no such local path, so it's left untouched.
                    if spec_str.startswith(("./", "../", "~")) or candidate.exists():
                        bench_source["source"] = str(candidate.resolve())

        # Apply environment variable overrides
        if model_env := _get_env_alias("TEICH_MODEL"):
            data.setdefault("model", {})["model"] = model_env
        if base_url_env := _get_env_alias("TEICH_BASE_URL"):
            data.setdefault("api", {})["base_url"] = base_url_env
        api_section = data.setdefault("api", {})
        provider_for_env = api_section.get("provider") if isinstance(api_section, dict) else None
        configured_api_key = api_section.get("api_key") if isinstance(api_section, dict) else None
        if api_key_env := _get_env_alias("TEICH_API_KEY"):
            api_section["api_key"] = api_key_env
        elif not (isinstance(configured_api_key, str) and configured_api_key.strip()):
            provider_aliases = [
                alias
                for alias in _api_key_env_aliases(provider_for_env)
                if alias != "TEICH_API_KEY"
            ]
            if api_key_env := _get_env_alias(*provider_aliases):
                api_section["api_key"] = api_key_env
        if provider_env := _get_env_alias("TEICH_PROVIDER"):
            data.setdefault("api", {})["provider"] = provider_env

        return cls(**data)

    def get_api_key(self) -> str | None:
        """Get effective API key (from api config or global)."""
        return (
            _normalize_api_key(self.api.api_key)
            or _normalize_api_key(self.openai_api_key)
            or _get_env_alias(*_api_key_env_aliases(self.api.provider))
        )

    def get_effective_model(self) -> str:
        """Get effective model identifier."""
        return self.model.model

    def get_base_url(self) -> str | None:
        """Get effective base URL."""
        return self.api.base_url

    def get_agent_provider(self) -> str:
        """Get active agent provider."""
        return self.agent.provider.strip().lower() or "codex"

    def get_codex_host_auth_source(self) -> Path:
        """Resolve the host Codex auth file to seed the shared snapshot from.

        Honors an explicit ``host_auth_file`` override, then ``$CODEX_HOME``,
        and finally the default ``~/.codex/auth.json``.
        """
        override = self.agent.codex.host_auth_file
        if override is not None:
            return Path(override).expanduser()
        codex_home = os.environ.get("CODEX_HOME")
        base = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
        return base / "auth.json"

    def get_dataset_tags(self) -> list[str]:
        """Get auto-generated dataset tags for README frontmatter and uploads."""
        provider = self.get_agent_provider()
        model_name = self.get_effective_model().strip() or "unknown-model"
        if provider == "chat":
            ordered_tags = ["conversational", "distillation", "teich", model_name]
        else:
            ordered_tags = ["agent-traces", "format:agent-traces", provider, "distillation", model_name, "teich"]
        tags: list[str] = []
        seen: set[str] = set()
        for tag in ordered_tags:
            normalized = tag.strip() if isinstance(tag, str) else str(tag).strip()
            if not normalized or normalized in seen:
                continue
            tags.append(normalized)
            seen.add(normalized)
        return tags

    def get_publish_repo_id(self) -> str | None:
        """Get the configured Hugging Face dataset repo id, if any."""
        return self.publish.repo_id

    def get_hf_token(self) -> str | None:
        """Get effective Hugging Face token from config or common env vars."""
        return self.publish.hf_token or _get_env_alias("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "TEICH_HF_TOKEN")

    def get_prompts(self) -> list[str]:
        """Get prompt text only for all configured prompt inputs."""
        return [prompt_input.prompt for prompt_input in self.get_prompt_inputs()]

    def get_prompt_inputs(self) -> list[PromptInput]:
        """Get structured prompt inputs from config and prompts_file."""
        prompt_inputs: list[PromptInput] = []
        for index, prompt in enumerate(self.prompts, start=1):
            if isinstance(prompt, str):
                prompt_inputs.append(PromptInput(prompt=prompt))
                continue
            if not isinstance(prompt, dict):
                raise ValueError(f"Inline prompt entry {index} must be a string or object")
            prompt_input = self._prompt_input_from_mapping(prompt, source=f"inline prompt entry {index}")
            if prompt_input is not None:
                prompt_inputs.append(prompt_input)
        if self.prompts_file:
            prompt_inputs.extend(self._load_prompt_inputs_from_file(self.prompts_file))
        for prompt_input in prompt_inputs:
            if prompt_input.image is not None:
                raise ValueError(
                    "Prompt image inputs are not supported yet. Leave the image column blank or set it to None."
                )
        return prompt_inputs

    @staticmethod
    def _load_prompt_inputs_from_file(path: Path) -> list[PromptInput]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return Config._load_prompt_inputs_from_csv(path)
        if suffix in {".jsonl", ".ndjson"}:
            return Config._load_prompt_inputs_from_jsonl(path)
        if suffix == ".json":
            return Config._load_prompt_inputs_from_json(path)
        return Config._load_prompt_inputs_from_text(path)

    @staticmethod
    def _load_prompt_inputs_from_text(path: Path) -> list[PromptInput]:
        with path.open("r", encoding="utf-8") as handle:
            return [
                PromptInput(prompt=line.strip())
                for line in handle
                if line.strip() and not line.startswith("#")
            ]

    @staticmethod
    def _load_prompt_inputs_from_csv(path: Path) -> list[PromptInput]:
        _raise_csv_field_limit()
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, restkey="__extra_columns__", strict=True)
                raw_fieldnames = reader.fieldnames or []
                fieldnames = [
                    name.strip().lower()
                    for name in raw_fieldnames
                    if isinstance(name, str) and name.strip()
                ]
                if "prompt" not in fieldnames:
                    raise ValueError("Prompt CSV must include a 'prompt' column")
                if len(fieldnames) != len(set(fieldnames)):
                    raise ValueError("Prompt CSV contains duplicate column names after normalization")
                prompt_inputs: list[PromptInput] = []
                for row_number, row in enumerate(reader, start=2):
                    normalized_row = {
                        key.strip().lower(): value
                        for key, value in row.items()
                        if isinstance(key, str)
                    }
                    if normalized_row.get("__extra_columns__"):
                        raise ValueError(
                            f"Prompt CSV row {row_number} has more columns than the header. "
                            "If a prompt contains commas or newlines, quote the entire prompt field."
                        )
                    if not any(
                        isinstance(value, str) and value.strip()
                        for value in normalized_row.values()
                    ):
                        continue
                    prompt = normalized_row.get("prompt")
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f"Prompt CSV row {row_number} has an empty 'prompt' value")
                    try:
                        prompt_inputs.append(
                            PromptInput(
                                image=normalized_row.get("image"),
                                github_repo=normalized_row.get("github_repo"),
                                system=normalized_row.get("system"),
                                prompt=prompt,
                            )
                        )
                    except ValueError as exc:
                        raise ValueError(f"Invalid prompt CSV row {row_number}: {exc}") from exc
        except csv.Error as exc:
            raise ValueError(
                f"Failed to parse prompt CSV {path}: {exc}. "
                "Multiline prompts must be inside a quoted prompt field; for very long prompts, prefer JSONL."
            ) from exc
        return prompt_inputs

    @staticmethod
    def _prompt_input_from_mapping(row: dict[str, object], *, source: str) -> PromptInput | None:
        normalized_row = {
            key.strip().lower(): value
            for key, value in row.items()
            if isinstance(key, str)
        }
        if not any(
            isinstance(value, str) and value.strip()
            for value in normalized_row.values()
        ):
            return None
        prompt = normalized_row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{source} has an empty or missing 'prompt' value")
        try:
            return PromptInput(
                image=normalized_row.get("image"),
                github_repo=normalized_row.get("github_repo"),
                system=normalized_row.get("system"),
                prompt=prompt,
                follow_up_prompts=normalized_row.get("follow_up_prompts"),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid {source}: {exc}") from exc

    @staticmethod
    def _load_prompt_inputs_from_jsonl(path: Path) -> list[PromptInput]:
        prompt_inputs: list[PromptInput] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid prompt JSONL line {line_number}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Prompt JSONL line {line_number} must be a JSON object")
                prompt_input = Config._prompt_input_from_mapping(row, source=f"prompt JSONL line {line_number}")
                if prompt_input is not None:
                    prompt_inputs.append(prompt_input)
        return prompt_inputs

    @staticmethod
    def _load_prompt_inputs_from_json(path: Path) -> list[PromptInput]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid prompt JSON file {path}: {exc.msg}") from exc
        rows = payload.get("prompts") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Prompt JSON must be a list of objects or an object with a 'prompts' list")
        prompt_inputs: list[PromptInput] = []
        for index, row in enumerate(rows, start=1):
            if isinstance(row, str):
                prompt_inputs.append(PromptInput(prompt=row))
                continue
            if not isinstance(row, dict):
                raise ValueError(f"Prompt JSON entry {index} must be an object or string")
            prompt_input = Config._prompt_input_from_mapping(row, source=f"prompt JSON entry {index}")
            if prompt_input is not None:
                prompt_inputs.append(prompt_input)
        return prompt_inputs


def load_config(path: Path) -> Config:
    """Load configuration from YAML file.

    Environment variables (for testing):
        TEICH_MODEL - Override model ID
        TEICH_BASE_URL - Override API base URL
        TEICH_API_KEY - Override API key
        TEICH_PROVIDER - Override provider name
    """
    return Config.from_yaml(path)
