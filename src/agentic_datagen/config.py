"""Configuration management for teich."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


GITHUB_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _get_env_alias(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


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


class AgentConfig(BaseModel):
    """Agent runtime selection."""
    provider: str = "codex"


class ModelConfig(BaseModel):
    """Model configuration."""
    model: str = "codex-mini-latest"
    approval_policy: str = "never"
    sandbox: str = "danger-full-access"
    reasoning_effort: str | None = None
    approval_mode: str | None = "none"
    pi_model_overrides: dict[str, object] = Field(default_factory=dict)

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
    pretty_name: str = "Agentic Training Traces"
    tags: list[str] = Field(default_factory=lambda: ["agent-traces", "codex"])
    readme_file_name: str = "README.md"


class PromptInput(BaseModel):
    """Structured prompt input row."""
    image: str | None = None
    github_repo: str | None = None
    prompt: str

    @staticmethod
    def _normalize_optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = value if isinstance(value, str) else str(value)
        normalized = text.strip()
        if not normalized or normalized.lower() == "none":
            return None
        return normalized

    @field_validator("image", "github_repo", mode="before")
    @classmethod
    def normalize_optional_fields(cls, value: object) -> str | None:
        return cls._normalize_optional_text(value)

    @field_validator("prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value: object) -> str:
        if value is None:
            return ""
        return (value if isinstance(value, str) else str(value)).strip()

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value:
            raise ValueError("Prompt cannot be empty")
        return value

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


class Config(BaseModel):
    """Main configuration."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mcp_servers: list[MCPConfig] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    prompts_file: Path | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
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

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load config from YAML file.

        Environment variables override config file values (for testing):
            TEICH_MODEL - Override model ID
            TEICH_BASE_URL - Override API base URL
            TEICH_API_KEY - Override API key
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
            prompts_file_path = Path(prompts_file)
            if not prompts_file_path.is_absolute():
                data["prompts_file"] = (path.parent / prompts_file_path).resolve()

        # Apply environment variable overrides
        if model_env := _get_env_alias("TEICH_MODEL", "AGENTIC_DATAGEN_MODEL"):
            data.setdefault("model", {})["model"] = model_env
        if base_url_env := _get_env_alias("TEICH_BASE_URL", "AGENTIC_DATAGEN_BASE_URL"):
            data.setdefault("api", {})["base_url"] = base_url_env
        if api_key_env := _get_env_alias("TEICH_API_KEY", "AGENTIC_DATAGEN_API_KEY"):
            data.setdefault("api", {})["api_key"] = api_key_env
        if provider_env := _get_env_alias("TEICH_PROVIDER", "AGENTIC_DATAGEN_PROVIDER"):
            data.setdefault("api", {})["provider"] = provider_env

        return cls(**data)

    def get_api_key(self) -> str | None:
        """Get effective API key (from api config or global)."""
        return self.api.api_key or self.openai_api_key

    def get_effective_model(self) -> str:
        """Get effective model identifier."""
        return self.model.model

    def get_base_url(self) -> str | None:
        """Get effective base URL."""
        return self.api.base_url

    def get_agent_provider(self) -> str:
        """Get active agent provider."""
        return self.agent.provider.strip().lower() or "codex"

    def get_prompts(self) -> list[str]:
        """Get prompt text only for all configured prompt inputs."""
        return [prompt_input.prompt for prompt_input in self.get_prompt_inputs()]

    def get_prompt_inputs(self) -> list[PromptInput]:
        """Get structured prompt inputs from config and prompts_file."""
        prompt_inputs = [PromptInput(prompt=prompt) for prompt in self.prompts]
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
        if path.suffix.lower() == ".csv":
            return Config._load_prompt_inputs_from_csv(path)
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
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [name.strip().lower() for name in reader.fieldnames or [] if isinstance(name, str)]
            if "prompt" not in fieldnames:
                raise ValueError("Prompt CSV must include a 'prompt' column")
            prompt_inputs: list[PromptInput] = []
            for row in reader:
                normalized_row = {
                    key.strip().lower(): value
                    for key, value in row.items()
                    if isinstance(key, str)
                }
                if not any(
                    isinstance(value, str) and value.strip()
                    for value in normalized_row.values()
                ):
                    continue
                prompt_inputs.append(
                    PromptInput(
                        image=normalized_row.get("image"),
                        github_repo=normalized_row.get("github_repo"),
                        prompt=normalized_row.get("prompt") or "",
                    )
                )
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
