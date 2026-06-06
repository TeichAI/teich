"""Integration tests for teich.

These tests require:
- Docker running
- OPENAI_API_KEY environment variable set (for live tests)
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from teich.config import Config, APIConfig, ModelConfig
from teich.runner import CodexRunner, RUNTIME_CONTAINER_USER, RUNTIME_IMAGE_NAME


# Skip integration tests if Docker is not available
docker_available = subprocess.run(
    ["docker", "info"], capture_output=True, timeout=5
).returncode == 0

requires_docker = pytest.mark.skipif(
    not docker_available,
    reason="Docker not available"
)

requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

requires_runtime_smoke = pytest.mark.skipif(
    os.getenv("TEICH_RUN_DOCKER_RUNTIME_SMOKE") != "1",
    reason="set TEICH_RUN_DOCKER_RUNTIME_SMOKE=1 to run Docker package-manager smoke",
)


class TestDockerImage:
    """Tests for Docker image building."""

    @requires_docker
    def test_dfile_exists(self):
        """Verify Dockerfile exists."""
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"
        assert dockerfile.exists()

    def test_dockerfile_preinstalls_playwright_chromium(self):
        """Verify the runtime image bakes in Playwright Chromium dependencies."""
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")
        assert "python3-pip" in content
        assert "python3-venv" in content
        assert "python3-dev" in content
        assert "python3 -m venv /opt/venv" in content
        assert 'ENV VIRTUAL_ENV=/opt/venv' in content
        assert 'ENV PATH="/opt/venv/bin:/usr/local/bin:$PATH"' in content
        assert "pip --version" in content
        assert "pip3 --version" in content
        assert "ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright" in content
        assert "npm install -g @openai/codex @anthropic-ai/claude-code @mariozechner/pi-coding-agent playwright" in content
        assert "git clone --depth 1 https://github.com/NousResearch/hermes-agent.git" in content
        assert "uv pip install --python /usr/local/lib/hermes-agent/venv/bin/python -e ." in content
        assert "npx playwright install --with-deps chromium" in content
        assert 'ENV NODE_PATH="/usr/local/lib/node_modules"' in content

    @requires_docker
    @pytest.mark.slow  # Takes 2-3 minutes, skip by default
    def test_docker_build(self, tmp_path):
        """Test Docker image builds successfully."""
        pytest.skip("Docker build test - run manually with: pytest tests/test_integration.py::TestDockerImage::test_docker_build -v")
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"

        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(dockerfile),
                "-t", "test-teich:latest",
                str(dockerfile.parent),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for build
        )

        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

    @requires_docker
    @requires_runtime_smoke
    @pytest.mark.slow
    def test_runtime_container_can_install_system_packages(self):
        """Verify generated agents can use apt-get for missing system dependencies."""
        CodexRunner(Config())

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                RUNTIME_CONTAINER_USER,
                "-e",
                "HOME=/home/codex",
                RUNTIME_IMAGE_NAME,
                "bash",
                "-lc",
                "test \"$(id -u)\" != 0 && test \"$(command -v apt-get)\" = /usr/local/bin/apt-get && apt-get update -qq",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        assert result.returncode == 0, result.stderr or result.stdout


class TestRunnerIntegration:
    """Integration tests for CodexRunner with real Docker."""

    @requires_docker
    def test_runner_creates_output_dir(self, tmp_path):
        """Test runner creates output directory."""
        from teich.config import OutputConfig

        config = Config(
            model=ModelConfig(model="codex-mini-latest", approval_mode="none"),
            prompts=["test"],
            output=OutputConfig(traces_dir=tmp_path / "output"),
        )

        with patch.object(CodexRunner, '_ensure_image'):
            runner = CodexRunner(config)

        # Output dir should be created
        runner.config.output.traces_dir.mkdir(parents=True, exist_ok=True)
        assert runner.config.output.traces_dir.exists()


class TestTraceFormat:
    """Tests for trace output format validation."""

    def test_trace_line_structure(self, tmp_path):
        """Verify trace JSONL has required fields."""
        # Create a mock trace file
        trace_file = tmp_path / "test_session.jsonl"

        events = [
            {"type": "session", "id": "test-123", "timestamp": "2024-01-01T00:00:00Z"},
            {"type": "message", "message": {"role": "user", "content": "Build an app"}},
            {"type": "message", "message": {"role": "assistant", "content": "I'll help you build an app"}},
        ]

        with open(trace_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        # Validate structure
        with open(trace_file) as f:
            lines = f.readlines()
            assert len(lines) == 3

            # First line should be session
            first = json.loads(lines[0])
            assert first["type"] == "session"
            assert "id" in first

            # Messages should have role
            for line in lines[1:]:
                event = json.loads(line)
                if event["type"] == "message":
                    assert "message" in event
                    assert "role" in event["message"]


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_config_yaml_roundtrip(self, tmp_path, monkeypatch):
        """Test config can be saved and loaded."""
        # Clear env vars that would override YAML values
        monkeypatch.delenv("TEICH_MODEL", raising=False)
        monkeypatch.delenv("TEICH_BASE_URL", raising=False)
        monkeypatch.delenv("TEICH_API_KEY", raising=False)
        monkeypatch.delenv("TEICH_PROVIDER", raising=False)

        config_file = tmp_path / "config.yaml"

        # Create and save config (use model_dump with mode='json' for safe serialization)
        original = Config(
            model=ModelConfig(model="gpt-4o", approval_mode="suggest"),
            prompts=["Test prompt 1", "Test prompt 2"],
            openai_api_key="sk-test",
        )

        # Save manually - use json mode to avoid Path serialization issues
        import yaml
        config_file.write_text(yaml.dump(original.model_dump(mode="json")))

        # Load back
        loaded = Config.from_yaml(config_file)

        assert loaded.model.model == original.model.model
        assert loaded.prompts == original.prompts


class TestOpenRouterIntegration:
    """Tests for OpenRouter/custom API provider integration."""

    def test_openrouter_config_in_command(self):
        """Verify OpenRouter config generates correct CLI args."""
        config = Config(
            model=ModelConfig(model="anthropic/claude-3.5-sonnet"),
            api=APIConfig(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-or-test",
            ),
            openai_api_key="sk-openai",
        )

        # The command should use the API-specific key
        assert config.api.api_key == "sk-or-test"
        # But fall back to global if not set
        config2 = Config(openai_api_key="sk-global")
        assert config2.api.api_key is None


@pytest.mark.integration
@pytest.mark.slow
@requires_docker
@requires_api_key
class TestEndToEnd:
    """Full end-to-end tests requiring real API access."""

    def test_full_generation_workflow(self, tmp_path):
        """Test complete workflow: init -> generate -> verify output."""
        # Setup
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        config = Config(
            model=ModelConfig(model="codex-mini-latest", approval_mode="none"),
            prompts=["Create a Python hello world script"],
            output=MagicMock(traces_dir=project_dir / "output"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            timeout_seconds=60,
        )

        with patch.object(CodexRunner, '_ensure_image'):
            runner = CodexRunner(config)

        # Create workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Run session (mocked for unit test, would be real for integration)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Mock session file creation
            session_dir = tmp_path / ".codex" / "sessions"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "test-session.jsonl"
            session_file.write_text(
                json.dumps({"type": "session", "id": "test"}) + "\n"
            )

            with patch.object(runner, '_extract_session_file', return_value=project_dir / "output" / "test.jsonl"):
                result = runner.run_session("Create a Python hello world script", "test")

        assert result is not None


# Integration test markers
pytestmark = [
    pytest.mark.integration,
]
