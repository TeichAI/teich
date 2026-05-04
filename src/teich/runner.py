"""Docker-based runners for non-interactive Codex and Pi sessions."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config, PromptInput

RUNTIME_IMAGE_NAME = "teich-runtime:v3"
RUNTIME_DOCKERFILE_NAME = "codex-runtime.Dockerfile"
CODEX_HOME_IN_CONTAINER = "/home/codex/.codex"
PI_AGENT_DIR_IN_CONTAINER = "/home/codex/.pi/agent"
PI_SESSIONS_DIR_IN_CONTAINER = "/home/codex/pi-sessions"
WORKSPACE_IN_CONTAINER = "/workspace"
LOCAL_PROVIDER_PROXY_SCRIPT_NAME = "local_provider_proxy.js"
PI_SYSTEM_PROMPT_CUSTOM_TYPE = "teich-system-prompt"
PI_EMPTY_TOOL_NOT_FOUND_TEXT = "Tool  not found"

LOCAL_PROVIDER_PROXY_SCRIPT = """
const http = require('node:http');
const https = require('node:https');
const { Readable } = require('node:stream');

const target = new URL(process.env.TEICH_LOCAL_PROVIDER_TARGET);
const listenPort = Number(process.env.TEICH_LOCAL_PROVIDER_PORT || '1234');

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  return chunks.length ? Buffer.concat(chunks) : undefined;
}

const server = http.createServer(async (req, res) => {
  const upstreamUrl = new URL(req.url || '/', target);
  const headers = { ...req.headers };
  delete headers.host;
  delete headers['content-length'];

  try {
    const body = req.method === 'GET' || req.method === 'HEAD' ? undefined : await readRequestBody(req);
    const upstream = await fetch(upstreamUrl, {
      method: req.method,
      headers,
      body,
      duplex: body ? 'half' : undefined,
      dispatcher: upstreamUrl.protocol === 'https:' ? undefined : undefined,
    });

    res.writeHead(upstream.status, Object.fromEntries(upstream.headers.entries()));
    if (!upstream.body) {
      res.end();
      return;
    }
    Readable.fromWeb(upstream.body).pipe(res);
  } catch (error) {
    res.writeHead(502, { 'content-type': 'text/plain; charset=utf-8' });
    res.end(String(error));
  }
});

server.listen(listenPort, '127.0.0.1');
""".strip()


PI_SYSTEM_PROMPT_EXTENSION = f"""
export default function (pi) {{
  pi.on("before_agent_start", async (_event, ctx) => {{
    const systemPrompt = ctx.getSystemPrompt();
    if (typeof systemPrompt !== "string" || !systemPrompt.trim()) {{
      return;
    }}
    const entries = typeof ctx.sessionManager.getEntries === "function"
      ? ctx.sessionManager.getEntries()
      : [];
    if (Array.isArray(entries) && entries.some(
      (entry) => entry?.type === "custom"
        && entry?.customType === "{PI_SYSTEM_PROMPT_CUSTOM_TYPE}"
        && entry?.data?.systemPrompt === systemPrompt,
    )) {{
      return;
    }}
    if (typeof ctx.sessionManager.appendCustomEntry === "function") {{
      ctx.sessionManager.appendCustomEntry("{PI_SYSTEM_PROMPT_CUSTOM_TYPE}", {{
        systemPrompt,
      }});
    }}
  }});
}}
""".strip()


@dataclass(slots=True)
class TraceMetrics:
    model: str | None = None
    provider: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    est_total_tokens: int = 0
    total_cost: float = 0.0

    @staticmethod
    def _int_value(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    @staticmethod
    def _float_value(value: Any) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def add_pi_usage(self, usage: dict[str, Any]) -> None:
        self.input_tokens += self._int_value(usage.get("input"))
        self.output_tokens += self._int_value(usage.get("output"))
        self.cache_read_tokens += self._int_value(usage.get("cacheRead"))
        self.cache_write_tokens += self._int_value(usage.get("cacheWrite"))
        total_tokens = self._int_value(usage.get("totalTokens"))
        self.total_tokens += total_tokens
        self.est_total_tokens += total_tokens
        cost = usage.get("cost")
        if isinstance(cost, dict):
            self.total_cost += self._float_value(cost.get("total"))

    def add_codex_last_usage(self, usage: dict[str, Any]) -> None:
        self.input_tokens += self._int_value(usage.get("input_tokens"))
        self.output_tokens += self._int_value(usage.get("output_tokens"))
        self.reasoning_tokens += self._int_value(usage.get("reasoning_output_tokens"))
        self.cache_read_tokens += self._int_value(usage.get("cached_input_tokens"))
        total_tokens = self._int_value(usage.get("total_tokens"))
        if total_tokens:
            self.total_tokens += total_tokens

    def apply_codex_total_usage(self, usage: dict[str, Any]) -> None:
        self.input_tokens = self._int_value(usage.get("input_tokens"))
        self.output_tokens = self._int_value(usage.get("output_tokens"))
        self.reasoning_tokens = self._int_value(usage.get("reasoning_output_tokens"))
        self.cache_read_tokens = self._int_value(usage.get("cached_input_tokens"))
        total_tokens = self._int_value(usage.get("total_tokens"))
        if total_tokens:
            self.total_tokens = total_tokens

    def apply_codex_estimated_usage(self, usage: dict[str, Any]) -> None:
        total_tokens = self._int_value(usage.get("total_tokens"))
        if total_tokens:
            self.est_total_tokens = total_tokens
            return
        self.est_total_tokens = (
            self._int_value(usage.get("input_tokens"))
            + self._int_value(usage.get("output_tokens"))
            + self._int_value(usage.get("reasoning_output_tokens"))
            + self._int_value(usage.get("cached_input_tokens"))
        )

    def finalize(self) -> None:
        if not self.total_tokens:
            self.total_tokens = (
                self.input_tokens
                + self.output_tokens
                + self.reasoning_tokens
                + self.cache_read_tokens
                + self.cache_write_tokens
            )
        if not self.est_total_tokens:
            self.est_total_tokens = self.total_tokens


@dataclass(slots=True)
class SessionProgressUpdate:
    prompt_id: str
    prompt_index: int
    total_prompts: int
    prompt: str
    prompt_preview: str
    status: str
    session_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trace_path: Path | None = None
    sandbox_path: Path | None = None
    error: str | None = None
    metrics: TraceMetrics | None = None


SessionProgressCallback = Callable[[SessionProgressUpdate], None]


class DockerRuntimeRunner:
    """Shared Docker runtime used by agent runners."""

    def __init__(self, config: Config):
        self.config = config
        self.image_name = RUNTIME_IMAGE_NAME
        self._ensure_image()

    @staticmethod
    def _runtime_dockerfile_path() -> Path:
        return Path(__file__).parent.parent.parent / "docker" / RUNTIME_DOCKERFILE_NAME

    def _image_created_at(self) -> datetime | None:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name, "--format", "{{.Created}}"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return None

        created_text = result.stdout.strip()
        if not created_text:
            return None
        try:
            return datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _ensure_image(self) -> None:
        """Build Docker image if needed."""
        dockerfile_path = self._runtime_dockerfile_path()
        if not dockerfile_path.exists():
            raise RuntimeError(f"Dockerfile not found: {dockerfile_path}")
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.image_name],
                capture_output=True,
                text=True,
                check=True,
            )
            if not result.stdout.strip():
                self._build_image()
                return

            image_created_at = self._image_created_at()
            dockerfile_mtime = datetime.fromtimestamp(dockerfile_path.stat().st_mtime, tz=timezone.utc)
            if image_created_at is None or image_created_at < dockerfile_mtime:
                self._build_image()
        except subprocess.CalledProcessError:
            self._build_image()

    def _build_image(self) -> None:
        """Build the Docker image."""
        dockerfile_path = self._runtime_dockerfile_path()
        if not dockerfile_path.exists():
            raise RuntimeError(f"Dockerfile not found: {dockerfile_path}")

        context = dockerfile_path.parent
        subprocess.run(
            ["docker", "build", "-t", self.image_name, "-f", str(dockerfile_path), str(context)],
            check=True,
        )

    @staticmethod
    def _container_base_url(base_url: str | None) -> str | None:
        if not base_url:
            return None
        parsed = urlsplit(base_url)
        hostname = parsed.hostname or ""
        if hostname not in {"localhost", "127.0.0.1"}:
            return base_url
        netloc = parsed.netloc.replace(hostname, "host.docker.internal", 1)
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    @staticmethod
    def _prompt_preview(prompt: str, limit: int = 60) -> str:
        normalized = " ".join(prompt.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @staticmethod
    def _copy_workspace_snapshot(workspace: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(workspace, destination, dirs_exist_ok=False)

    @staticmethod
    def _github_repo_checkout_name(github_repo: str) -> str:
        return github_repo.rsplit("/", maxsplit=1)[-1]

    @staticmethod
    def _clone_github_repo(github_repo: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        repo_url = f"https://github.com/{github_repo}.git"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(destination)],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
            details = stderr.strip() or stdout.strip() or str(exc)
            raise RuntimeError(f"Failed to clone github repo {github_repo}: {details}") from exc

    def _prepare_workspace(
        self,
        session_id: str,
        prompt_input: PromptInput | None,
        prefix: str,
    ) -> tuple[Path, Path]:
        workspace_root = Path(tempfile.mkdtemp(prefix=f"{prefix}-{session_id}-"))
        workspace = workspace_root
        if prompt_input is None:
            return workspace_root, workspace
        if prompt_input.image is not None:
            raise RuntimeError(
                "Prompt image inputs are not supported yet. Leave the image column blank or set it to None."
            )
        if prompt_input.github_repo:
            workspace = workspace_root / self._github_repo_checkout_name(prompt_input.github_repo)
            self._clone_github_repo(prompt_input.github_repo, workspace)
        return workspace_root, workspace

    def _sandbox_destination(self, trace_path: Path) -> Path:
        return self.config.output.sandbox_dir / trace_path.name

    @staticmethod
    def _latest_session_file(session_dir: Path, started_at: datetime) -> Path | None:
        session_files = sorted(path for path in session_dir.rglob("*.jsonl") if path.is_file())
        fresh_files = [
            path
            for path in session_files
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= started_at
        ]
        candidates = fresh_files or session_files
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _runtime_trace_guard_error(self, trace_path: Path) -> str | None:
        return None

    def _monitor_process(
        self,
        process: subprocess.Popen[str],
        session_id: str,
        started_at: datetime,
        session_dir: Path | None,
        progress_callback: SessionProgressCallback | None,
        progress_base: SessionProgressUpdate | None,
        stdout_handle,
        stderr_handle,
    ) -> None:
        deadline = time.monotonic() + self.config.timeout_seconds
        last_signature: tuple[str | None, int, float] | None = None
        while True:
            return_code = process.poll()
            trace_path: Path | None = None
            metrics: TraceMetrics | None = None
            if session_dir is not None:
                trace_path = self._latest_session_file(session_dir, started_at)
                if trace_path and trace_path.exists():
                    stat = trace_path.stat()
                    signature = (str(trace_path), stat.st_size, stat.st_mtime)
                    if signature != last_signature:
                        last_signature = signature
                        try:
                            metrics = self._summarize_trace_file(trace_path)
                            trace_guard_error = self._runtime_trace_guard_error(trace_path)
                        except (OSError, json.JSONDecodeError):
                            metrics = None
                            trace_guard_error = None
                        if trace_guard_error:
                            process.kill()
                            process.wait()
                            raise RuntimeError(f"Session {session_id[:8]} failed: {trace_guard_error}")
                        if progress_callback and progress_base and metrics is not None:
                            progress_callback(
                                SessionProgressUpdate(
                                    prompt_id=progress_base.prompt_id,
                                    prompt_index=progress_base.prompt_index,
                                    total_prompts=progress_base.total_prompts,
                                    prompt=progress_base.prompt,
                                    prompt_preview=progress_base.prompt_preview,
                                    status="running",
                                    session_id=session_id,
                                    started_at=progress_base.started_at,
                                    trace_path=trace_path,
                                    metrics=metrics,
                                )
                            )
            if return_code is not None:
                stdout_handle.flush()
                stderr_handle.flush()
                stdout_handle.seek(0)
                stderr_handle.seek(0)
                stdout = stdout_handle.read()
                stderr = stderr_handle.read()
                if return_code != 0:
                    details = (stderr or "").strip() or (stdout or "").strip()
                    raise subprocess.CalledProcessError(
                        return_code,
                        process.args,
                        output=stdout,
                        stderr=stderr or details,
                    )
                return
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(process.args, self.config.timeout_seconds)
            time.sleep(0.5)

    @staticmethod
    def _codex_usage_has_tokens(usage: dict[str, Any]) -> bool:
        return any(
            TraceMetrics._int_value(usage.get(key))
            for key in (
                "input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "cached_input_tokens",
                "total_tokens",
            )
        )

    @staticmethod
    def _codex_usage_delta(
        previous_usage: dict[str, Any],
        current_usage: dict[str, Any],
    ) -> dict[str, int]:
        delta: dict[str, int] = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "cached_input_tokens",
            "total_tokens",
        ):
            current_value = TraceMetrics._int_value(current_usage.get(key))
            previous_value = TraceMetrics._int_value(previous_usage.get(key))
            delta[key] = max(0, current_value - previous_value)
        return delta

    @classmethod
    def _summarize_trace_file(cls, trace_file: Path) -> TraceMetrics:
        metrics = TraceMetrics()
        codex_total_usage: dict[str, Any] | None = None
        codex_estimated_usage: dict[str, Any] | None = None
        previous_codex_total_usage: dict[str, Any] | None = None
        with trace_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue

                event_type = event.get("type")
                if event_type == "session_meta":
                    payload = event.get("payload")
                    if isinstance(payload, dict) and not metrics.provider:
                        provider = payload.get("model_provider")
                        if isinstance(provider, str) and provider.strip():
                            metrics.provider = provider.strip()
                    continue

                if event_type == "model_change":
                    provider = event.get("provider")
                    model = event.get("modelId")
                    if isinstance(provider, str) and provider.strip() and not metrics.provider:
                        metrics.provider = provider.strip()
                    if isinstance(model, str) and model.strip() and not metrics.model:
                        metrics.model = model.strip()
                    continue

                if event_type == "event_msg":
                    payload = event.get("payload")
                    if not isinstance(payload, dict) or payload.get("type") != "token_count":
                        continue
                    info = payload.get("info")
                    if not isinstance(info, dict):
                        continue
                    total_usage = info.get("total_token_usage")
                    if isinstance(total_usage, dict):
                        codex_total_usage = total_usage
                        if previous_codex_total_usage is None:
                            if cls._codex_usage_has_tokens(total_usage):
                                codex_estimated_usage = total_usage
                        else:
                            delta_usage = cls._codex_usage_delta(previous_codex_total_usage, total_usage)
                            if cls._codex_usage_has_tokens(delta_usage):
                                codex_estimated_usage = delta_usage
                        previous_codex_total_usage = total_usage
                    last_usage = info.get("last_token_usage")
                    if isinstance(last_usage, dict):
                        metrics.add_codex_last_usage(last_usage)
                        if cls._codex_usage_has_tokens(last_usage):
                            codex_estimated_usage = last_usage
                    continue

                if event_type != "message":
                    continue

                payload = event.get("message")
                if not isinstance(payload, dict):
                    continue
                provider = payload.get("provider")
                model = payload.get("model")
                if isinstance(provider, str) and provider.strip() and not metrics.provider:
                    metrics.provider = provider.strip()
                if isinstance(model, str) and model.strip() and not metrics.model:
                    metrics.model = model.strip()
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    metrics.add_pi_usage(usage)

        if codex_total_usage:
            metrics.apply_codex_total_usage(codex_total_usage)
        if codex_estimated_usage:
            metrics.apply_codex_estimated_usage(codex_estimated_usage)
        metrics.finalize()
        return metrics

    def _run_prompt_task(
        self,
        prompt_id: str,
        prompt_index: int,
        total_prompts: int,
        prompt_input: PromptInput,
        progress_callback: SessionProgressCallback | None,
    ) -> Path:
        session_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        prompt_preview = self._prompt_preview(prompt_input.prompt)
        progress_base = SessionProgressUpdate(
            prompt_id=prompt_id,
            prompt_index=prompt_index,
            total_prompts=total_prompts,
            prompt=prompt_input.prompt,
            prompt_preview=prompt_preview,
            status="running",
            session_id=session_id,
            started_at=started_at,
        )
        if progress_callback:
            progress_callback(progress_base)
        try:
            result = self.run_session(
                prompt_input.prompt,
                session_id,
                progress_callback=progress_callback,
                progress_base=progress_base,
                prompt_input=prompt_input,
            )
            metrics = self._summarize_trace_file(result)
            sandbox_path = self._sandbox_destination(result)
            if progress_callback:
                progress_callback(
                    SessionProgressUpdate(
                        prompt_id=prompt_id,
                        prompt_index=prompt_index,
                        total_prompts=total_prompts,
                        prompt=prompt_input.prompt,
                        prompt_preview=prompt_preview,
                        status="completed",
                        session_id=session_id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                        trace_path=result,
                        sandbox_path=sandbox_path,
                        metrics=metrics,
                    )
                )
            return result
        except Exception as exc:
            if progress_callback:
                progress_callback(
                    SessionProgressUpdate(
                        prompt_id=prompt_id,
                        prompt_index=prompt_index,
                        total_prompts=total_prompts,
                        prompt=prompt_input.prompt,
                        prompt_preview=prompt_preview,
                        status="failed",
                        session_id=session_id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                        error=str(exc),
                    )
                )
            raise

    def run_all(
        self,
        max_concurrency: int = 1,
        progress_callback: SessionProgressCallback | None = None,
    ) -> list[Path]:
        prompt_inputs = self.config.get_prompt_inputs()
        if not prompt_inputs:
            raise ValueError("No prompts configured")

        total_prompts = len(prompt_inputs)
        worker_count = max(1, min(max_concurrency, total_prompts))
        for prompt_index, prompt_input in enumerate(prompt_inputs, start=1):
            if progress_callback:
                progress_callback(
                    SessionProgressUpdate(
                        prompt_id=f"prompt-{prompt_index}",
                        prompt_index=prompt_index,
                        total_prompts=total_prompts,
                        prompt=prompt_input.prompt,
                        prompt_preview=self._prompt_preview(prompt_input.prompt),
                        status="queued",
                    )
                )

        results_by_index: dict[int, Path] = {}
        errors: list[Exception] = []
        if worker_count == 1:
            for prompt_index, prompt_input in enumerate(prompt_inputs, start=1):
                try:
                    results_by_index[prompt_index] = self._run_prompt_task(
                        f"prompt-{prompt_index}",
                        prompt_index,
                        total_prompts,
                        prompt_input,
                        progress_callback,
                    )
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
            return [results_by_index[index] for index in range(1, total_prompts + 1)]

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._run_prompt_task,
                    f"prompt-{prompt_index}",
                    prompt_index,
                    total_prompts,
                    prompt_input,
                    progress_callback,
                ): prompt_index
                for prompt_index, prompt_input in enumerate(prompt_inputs, start=1)
            }
            for future in as_completed(futures):
                prompt_index = futures[future]
                try:
                    results_by_index[prompt_index] = future.result()
                except Exception as exc:
                    errors.append(exc)

        if errors:
            raise errors[0]
        return [results_by_index[index] for index in range(1, total_prompts + 1)]

    def _run_process(
        self,
        command: list[str],
        session_id: str,
        started_at: datetime,
        session_dir: Path | None = None,
        progress_callback: SessionProgressCallback | None = None,
        progress_base: SessionProgressUpdate | None = None,
    ) -> None:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_handle, tempfile.TemporaryFile(
            mode="w+", encoding="utf-8"
        ) as stderr_handle:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Docker runtime not available. Ensure Docker is installed and the runtime image can be built."
                ) from exc
            self._monitor_process(
                process,
                session_id,
                started_at,
                session_dir,
                progress_callback,
                progress_base,
                stdout_handle,
                stderr_handle,
            )


class CodexRunner(DockerRuntimeRunner):
    """Manages Docker-based Codex sessions."""

    @staticmethod
    def _toml_string(value: str) -> str:
        return json.dumps(value)

    @classmethod
    def _toml_list(cls, values: list[str]) -> str:
        return "[" + ", ".join(cls._toml_string(value) for value in values) + "]"

    @classmethod
    def _toml_inline_table(cls, values: dict[str, str]) -> str:
        items = ", ".join(
            f"{cls._toml_string(key)} = {cls._toml_string(value)}"
            for key, value in values.items()
        )
        return "{" + items + "}"

    @staticmethod
    def _provider_key(provider: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", provider.strip().lower())
        return normalized or "openai"

    @classmethod
    def _provider_env_key(cls, provider: str) -> str:
        provider_key = cls._provider_key(provider).upper()
        return f"{provider_key}_API_KEY"

    @classmethod
    def _custom_provider_key(cls, provider: str) -> str:
        provider_key = cls._provider_key(provider)
        if provider_key in {"openai", "oss"}:
            return f"{provider_key}_compatible"
        return provider_key

    @staticmethod
    def _is_oss_local_provider(provider: str) -> bool:
        normalized = provider.strip().lower()
        return normalized in {"lmstudio", "lm_studio", "ollama"}

    @staticmethod
    def _normalize_oss_local_provider(provider: str) -> str:
        normalized = provider.strip().lower()
        if normalized in {"lmstudio", "lm_studio"}:
            return "lmstudio"
        return normalized

    @staticmethod
    def _local_provider_default_port(provider: str) -> int:
        normalized = provider.strip().lower()
        if normalized in {"lmstudio", "lm_studio"}:
            return 1234
        if normalized == "ollama":
            return 11434
        return 1234

    def _local_provider_proxy_target(self, provider: str, base_url: str | None) -> str | None:
        if not base_url or not self._is_oss_local_provider(provider):
            return None

        parsed = urlsplit(base_url)
        hostname = parsed.hostname or ""
        if hostname not in {"localhost", "127.0.0.1"}:
            return base_url

        netloc = parsed.netloc.replace(hostname, "host.docker.internal", 1)
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    def _write_local_provider_proxy(self, codex_home: Path) -> Path:
        proxy_script = codex_home / LOCAL_PROVIDER_PROXY_SCRIPT_NAME
        proxy_script.write_text(LOCAL_PROVIDER_PROXY_SCRIPT + "\n", encoding="utf-8")
        return proxy_script

    @staticmethod
    def _is_likely_incompatible_custom_provider(provider: str) -> bool:
        normalized = provider.strip().lower()
        return normalized in {"llama.cpp", "llama_cpp", "llamacpp"}

    def _write_codex_config(self, codex_home: Path) -> None:
        codex_home.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if self.config.developer_instructions:
            lines.append(
                f"developer_instructions = {self._toml_string(self.config.developer_instructions)}"
            )
        if self.config.model.reasoning_effort:
            lines.append(
                f"model_reasoning_effort = {self._toml_string(self.config.model.reasoning_effort)}"
            )

        for mcp in self.config.mcp_servers:
            server_header = f"[mcp_servers.{self._toml_string(mcp.name)}]"
            lines.extend(["", server_header])
            if mcp.command:
                lines.append(f"command = {self._toml_string(mcp.command)}")
            if mcp.args:
                lines.append(f"args = {self._toml_list(mcp.args)}")
            if mcp.env_vars:
                lines.append(f"env_vars = {self._toml_list(mcp.env_vars)}")
            if mcp.cwd:
                lines.append(f"cwd = {self._toml_string(mcp.cwd)}")
            if mcp.url:
                lines.append(f"url = {self._toml_string(mcp.url)}")
            if mcp.bearer_token_env_var:
                lines.append(
                    f"bearer_token_env_var = {self._toml_string(mcp.bearer_token_env_var)}"
                )
            if mcp.http_headers:
                lines.append(
                    f"http_headers = {self._toml_inline_table(mcp.http_headers)}"
                )
            if mcp.env_http_headers:
                lines.append(
                    f"env_http_headers = {self._toml_inline_table(mcp.env_http_headers)}"
                )
            if mcp.startup_timeout_sec is not None:
                lines.append(f"startup_timeout_sec = {mcp.startup_timeout_sec}")
            if mcp.tool_timeout_sec is not None:
                lines.append(f"tool_timeout_sec = {mcp.tool_timeout_sec}")
            if mcp.enabled is not True:
                lines.append(f"enabled = {str(mcp.enabled).lower()}")
            if mcp.required is True:
                lines.append("required = true")
            if mcp.enabled_tools:
                lines.append(f"enabled_tools = {self._toml_list(mcp.enabled_tools)}")
            if mcp.disabled_tools:
                lines.append(f"disabled_tools = {self._toml_list(mcp.disabled_tools)}")
            if mcp.env:
                lines.append("")
                lines.append(f"[mcp_servers.{self._toml_string(mcp.name)}.env]")
                for key, value in mcp.env.items():
                    lines.append(f"{self._toml_string(key)} = {self._toml_string(value)}")

        config_text = "\n".join(lines).strip()
        (codex_home / "config.toml").write_text(
            (config_text + "\n") if config_text else "",
            encoding="utf-8",
        )

    @staticmethod
    def _list_session_files(codex_home: Path) -> list[Path]:
        sessions_dir = codex_home / "sessions"
        if not sessions_dir.exists():
            return []
        return sorted(path for path in sessions_dir.rglob("*.jsonl") if path.is_file())

    @staticmethod
    def _reasoning_summary_from_content(payload: dict[str, object]) -> list[dict[str, str]]:
        content = payload.get("content")
        if not isinstance(content, list):
            return []
        summary: list[dict[str, str]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "reasoning_text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                summary.append({"type": "summary_text", "text": text.strip()})
        return summary

    @classmethod
    def _normalize_trace_event(cls, event: dict[str, object]) -> dict[str, object]:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return event
        if event.get("type") != "response_item" or payload.get("type") != "reasoning":
            return event
        summary = payload.get("summary")
        if isinstance(summary, list) and any(
            isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
            for item in summary
        ):
            return event
        normalized_event = dict(event)
        normalized_payload = dict(payload)
        normalized_payload["summary"] = cls._reasoning_summary_from_content(normalized_payload)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    @classmethod
    def _copy_normalized_session_file(cls, source_path: Path, destination: Path) -> None:
        normalized_lines: list[str] = []
        with source_path.open("r", encoding="utf-8") as source_handle:
            for raw_line in source_handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                normalized_lines.append(json.dumps(cls._normalize_trace_event(event), separators=(",", ":")))
        destination.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")

    def _build_codex_command(self, prompt: str, workspace: Path, codex_home: Path) -> list[str]:
        api_key = self.config.get_api_key() or ""
        configured_base_url = self.config.get_base_url()
        base_url = configured_base_url
        model = self.config.get_effective_model()
        provider_name = self.config.api.provider
        provider_env_key = self._provider_env_key(provider_name)
        proxy_target = self._local_provider_proxy_target(provider_name, configured_base_url)
        if configured_base_url and not self._is_oss_local_provider(provider_name):
            base_url = self._container_base_url(configured_base_url)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--user",
            "codex",
            "-e",
            f"CODEX_HOME={CODEX_HOME_IN_CONTAINER}",
            "-e",
            "HOME=/home/codex",
            "-v",
            f"{workspace}:{WORKSPACE_IN_CONTAINER}",
            "-v",
            f"{codex_home}:{CODEX_HOME_IN_CONTAINER}",
            "-w",
            WORKSPACE_IN_CONTAINER,
        ]
        if proxy_target or (configured_base_url and base_url != configured_base_url):
            cmd.extend([
                "--add-host",
                "host.docker.internal:host-gateway",
            ])
        if proxy_target:
            cmd.extend(
                [
                    "-e",
                    f"TEICH_LOCAL_PROVIDER_TARGET={proxy_target}",
                    "-e",
                    f"TEICH_LOCAL_PROVIDER_PORT={self._local_provider_default_port(provider_name)}",
                ]
            )
        if api_key:
            cmd.extend(["-e", f"{provider_env_key}={api_key}"])
        codex_cmd = [
            "codex",
            "--ask-for-approval",
            self.config.model.approval_policy,
        ]
        if self._is_oss_local_provider(provider_name):
            codex_cmd.extend(
                [
                    "--oss",
                    "--local-provider",
                    self._normalize_oss_local_provider(provider_name),
                ]
            )
        codex_cmd.extend(
            [
                "exec",
                "--model",
                model,
                "--sandbox",
                self.config.model.sandbox,
                "--skip-git-repo-check",
            ]
        )
        if base_url and not self._is_oss_local_provider(provider_name):
            provider_key = self._custom_provider_key(provider_name)
            provider_literal = (
                "{"
                f"name={self._toml_string(provider_key)},"
                f"base_url={self._toml_string(base_url)},"
                f"env_key={self._toml_string(provider_env_key)},"
                f"wire_api={self._toml_string(self.config.api.wire_api)}"
                "}"
            )
            cmd.extend(
                [
                    "--config",
                    f"model_providers.{provider_key}={provider_literal}",
                    "--config",
                    f"model_provider={self._toml_string(provider_key)}",
                ]
            )
            codex_cmd.extend(cmd[-4:])
            cmd = cmd[:-4]
        codex_cmd.append(prompt)
        cmd.append(self.image_name)
        if proxy_target:
            proxy_script = f"{CODEX_HOME_IN_CONTAINER}/{LOCAL_PROVIDER_PROXY_SCRIPT_NAME}"
            shell_command = (
                f"node {shlex.quote(proxy_script)} >/tmp/local-provider-proxy.log 2>&1 & "
                f"sleep 1 && exec {shlex.join(codex_cmd)}"
            )
            cmd.extend(["bash", "-lc", shell_command])
        else:
            cmd.extend(codex_cmd)
        return cmd

    def _extract_session_file(
        self,
        session_id: str,
        codex_home: Path,
        existing_sessions: set[Path],
        started_at: datetime,
    ) -> Path:
        """Extract a session file from the mounted CODEX_HOME and save it to output."""
        session_files = self._list_session_files(codex_home)
        fresh_files = [path for path in session_files if path.resolve() not in existing_sessions]
        if not fresh_files:
            fresh_files = [
                path
                for path in session_files
                if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= started_at
            ]
        if not fresh_files:
            raise RuntimeError(f"No session file found for {session_id}")
        source_path = max(fresh_files, key=lambda path: path.stat().st_mtime)
        destination = self._resolve_output_path(source_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._copy_normalized_session_file(source_path, destination)
        return destination

    def _resolve_output_path(self, file_name: str) -> Path:
        destination = self.config.output.traces_dir / file_name
        if not destination.exists():
            return destination
        stem = destination.stem
        suffix = destination.suffix
        counter = 1
        while True:
            candidate = destination.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def run_session(
        self,
        prompt: str,
        session_id: str | None = None,
        progress_callback: SessionProgressCallback | None = None,
        progress_base: SessionProgressUpdate | None = None,
        prompt_input: PromptInput | None = None,
    ) -> Path:
        """Run a single codex session and return the exported session file path."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        workspace_root, workspace = self._prepare_workspace(session_id, prompt_input, "codex")
        codex_home = Path(tempfile.mkdtemp(prefix=f"codex-home-{session_id}-"))
        started_at = datetime.now(timezone.utc)

        try:
            self._write_codex_config(codex_home)
            if self._local_provider_proxy_target(self.config.api.provider, self.config.get_base_url()):
                self._write_local_provider_proxy(codex_home)
            existing_sessions = {path.resolve() for path in self._list_session_files(codex_home)}
            cmd = self._build_codex_command(prompt, workspace, codex_home)
            try:
                self._run_process(
                    cmd,
                    session_id,
                    started_at,
                    codex_home / "sessions",
                    progress_callback,
                    progress_base,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Session {session_id[:8]} timed out after {self.config.timeout_seconds}s"
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or "")
                if "'type' of tool must be 'function'" in stderr:
                    stderr = (
                        f"{stderr}\nHint: this custom provider endpoint is not fully compatible with Codex's Responses API tool format. "
                        "Use a provider that supports Codex Responses semantics, or use Codex OSS mode with a supported local provider."
                    )
                raise RuntimeError(f"Session {session_id[:8]} failed: {stderr}")
            trace_path = self._extract_session_file(
                session_id,
                codex_home,
                existing_sessions,
                started_at,
            )
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
            return trace_path
        finally:
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(codex_home, ignore_errors=True)

    def run_all(
        self,
        max_concurrency: int = 1,
        progress_callback: SessionProgressCallback | None = None,
    ) -> list[Path]:
        return super().run_all(max_concurrency=max_concurrency, progress_callback=progress_callback)


class PiRunner(DockerRuntimeRunner):
    """Manages Docker-based Pi sessions."""

    def _runtime_trace_guard_error(self, trace_path: Path) -> str | None:
        try:
            self._normalized_pi_trace_events(trace_path)
        except RuntimeError as exc:
            return str(exc).replace("This trace was not exported because ", "")
        return None

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        return provider.strip().lower()

    def _pi_uses_builtin_provider_override(self) -> bool:
        provider = self._normalize_provider(self.config.api.provider)
        return bool(self.config.get_base_url()) and provider == "openrouter"

    def _pi_provider_name(self) -> str:
        provider = self._normalize_provider(self.config.api.provider)
        if self._pi_uses_builtin_provider_override():
            return provider
        if self.config.get_base_url():
            return f"teich-{provider}"
        return provider

    @staticmethod
    def _normalize_thinking_level(reasoning_effort: str | None) -> str | None:
        if not isinstance(reasoning_effort, str):
            return None
        normalized = reasoning_effort.strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return None

    @staticmethod
    def _list_session_files(session_dir: Path) -> list[Path]:
        if not session_dir.exists():
            return []
        return sorted(path for path in session_dir.rglob("*.jsonl") if path.is_file())

    def _pi_provider_api(self) -> str:
        wire_api = self.config.api.wire_api.strip().lower()
        if wire_api in {"completions", "chat_completions", "chat-completions", "openai-completions"}:
            return "openai-completions"
        return "openai-responses"

    def _pi_model_overrides(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.config.model.pi_model_overrides.items()
            if key != "id"
        }

    def _pi_provider_settings(self) -> dict[str, Any] | None:
        base_url = self._container_base_url(self.config.get_base_url())
        if not base_url:
            return None
        api_key = self.config.get_api_key() or "local"
        provider_settings: dict[str, Any] = {
            "baseUrl": base_url,
            "api": self._pi_provider_api(),
            "authHeader": True,
        }
        model_overrides = self._pi_model_overrides()
        if self._pi_uses_builtin_provider_override():
            if model_overrides:
                provider_settings["modelOverrides"] = {
                    self.config.get_effective_model(): model_overrides
                }
        else:
            model_config: dict[str, Any] = {"id": self.config.get_effective_model()}
            model_config.update(model_overrides)
            provider_settings["models"] = [model_config]
        provider_settings["apiKey"] = api_key
        return provider_settings

    def _project_settings(self) -> dict[str, Any]:
        provider = self._pi_provider_name()
        settings: dict[str, Any] = {
            "defaultProvider": provider,
            "defaultModel": self.config.get_effective_model(),
        }
        thinking_level = self._normalize_thinking_level(self.config.model.reasoning_effort)
        if thinking_level:
            settings["defaultThinkingLevel"] = thinking_level
        return settings

    def _pi_models_config(self) -> dict[str, Any] | None:
        provider_settings = self._pi_provider_settings()
        if not provider_settings:
            return None
        return {"providers": {self._pi_provider_name(): provider_settings}}

    def _write_pi_agent_settings(self, agent_dir: Path) -> None:
        agent_dir.mkdir(parents=True, exist_ok=True)
        settings_file = agent_dir / "settings.json"
        settings_file.write_text(
            json.dumps(self._project_settings(), indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_pi_extension(agent_dir)
        models_config = self._pi_models_config()
        if models_config:
            models_file = agent_dir / "models.json"
            models_file.write_text(
                json.dumps(models_config, indent=2) + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _write_pi_extension(agent_dir: Path) -> None:
        extensions_dir = agent_dir / "extensions"
        extensions_dir.mkdir(parents=True, exist_ok=True)
        extension_file = extensions_dir / "teich_system_prompt.ts"
        extension_file.write_text(PI_SYSTEM_PROMPT_EXTENSION + "\n", encoding="utf-8")

    def _write_pi_project_settings(self, workspace: Path) -> None:
        settings_dir = workspace / ".pi"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_file = settings_dir / "settings.json"
        settings_file.write_text(
            json.dumps(self._project_settings(), indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _resolve_pi_executable() -> str:
        return "@mariozechner/pi-coding-agent"

    def _build_pi_command(
        self,
        prompt: str,
        workspace: Path,
        agent_dir: Path,
        session_dir: Path,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            "codex",
            "-e",
            "HOME=/home/codex",
            "-e",
            f"PI_CODING_AGENT_DIR={PI_AGENT_DIR_IN_CONTAINER}",
            "-v",
            f"{workspace}:{WORKSPACE_IN_CONTAINER}",
            "-v",
            f"{agent_dir}:{PI_AGENT_DIR_IN_CONTAINER}",
            "-v",
            f"{session_dir}:{PI_SESSIONS_DIR_IN_CONTAINER}",
            "-w",
            WORKSPACE_IN_CONTAINER,
        ]
        configured_base_url = self.config.get_base_url()
        if configured_base_url and self._container_base_url(configured_base_url) != configured_base_url:
            command.extend(["--add-host", "host.docker.internal:host-gateway"])
        pi_command = [
            "npx",
            "-y",
            self._resolve_pi_executable(),
            "--mode",
            "json",
            "--session-dir",
            PI_SESSIONS_DIR_IN_CONTAINER,
        ]
        provider = self._pi_provider_name()
        pi_command.extend(
            [
                "--provider",
                provider,
                "--model",
                self.config.get_effective_model(),
            ]
        )
        thinking_level = self._normalize_thinking_level(self.config.model.reasoning_effort)
        if thinking_level:
            pi_command.extend(["--thinking", thinking_level])
        api_key = self.config.get_api_key()
        if api_key and not configured_base_url:
            pi_command.extend(["--api-key", api_key])
        pi_command.append(prompt)
        command.append(self.image_name)
        command.extend(pi_command)
        return command

    @classmethod
    def _normalize_pi_trace_event(cls, event: dict[str, object]) -> dict[str, object]:
        if event.get("type") == "model_change":
            normalized_event = dict(event)
            normalized_event.pop("provider", None)
            return normalized_event
        if event.get("type") != "message":
            return event
        payload = event.get("message")
        if not isinstance(payload, dict) or "provider" not in payload:
            return event
        normalized_event = dict(event)
        normalized_payload = dict(payload)
        normalized_payload.pop("provider", None)
        normalized_event["message"] = normalized_payload
        return normalized_event

    @staticmethod
    def _pi_system_prompt_from_event(event: dict[str, object]) -> str | None:
        if event.get("type") != "custom" or event.get("customType") != PI_SYSTEM_PROMPT_CUSTOM_TYPE:
            return None
        data = event.get("data")
        if not isinstance(data, dict):
            return None
        system_prompt = data.get("systemPrompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return None
        return system_prompt.strip()

    @staticmethod
    def _pi_has_system_message(events: list[dict[str, object]]) -> bool:
        for event in events:
            if event.get("type") != "message":
                continue
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            role = payload.get("role")
            if role in {"system", "developer"}:
                return True
        return False

    @staticmethod
    def _pi_system_message_event(system_prompt: str, timestamp: str | None) -> dict[str, object]:
        event_timestamp = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "type": "message",
            "id": f"system-{uuid.uuid4().hex[:8]}",
            "parentId": None,
            "timestamp": event_timestamp,
            "message": {
                "role": "developer",
                "content": [{"type": "text", "text": system_prompt}],
            },
        }

    @staticmethod
    def _pi_message_text(payload: dict[str, object]) -> str:
        content = payload.get("content")
        if not isinstance(content, list):
            return ""
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                return text
        return ""

    @classmethod
    def _validate_pi_trace_events(cls, events: list[dict[str, object]]) -> None:
        empty_tool_calls = 0
        empty_tool_results = 0
        empty_argument_validation_errors = 0
        for event in events:
            if event.get("type") != "message":
                continue
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            role = payload.get("role")
            if role == "assistant":
                content = payload.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "toolCall":
                            continue
                        tool_id = block.get("id")
                        tool_name = block.get("name")
                        if not isinstance(tool_id, str) or not tool_id.strip():
                            empty_tool_calls += 1
                            continue
                        if not isinstance(tool_name, str) or not tool_name.strip():
                            empty_tool_calls += 1
            elif role == "toolResult":
                tool_name = payload.get("toolName") if isinstance(payload.get("toolName"), str) else ""
                tool_call_id = payload.get("toolCallId") if isinstance(payload.get("toolCallId"), str) else ""
                text = cls._pi_message_text(payload).strip()
                if not tool_name.strip() or not tool_call_id.strip() or text == PI_EMPTY_TOOL_NOT_FOUND_TEXT:
                    empty_tool_results += 1
                if text.startswith("Validation failed for tool ") and "Received arguments:\n{}" in text:
                    empty_argument_validation_errors += 1
        if not empty_tool_calls and not empty_tool_results and not empty_argument_validation_errors:
            return
        raise RuntimeError(
            "Pi session produced malformed tool calls/results "
            f"(empty_tool_calls={empty_tool_calls}, empty_tool_results={empty_tool_results}, "
            f"empty_argument_validation_errors={empty_argument_validation_errors}). "
            "This trace was not exported because the model/provider emitted corrupted tool invocations."
        )

    def _normalized_pi_trace_events(self, source_path: Path) -> list[dict[str, object]]:
        normalized_events: list[dict[str, object]] = []
        system_prompt: str | None = None
        system_prompt_timestamp: str | None = None
        with source_path.open("r", encoding="utf-8") as source_handle:
            for raw_line in source_handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if system_prompt is None:
                    extracted_system_prompt = self._pi_system_prompt_from_event(event)
                    if extracted_system_prompt:
                        system_prompt = extracted_system_prompt
                        timestamp = event.get("timestamp")
                        if isinstance(timestamp, str) and timestamp.strip():
                            system_prompt_timestamp = timestamp.strip()
                        continue
                normalized_events.append(self._normalize_pi_trace_event(event))
        if system_prompt and not self._pi_has_system_message(normalized_events):
            system_message = self._pi_system_message_event(system_prompt, system_prompt_timestamp)
            insert_at = 1 if normalized_events and normalized_events[0].get("type") == "session" else 0
            normalized_events.insert(insert_at, system_message)
        self._validate_pi_trace_events(normalized_events)
        return normalized_events

    def _copy_normalized_session_file(self, source_path: Path, destination: Path) -> None:
        normalized_events = self._normalized_pi_trace_events(source_path)
        destination.write_text(
            "\n".join(json.dumps(event, separators=(",", ":")) for event in normalized_events) + "\n",
            encoding="utf-8",
        )

    def _extract_session_file(self, session_id: str, session_dir: Path, started_at: datetime) -> Path:
        session_files = self._list_session_files(session_dir)
        fresh_files = [
            path
            for path in session_files
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= started_at
        ]
        if not fresh_files:
            fresh_files = session_files
        if not fresh_files:
            raise RuntimeError(f"No Pi session file found for {session_id}")
        source_path = max(fresh_files, key=lambda path: path.stat().st_mtime)
        destination = self._resolve_output_path(source_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._copy_normalized_session_file(source_path, destination)
        return destination

    def _resolve_output_path(self, file_name: str) -> Path:
        destination = self.config.output.traces_dir / file_name
        if not destination.exists():
            return destination
        stem = destination.stem
        suffix = destination.suffix
        counter = 1
        while True:
            candidate = destination.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def run_session(
        self,
        prompt: str,
        session_id: str | None = None,
        progress_callback: SessionProgressCallback | None = None,
        progress_base: SessionProgressUpdate | None = None,
        prompt_input: PromptInput | None = None,
    ) -> Path:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if self.config.mcp_servers:
            raise RuntimeError("Pi runner does not support mcp_servers in v2 yet")

        workspace_root, workspace = self._prepare_workspace(session_id, prompt_input, "pi")
        agent_dir = Path(tempfile.mkdtemp(prefix=f"pi-agent-{session_id}-"))
        session_dir = Path(tempfile.mkdtemp(prefix=f"pi-sessions-{session_id}-"))
        started_at = datetime.now(timezone.utc)
        try:
            self._write_pi_agent_settings(agent_dir)
            self._write_pi_project_settings(workspace)
            command = self._build_pi_command(prompt, workspace, agent_dir, session_dir)
            try:
                self._run_process(
                    command,
                    session_id,
                    started_at,
                    session_dir,
                    progress_callback,
                    progress_base,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Session {session_id[:8]} timed out after {self.config.timeout_seconds}s"
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
                details = stderr.strip() or stdout.strip()
                raise RuntimeError(f"Session {session_id[:8]} failed: {details}")
            trace_path = self._extract_session_file(session_id, session_dir, started_at)
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
            return trace_path
        finally:
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(agent_dir, ignore_errors=True)
            shutil.rmtree(session_dir, ignore_errors=True)

    def run_all(
        self,
        max_concurrency: int = 1,
        progress_callback: SessionProgressCallback | None = None,
    ) -> list[Path]:
        return super().run_all(max_concurrency=max_concurrency, progress_callback=progress_callback)
