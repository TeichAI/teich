"""Background batch generation jobs for Teich Studio."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from ..runner import (
    ChatRunner,
    ClaudeCodeRunner,
    CodexRunner,
    HermesRunner,
    PiRunner,
    SessionProgressUpdate,
    prompt_inputs_for_run,
    unique_prompt_inputs_by_completion_key,
)
from .interactive import EventLog

RUNNER_CLASSES = {
    "codex": CodexRunner,
    "pi": PiRunner,
    "claude": ClaudeCodeRunner,
    "claude-code": ClaudeCodeRunner,
    "claude_code": ClaudeCodeRunner,
    "hermes": HermesRunner,
    "hermes-agent": HermesRunner,
    "hermes_agent": HermesRunner,
    "chat": ChatRunner,
}


class GenerationStopped(RuntimeError):
    """Raised inside runner progress callbacks to prevent queued prompts from starting."""


def _metrics_dict(update: SessionProgressUpdate) -> dict[str, Any] | None:
    metrics = update.metrics
    if metrics is None:
        return None
    return {
        "model": metrics.model,
        "total_tokens": metrics.total_tokens if metrics.has_token_usage else None,
        "total_cost": metrics.total_cost if metrics.has_cost else None,
    }


class GenerationJob:
    """One batch generation run executing in a background thread."""

    def __init__(self, config: Config, *, resume: bool = False):
        self.id = str(uuid.uuid4())
        self.config = config
        self.resume = resume
        self.status = "starting"
        self.error: str | None = None
        self.events = EventLog()
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.result_files: list[str] = []
        self._prompt_states: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._runner: Any = None
        self._stop_requested = False

    def start(self) -> None:
        threading.Thread(target=self._run, name=f"studio-generate-{self.id[:8]}", daemon=True).start()

    def _emit_status(self, status: str, message: str | None = None) -> None:
        self.status = status
        event: dict[str, Any] = {"kind": "job_status", "status": status}
        if message:
            event["text"] = message
        if status in {"completed", "failed", "stopped"}:
            event["result_files"] = self.result_files
            if self.error:
                event["error"] = self.error
        self.events.append(event)

    def _progress(self, update: SessionProgressUpdate) -> None:
        if self._should_stop() and update.status in {"queued", "running"}:
            raise GenerationStopped("Run stopped.")
        with self._lock:
            previous = self._prompt_states.get(update.prompt_id, {})
            state = {
                "prompt_id": update.prompt_id,
                "prompt_index": update.prompt_index,
                "total_prompts": update.total_prompts,
                "prompt_preview": update.prompt_preview,
                "status": update.status,
                "error": update.error,
                "details": update.details,
                "trace": update.trace_path.name if update.trace_path else previous.get("trace"),
                "metrics": _metrics_dict(update) or previous.get("metrics"),
            }
            self._prompt_states[update.prompt_id] = state
        self.events.append({"kind": "prompt_update", **state})

    def _run(self) -> None:
        try:
            provider = self.config.get_agent_provider()
            runner_cls = RUNNER_CLASSES.get(provider)
            if runner_cls is None:
                raise RuntimeError(
                    f"Unsupported agent provider: {provider}. "
                    "Supported providers: codex, pi, claude-code, hermes, chat."
                )
            prompt_inputs = unique_prompt_inputs_by_completion_key(self.config.get_prompt_inputs())
            if not prompt_inputs:
                raise RuntimeError("No prompts configured. Add prompts before generating.")
            self.config.output.traces_dir.mkdir(parents=True, exist_ok=True)
            if self.resume:
                prompt_inputs = prompt_inputs_for_run(
                    prompt_inputs,
                    self.config.output.traces_dir,
                    resume=True,
                    excluded_dirs=[self.config.output.failures_dir],
                )
                if not prompt_inputs:
                    self._emit_status("completed", "All prompts already have completed outputs.")
                    self._finish()
                    return
            if provider != "chat":
                self._emit_status("starting", "Preparing Docker runtime (first run may build the image)…")
            self._runner = runner_cls(self.config)
            if self._stop_requested:
                self._emit_status("stopped", "Run cancelled before it started.")
                self._finish()
                return
            self._emit_status(
                "running",
                f"Running {len(prompt_inputs)} prompt(s) with concurrency {self.config.max_concurrency}.",
            )
            results = self._runner.run_all(
                max_concurrency=self.config.max_concurrency,
                progress_callback=self._progress,
                prompt_inputs=prompt_inputs,
                resume=self.resume,
            )
            self.result_files = [path.name for path in results if path]
            self._write_readme()
            if self._stop_requested:
                self._emit_status("stopped", "Run stopped. Completed traces were kept.")
            else:
                self._emit_status("completed", f"Generated {len(self.result_files)} trace file(s).")
        except GenerationStopped:
            self.error = None
            self._write_readme()
            self._emit_status("stopped", "Run stopped. Completed traces were kept.")
        except Exception as exc:
            self._write_readme()
            if self._should_stop():
                self.error = None
                self._emit_status("stopped", "Run stopped. Completed traces were kept.")
            else:
                self.error = str(exc)
                self._emit_status("failed", str(exc))
        finally:
            self._finish()

    def _finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc)
        self.events.close()

    def _write_readme(self) -> None:
        try:
            from ..tool_schema import snapshot_configured_tools
            from ..trace_readme import write_traces_readme

            has_outputs = any(
                path.is_file() and path.stat().st_size > 0
                for path in self.config.output.traces_dir.rglob("*.jsonl")
            ) if self.config.output.traces_dir.exists() else False
            if not has_outputs:
                return
            write_traces_readme(
                self.config.output.traces_dir,
                pretty_name=self.config.output.pretty_name,
                tags=self.config.get_dataset_tags(),
                model_id=self.config.model.model,
                repo_id=self.config.get_publish_repo_id(),
                tools=snapshot_configured_tools(self.config),
                excluded_dirs=[self.config.output.failures_dir],
                license=self.config.output.license,
                card_extra=self.config.output.card_extra,
                readme_template=self.config.output.readme_template,
            )
        except Exception:
            pass

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
        runner = self._runner
        if runner is not None:
            try:
                runner._terminate_active_processes()
            except Exception:
                pass

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            prompts = sorted(self._prompt_states.values(), key=lambda item: item.get("prompt_index") or 0)
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "resume": self.resume,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result_files": self.result_files,
            "prompts": prompts,
        }


class GenerationManager:
    """Allows one active generation job at a time, keeping recent history."""

    def __init__(self) -> None:
        self._jobs: dict[str, GenerationJob] = {}
        self._current: GenerationJob | None = None
        self._lock = threading.Lock()

    def start(self, config: Config, *, resume: bool = False) -> GenerationJob:
        with self._lock:
            if self._current is not None and self._current.status in {"starting", "running"}:
                raise RuntimeError("A generation run is already in progress")
            job = GenerationJob(config, resume=resume)
            self._jobs[job.id] = job
            self._current = job
        job.start()
        return job

    def current(self) -> GenerationJob | None:
        with self._lock:
            return self._current

    def get(self, job_id: str) -> GenerationJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def shutdown(self) -> None:
        with self._lock:
            job = self._current
        if job is not None and job.status in {"starting", "running"}:
            job.stop()


def _now() -> float:
    return time.time()
