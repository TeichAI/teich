"""Background extraction jobs for Teich Studio."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..anonymize import anonymize_path
from ..config import Config
from ..extract import CURSOR_EXTRACTION_NOTICE, ExtractProvider, extract_local_sessions
from ..trace_readme import write_traces_readme
from .interactive import EventLog


def _extract_dataset_config(
    provider: ExtractProvider,
    output: Path,
    *,
    model_filter: str | None = None,
) -> Config:
    model_id = model_filter or f"{provider}-local-sessions"
    pretty_model = model_filter or provider
    return Config.model_validate(
        {
            "agent": {"provider": provider},
            "model": {"model": model_id},
            "output": {
                "traces_dir": output,
                "pretty_name": f"{pretty_model} Agent Traces",
            },
        }
    )


class ExtractionJob:
    """One local-session extraction run executing in a background thread."""

    def __init__(
        self,
        provider: ExtractProvider,
        *,
        output_dir: Path,
        source_paths: list[Path] | None = None,
        model_filter: str | None = None,
        skip_anonymize: bool = False,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.provider = provider
        self.output_dir = output_dir
        self.source_paths = source_paths or []
        self.model_filter = model_filter
        self.skip_anonymize = skip_anonymize
        self.status = "starting"
        self.error: str | None = None
        self.events = EventLog()
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.result_files: list[str] = []
        self.result_rows = 0
        self.detected_sources: list[str] = []
        self.anonymize_totals: dict[str, int] | None = None
        self._lock = threading.RLock()

    def start(self) -> None:
        threading.Thread(target=self._run, name=f"studio-extract-{self.id[:8]}", daemon=True).start()

    def _emit_status(self, status: str, message: str | None = None) -> None:
        with self._lock:
            self.status = status
            event: dict[str, Any] = {"kind": "extract_status", "status": status}
            if message:
                event["text"] = message
            if status in {"completed", "failed"}:
                event["result_files"] = self.result_files
                event["result_rows"] = self.result_rows
                event["sources"] = self.detected_sources
                event["anonymize_totals"] = self.anonymize_totals
                if self.error:
                    event["error"] = self.error
            self.events.append(event)

    def _run(self) -> None:
        try:
            self._emit_status("running", f"Extracting local {self.provider} sessions...")
            if self.provider == "cursor":
                self.events.append({"kind": "extract_warning", "text": CURSOR_EXTRACTION_NOTICE})
            last_progress_at = 0.0

            def emit_progress(event: dict[str, Any]) -> None:
                nonlocal last_progress_at
                now = time.monotonic()
                if now - last_progress_at < 2.0:
                    return
                last_progress_at = now
                self.events.append(event)

            result = extract_local_sessions(
                self.provider,
                output_dir=self.output_dir,
                sources=self.source_paths or None,
                model_filter=self.model_filter,
                clear_destination=True,
                progress=emit_progress,
            )
            self.detected_sources = [str(path) for path in result.source_paths]
            self.result_files = [path.name for path in result.copied_files]
            self.result_rows = _jsonl_row_count(result.copied_files)
            if not result.source_paths:
                raise RuntimeError(
                    f"No local {self.provider} session folders found. "
                    "Choose one or more source folders/files in the extraction form."
                )
            if not result.copied_files:
                model_clause = f" matching model '{self.model_filter}'" if self.model_filter else ""
                raise RuntimeError(
                    f"Found {len(result.source_paths)} source path(s), but no "
                    f"{self.provider} sessions{model_clause} were extracted."
                )
            self.events.append(
                {
                    "kind": "extract_files",
                    "text": f"Copied {result.count} trace file(s) containing {self.result_rows} row(s).",
                    "files": self.result_files,
                    "rows": self.result_rows,
                    "sources": self.detected_sources,
                }
            )
            if self.skip_anonymize:
                self.events.append(
                    {
                        "kind": "extract_warning",
                        "text": "Skipped anonymization. Review the output before sharing or uploading it.",
                    }
                )
            else:
                self._emit_status("running", "Anonymizing staged traces...")
                report = anonymize_path(self.output_dir, self.output_dir, in_place=True)
                self.anonymize_totals = report.totals
                self.events.append(
                    {
                        "kind": "extract_anonymize",
                        "text": (
                            "Scrambled "
                            f"{self.anonymize_totals.get('api_key', 0)} API keys, "
                            f"{self.anonymize_totals.get('email', 0)} emails, and "
                            f"{self.anonymize_totals.get('username', 0)} usernames."
                        ),
                        "totals": self.anonymize_totals,
                    }
                )
            self._write_readme()
            self._emit_status(
                "completed",
                f"Extracted {self.result_rows} trace row(s) across {len(self.result_files)} file(s).",
            )
        except Exception as exc:
            self.error = str(exc)
            self._emit_status("failed", self.error)
        finally:
            self.finished_at = datetime.now(timezone.utc)
            self.events.close()

    def _write_readme(self) -> None:
        cfg = _extract_dataset_config(self.provider, self.output_dir, model_filter=self.model_filter)
        readme_path = write_traces_readme(
            cfg.output.traces_dir,
            pretty_name=cfg.output.pretty_name,
            tags=cfg.get_dataset_tags(),
            model_id=cfg.model.model,
            repo_id=cfg.get_publish_repo_id(),
        )
        self.events.append({"kind": "extract_readme", "text": f"Wrote {readme_path.name}.", "path": str(readme_path)})

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "provider": self.provider,
                "status": self.status,
                "error": self.error,
                "output_dir": str(self.output_dir),
                "sources": self.detected_sources or [str(path) for path in self.source_paths],
                "model_filter": self.model_filter,
                "skip_anonymize": self.skip_anonymize,
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "result_files": self.result_files,
                "result_rows": self.result_rows,
                "anonymize_totals": self.anonymize_totals,
            }


class ExtractionManager:
    """Allows one active extraction job at a time, keeping recent history."""

    def __init__(self) -> None:
        self._jobs: dict[str, ExtractionJob] = {}
        self._current: ExtractionJob | None = None
        self._lock = threading.Lock()

    def start(
        self,
        provider: ExtractProvider,
        *,
        output_dir: Path,
        source_paths: list[Path] | None = None,
        model_filter: str | None = None,
        skip_anonymize: bool = False,
    ) -> ExtractionJob:
        with self._lock:
            if self._current is not None and self._current.status in {"starting", "running"}:
                raise RuntimeError("An extraction run is already in progress")
            job = ExtractionJob(
                provider,
                output_dir=output_dir,
                source_paths=source_paths,
                model_filter=model_filter,
                skip_anonymize=skip_anonymize,
            )
            self._jobs[job.id] = job
            self._current = job
        job.start()
        return job

    def current(self) -> ExtractionJob | None:
        with self._lock:
            return self._current

    def get(self, job_id: str) -> ExtractionJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def shutdown(self) -> None:
        pass


def _jsonl_row_count(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                count += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return count
