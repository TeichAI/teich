"""Pluggable bench-backend abstraction.

A benchmark *source* (declared in ``bench.sources`` with a ``type``) is executed by a
backend (harbor, swe-bench). Each backend turns a task into a ``BenchRun`` — the agent's
plain native trace plus a rewards dict — and the *harvest* here (route by score into
``passed``/``failed``/``borderline`` + a per-task ``metadata/`` sidecar) is backend-agnostic.

Trace/metadata filenames are namespaced by source (``bench-<source>-<task>``) so two sources
in one dataset can never collide; an unfinished task writes nothing and is retried on resume.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol

from ...converter import convert_traces_to_training_data

if TYPE_CHECKING:
    from ...config import BenchSource, Config

BENCH_SPLITS = ("passed", "failed", "borderline")


@dataclass
class BenchTask:
    """One unit of work within a source (a harbor task dir, a swe-bench instance, ...)."""

    id: str
    raw: Any = None


@dataclass
class BenchRun:
    """A backend's result for one task: the native trace + scores + provenance metadata."""

    native_lines: list[str]
    rewards: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchBackend(Protocol):
    """What a benchmark backend must provide. The driver + harvest are shared."""

    type: str

    def require(self) -> None:
        """Raise RuntimeError with an install hint if the backend's extra is missing."""

    def tasks(self, cfg: Config, source: BenchSource, *, refresh: bool = False) -> Iterable[BenchTask]:
        """Resolve a source into its tasks/instances (``refresh`` re-fetches remote data)."""

    def run(self, cfg: Config, source: BenchSource, task: BenchTask) -> BenchRun:
        """Run the agent on one task in its environment and return its trace + rewards."""


def bench_root(cfg: Config) -> Path:
    """Working dir for backends' intermediates (downloads, sessions, trials).

    A ``bench`` dir beside ``traces_dir`` (parallel to sandbox/failures, outside the
    dataset); overridable via ``output.bench_dir``. Re-checked here (not just at config
    load) so a ``--output`` override can't leave ``bench_dir`` inside the dataset.
    """
    root = Path(cfg.output.bench_dir) if cfg.output.bench_dir is not None else cfg.output.traces_dir.parent / "bench"
    # Guard both the explicit override and the computed default: a bench root at/under traces_dir
    # (e.g. output.traces_dir named ``bench``, so the sibling default collides with it) would get
    # raw trials/sessions uploaded and misclassified as dataset rows.
    traces = cfg.output.traces_dir.resolve()
    if traces == root.resolve() or traces in root.resolve().parents:
        raise RuntimeError(
            f"bench working dir ({root}) must be outside output.traces_dir "
            f"({cfg.output.traces_dir}); raw trials/sessions there would be uploaded and "
            "misclassified as dataset rows. Set output.bench_dir explicitly or rename traces_dir."
        )
    return root


def slug(value: str) -> str:
    """Filesystem-safe slug (``terminal-bench@2.0`` -> ``terminal-bench-2.0``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "x"


def _instances_key(instances: list[str] | None) -> str | None:
    """A bounded, order-independent discriminator for an ``instances`` subset.

    The full comma-joined list would otherwise flow through ``source_id`` into per-task
    filenames, Docker tags, and container/env-file names; a dozen ~22-char SWE-bench IDs
    blow past the 255-byte path-component and 128-char Docker-tag limits and crash the run
    before any result is written. Short lists stay readable; longer ones collapse to a
    stable hash (sorted, so listing order doesn't spawn a distinct id).
    """
    if not instances:
        return None
    joined = ",".join(sorted(instances))
    if len(joined) <= 40:
        return joined
    return "i" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def source_id(source: BenchSource) -> str:
    """Stable identifier for a source, used to namespace its output files.

    Keyed on the discriminating knobs (not just ``source``): two sources that share a spec
    but differ in ``repo``/``version``/``split``/``instances``/``backend`` get distinct ids,
    so they can't overwrite each other's traces or wrongly resume-skip. A suffix is appended
    only when a field diverges from its default, so the common single-field source keeps a
    clean, stable id. ``type`` is intentionally excluded (it is recorded in ``metadata`` and
    the existing namespacing contract is type-independent).
    """
    extras = [
        slug(part)
        for part in (
            source.repo,
            source.version,
            source.split,
            _instances_key(source.instances),
            source.backend if source.backend != "docker" else None,
        )
        if part
    ]
    base_id = slug(source.source)
    return base_id if not extras else f"{base_id}-{slug('-'.join(extras))}"


def bench_stem(source: BenchSource, task_id: str) -> str:
    """Per-task dataset stem, namespaced by source (the co-mingle guard keys on ``bench-``)."""
    return f"bench-{source_id(source)}-{slug(task_id)}"


def existing_output(cfg: Config, stem: str) -> Path | None:
    """The harvested trace for ``stem`` (in any split), or None — used for ``--resume``."""
    for split in BENCH_SPLITS:
        path = cfg.output.traces_dir / split / f"{stem}.jsonl"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def numeric(value: Any) -> float | None:
    """A real number (bools excluded) as float, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def rewards_from_mapping(data: Any) -> dict[str, float] | None:
    """The full dict of numeric scores (no clamping). Accepts ``{"rewards": {...}}`` or flat."""
    if not isinstance(data, dict):
        return None
    rewards = data.get("rewards") if isinstance(data.get("rewards"), dict) else data
    scores = {key: numeric(val) for key, val in rewards.items() if numeric(val) is not None}
    return scores or None


def primary_score(rewards: dict[str, float] | None) -> float | None:
    """The scalar used for routing: the ``reward`` key, else the first numeric value."""
    if not rewards:
        return None
    primary = numeric(rewards.get("reward"))
    if primary is not None:
        return primary
    for value in rewards.values():
        primary = numeric(value)
        if primary is not None:
            return primary
    return None


def route_split(primary: float | None) -> str:
    """passed = score 1, failed = score 0 / unscored, borderline = any other value."""
    if primary is None or primary == 0:
        return "failed"
    if primary == 1:
        return "passed"
    return "borderline"


def trace_metadata(native_dir: Path | None) -> dict[str, Any]:
    """Metadata teich's converter recovers from a native trace (model, session, cwd, ...)."""
    if native_dir is None:
        return {}
    try:
        rows = convert_traces_to_training_data(native_dir)
    except Exception:  # metadata is best-effort; never fail the harvest over it
        return {}
    return rows[0].get("metadata", {}) if rows else {}


def harvest(cfg: Config, source: BenchSource, task: BenchTask, run: BenchRun) -> tuple[list[Path], str]:
    """Write a backend's native trace (routed by score) + a per-task metadata sidecar.

    Returns (written paths, split). The trace is written verbatim (plain agent-trace data);
    scores + provenance go to ``output/metadata/<stem>.json``.
    """
    primary = primary_score(run.rewards)
    split = route_split(primary)
    stem = bench_stem(source, task.id)

    trace_path = cfg.output.traces_dir / split / f"{stem}.jsonl"
    # A re-run (without --resume) whose score crosses a routing boundary would otherwise leave a
    # stale copy in the old split; the dataset scanners read every split, so a task would appear
    # twice with contradictory labels. Drop any sibling copy before writing the new one.
    for other in BENCH_SPLITS:
        if other != split:
            (cfg.output.traces_dir / other / f"{stem}.jsonl").unlink(missing_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("\n".join(run.native_lines) + "\n", encoding="utf-8")

    metadata: dict[str, Any] = {
        "task": task.id,
        "source": source_id(source),
        "type": source.type,
        "split": split,
        "reward": primary,
        "rewards": run.rewards or {},
        "agent": cfg.get_agent_provider(),
        "trace_file": f"{split}/{stem}.jsonl",
        **run.metadata,
    }
    meta_path = cfg.output.traces_dir / "metadata" / f"{stem}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return [trace_path], split
