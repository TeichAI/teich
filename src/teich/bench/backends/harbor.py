"""Harbor bench backend.

Drives the optional ``harbor`` package over Harbor-format tasks: resolve a source
(local dir, registry spec, or git/HF registry) into task dirs, run each in its own
environment image via harbor's built-in agent, and return the native trace + the task
verifier's reward dict. harbor is imported lazily (the ``teich[harbor]`` extra, py>=3.12).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import agent_cfg
from . import base

if TYPE_CHECKING:
    from ...config import BenchSource, Config

HARBOR_INSTALL_HINT = (
    "The harbor bench backend needs the optional 'harbor' extra: install with "
    "`pip install 'teich[harbor]'` (requires Python 3.12+)."
)

# teich agent provider -> harbor AgentName value.
_PROVIDER_TO_AGENT: dict[str, str] = {
    "codex": "codex",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "claude_code": "claude-code",
    "pi": "pi",
    "hermes": "hermes",
}


def _agent_name_for(provider: str) -> str:
    name = _PROVIDER_TO_AGENT.get(provider.strip().lower())
    if not name:
        raise RuntimeError(
            f"Bench mode does not support agent provider {provider!r}; "
            "use one of: codex, claude-code, pi, hermes."
        )
    return name


def _agent_auth_env(cfg: Config) -> dict[str, str]:
    """Model credentials + base URL for the in-container agent — shared ``agent_cfg`` derivation
    (the same one prompt mode uses), so every provider gets the right key var (not just
    openai/openrouter/anthropic) and a host-local base_url reaches the container."""
    return agent_cfg.bench_auth_env(cfg)


def _bench_model_name(cfg: Config) -> str:
    """Model name for the in-container agent; pi requires a ``<provider>/<model>`` prefix so it
    splits on the first ``/`` to pick the credential env var (shared ``agent_cfg`` helper)."""
    return agent_cfg.pi_prefixed_model(cfg)


def _resolve_task_dirs(root: str | Path) -> list[Path]:
    """Resolve a local Harbor task root to one or more task dirs (single-task or dir-of-tasks)."""
    path = Path(root).expanduser()
    if not path.exists():
        raise RuntimeError(f"bench task directory not found: {path}.")
    if (path / "task.toml").is_file():
        return [path]
    tasks = sorted(d for d in path.iterdir() if d.is_dir() and (d / "task.toml").is_file())
    if not tasks:
        raise RuntimeError(f"No Harbor tasks (a task.toml) found under {path}.")
    return tasks


def _classify_remote_source(source: str, repo: str | None, version: str | None) -> tuple[str, str]:
    """Map a remote source to (client kind, harbor dataset ref), mirroring ``harbor download``."""
    has_version = "@" in source
    name = source.split("@", 1)[0]

    def _ref(default_version: str | None = None) -> str:
        if has_version:
            return source
        if version:
            return f"{source}@{version}"
        return f"{source}@{default_version}" if default_version else source

    if repo:
        return "repo", _ref()
    if "/" in name:
        return "package", _ref(default_version="latest")
    return "registry", _ref()


def _source_slug(source: str, version: str | None, repo: str | None = None) -> str:
    raw = source if "@" in source else (f"{source}@{version}" if version else source)
    # ``repo`` selects the download origin in ``_download_async`` and is a discriminating
    # knob in ``source_id``; fold it into the cache key too, else two sources with the same
    # spec/version but different repos share a cache dir and the second reuses the first's tasks.
    if repo:
        raw = f"{repo}#{raw}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-") or "source"


def _task_root(cache_dir: Path) -> Path | None:
    """The common parent of every exported ``task.toml`` (harbor exports ``<ds>/<task>/``)."""
    parents = sorted({toml.parent for toml in cache_dir.rglob("task.toml")})
    if not parents:
        return None
    if len(parents) == 1:
        return parents[0]
    return Path(os.path.commonpath([str(p) for p in parents]))


async def _download_async(source: BenchSource, cache_dir: Path) -> None:
    kind, ref = _classify_remote_source(source.source.strip(), source.repo, source.version)
    if kind == "repo":
        from harbor.registry.client.factory import RegistryClientFactory

        client = RegistryClientFactory.create(repo=source.repo)
    elif kind == "package":
        from harbor.registry.client.package import PackageDatasetClient

        client = PackageDatasetClient()
    else:
        from harbor.registry.client.factory import RegistryClientFactory

        client = RegistryClientFactory.create()
    cache_dir.mkdir(parents=True, exist_ok=True)
    await client.download_dataset(ref, overwrite=True, output_dir=cache_dir, export=True)


def _fetch_remote(source: BenchSource, cache_dir: Path) -> None:
    try:
        asyncio.run(_download_async(source, cache_dir))
    except Exception as exc:  # network / unknown dataset / auth -> a clean bench error
        raise RuntimeError(
            f"Failed to download bench source {source.source!r}: {type(exc).__name__}: {exc}. "
            "Check the spec/version, network, and (for HF/private registries) HF_TOKEN."
        ) from exc


def _resolve_source(cfg: Config, source: BenchSource, *, refresh: bool) -> Path:
    """Return a local task root: a local path as-is, else download the remote spec."""
    spec = source.source.strip()
    local = Path(spec).expanduser()
    if local.exists():
        return local
    cache_dir = base.bench_root(cfg) / "sources" / _source_slug(spec, source.version, source.repo)
    root = None if refresh else _task_root(cache_dir)
    if root is None:
        if refresh and cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        _fetch_remote(source, cache_dir)
        root = _task_root(cache_dir)
    if root is None:
        raise RuntimeError(f"bench source {spec!r}: no Harbor tasks found after download into {cache_dir}.")
    return root


def _build_trial_config(cfg: Config, source: BenchSource, task_dir: Path, trials_dir: Path) -> Any:
    from harbor.models.agent.name import AgentName
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import TaskConfig, TrialConfig

    config = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials_dir)
    config.agent.name = AgentName(_agent_name_for(cfg.get_agent_provider()))
    model = _bench_model_name(cfg)
    if model:
        config.agent.model_name = model
    config.agent.env.update(_agent_auth_env(cfg))
    try:
        config.environment.type = EnvironmentType(source.backend)
    except ValueError as exc:
        supported = ", ".join(t.value for t in EnvironmentType)
        raise RuntimeError(
            f"Unknown backend {source.backend!r}; harbor supports: {supported}."
        ) from exc
    return config


async def _create_and_run(config: Any) -> tuple[Any, Any]:
    from harbor.trial.trial import Trial

    trial = await Trial.create(config)
    result = await trial.run()
    return trial, result


def _agent_dir(trial: Any) -> Path | None:
    paths = getattr(trial, "paths", None)
    agent_dir = getattr(paths, "agent_dir", None)
    return Path(agent_dir) if agent_dir else None


def _pi_stream_to_session_events(pi_txt: Path) -> list[dict[str, Any]]:
    """Normalize pi's ``--mode json`` stream into pi session events.

    harbor runs pi with ``--no-session`` (only a streaming log + a leading non-JSON
    ``Warning:`` line). Keep the ``session`` header, turn each ``message_end`` into a
    ``message``, and synthesize a ``model_change`` from the first assistant message.
    """
    events: list[dict[str, Any]] = []
    model_change_added = False
    for raw in pi_txt.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session":
            events.append(event)
        elif event_type == "message_end":
            message = event.get("message")
            if not isinstance(message, dict) or not message.get("role"):
                continue
            if not model_change_added and message.get("role") == "assistant":
                provider, model_id = message.get("provider"), message.get("model")
                if isinstance(provider, str) and isinstance(model_id, str):
                    events.append({"type": "model_change", "provider": provider, "modelId": model_id})
                    model_change_added = True
            events.append({"type": "message", "message": message})
    return events


def _native_trace(
    cfg: Config, source: BenchSource, agent_dir: Path, task_id: str
) -> tuple[list[str], Path | None]:
    """The agent's plain native trace as JSONL lines + an isolated dir (for metadata recovery)."""
    sessions = agent_dir / "sessions"  # codex / claude-code export a native session dir
    if sessions.is_dir() and any(sessions.glob("*.jsonl")):
        lines = [
            line
            for path in sorted(sessions.glob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return lines, sessions
    pi_txt = agent_dir / "pi.txt"  # pi runs --no-session: normalize its --mode json stream
    if pi_txt.is_file():
        events = _pi_stream_to_session_events(pi_txt)
        if events:
            # Namespace by source so concurrent same-named tasks from different sources can't race.
            norm_dir = base.bench_root(cfg) / "sessions" / base.bench_stem(source, task_id)
            norm_dir.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(event, ensure_ascii=False) for event in events]
            (norm_dir / "pi.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            return lines, norm_dir
    jsonls = sorted(agent_dir.rglob("*.jsonl"))  # last resort: anything the agent left
    if jsonls:
        lines = [
            line
            for path in jsonls
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return lines, agent_dir
    return [], None


def _rewards_from_result(result: Any) -> dict[str, float] | None:
    verifier = getattr(result, "verifier_result", None)
    if isinstance(verifier, dict):
        return base.rewards_from_mapping(verifier)
    return base.rewards_from_mapping(getattr(verifier, "rewards", None))


def _rewards_from_files(base_dir: Path | None) -> dict[str, float] | None:
    if base_dir is None:
        return None
    for name in ("rewards.json", "reward.json"):
        for path in sorted(base_dir.rglob(name)):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scores = base.rewards_from_mapping(data)
            if scores is not None:
                return scores
    for path in sorted(base_dir.rglob("reward.txt")):
        try:
            value = base.numeric(float(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            continue
        if value is not None:
            return {"reward": value}
    return None


def _sanitize_hb_image(name: str) -> str:
    """Replicate harbor's docker image-name sanitizer (environments/docker/docker.py)."""
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9._-]", "-", name)


def _task_prebuilt_images(task_dir: Path | None) -> list[str]:
    """The task's prebuilt image name(s) from task.toml: ``[environment].docker_image``
    plus a separate ``[verifier.environment].docker_image`` when declared.

    A task that declares one runs in harbor's prebuilt mode: compose pulls that image and
    runs it directly — no ``hb__<task>`` image is ever built — and harbor's teardown
    (``down --rmi local``) never removes a pulled tagged image. All terminal-bench 2.0
    tasks are prebuilt, so without these names the purge is a no-op and each task leaks
    its (often multi-GB) image.
    """
    if task_dir is None:
        return []
    try:
        import tomllib  # py3.11+; the harbor backend already requires 3.12

        with (Path(task_dir) / "task.toml").open("rb") as fh:
            data = tomllib.load(fh)
        candidates = [
            data.get("environment", {}).get("docker_image"),
            data.get("verifier", {}).get("environment", {}).get("docker_image"),
        ]
    except Exception:  # missing/unparseable task.toml — stay best-effort like the purge itself
        return []
    return [image for image in candidates if isinstance(image, str) and image.strip()]


def _harbor_image_names(trial: Any, task_id: str, task_dir: Path | None = None) -> list[str]:
    """Per-task image name(s) harbor leaves behind after ``compose down --rmi local``:
    the custom-tagged ``hb__<task>`` image (built mode) and/or the pulled prebuilt
    image(s) declared in task.toml, so we remove them ourselves.

    Prefers the actual name off the trial's environments; always includes the deterministic
    ``sanitize("hb__" + short_name)`` (short_name == the task dir name == task_id) and the
    task.toml prebuilt names, which also cover the error/interrupt path where the trial
    object was never returned.
    """
    names: set[str] = set()
    for attr in ("agent_environment", "verifier_environment"):
        env = getattr(trial, attr, None) if trial is not None else None
        try:
            value = getattr(env, "_main_image_name", None)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            names.add(value)
    short = getattr(getattr(trial, "task", None), "short_name", None) if trial is not None else None
    names.add(_sanitize_hb_image(f"hb__{short or task_id}"))
    names.update(_task_prebuilt_images(task_dir))
    return sorted(names)


def _purge_images(names: list[str]) -> None:
    """Best-effort ``docker rmi -f`` of the given images (ignore not-found / in-use / no docker)."""
    for name in names:
        try:
            # Short timeout: this also runs in the finally after a Ctrl-C, so a slow/unhappy
            # docker must not stall the exit for 2 minutes per image.
            subprocess.run(["docker", "rmi", "-f", name], capture_output=True, timeout=30)
        except Exception:
            pass


class HarborBackend:
    """Runs Harbor-format tasks via the harbor package (one container per task)."""

    type = "harbor"

    def __init__(self) -> None:
        # Prebuilt images declared by more than one task of this source (see tasks()).
        # rmi'ing a shared image after one task can race a concurrent sibling's
        # `compose up` (No such image -> that task fails) and forces a re-pull per task,
        # so those are kept; only images unique to a task are purged after it.
        self._shared_prebuilt: frozenset[str] = frozenset()

    def require(self) -> None:
        if sys.version_info < (3, 12):
            raise RuntimeError(
                "The harbor bench backend requires Python 3.12+ "
                f"(current: {sys.version_info.major}.{sys.version_info.minor}). {HARBOR_INSTALL_HINT}"
            )
        try:
            import harbor  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise RuntimeError(HARBOR_INSTALL_HINT) from exc

    def tasks(self, cfg: Config, source: BenchSource, *, refresh: bool = False) -> list[base.BenchTask]:
        root = _resolve_source(cfg, source, refresh=refresh)
        task_dirs = _resolve_task_dirs(root)
        # Computed once, before any run() is dispatched (runner threads only call run()).
        # Deduped per task: one task using the same image for agent + verifier isn't sharing.
        counts = Counter(image for task_dir in task_dirs for image in set(_task_prebuilt_images(task_dir)))
        self._shared_prebuilt = frozenset(image for image, n in counts.items() if n > 1)
        return [base.BenchTask(id=task_dir.name, raw=task_dir) for task_dir in task_dirs]

    def run(self, cfg: Config, source: BenchSource, task: base.BenchTask) -> base.BenchRun:
        trials_dir = base.bench_root(cfg) / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        config = _build_trial_config(cfg, source, task.raw, trials_dir)
        trial: Any = None
        try:
            trial, result = asyncio.run(_create_and_run(config))

            exc_info = getattr(result, "exception_info", None)
            exception: str | None = None
            if exc_info:
                exception = (
                    exc_info.get("exception_type") if isinstance(exc_info, dict)
                    else getattr(exc_info, "exception_type", None)
                ) or "agent error"

            agent_dir = _agent_dir(trial)
            lines: list[str] = []
            native_dir: Path | None = None
            if agent_dir is not None and agent_dir.exists():
                lines, native_dir = _native_trace(cfg, source, agent_dir, task.id)

            rewards = _rewards_from_result(result) or _rewards_from_files(
                agent_dir.parent if agent_dir else None
            )
            metadata = {"exception": exception, **base.trace_metadata(native_dir)}
            return base.BenchRun(native_lines=lines, rewards=rewards, metadata=metadata)
        finally:
            # harbor's `down --rmi local` leaves the `hb__<task>` image (built mode) and the
            # pulled task image(s) (prebuilt mode); remove them on every outcome
            # (passed/failed/error/Ctrl-C) so per-task images don't pile up and fill the
            # disk. Images shared with sibling tasks are kept (see __init__).
            if not cfg.output.keep_bench_images:
                names = _harbor_image_names(trial, task.id, task.raw)
                _purge_images([name for name in names if name not in self._shared_prebuilt])
