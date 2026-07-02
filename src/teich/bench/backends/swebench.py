"""SWE-bench bench backend.

SWE-bench ships an *evaluation* harness (it grades a patch against held-out tests); it
does not run agents. So teich supplies the agent: load instances from a SWE-bench dataset,
run teich's agent against the clean repo @ base_commit inside swebench's prebuilt instance
image (with a thin Jinja-rendered agent layer on top), take the agent's ``git diff`` as the
candidate patch, and hand that to swebench's ``run_instance`` for grading. The ``resolved``
verdict becomes the reward. swebench is imported lazily (the ``teich[swe]`` extra, py>=3.10).

Two seams here are inherently Docker-bound and validated by an integration run rather than
unit tests: building the agent image (``_build_image``) + running the agent (``_run_agent``),
and grading (``_grade``). Everything else — dataset loading, the Dockerfile render, and the
reward mapping — is plain logic with unit coverage.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import base

if TYPE_CHECKING:
    from ...config import BenchSource, Config

SWE_INSTALL_HINT = (
    "The swe-bench bench backend needs the optional 'swe' extra: install with "
    "`pip install 'teich[swe]'` (requires Python 3.10+)."
)

# SWE-bench publishes prebuilt instance images under this Docker namespace; using them
# means we pull one image and add a thin agent layer instead of rebuilding base+env+instance.
DEFAULT_NAMESPACE = "swebench"
TESTBED = "/testbed"  # swebench DOCKER_WORKDIR: the repo @ base_commit lives here
CAPTURE = "/teich-out"  # mounted host dir for the prompt, the agent session, and the diff

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


# --------------------------------------------------------------------------- agent layer

# Installs Node.js (for the JS agent CLIs) onto a swebench instance image (Debian + conda).
_NODE_RUNTIME = (
    "RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git && \\\n"
    "    (command -v node >/dev/null 2>&1 || \\\n"
    "      (curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \\\n"
    "       apt-get install -y --no-install-recommends nodejs)) && \\\n"
    "    rm -rf /var/lib/apt/lists/*"
)


@dataclass(frozen=True)
class AgentLayer:
    """How to install and invoke one agent inside the rendered image.

    ``run`` is the in-container command; it reads the task prompt from ``CAPTURE/prompt.txt``
    and runs non-interactively in ``/testbed``. ``env`` points the agent's session/home at
    ``CAPTURE`` so the native trace lands on the mounted host dir; ``session_glob`` finds it.
    ``model_flag`` is the CLI flag used to pin teich's configured model (all current agents
    use ``--model``); ``_run_command`` appends ``<model_flag> <model>`` so the container runs
    teich's model rather than each CLI's own default.
    """

    provider: str
    runtime_install: str
    agent_install: str
    env: dict[str, str] = field(default_factory=dict)
    session_glob: str = "**/*.jsonl"
    run: str = ""
    model_flag: str = "--model"


_PROMPT = f"$(cat {CAPTURE}/prompt.txt)"

# Per-provider recipes mirror teich's own non-interactive agent invocations. The exact CLI
# flags + session locations are what the Docker integration pass validates.
_LAYERS: dict[str, AgentLayer] = {
    "pi": AgentLayer(
        provider="pi",
        runtime_install=_NODE_RUNTIME,
        agent_install="RUN npm install -g @mariozechner/pi-coding-agent",
        env={"PI_SESSION_DIR": f"{CAPTURE}/sessions"},
        session_glob="sessions/**/*.jsonl",
        run=f'pi --yolo --print "{_PROMPT}"',
    ),
    "codex": AgentLayer(
        provider="codex",
        runtime_install=_NODE_RUNTIME,
        agent_install="RUN npm install -g @openai/codex",
        env={"CODEX_HOME": f"{CAPTURE}/codex"},
        session_glob="codex/sessions/**/*.jsonl",
        run=f'codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "{_PROMPT}"',
    ),
    "claude-code": AgentLayer(
        provider="claude-code",
        runtime_install=_NODE_RUNTIME,
        agent_install="RUN npm install -g @anthropic-ai/claude-code",
        env={"CLAUDE_CONFIG_DIR": f"{CAPTURE}/claude"},
        session_glob="claude/projects/**/*.jsonl",
        run=f'claude -p "{_PROMPT}" --output-format stream-json --verbose --dangerously-skip-permissions',
    ),
}

# teich provider aliases -> the swe-bench layer key.
_PROVIDER_ALIASES = {"claude": "claude-code", "claude_code": "claude-code"}


def _agent_layer(cfg: Config) -> AgentLayer:
    provider = cfg.get_agent_provider().strip().lower()
    key = _PROVIDER_ALIASES.get(provider, provider)
    layer = _LAYERS.get(key)
    if layer is None:
        raise RuntimeError(
            f"The swe-bench backend does not support agent provider {provider!r}; "
            f"use one of: {', '.join(sorted(_LAYERS))}."
        )
    # The claude-code layer runs the Anthropic ``claude`` CLI, which authenticates via
    # ``ANTHROPIC_*``; ``_auth_env`` only seeds those for an ``anthropic`` project. With any other
    # api.provider the container has no usable Claude credentials, so the run would silently
    # produce empty traces — fail fast instead. (Routing Claude through an OpenRouter/OpenAI proxy
    # is the larger prompt-mode-parity follow-up.)
    if layer.provider == "claude-code" and cfg.api.provider != "anthropic":
        raise RuntimeError(
            "The swe-bench claude-code agent runs the Anthropic `claude` CLI, which needs an "
            f"anthropic-provider config; api.provider is {cfg.api.provider!r}. Set api.provider: "
            "anthropic for swe-bench claude-code runs, or use a different agent provider."
        )
    return layer


def _model_name(cfg: Config) -> str:
    """Model for the in-container agent, with the pi ``<provider>/<model>`` prefix when needed.

    Mirrors the harbor backend's prefix rule: pi splits ``<provider>/<model>`` to pick the
    credential env var, so ``model: z-ai/glm-5.2`` + ``api.provider: openrouter`` becomes
    ``openrouter/z-ai/glm-5.2``.
    """
    model = cfg.get_effective_model().strip()
    if not model:
        return model
    api_provider = (cfg.api.provider or "").strip()
    if cfg.get_agent_provider() == "pi" and api_provider and not model.startswith(f"{api_provider}/"):
        return f"{api_provider}/{model}"
    return model


def _run_command(cfg: Config, layer: AgentLayer) -> str:
    """The agent's in-container command, with teich's configured model pinned when set."""
    model = _model_name(cfg)
    if not model:
        return layer.run
    return f"{layer.run} {layer.model_flag} {shlex.quote(model)}"


def _auth_env(cfg: Config) -> dict[str, str]:
    """Model credentials for the in-container agent, mirroring the harbor backend.

    An ``anthropic`` project sets ``ANTHROPIC_API_KEY`` only (not shadowed under
    ``OPENAI_API_KEY``); otherwise the key goes to ``OPENAI_API_KEY`` plus
    ``OPENROUTER_API_KEY`` for an OpenRouter project.
    """
    env: dict[str, str] = {}
    api_key = cfg.get_api_key()
    if api_key:
        if cfg.api.provider == "anthropic":
            env["ANTHROPIC_API_KEY"] = api_key
        else:
            env["OPENAI_API_KEY"] = api_key
            if cfg.api.provider == "openrouter":
                env["OPENROUTER_API_KEY"] = api_key
    base_url = cfg.get_base_url()
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    return env


def render_agent_dockerfile(
    base_image: str, layer: AgentLayer, *, langfuse: bool = False, langfuse_install: str = ""
) -> str:
    """Render the agent-layer Dockerfile (FROM the instance image) from the Jinja template."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template("agent.dockerfile.j2")
    return template.render(
        base_image=base_image,
        provider=layer.provider,
        runtime_install=layer.runtime_install,
        agent_install=layer.agent_install,
        langfuse=langfuse,
        langfuse_install=langfuse_install,
    )


# --------------------------------------------------------------------------- dataset


def _load_instances(source: BenchSource) -> list[dict[str, Any]]:
    """Load SWE-bench instances for a source (HF dataset id or a local json/jsonl path)."""
    from swebench.harness.utils import load_swebench_dataset

    split = source.split or "test"
    instance_ids = list(source.instances) if source.instances else None
    try:
        dataset = load_swebench_dataset(source.source, split, instance_ids)
    except Exception as exc:  # bad name/split/ids, network, HF auth -> a clean bench error
        raise RuntimeError(
            f"Failed to load swe-bench dataset {source.source!r} (split {split!r}): "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    return [dict(instance) for instance in dataset]


# --------------------------------------------------------------------------- rewards


def _rewards_from_report(entry: dict[str, Any]) -> dict[str, float]:
    """Map one swebench report entry to a rewards dict (``reward`` = resolved, for routing)."""
    resolved = 1.0 if entry.get("resolved") else 0.0
    rewards: dict[str, float] = {
        "reward": resolved,
        "resolved": resolved,
        "patch_applied": 1.0 if entry.get("patch_successfully_applied") else 0.0,
    }
    tests = entry.get("tests_status")
    if isinstance(tests, dict):
        for key, name in (("FAIL_TO_PASS", "fail_to_pass"), ("PASS_TO_PASS", "pass_to_pass")):
            group = tests.get(key)
            if isinstance(group, dict):
                passed = len(group.get("success") or [])
                failed = len(group.get("failure") or [])
                if passed + failed:
                    rewards[name] = passed / (passed + failed)
    return rewards


# ----------------------------------------------------------------- docker seam (integration)


def _docker(args: list[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=check
    )


def _ensure_instance_image(spec: Any, *, namespace: str | None, timeout: int | None = None) -> None:
    """Make the swebench instance image available locally (pull remote, else build)."""
    image = spec.instance_image_key
    if _docker(["images", "-q", image], check=False).stdout.strip():
        return
    if namespace:  # remote (published) image -> pull it
        _docker(["pull", image], timeout=timeout)
        return
    import docker  # local build path
    from swebench.harness.docker_build import build_instance_images

    build_instance_images(docker.from_env(), [spec], namespace=None, max_workers=1)


def _build_image(dockerfile: str, tag: str, *, timeout: int | None = None) -> None:
    # Build from stdin (no Dockerfile written into the repo/context); capture logs and bound the
    # build by the run timeout, surfacing the tail on failure instead of flooding stdout.
    try:
        subprocess.run(
            ["docker", "build", "-t", tag, "-"],
            input=dockerfile, text=True, capture_output=True, check=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "")[-2000:]
        raise RuntimeError(f"docker build for {tag} failed:\n{tail}") from exc


def _native_trace(capture_dir: Path, session_glob: str) -> tuple[list[str], Path | None]:
    """Collect the agent's native session JSONL from the mounted capture dir."""
    files = sorted(capture_dir.glob(session_glob))
    lines = [
        line
        for path in files
        if path.is_file()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return [], None
    return lines, files[0].parent


def _run_agent(
    cfg: Config, source: BenchSource, instance: dict[str, Any], spec: Any,
    layer: AgentLayer, capture_dir: Path,
) -> tuple[list[str], Path | None, str]:
    """Build the agent layer, run the agent on the problem, return (trace, dir, model_patch)."""
    # Reset the per-task capture dir: it's a stable path (keyed by source+task.id), so a rerun
    # that aborts before writing a fresh session/model.patch would otherwise harvest and grade
    # the previous attempt's stale artifacts.
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    (capture_dir / "prompt.txt").write_text(instance.get("problem_statement") or "", encoding="utf-8")

    # Namespace all scratch (image tag, container, env-file) by source so two sources that
    # contain the same instance can't collide under concurrency; lowercased for Docker.
    key = base.slug(f"{base.source_id(source)}-{instance['instance_id']}").lower()
    # Langfuse is intentionally not wired here (guarded against in tasks()); render it off so
    # the image never carries a phantom tracing block.
    dockerfile = render_agent_dockerfile(spec.instance_image_key, layer)
    agent_image = f"teich-swe-{key}:latest"
    timeout = cfg.timeout_seconds if cfg.timeout_seconds and cfg.timeout_seconds > 0 else None
    _build_image(dockerfile, agent_image, timeout=timeout)

    # Pass credentials via an env-file (host-only, outside the mounted dir, mode 0600) rather
    # than `-e KEY=VALUE`, so keys aren't visible in the host process list. Note: they still
    # become container env vars (readable via `docker inspect` / inside the container), so this
    # guards the host's `ps`, not a hostile Docker-socket user.
    env = {**layer.env, **_auth_env(cfg)}
    env_file = capture_dir.parent / f"{key}.env"
    fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)  # O_CREAT's mode only applies on create (and is umask-masked); enforce 0600 either way
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("".join(f"{name}={value}\n" for name, value in env.items()))

    container = f"teich-swe-{key}"
    capture = f"{CAPTURE}/model.patch"
    run = _run_command(cfg, layer)
    shell = (
        f"cd {TESTBED} && ({run} || true) && git -C {TESTBED} add -A && "
        f"git -C {TESTBED} diff --cached --no-color > {capture} 2>/dev/null || true; "
        # The container runs as root and writes the agent's session dir into the bind-mounted
        # capture tree. On native-Linux Docker those files land root-owned, so the next no-resume
        # run's shutil.rmtree(capture_dir) fails; hand the tree back to the host UID before exit.
        f"chmod -R a+rwX {CAPTURE} 2>/dev/null || true"
    )
    cmd = [
        "docker", "run", "--rm", "--name", container, "--env-file", str(env_file),
        "-v", f"{capture_dir}:{CAPTURE}",
        agent_image, "bash", "-lc", shell,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        _docker(["rm", "-f", container], check=False)  # reap the named container, then fail the task
        raise
    finally:
        env_file.unlink(missing_ok=True)
        # Remove the per-task agent-layer image so it doesn't accumulate (the shared swebench
        # instance image is left in place — it's reused and expensive to re-pull). Best-effort:
        # check=False only swallows non-zero exits, so a missing docker / hang must not mask the
        # real task result during this finally.
        if not cfg.output.keep_bench_images:
            try:
                _docker(["rmi", "-f", agent_image], timeout=30, check=False)
            except Exception:
                pass
    if result.returncode != 0:
        # The shell masks the *agent's* exit with `|| true`, so a non-zero exit here is a Docker/
        # infra failure — fail the task rather than grade an empty patch as a real result.
        raise RuntimeError(
            f"docker run failed for {container} (exit {result.returncode}): "
            f"{(result.stderr or result.stdout or '').strip()[-1000:]}"
        )

    patch_file = capture_dir / "model.patch"
    model_patch = patch_file.read_text(encoding="utf-8") if patch_file.is_file() else ""
    lines, native_dir = _native_trace(capture_dir, layer.session_glob)
    return lines, native_dir, model_patch


def _grade(cfg: Config, instance: dict[str, Any], spec: Any, model_patch: str) -> dict[str, Any]:
    """Grade a candidate patch with swebench's harness; return the report entry for the instance."""
    import docker
    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR
    from swebench.harness.run_evaluation import run_instance

    instance_id = instance["instance_id"]
    model_name = "teich"
    run_id = f"teich-{base.slug(instance_id)}"
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": model_patch,
    }
    timeout = cfg.timeout_seconds if cfg.timeout_seconds and cfg.timeout_seconds > 0 else None
    report_path = (
        Path(RUN_EVALUATION_LOG_DIR) / run_id / model_name / instance_id / "report.json"
    )
    # run_id is deterministic, so drop any prior report first: if this rerun fails before
    # writing a fresh one, we must not grade off a stale verdict from a previous attempt.
    report_path.unlink(missing_ok=True)
    run_instance(
        spec, prediction, rm_image=False, force_rebuild=False,
        client=docker.from_env(), run_id=run_id, timeout=timeout,
    )
    if not report_path.is_file():
        return {"resolved": False, "patch_successfully_applied": bool(model_patch)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get(instance_id, {"resolved": False})


# --------------------------------------------------------------------------- backend


class SweBenchBackend:
    """Runs teich's agent against SWE-bench instances and grades the diff with swebench."""

    type = "swe-bench"

    def require(self) -> None:
        if sys.version_info < (3, 10):
            raise RuntimeError(
                "The swe-bench bench backend requires Python 3.10+ "
                f"(current: {sys.version_info.major}.{sys.version_info.minor}). {SWE_INSTALL_HINT}"
            )
        try:
            import swebench  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise RuntimeError(SWE_INSTALL_HINT) from exc

    def tasks(self, cfg: Config, source: BenchSource, *, refresh: bool = False) -> list[base.BenchTask]:
        # refresh is a no-op here: HF datasets handle their own cache.
        if cfg.agent.langfuse.enabled:
            # The swe-bench backend does not wire Langfuse into the agent container (no
            # creds in the env-file, no install layer), so tracing would silently no-op.
            # Fail loudly rather than pretend support.
            raise RuntimeError(
                "The swe-bench bench backend does not support Langfuse tracing; "
                "set agent.langfuse.enabled = false for swe-bench sources."
            )
        return [
            base.BenchTask(id=instance["instance_id"], raw=instance)
            for instance in _load_instances(source)
        ]

    def run(self, cfg: Config, source: BenchSource, task: base.BenchTask) -> base.BenchRun:
        from swebench.harness.test_spec.test_spec import make_test_spec

        instance = task.raw
        layer = _agent_layer(cfg)  # fail fast on an unsupported provider, before any Docker work
        namespace = DEFAULT_NAMESPACE
        spec = make_test_spec(instance, namespace=namespace)
        timeout = cfg.timeout_seconds if cfg.timeout_seconds and cfg.timeout_seconds > 0 else None
        _ensure_instance_image(spec, namespace=namespace, timeout=timeout)

        capture_dir = base.bench_root(cfg) / "swe" / base.bench_stem(source, task.id)
        lines, native_dir, model_patch = _run_agent(cfg, source, instance, spec, layer, capture_dir)

        entry = _grade(cfg, instance, spec, model_patch)
        rewards = _rewards_from_report(entry)
        metadata = {
            "instance_id": task.id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "model_patch": model_patch,
            "patch_applied": bool(entry.get("patch_successfully_applied")),
            **base.trace_metadata(native_dir),
        }
        return base.BenchRun(native_lines=lines, rewards=rewards, metadata=metadata)
