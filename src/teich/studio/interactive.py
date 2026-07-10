"""Interactive agent sessions for Teich Studio.

Agent providers (pi, codex, claude-code, hermes) run as *native* terminal
sessions: the studio starts the same persistent Docker container the batch
runners use, then launches the agent CLI inside it with a real PTY (via
util-linux `script`).  The browser attaches over a WebSocket and renders the
CLI exactly as a terminal would.  When the user is happy with the session,
the trace is exported from the mounted session directories into the output
dataset — identical to a batch-generated trace.

The chat provider has no terminal; it keeps a simple message-by-message loop
over the OpenAI-compatible API, streamed to the browser via SSE.
"""

from __future__ import annotations

import codecs
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config, PromptInput
from ..runner import (
    HERMES_DEFAULT_TOOLSETS,
    PI_SESSIONS_DIR_IN_CONTAINER,
    RUNTIME_CONTAINER_USER,
    WORKSPACE_IN_CONTAINER,
    ChatRunner,
    ClaudeCodeRunner,
    CodexRunner,
    HermesRunner,
    PiRunner,
    _make_tree_world_writable,
)
from .events import display_event

TEXT_SUBPROCESS_KWARGS: dict[str, Any] = {
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
}

SCROLLBACK_LIMIT = 2 * 1024 * 1024  # bytes of terminal history kept for reconnects


class EventLog:
    """Append-only event list with a condition for SSE long-polling."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self.closed = False

    def append(self, event: dict[str, Any]) -> None:
        with self._condition:
            event = {**event, "seq": len(self._events), "ts": time.time()}
            self._events.append(event)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()

    def wait_for(self, start_index: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        """Return events at/after start_index, blocking up to timeout if none yet."""
        with self._condition:
            if len(self._events) <= start_index and not self.closed:
                self._condition.wait(timeout=timeout)
            return self._events[start_index:]

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            return list(self._events)


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    except OSError:
        pass


def _remove_container(container_name: str | None) -> None:
    if not container_name:
        return
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        **TEXT_SUBPROCESS_KWARGS,
    )


class TerminalBridge:
    """Bridges a PTY-backed `docker exec` process to WebSocket listeners.

    Output bytes are decoded incrementally, broadcast to all attached
    listeners, and kept in a capped scrollback so reconnects can replay
    history.
    """

    def __init__(self, on_exit: Callable[[int], None] | None = None) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._listeners: set[Callable[[str], None]] = set()
        self._scrollback: list[str] = []
        self._scrollback_size = 0
        self._lock = threading.Lock()
        self._on_exit = on_exit
        self.started = False
        self.exit_code: int | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, command: list[str]) -> None:
        with self._lock:
            if self.started:
                return
            self.started = True
        try:
            # bufsize=0 keeps stdout unbuffered so partial TUI frames are
            # delivered immediately instead of waiting for a full read block.
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker is not available. Install Docker and make sure it is running."
            ) from exc
        threading.Thread(target=self._read_loop, name="studio-term-reader", daemon=True).start()

    def _read_loop(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        fd = process.stdout.fileno()
        while True:
            try:
                # os.read returns as soon as *any* bytes are available, unlike
                # BufferedReader.read(n) which blocks until n bytes or EOF.
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                self._broadcast(text)
        tail = decoder.decode(b"", final=True)
        if tail:
            self._broadcast(tail)
        self.exit_code = process.wait()
        if self._on_exit is not None:
            try:
                self._on_exit(self.exit_code)
            except Exception:
                pass

    def _broadcast(self, text: str) -> None:
        with self._lock:
            self._scrollback.append(text)
            self._scrollback_size += len(text)
            while self._scrollback_size > SCROLLBACK_LIMIT and len(self._scrollback) > 1:
                dropped = self._scrollback.pop(0)
                self._scrollback_size -= len(dropped)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(text)
            except Exception:
                pass

    def attach(self, listener: Callable[[str], None]) -> str:
        """Register a listener and return the scrollback to replay."""
        with self._lock:
            self._listeners.add(listener)
            return "".join(self._scrollback)

    def detach(self, listener: Callable[[str], None]) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def write_stdin(self, data: str) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            return
        try:
            process.stdin.write(data.encode("utf-8"))
            process.stdin.flush()
        except OSError:
            pass

    def close(self) -> None:
        _terminate(self._process)


class InteractiveSession:
    """One steerable agent session.

    mode "terminal": native CLI in a persistent Docker container (PTY).
    mode "chat": direct API loop for the chat provider.
    """

    def __init__(self, config: Config, *, github_repo: str | None = None, system: str | None = None):
        self.id = str(uuid.uuid4())
        self.config = config
        self.provider = config.get_agent_provider()
        if self.provider in {"claude", "claude_code"}:
            self.provider = "claude-code"
        if self.provider in {"hermes-agent", "hermes_agent"}:
            self.provider = "hermes"
        self.mode = "chat" if self.provider == "chat" else "terminal"
        self.github_repo = github_repo
        self.system = system
        self.events = EventLog()
        self.status = "starting"
        self.started_at = datetime.now(timezone.utc)
        self.saved_trace: Path | None = None
        self.terminal = TerminalBridge(on_exit=self._on_cli_exit)
        self._lock = threading.RLock()
        self._cleanup_paths: list[Path] = []
        self.container_name: str | None = None
        self.workspace: Path | None = None
        self._runner: Any = None
        # Provider-specific state
        self._codex_home: Path | None = None
        self._codex_existing_sessions: set[Path] = set()
        self._pi_agent_dir: Path | None = None
        self._pi_session_dir: Path | None = None
        self._home_dir: Path | None = None
        self._claude_existing_sessions: set[Path] = set()
        # Chat state
        self._busy = False
        self.turn_prompts: list[str] = []
        self._chat_history: list[dict[str, str]] = []
        self._chat_messages: list[dict[str, Any]] = []
        self._chat_usages: list[dict[str, Any] | None] = []
        self._chat_model: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_async(self) -> None:
        threading.Thread(target=self._start, name=f"studio-session-{self.id[:8]}", daemon=True).start()

    def _set_status(self, status: str, message: str | None = None) -> None:
        self.status = status
        event: dict[str, Any] = {"kind": "session_status", "status": status}
        if message:
            event["text"] = message
        self.events.append(event)

    def _start(self) -> None:
        try:
            if self.mode == "chat":
                self._runner = ChatRunner(self.config)
                if self.system:
                    self._chat_messages.append({"role": "system", "content": self.system, "thinking": None})
                self._set_status("ready", "Chat session ready. Send a message to begin.")
                return
            self._set_status("starting", "Preparing Docker runtime (first run may build the image)…")
            runner_cls = {
                "codex": CodexRunner,
                "pi": PiRunner,
                "claude-code": ClaudeCodeRunner,
                "hermes": HermesRunner,
            }.get(self.provider)
            if runner_cls is None:
                raise RuntimeError(f"Unsupported provider for interactive mode: {self.provider}")
            self._runner = runner_cls(self.config)
            self._prepare_environment()
            self._set_status("ready", "Container is running. Attaching terminal…")
        except Exception as exc:
            self.events.append(display_event("error", f"Failed to start session: {exc}"))
            self._set_status("error")
            self._cleanup()

    def _prepare_environment(self) -> None:
        runner = self._runner
        prompt_input = (
            PromptInput(prompt="interactive session", github_repo=self.github_repo)
            if self.github_repo
            else None
        )
        workspace_root, workspace = runner._prepare_workspace(self.id, prompt_input, self.provider.replace("-", ""))
        self._cleanup_paths.append(workspace_root)
        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        self.container_name = runner._container_name(f"studio-{self.provider}"[:20], self.id)

        if self.provider == "codex":
            self._codex_home = Path(tempfile_mkdtemp(f"codex-home-{self.id}-"))
            self._cleanup_paths.append(self._codex_home)
            runner._write_codex_config(self._codex_home)
            if runner._local_provider_proxy_target(self.config.api.provider, self.config.get_base_url()):
                runner._write_local_provider_proxy(self._codex_home)
            self._codex_existing_sessions = {p.resolve() for p in runner._list_session_files(self._codex_home)}
            command = runner._build_codex_persistent_container_command(workspace, self._codex_home, self.container_name)
        elif self.provider == "pi":
            self._pi_agent_dir = Path(tempfile_mkdtemp(f"pi-agent-{self.id}-"))
            self._pi_session_dir = Path(tempfile_mkdtemp(f"pi-sessions-{self.id}-"))
            self._cleanup_paths.extend([self._pi_agent_dir, self._pi_session_dir])
            runner._write_pi_agent_settings(self._pi_agent_dir)
            runner._write_pi_project_settings(workspace)
            for path in (self._pi_agent_dir, self._pi_session_dir):
                _make_tree_world_writable(path)
            command = runner._build_pi_persistent_container_command(
                workspace, self._pi_agent_dir, self._pi_session_dir, self.container_name
            )
        else:  # claude-code, hermes
            self._home_dir = Path(tempfile_mkdtemp(f"{self.provider}-home-{self.id}-"))
            self._home_dir.chmod(0o777)
            self._cleanup_paths.append(self._home_dir)
            if self.provider == "hermes":
                runner._write_hermes_runtime_config(self._home_dir)
            else:
                self._claude_existing_sessions = {
                    p.resolve() for p in runner._list_native_session_files(self._home_dir)
                }
            command = runner._build_external_persistent_container_command(workspace, self._home_dir, self.container_name)

        runner._start_container(command)

    # ------------------------------------------------------------------
    # Native terminal
    # ------------------------------------------------------------------

    def _native_cli_command(self) -> list[str]:
        """The agent CLI exactly as a user would run it inside the container."""
        runner = self._runner
        model = self.config.get_effective_model()
        if self.provider == "codex":
            # Reuse the exec command builder, then strip the non-interactive parts
            # so we get the full codex TUI with identical provider configuration.
            cli = [
                token
                for token in runner._build_codex_agent_command(resume=False)
                if token not in {"exec", "-", "--skip-git-repo-check"}
            ]
            return cli
        if self.provider == "pi":
            cli = [
                "pi",
                "--session-dir",
                PI_SESSIONS_DIR_IN_CONTAINER,
                "--provider",
                runner._pi_provider_name(),
                "--model",
                model,
            ]
            thinking = runner._normalize_thinking_level(self.config.model.reasoning_effort)
            if thinking:
                cli.extend(["--thinking", thinking])
            api_key = self.config.get_api_key()
            if api_key and not self.config.get_base_url():
                cli.extend(["--api-key", api_key])
            return cli
        if self.provider == "claude-code":
            cli = ["claude", "--model", runner._claude_visible_model()]
            permission_mode = runner._permission_mode()
            if permission_mode:
                cli.extend(["--permission-mode", permission_mode])
            cli.extend(runner._claude_passthrough_args())
            return cli
        # hermes
        return [
            "hermes",
            "chat",
            "--provider",
            runner._hermes_cli_provider(),
            "--model",
            model,
            "--toolsets",
            HERMES_DEFAULT_TOOLSETS,
            "--yolo",
            "--ignore-user-config",
            "--source",
            "teich",
        ]

    def _terminal_exec_command(self, cols: int, rows: int) -> list[str]:
        cli = shlex.join(self._native_cli_command())
        inner = f"stty cols {int(cols)} rows {int(rows)} 2>/dev/null; exec {cli}"
        return [
            "docker",
            "exec",
            "-i",
            "-e",
            "TERM=xterm-256color",
            "--user",
            RUNTIME_CONTAINER_USER,
            "-w",
            WORKSPACE_IN_CONTAINER,
            str(self.container_name),
            "script",
            "-qfec",
            inner,
            "/dev/null",
        ]

    def start_terminal(self, cols: int = 120, rows: int = 32) -> None:
        """Launch the native CLI with a PTY. Called on first WebSocket attach."""
        with self._lock:
            if self.mode != "terminal":
                raise RuntimeError("This session has no terminal (chat provider)")
            if self.status not in {"ready", "live"}:
                raise RuntimeError(f"Session is not ready (status: {self.status})")
            if self.terminal.started:
                return
            self.terminal.start(self._terminal_exec_command(cols, rows))
            self._set_status("live", "Terminal attached — you are talking to the real agent CLI.")

    def _on_cli_exit(self, exit_code: int) -> None:
        if self.status in {"finished", "discarded", "saving"}:
            return
        self._set_status(
            "exited",
            f"Agent CLI exited (code {exit_code}). Save the trace to keep this session, or discard it.",
        )

    # ------------------------------------------------------------------
    # Chat turns (chat provider only)
    # ------------------------------------------------------------------

    def send_async(self, prompt: str) -> None:
        if self.mode != "chat":
            raise RuntimeError("This session is a native terminal — type directly into it")
        prompt = prompt.replace("\r\n", "\n").strip()
        if not prompt:
            raise ValueError("Message cannot be empty")
        with self._lock:
            if self.status not in {"ready", "idle"}:
                raise RuntimeError(f"Session is not ready for input (status: {self.status})")
            if self._busy:
                raise RuntimeError("The agent is still working on the previous message")
            self._busy = True
        self.events.append(display_event("user", prompt))
        threading.Thread(
            target=self._run_chat_turn_thread,
            args=(prompt,),
            name=f"studio-turn-{self.id[:8]}",
            daemon=True,
        ).start()

    def _run_chat_turn_thread(self, prompt: str) -> None:
        self._set_status("running")
        try:
            runner = self._runner
            content, thinking, usage, model = runner._request_chat_turn_with_optional_system(
                prompt, self._chat_history or None, self.system
            )
            self._chat_messages.append({"role": "user", "content": prompt, "thinking": None})
            self._chat_messages.append({"role": "assistant", "content": content, "thinking": thinking})
            self._chat_history.append({"role": "user", "content": prompt})
            self._chat_history.append({"role": "assistant", "content": content})
            self._chat_usages.append(usage)
            self._chat_model = model
            self.turn_prompts.append(prompt)
            if isinstance(thinking, str) and thinking.strip():
                self.events.append(display_event("thinking", thinking))
            self.events.append(display_event("assistant", content))
            self._set_status("idle")
        except Exception as exc:
            self.events.append(display_event("error", str(exc)))
            self._set_status("idle" if self.turn_prompts else "ready")
        finally:
            with self._lock:
                self._busy = False

    # ------------------------------------------------------------------
    # Save / discard
    # ------------------------------------------------------------------

    def save(self) -> Path:
        with self._lock:
            if self.mode == "chat" and self._busy:
                raise RuntimeError("Wait for the agent to finish the current turn before saving")
            if self.status in {"finished", "discarded"}:
                raise RuntimeError("Session is already closed")
            if self.mode == "chat" and not self.turn_prompts:
                raise RuntimeError("Nothing to save yet — send at least one message first")
            if self.mode == "terminal" and not self.terminal.started:
                raise RuntimeError("Nothing to save yet — the terminal has not been used")
            self._set_status("saving")
        try:
            if self.mode == "terminal":
                # Stop the CLI so it flushes its session files before export.
                self.terminal.close()
                time.sleep(0.3)
            trace_path = self._export_trace()
            self.saved_trace = trace_path
            self.events.append(
                {"kind": "session_saved", "text": trace_path.name, "path": str(trace_path)}
            )
            self._set_status("finished")
            return trace_path
        except Exception as exc:
            self.events.append(display_event("error", f"Failed to save trace: {exc}"))
            self._set_status("exited" if self.mode == "terminal" else "idle")
            raise
        finally:
            if self.status == "finished":
                self._cleanup()
                self.events.close()

    def _export_trace(self) -> Path:
        runner = self._runner
        self.config.output.traces_dir.mkdir(parents=True, exist_ok=True)
        if self.provider == "chat":
            destination = runner._resolve_output_path(f"{self.id}.jsonl")
            destination.parent.mkdir(parents=True, exist_ok=True)
            row = self._chat_training_row()
            destination.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            return destination
        if self.provider == "codex":
            trace_path = runner._extract_session_file(
                self.id, self._codex_home, self._codex_existing_sessions, self.started_at
            )
        elif self.provider == "pi":
            trace_path = runner._extract_session_file(self.id, self._pi_session_dir, self.started_at)
        elif self.provider == "claude-code":
            trace_path = runner._extract_native_session_file(
                self.id, self._home_dir, self._claude_existing_sessions, self.started_at
            )
        else:  # hermes
            exports = runner._export_hermes_state_sessions(self._home_dir, self.workspace)
            if not exports:
                raise RuntimeError("Hermes did not write any sessions to export")
            trace_path = (
                next((path for sid, path in exports.items() if runner._trace_has_no_parent(path)), None)
                or next(iter(exports.values()))
            )
        runner._finalize_trace_export(trace_path, None)
        if self.workspace is not None and self.workspace.exists():
            try:
                runner._copy_workspace_snapshot(self.workspace, runner._sandbox_destination(trace_path))
            except OSError:
                pass
        return trace_path

    def _chat_training_row(self) -> dict[str, Any]:
        runner = self._runner
        responses = [
            message["content"]
            for message in self._chat_messages
            if message.get("role") == "assistant" and isinstance(message.get("content"), str)
        ]
        thinking_parts = [
            message["thinking"]
            for message in self._chat_messages
            if isinstance(message.get("thinking"), str) and message["thinking"].strip()
        ]
        model = self._chat_model or self.config.get_effective_model()
        row: dict[str, Any] = {
            "messages": self._chat_messages,
            "prompt": self.turn_prompts[0],
            "follow_up_prompts": self.turn_prompts[1:],
            "thinking": "\n\n".join(thinking_parts) or None,
            "response": responses[-1] if responses else "",
            "responses": responses,
            "model": model,
            "provider": self.config.api.provider,
            "usage": runner._merge_usage_totals(self._chat_usages),
            "metadata": {
                "trace_type": "chat",
                "model_provider": self.config.api.provider,
                "model": model,
                "turn_count": len(self.turn_prompts),
                "interactive": True,
            },
        }
        if self.system:
            row["system"] = self.system
        return row

    def discard(self) -> None:
        with self._lock:
            if self.status == "finished":
                return
            if self.mode == "chat" and self._busy:
                raise RuntimeError("Wait for the agent to finish the current turn before discarding")
            self._set_status("discarded")
        self.terminal.close()
        self._cleanup()
        self.events.close()

    def _cleanup(self) -> None:
        self.terminal.close()
        _remove_container(self.container_name)
        for path in self._cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)
        self._cleanup_paths = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "mode": self.mode,
            "model": self.config.get_effective_model(),
            "status": self.status,
            "turns": len(self.turn_prompts),
            "github_repo": self.github_repo,
            "system": self.system,
            "saved_trace": str(self.saved_trace) if self.saved_trace else None,
            "started_at": self.started_at.isoformat(),
        }


def tempfile_mkdtemp(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


class SessionManager:
    """Tracks live interactive sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, InteractiveSession] = {}
        self._lock = threading.Lock()

    def create(self, config: Config, *, github_repo: str | None = None, system: str | None = None) -> InteractiveSession:
        session = InteractiveSession(config, github_repo=github_repo, system=system)
        with self._lock:
            self._sessions[session.id] = session
        session.start_async()
        return session

    def get(self, session_id: str) -> InteractiveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.to_dict() for session in sessions]

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                session.discard()
            except Exception:
                pass
