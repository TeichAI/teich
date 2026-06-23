"""In-process token broker for Codex ChatGPT-subscription host auth.

When ``agent.codex.use_host_auth`` is enabled, teich runs a single host-side
broker that owns the rotating ChatGPT OAuth refresh token for the whole run.
Every Codex container is pointed at the broker via
``CODEX_REFRESH_TOKEN_URL_OVERRIDE`` and seeded with its own ``auth.json`` whose
``refresh_token`` is a per-run secret -- the real refresh token never enters a
container. The broker hands the same live access token to every container and
performs the single-use refresh-token rotation centrally, so concurrent
containers can no longer invalidate one another (which is what happens when N
containers each rotate a shared ``auth.json`` against ``auth.openai.com``).

The broker stores and rotates only the copy in ``auth_dir`` (next to the teich
config); it never touches the host ``~/.codex/auth.json``.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Real OpenAI OAuth refresh endpoint + the Codex CLI client id. Both are
# injectable so tests can point at a local fake without network access.
REAL_REFRESH_TOKEN_URL = "https://auth.openai.com/oauth/token"
REAL_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Codex proactively refreshes when the access token is within 5 minutes of
# expiry. Rotate a little earlier so a token we hand back is never already
# inside Codex's window -- otherwise Codex would refresh again on the very next
# request and loop.
ROTATE_WINDOW_SECONDS = 6 * 60

REFRESH_PATH = "/oauth/token"

# A refresh request body is tiny; cap reads so a bogus Content-Length can't
# exhaust memory (the broker listens on 0.0.0.0).
MAX_REQUEST_BYTES = 64 * 1024


def _decode_jwt_exp(token: str) -> int | None:
    """Return the ``exp`` claim (unix seconds) from a JWT without verifying it."""
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        # binascii.Error (malformed base64) is a subclass of ValueError.
        decoded = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return None
    try:
        return int(exp)  # int(float("inf")) / NaN raise OverflowError/ValueError
    except (OverflowError, ValueError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _UpstreamRefreshError(Exception):
    """Carries an upstream OAuth failure so the broker can relay it to Codex.

    Codex classifies the response body's ``error.code`` (e.g.
    ``refresh_token_reused``) to decide whether the failure is permanent, so we
    pass the upstream status + body straight through.
    """

    def __init__(self, status: int, payload: dict[str, Any]):
        super().__init__(f"upstream refresh failed: {status}")
        self.status = status
        self.payload = payload


class CodexTokenBroker:
    """Single owner of the rotating ChatGPT refresh token for a teich run."""

    def __init__(
        self,
        auth_json_path: Path | str,
        *,
        refresh_url: str = REAL_REFRESH_TOKEN_URL,
        client_id: str = REAL_OAUTH_CLIENT_ID,
        host: str = "0.0.0.0",
        port: int = 0,
        rotate_window_seconds: int = ROTATE_WINDOW_SECONDS,
        upstream_timeout: float = 30.0,
    ) -> None:
        self._auth_json_path = Path(auth_json_path)
        self._refresh_url = refresh_url
        self._client_id = client_id
        self._host = host
        self._requested_port = port
        self._rotate_window_seconds = rotate_window_seconds
        self._upstream_timeout = upstream_timeout
        self._lock = threading.Lock()
        # Per-run shared secret. Containers receive this as their auth.json
        # refresh_token; the broker only serves callers that present it.
        self.secret = secrets.token_urlsafe(32)
        self._auth = self._load_auth()
        self._validate_chatgpt_auth(self._auth)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._server is not None:
            return
        server = _BrokerServer((self._host, self._requested_port), _BrokerHandler, self)
        self.port = server.server_address[1]
        thread = threading.Thread(
            target=server.serve_forever,
            name="teich-codex-token-broker",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self.port = None

    @property
    def override_url(self) -> str:
        """The ``CODEX_REFRESH_TOKEN_URL_OVERRIDE`` value for containers."""
        if self.port is None:
            raise RuntimeError("CodexTokenBroker.start() has not been called")
        return f"http://host.docker.internal:{self.port}{REFRESH_PATH}"

    # -- container seed ----------------------------------------------------
    def seed_auth_json(self) -> dict[str, Any]:
        """A container-safe ``auth.json``: real tokens but the refresh token
        replaced by the per-run secret, so the durable refresh token stays on
        the host."""
        with self._lock:
            auth = json.loads(json.dumps(self._auth))  # deep copy
        tokens = auth.setdefault("tokens", {})
        tokens["refresh_token"] = self.secret
        return auth

    # -- refresh handling --------------------------------------------------
    def handle_refresh(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Core of the broker: validate the caller, return the current access
        token, rotating once upstream only when it is near expiry."""
        incoming = body.get("refresh_token")
        if not isinstance(incoming, str) or not secrets.compare_digest(incoming, self.secret):
            return 401, {
                "error": {
                    "code": "invalid_grant",
                    "message": "unrecognized refresh token",
                }
            }
        with self._lock:
            tokens = self._auth.get("tokens", {})
            if self._needs_rotation(tokens.get("access_token", "")):
                try:
                    self._rotate()
                except _UpstreamRefreshError as exc:
                    return exc.status, exc.payload
            return 200, self._refresh_response()

    def _needs_rotation(self, access_token: str) -> bool:
        exp = _decode_jwt_exp(access_token)
        if exp is not None:
            return exp <= time.time() + self._rotate_window_seconds
        # No parseable exp -> rotate once to obtain a fresh JWT.
        return True

    def _refresh_response(self) -> dict[str, Any]:
        tokens = self._auth.get("tokens", {})
        return {
            "id_token": tokens.get("id_token"),
            "access_token": tokens.get("access_token"),
            # Hand back the secret, never the real refresh token.
            "refresh_token": self.secret,
        }

    def _rotate(self) -> None:
        tokens = self._auth.setdefault("tokens", {})
        request_body = json.dumps(
            {
                "client_id": self._client_id,
                "grant_type": "refresh_token",
                "refresh_token": tokens.get("refresh_token", ""),
            }
        ).encode("utf-8")
        request = Request(
            self._refresh_url,
            data=request_body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self._upstream_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace") if exc.fp is not None else ""
            try:
                err_payload = json.loads(raw) if raw else {}
            except ValueError:
                err_payload = {"error": {"message": raw}}
            raise _UpstreamRefreshError(exc.code, err_payload)
        except URLError as exc:
            raise _UpstreamRefreshError(
                502, {"error": {"message": f"refresh upstream unreachable: {exc.reason}"}}
            )

        if not isinstance(payload, dict):
            raise _UpstreamRefreshError(
                502, {"error": {"message": "refresh upstream returned a non-object body"}}
            )
        new_access = payload.get("access_token")
        if not isinstance(new_access, str) or not new_access:
            raise _UpstreamRefreshError(
                502, {"error": {"message": "refresh upstream returned no access_token"}}
            )
        # Only persist after the response validates, and only string fields, so a
        # malformed 2xx body can never overwrite good token state with junk.
        for field in ("id_token", "access_token", "refresh_token"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                tokens[field] = value
        self._auth["tokens"] = tokens
        self._auth["last_refresh"] = _utc_now_iso()
        self._persist()

    # -- storage -----------------------------------------------------------
    def _load_auth(self) -> dict[str, Any]:
        try:
            data = json.loads(self._auth_json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to read Codex auth snapshot {self._auth_json_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Codex auth snapshot {self._auth_json_path} is not a JSON object"
            )
        return data

    def _persist(self) -> None:
        path = self._auth_json_path
        tmp = path.with_name(path.name + ".tmp")
        data = json.dumps(self._auth, indent=2) + "\n"
        # Create the temp file owner-only (0o600) at creation time so the
        # secret-bearing file is never momentarily readable by other users.
        # os.replace preserves the source's permissions onto the final path.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)

    @staticmethod
    def _validate_chatgpt_auth(auth: dict[str, Any]) -> None:
        tokens = auth.get("tokens")
        access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
        refresh_token = tokens.get("refresh_token") if isinstance(tokens, dict) else None
        if (
            not isinstance(access_token, str)
            or not access_token
            or not isinstance(refresh_token, str)
            or not refresh_token
        ):
            raise RuntimeError(
                "Codex host auth is not a ChatGPT login (auth.json has no "
                "tokens.refresh_token/access_token). Run `codex login` with a "
                "ChatGPT subscription, or disable agent.codex.use_host_auth."
            )


class _BrokerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, broker: CodexTokenBroker):
        super().__init__(address, handler)
        self.broker = broker


class _BrokerHandler(BaseHTTPRequestHandler):
    server: _BrokerServer  # type: ignore[assignment]

    def log_message(self, *_args) -> None:  # silence default stderr logging
        pass

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0].rstrip("/") != REFRESH_PATH:
            self._write_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(400, {"error": {"message": "invalid Content-Length"}})
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._write_json(413, {"error": {"message": "request body too large"}})
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        status, payload = self.server.broker.handle_refresh(body)
        self._write_json(status, payload)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._write_json(200, {"status": "ok"})
        else:
            self._write_json(404, {"error": {"message": "not found"}})
