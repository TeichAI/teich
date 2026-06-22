"""Hermetic tests for the Codex host-auth token broker.

These tests use fake JWTs and a local fake OAuth upstream; they never touch the
network, Docker, or any real auth.json.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

from teich.codex_auth_broker import CodexTokenBroker, _decode_jwt_exp


def _make_jwt(exp_offset_seconds: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = json.dumps({"exp": int(time.time()) + exp_offset_seconds}).encode()
    body = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def _write_auth(path: Path, *, access_exp_offset: int = 3600, refresh_token: str = "real-refresh-0") -> None:
    auth = {
        "tokens": {
            "id_token": _make_jwt(99_999),
            "access_token": _make_jwt(access_exp_offset),
            "refresh_token": refresh_token,
            "account_id": "acct-123",
        },
        "last_refresh": "2026-06-01T00:00:00+00:00",
    }
    path.write_text(json.dumps(auth), encoding="utf-8")


class _UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # noqa: D401 - silence test server logging
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.server.calls.append(json.loads(raw.decode("utf-8")))  # type: ignore[attr-defined]
        if self.server.delay:  # type: ignore[attr-defined]
            time.sleep(self.server.delay)  # type: ignore[attr-defined]
        data = json.dumps(self.server.response).encode("utf-8")  # type: ignore[attr-defined]
        self.send_response(self.server.status)  # type: ignore[attr-defined]
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Upstream(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def upstream():
    server = _Upstream(("127.0.0.1", 0), _UpstreamHandler)
    server.calls = []
    server.status = 200
    server.delay = 0.0
    server.response = {
        "id_token": _make_jwt(99_999),
        "access_token": _make_jwt(3600),
        "refresh_token": "real-refresh-rotated",
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/oauth/token"
    try:
        yield server, url
    finally:
        server.shutdown()
        server.server_close()


def _broker(auth_path: Path, url: str, **kwargs) -> CodexTokenBroker:
    return CodexTokenBroker(auth_path, refresh_url=url, client_id="test-client", **kwargs)


def test_decode_jwt_exp_roundtrip():
    token = _make_jwt(1000)
    exp = _decode_jwt_exp(token)
    assert exp is not None and abs(exp - (time.time() + 1000)) < 5
    assert _decode_jwt_exp("not-a-jwt") is None


def test_rejects_non_chatgpt_auth(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"OPENAI_API_KEY": "sk-x"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a ChatGPT login"):
        _broker(path, "http://127.0.0.1:1/oauth/token")


def test_wrong_secret_is_rejected(tmp_path, upstream):
    _server, url = upstream
    path = tmp_path / "auth.json"
    _write_auth(path)
    broker = _broker(path, url)
    status, payload = broker.handle_refresh({"refresh_token": "wrong"})
    assert status == 401
    assert payload["error"]["code"] == "invalid_grant"


def test_valid_token_served_without_rotation(tmp_path, upstream):
    server, url = upstream
    path = tmp_path / "auth.json"
    _write_auth(path, access_exp_offset=3600)  # far from expiry
    broker = _broker(path, url)
    original_access = json.loads(path.read_text())["tokens"]["access_token"]

    status, payload = broker.handle_refresh({"refresh_token": broker.secret})

    assert status == 200
    assert payload["access_token"] == original_access
    assert payload["refresh_token"] == broker.secret  # never the real one
    assert server.calls == []  # no upstream rotation


def test_near_expiry_triggers_single_rotation(tmp_path, upstream):
    server, url = upstream
    path = tmp_path / "auth.json"
    _write_auth(path, access_exp_offset=60)  # inside the 6-min window
    broker = _broker(path, url)

    status, payload = broker.handle_refresh({"refresh_token": broker.secret})

    assert status == 200
    assert payload["access_token"] == server.response["access_token"]
    assert len(server.calls) == 1
    # Upstream received the REAL refresh token, not the secret.
    assert server.calls[0]["refresh_token"] == "real-refresh-0"
    assert server.calls[0]["client_id"] == "test-client"
    # Rotation persisted to the auth_dir copy.
    persisted = json.loads(path.read_text())
    assert persisted["tokens"]["access_token"] == server.response["access_token"]
    assert persisted["tokens"]["refresh_token"] == "real-refresh-rotated"


def test_concurrent_refresh_is_single_flight(tmp_path, upstream):
    server, url = upstream
    server.delay = 0.3
    path = tmp_path / "auth.json"
    _write_auth(path, access_exp_offset=60)
    broker = _broker(path, url)

    results: list[tuple[int, dict]] = []
    lock = threading.Lock()

    def worker():
        out = broker.handle_refresh({"refresh_token": broker.secret})
        with lock:
            results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 6  # every worker completed and recorded a result
    assert len(server.calls) == 1  # only one real rotation
    assert all(status == 200 for status, _ in results)
    tokens_returned = {payload["access_token"] for _, payload in results}
    assert tokens_returned == {server.response["access_token"]}


def test_upstream_failure_is_relayed(tmp_path, upstream):
    server, url = upstream
    server.status = 400
    server.response = {"error": {"code": "refresh_token_reused", "message": "reused"}}
    path = tmp_path / "auth.json"
    _write_auth(path, access_exp_offset=60)
    broker = _broker(path, url)

    status, payload = broker.handle_refresh({"refresh_token": broker.secret})

    assert status == 400
    assert payload["error"]["code"] == "refresh_token_reused"


def test_seed_auth_json_hides_real_refresh_token(tmp_path, upstream):
    _server, url = upstream
    path = tmp_path / "auth.json"
    _write_auth(path, refresh_token="real-refresh-secret")
    broker = _broker(path, url)

    seed = broker.seed_auth_json()

    assert seed["tokens"]["refresh_token"] == broker.secret
    assert seed["tokens"]["refresh_token"] != "real-refresh-secret"
    assert seed["tokens"]["access_token"] == json.loads(path.read_text())["tokens"]["access_token"]
    assert seed["tokens"]["account_id"] == "acct-123"
    # The on-disk copy still holds the real refresh token (broker did not mutate it).
    assert json.loads(path.read_text())["tokens"]["refresh_token"] == "real-refresh-secret"


def test_http_server_roundtrip_and_override_url(tmp_path, upstream):
    _server, url = upstream
    path = tmp_path / "auth.json"
    _write_auth(path, access_exp_offset=3600)
    broker = _broker(path, url, host="127.0.0.1")
    broker.start()
    try:
        assert broker.port is not None
        assert broker.override_url == f"http://host.docker.internal:{broker.port}/oauth/token"
        # Hit the broker's own HTTP endpoint the way Codex would.
        body = json.dumps(
            {"client_id": "app", "grant_type": "refresh_token", "refresh_token": broker.secret}
        ).encode("utf-8")
        endpoint = f"http://127.0.0.1:{broker.port}/oauth/token"
        with urlopen(endpoint, data=body, timeout=5) as resp:  # noqa: S310 - localhost test
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["refresh_token"] == broker.secret
    finally:
        broker.stop()
        assert broker.port is None
