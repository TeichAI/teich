"""Docker-based runners for non-interactive Codex and Pi sessions."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import json
import os
import queue
import re
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

from .config import PromptInput

from .codex_auth_broker import CodexTokenBroker

from .converter import (
    TEICH_AVAILABLE_TOOLS_CUSTOM_TYPE,
    convert_trace_to_training_example,
    convert_traces_to_training_data,
    normalize_claude_code_trace_events,
    normalize_codex_trace_event,
    normalize_codex_trace_events,
)
from .tool_schema import snapshot_configured_tools

RUNTIME_IMAGE_NAME = "teich-runtime:v3"
LANGFUSE_RUNTIME_IMAGE_NAME = "teich-runtime:v3-langfuse"
RUNTIME_DOCKERFILE_NAME = "codex-runtime.Dockerfile"
RUNTIME_CONTAINER_USER = "codex"
CODEX_HOME_IN_CONTAINER = "/home/codex/.codex"
CLAUDE_HOME_IN_CONTAINER = "/home/codex/.claude"
HERMES_HOME_IN_CONTAINER = "/home/codex/.hermes"

# Langfuse Codex observability plugin: baked into the tracing-enabled runtime
# image at a staging CODEX_HOME (see codex-runtime.Dockerfile) so Teich can seed
# it into each session offline. The default runtime image does not install these
# optional dependencies.
LANGFUSE_PLUGIN_STAGE_CODEX_HOME = "/opt/codex-langfuse/.codex"
LANGFUSE_PLUGIN_ID = "tracing@codex-observability-plugin"

# Claude strips /opt/venv from a hook's PATH, so call the venv python (which has
# the langfuse SDK) by absolute path. Script is baked into the tracing image.
CLAUDE_LANGFUSE_PLUGIN_DIR = "/opt/claude-langfuse-plugin"
CLAUDE_LANGFUSE_HOOK_COMMAND = (
    f"/opt/venv/bin/python3 {CLAUDE_LANGFUSE_PLUGIN_DIR}/hooks/langfuse_hook.py"
)
PI_AGENT_DIR_IN_CONTAINER = "/home/codex/.pi/agent"
PI_SESSIONS_DIR_IN_CONTAINER = "/home/codex/pi-sessions"
WORKSPACE_IN_CONTAINER = "/workspace"
HERMES_DEFAULT_TOOLSETS = "safe,terminal,file,skills,memory,session_search,delegation"
HERMES_TRACE_WRITE_LOCK = threading.Lock()
CHAT_REQUEST_MAX_ATTEMPTS = 3
AGENT_TURN_RETRY_LIMIT = 3
NON_DATA_TRACE_DIR_NAMES = {"partials", "failures"}


def _make_tree_world_writable(path: Path) -> None:
    """Allow the unprivileged container user to write host-created mount paths."""
    try:
        path.chmod(0o777 if path.is_dir() else 0o666)
    except OSError:
        return
    if not path.is_dir():
        return
    for child in path.rglob("*"):
        try:
            child.chmod(0o777 if child.is_dir() else 0o666)
        except OSError:
            continue


def _read_jsonl_dict_events(path: Path) -> list[dict[str, Any]] | None:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    return None
                events.append(event)
    except (OSError, json.JSONDecodeError):
        return None
    return events


def _write_jsonl_dict_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _is_non_data_trace_path(path: Path, traces_dir: Path, excluded_dirs: list[Path] | None = None) -> bool:
    try:
        if any(part in NON_DATA_TRACE_DIR_NAMES for part in path.relative_to(traces_dir).parts):
            return True
    except ValueError:
        pass
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path
    for excluded_dir in excluded_dirs or []:
        try:
            resolved_excluded_dir = excluded_dir.resolve()
        except OSError:
            continue
        try:
            resolved_path.relative_to(resolved_excluded_dir)
            return True
        except ValueError:
            continue
    return False


TEXT_SUBPROCESS_KWARGS = {"text": True, "encoding": "utf-8", "errors": "replace"}
LOCAL_PROVIDER_PROXY_SCRIPT_NAME = "local_provider_proxy.js"
CLAUDE_OPENROUTER_PROXY_SCRIPT_NAME = "claude_openrouter_proxy.js"
CLAUDE_OPENROUTER_PROXY_PORT = 17891
CLAUDE_OPENROUTER_SURROGATE_MODEL = "claude-sonnet-4-6"
OPENROUTER_GENERATION_STATS_ATTEMPTS = 2
OPENROUTER_GENERATION_STATS_TIMEOUT_SECONDS = 3
TEICH_PROMPT_FILE_NAME = ".teich-prompt.txt"
PI_SYSTEM_PROMPT_CUSTOM_TYPE = "teich-system-prompt"


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

function responseHeadersForClient(upstream) {
  const headers = Object.fromEntries(upstream.headers.entries());
  delete headers['content-encoding'];
  delete headers['content-length'];
  delete headers['transfer-encoding'];
  return headers;
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

    res.writeHead(upstream.status, responseHeadersForClient(upstream));
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

CLAUDE_OPENROUTER_PROXY_SCRIPT = """
const http = require('node:http');
const https = require('node:https');
const { Readable } = require('node:stream');

const target = new URL(process.env.TEICH_CLAUDE_PROXY_TARGET);
const targetModel = process.env.TEICH_CLAUDE_PROXY_TARGET_MODEL;
const listenPort = Number(process.env.TEICH_CLAUDE_PROXY_PORT || '17891');

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  return chunks.length ? Buffer.concat(chunks) : undefined;
}

function responseHeadersForClient(upstream) {
  const headers = Object.fromEntries(upstream.headers.entries());
  delete headers['content-encoding'];
  delete headers['content-length'];
  delete headers['transfer-encoding'];
  return headers;
}

function upstreamUrlFor(reqUrl) {
  const incoming = new URL(reqUrl || '/', 'http://127.0.0.1');
  let pathname = incoming.pathname;
  if (pathname.startsWith('/v1/')) {
    pathname = pathname.slice('/v1'.length);
  }
  const basePath = target.pathname.replace(/\\/$/, '');
  const upstream = new URL(target);
  upstream.pathname = `${basePath}${pathname.startsWith('/') ? pathname : `/${pathname}`}`;
  upstream.search = incoming.search;
  return upstream;
}

function rewriteJsonBody(headers, body) {
  if (!body || !targetModel) {
    return body;
  }
  const contentType = String(headers['content-type'] || '');
  if (!contentType.includes('json')) {
    return body;
  }
  try {
    const payload = JSON.parse(body.toString('utf8'));
    if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
      if (typeof payload.model === 'string') {
        payload.model = targetModel;
      }
      // Claude Code may send Anthropic-specific thinking controls that third-party
      // OpenRouter models reject. Keep Teich's configured model, not Claude's
      // surrogate model, as the provider-facing contract.
      if (Object.prototype.hasOwnProperty.call(payload, 'thinking')) {
        delete payload.thinking;
      }
      return Buffer.from(JSON.stringify(payload));
    }
  } catch {
    return body;
  }
  return body;
}

const server = http.createServer(async (req, res) => {
  const upstreamUrl = upstreamUrlFor(req.url);
  const headers = { ...req.headers };
  delete headers.host;
  delete headers['content-length'];
  const apiKey = headers['x-api-key'] || headers.authorization?.replace(/^Bearer\\s+/i, '');
  if (apiKey && !headers.authorization) {
    headers.authorization = `Bearer ${apiKey}`;
  }

  try {
    const rawBody = req.method === 'GET' || req.method === 'HEAD' ? undefined : await readRequestBody(req);
    const body = rewriteJsonBody(headers, rawBody);
    const upstream = await fetch(upstreamUrl, {
      method: req.method,
      headers,
      body,
      duplex: body ? 'half' : undefined,
    });

    res.writeHead(upstream.status, responseHeadersForClient(upstream));
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
    has_token_usage: bool = False
    has_cost: bool = False

    def __post_init__(self) -> None:
        if (
            self.input_tokens
            or self.output_tokens
            or self.reasoning_tokens
            or self.cache_read_tokens
            or self.cache_write_tokens
            or self.total_tokens
            or self.est_total_tokens
        ):
            self.has_token_usage = True
        if self.total_cost:
            self.has_cost = True

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

    @staticmethod
    def _has_any_key(mapping: dict[str, Any], keys: tuple[str, ...]) -> bool:
        return any(key in mapping and mapping.get(key) is not None for key in keys)

    def add_pi_usage(self, usage: dict[str, Any]) -> None:
        if self._has_any_key(
            usage,
            ("input", "output", "cacheRead", "cacheWrite", "totalTokens"),
        ):
            self.has_token_usage = True
        self.input_tokens += self._int_value(usage.get("input"))
        self.output_tokens += self._int_value(usage.get("output"))
        self.cache_read_tokens += self._int_value(usage.get("cacheRead"))
        self.cache_write_tokens += self._int_value(usage.get("cacheWrite"))
        total_tokens = self._int_value(usage.get("totalTokens"))
        self.total_tokens += total_tokens
        self.est_total_tokens += total_tokens
        cost = usage.get("cost")
        if isinstance(cost, dict):
            if "total" in cost and cost.get("total") is not None:
                self.has_cost = True
            self.total_cost += self._float_value(cost.get("total"))

    def add_structured_usage(self, usage: dict[str, Any]) -> None:
        if self._has_any_key(
            usage,
            (
                "input",
                "prompt_tokens",
                "input_tokens",
                "output",
                "completion_tokens",
                "output_tokens",
                "reasoning",
                "reasoning_tokens",
                "reasoning_output_tokens",
                "cacheRead",
                "cached_input_tokens",
                "cacheWrite",
                "prompt_tokens_details",
                "totalTokens",
                "total_tokens",
            ),
        ):
            self.has_token_usage = True
        self.input_tokens += self._int_value(usage.get("input") or usage.get("prompt_tokens") or usage.get("input_tokens"))
        self.output_tokens += self._int_value(usage.get("output") or usage.get("completion_tokens") or usage.get("output_tokens"))
        self.reasoning_tokens += self._int_value(
            usage.get("reasoning")
            or usage.get("reasoning_tokens")
            or usage.get("reasoning_output_tokens")
        )
        prompt_token_details = usage.get("prompt_tokens_details")
        if not isinstance(prompt_token_details, dict):
            prompt_token_details = {}
        self.cache_read_tokens += self._int_value(
            usage.get("cacheRead")
            or usage.get("cached_input_tokens")
            or prompt_token_details.get("cached_tokens")
        )
        self.cache_write_tokens += self._int_value(
            usage.get("cacheWrite")
            or prompt_token_details.get("cache_write_tokens")
        )
        total_tokens = self._int_value(usage.get("totalTokens") or usage.get("total_tokens"))
        if total_tokens:
            self.total_tokens += total_tokens
            self.est_total_tokens += total_tokens
        cost = usage.get("cost")
        if isinstance(cost, dict):
            if "total" in cost and cost.get("total") is not None:
                self.has_cost = True
            self.total_cost += self._float_value(cost.get("total"))
        elif isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.has_cost = True
            self.total_cost += float(cost)

    def add_codex_last_usage(self, usage: dict[str, Any]) -> None:
        if self._has_any_key(
            usage,
            (
                "input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "cached_input_tokens",
                "total_tokens",
            ),
        ):
            self.has_token_usage = True
        self.input_tokens += self._int_value(usage.get("input_tokens"))
        self.output_tokens += self._int_value(usage.get("output_tokens"))
        self.reasoning_tokens += self._int_value(usage.get("reasoning_output_tokens"))
        self.cache_read_tokens += self._int_value(usage.get("cached_input_tokens"))
        total_tokens = self._int_value(usage.get("total_tokens"))
        if total_tokens:
            self.total_tokens += total_tokens

    def apply_codex_total_usage(self, usage: dict[str, Any]) -> None:
        if self._has_any_key(
            usage,
            (
                "input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "cached_input_tokens",
                "total_tokens",
            ),
        ):
            self.has_token_usage = True
        self.input_tokens = self._int_value(usage.get("input_tokens"))
        self.output_tokens = self._int_value(usage.get("output_tokens"))
        self.reasoning_tokens = self._int_value(usage.get("reasoning_output_tokens"))
        self.cache_read_tokens = self._int_value(usage.get("cached_input_tokens"))
        total_tokens = self._int_value(usage.get("total_tokens"))
        if total_tokens:
            self.total_tokens = total_tokens

    def apply_codex_estimated_usage(self, usage: dict[str, Any]) -> None:
        if self._has_any_key(
            usage,
            (
                "input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "cached_input_tokens",
                "total_tokens",
            ),
        ):
            self.has_token_usage = True
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
        if self.has_token_usage and not self.total_tokens:
            self.total_tokens = (
                self.input_tokens
                + self.output_tokens
                + self.reasoning_tokens
                + self.cache_read_tokens
                + self.cache_write_tokens
            )
        if self.has_token_usage and not self.est_total_tokens:
            self.est_total_tokens = self.total_tokens

    def copy(self) -> TraceMetrics:
        return TraceMetrics(
            model=self.model,
            provider=self.provider,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            total_tokens=self.total_tokens,
            est_total_tokens=self.est_total_tokens,
            total_cost=self.total_cost,
            has_token_usage=self.has_token_usage,
            has_cost=self.has_cost,
        )

    def add_metrics(self, metrics: TraceMetrics) -> None:
        if metrics.model:
            self.model = metrics.model
        if metrics.provider:
            self.provider = metrics.provider
        if metrics.has_token_usage:
            self.has_token_usage = True
        self.input_tokens += metrics.input_tokens
        self.output_tokens += metrics.output_tokens
        self.reasoning_tokens += metrics.reasoning_tokens
        self.cache_read_tokens += metrics.cache_read_tokens
        self.cache_write_tokens += metrics.cache_write_tokens
        self.total_tokens += metrics.total_tokens
        self.est_total_tokens += metrics.est_total_tokens
        if metrics.has_cost:
            self.has_cost = True
        self.total_cost += metrics.total_cost


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
    details: str | None = None
    metrics: TraceMetrics | None = None
    metrics_delta: bool = False


SessionProgressCallback = Callable[[SessionProgressUpdate], None]


def _prompt_text_completion_key(prompt: str) -> str:
    prompt = _unwrap_teich_prompt_file(prompt)
    return "\n".join(prompt.replace("\r\n", "\n").replace("\r", "\n").strip().splitlines())


def _prompt_completion_key(prompt_input: PromptInput | str) -> str:
    if isinstance(prompt_input, str):
        return _prompt_text_completion_key(prompt_input)
    prompt_parts = [
        _prompt_text_completion_key(prompt)
        for prompt in prompt_input.turn_prompts()
    ]
    system_prompt = prompt_input.system.strip() if isinstance(prompt_input.system, str) and prompt_input.system.strip() else ""
    if system_prompt:
        return "\n\n--- system ---\n\n".join(
            [
                _prompt_text_completion_key(system_prompt),
                "\n\n--- follow-up ---\n\n".join(prompt_parts),
            ]
        )
    return "\n\n--- follow-up ---\n\n".join(prompt_parts)


def _prompt_resume_key(prompt_input: PromptInput | str) -> str:
    if isinstance(prompt_input, str):
        prompt_parts = [_prompt_text_completion_key(prompt_input)]
    else:
        prompt_parts = [
            _prompt_text_completion_key(prompt)
            for prompt in prompt_input.turn_prompts()
        ]
    return "\n\n--- follow-up ---\n\n".join(prompt_parts)


def _agent_turn_prompts(prompt: str, prompt_input: PromptInput | None) -> list[str]:
    if prompt_input is None or not prompt_input.follow_up_prompts:
        return [prompt]
    return PromptInput(prompt=prompt, follow_up_prompts=prompt_input.follow_up_prompts).turn_prompts()


def _unwrap_teich_prompt_file(prompt: str) -> str:
    normalized = prompt.strip()
    match = re.fullmatch(
        r'<file\s+name=["\'][^"\']*\.teich-prompt\.txt["\']>\s*(?P<prompt>.*?)\s*</file>',
        normalized,
        flags=re.DOTALL,
    )
    if match:
        return match.group("prompt")
    return prompt


def _normalize_teich_prompt_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        normalized = _unwrap_teich_prompt_file(content)
        return normalized, normalized != content
    if not isinstance(content, list):
        return content, False
    normalized_blocks: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, str):
            normalized = _unwrap_teich_prompt_file(block)
            normalized_blocks.append(normalized)
            changed = changed or normalized != block
            continue
        if not isinstance(block, dict):
            normalized_blocks.append(block)
            continue
        text = block.get("text")
        if not isinstance(text, str):
            normalized_blocks.append(block)
            continue
        normalized_text = _unwrap_teich_prompt_file(text)
        if normalized_text == text:
            normalized_blocks.append(block)
            continue
        normalized_block = dict(block)
        normalized_block["text"] = normalized_text
        normalized_blocks.append(normalized_block)
        changed = True
    return normalized_blocks if changed else content, changed


def _normalize_teich_prompt_user_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("role") == "user":
        normalized_content, changed = _normalize_teich_prompt_content(event.get("content"))
        if changed:
            normalized_event = dict(event)
            normalized_event["content"] = normalized_content
            return normalized_event
        return event

    for envelope_key in ("message", "payload"):
        payload = event.get(envelope_key)
        if not isinstance(payload, dict) or payload.get("role") != "user":
            continue
        normalized_content, changed = _normalize_teich_prompt_content(payload.get("content"))
        if not changed:
            return event
        normalized_event = dict(event)
        normalized_payload = dict(payload)
        normalized_payload["content"] = normalized_content
        normalized_event[envelope_key] = normalized_payload
        return normalized_event
    return event


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _training_example_has_answer(example: dict[str, Any]) -> bool:
    response = example.get("response")
    if isinstance(response, str) and response.strip():
        return True
    messages = example.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if _message_text(message.get("content")):
            return True
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return True
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def _assistant_message_has_training_signal(message: dict[str, Any]) -> bool:
    if message.get("role") not in {"assistant", "model"}:
        return False
    if _message_text(message.get("content")):
        return True
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return True
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _assistant_message_has_tool_calls(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _assistant_message_is_provider_error(message: dict[str, Any]) -> bool:
    if message.get("role") not in {"assistant", "model"}:
        return False
    if message.get("teich_provider_error"):
        return True
    content = _message_text(message.get("content")).strip()
    return content.startswith("API Error:") or content.startswith("OpenRouter upstream request failed.")


def _training_example_ends_on_provider_error(example: dict[str, Any]) -> bool:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"assistant", "model"}:
            return _assistant_message_is_provider_error(message)
        if role == "tool":
            return False
    return False


def _training_example_completed_turns(example: dict[str, Any], turn_prompts: list[str]) -> bool:
    if not turn_prompts or not _training_example_has_answer(example):
        return False
    messages = example.get("messages")
    if not isinstance(messages, list):
        return False

    expected_keys = [_prompt_text_completion_key(prompt) for prompt in turn_prompts]
    user_indexes: list[int] = []
    search_start = 0
    for expected_key in expected_keys:
        matched_index: int | None = None
        for index in range(search_start, len(messages)):
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            if _prompt_text_completion_key(_message_text(message.get("content"))) != expected_key:
                continue
            matched_index = index
            break
        if matched_index is None:
            return False
        user_indexes.append(matched_index)
        search_start = matched_index + 1

    for user_position, user_index in enumerate(user_indexes):
        next_user_index = user_indexes[user_position + 1] if user_position + 1 < len(user_indexes) else len(messages)
        turn_messages = [
            message
            for message in messages[user_index + 1:next_user_index]
            if isinstance(message, dict) and message.get("role") in {"assistant", "model", "tool"}
        ]
        if not any(_assistant_message_has_training_signal(message) for message in turn_messages):
            return False
        if user_position == len(user_indexes) - 1:
            if not turn_messages:
                return False
            last_message = turn_messages[-1]
            if last_message.get("role") not in {"assistant", "model"}:
                return False
            if _assistant_message_is_provider_error(last_message):
                return False
            if _assistant_message_has_tool_calls(last_message):
                return False
    return True


def _structured_rows_from_jsonl(path: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                return None
            rows.append(event)
            if isinstance(event.get("messages"), list):
                structured_rows.append(event)
    return structured_rows or None


def _prompt_from_training_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return next(
        (
            _message_text(message.get("content"))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        "",
    )


def _system_from_training_example(example: dict[str, Any]) -> str | None:
    system = example.get("system")
    if isinstance(system, str) and system.strip():
        return system.strip()
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = _message_text(message.get("content"))
        if content:
            return content
    return None


def completed_prompt_keys_from_outputs(traces_dir: Path, excluded_dirs: list[Path] | None = None) -> set[str]:
    if not traces_dir.exists():
        return set()
    completed: set[str] = set()
    for path in sorted(traces_dir.rglob("*.jsonl")):
        if not path.is_file():
            continue
        if _is_non_data_trace_path(path, traces_dir, excluded_dirs):
            continue
        try:
            structured_rows = _structured_rows_from_jsonl(path)
            if structured_rows is not None:
                examples = structured_rows
            else:
                examples = [convert_trace_to_training_example(path).to_dict()]
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for example in examples:
            prompt = example.get("prompt") if isinstance(example.get("prompt"), str) else ""
            if not prompt.strip():
                prompt = _prompt_from_training_messages(example.get("messages"))
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            prompt_input = _prompt_input_from_training_example(example, prompt)
            turn_prompt_input = prompt_input if isinstance(prompt_input, PromptInput) else None
            messages = example.get("messages")
            has_user_message = isinstance(messages, list) and any(
                isinstance(message, dict) and message.get("role") == "user"
                for message in messages
            )
            if has_user_message:
                is_complete = _training_example_completed_turns(example, _agent_turn_prompts(prompt, turn_prompt_input))
            else:
                is_complete = _training_example_has_answer(example) and not _training_example_ends_on_provider_error(example)
            if is_complete:
                completed.add(_prompt_resume_key(prompt_input))
    return completed


def _prompt_input_from_training_example(example: dict[str, Any], prompt: str) -> PromptInput | str:
    system = _system_from_training_example(example)
    follow_up_prompts = example.get("follow_up_prompts")
    if not isinstance(follow_up_prompts, list):
        messages = example.get("messages")
        if not isinstance(messages, list):
            return PromptInput(prompt=prompt, system=system) if system else prompt
        user_prompts = [
            _message_text(message.get("content"))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        user_prompts = [item for item in user_prompts if item]
        if len(user_prompts) <= 1 or _prompt_text_completion_key(user_prompts[0]) != _prompt_text_completion_key(prompt):
            return PromptInput(prompt=prompt, system=system) if system else prompt
        follow_up_prompts = user_prompts[1:]
    try:
        return PromptInput(prompt=prompt, system=system, follow_up_prompts=follow_up_prompts)
    except ValueError:
        return prompt


def unique_prompt_inputs_by_completion_key(prompt_inputs: list[PromptInput]) -> list[PromptInput]:
    unique: list[PromptInput] = []
    seen: set[str] = set()
    for prompt_input in prompt_inputs:
        key = _prompt_completion_key(prompt_input)
        if key in seen:
            continue
        seen.add(key)
        unique.append(prompt_input)
    return unique


def pending_prompt_inputs_for_resume(
    prompt_inputs: list[PromptInput],
    traces_dir: Path,
    excluded_dirs: list[Path] | None = None,
) -> list[PromptInput]:
    prompt_inputs = unique_prompt_inputs_by_completion_key(prompt_inputs)
    completed = completed_prompt_keys_from_outputs(traces_dir, excluded_dirs=excluded_dirs)
    if not completed:
        return prompt_inputs
    return [
        prompt_input
        for prompt_input in prompt_inputs
        if _prompt_resume_key(prompt_input) not in completed
    ]


def prompt_inputs_for_run(
    prompt_inputs: list[PromptInput],
    traces_dir: Path,
    *,
    resume: bool = False,
    excluded_dirs: list[Path] | None = None,
) -> list[PromptInput]:
    if resume:
        return pending_prompt_inputs_for_resume(prompt_inputs, traces_dir, excluded_dirs=excluded_dirs)
    return unique_prompt_inputs_by_completion_key(prompt_inputs)


class DockerRuntimeRunner:
    """Shared Docker runtime used by agent runners."""

    def __init__(self, config: Config):
        self.config = config
        self.image_name = self._runtime_image_name()
        self._configured_tools_snapshot: list[dict[str, Any]] | None = None
        self._active_processes: dict[subprocess.Popen[str], str | None] = {}
        self._active_processes_lock = threading.Lock()
        self._active_containers: set[str] = set()
        self._active_containers_lock = threading.Lock()
        self._ensure_image()

    def _runtime_image_name(self) -> str:
        return RUNTIME_IMAGE_NAME

    def _runtime_build_args(self) -> list[str]:
        return []

    def _configured_tools(self) -> list[dict[str, Any]]:
        if self._configured_tools_snapshot is None:
            self._configured_tools_snapshot = snapshot_configured_tools(self.config)
        return deepcopy(self._configured_tools_snapshot)

    @staticmethod
    def _append_jsonl_events(path: Path, events: list[dict[str, object]]) -> None:
        if not events:
            return
        with path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _finalize_trace_export(self, trace_path: Path, prompt_input: PromptInput | None = None) -> None:
        return

    @staticmethod
    def _runtime_dockerfile_path() -> Path:
        package_path = Path(__file__).parent / "docker" / RUNTIME_DOCKERFILE_NAME
        if package_path.exists():
            return package_path
        return Path(__file__).parent.parent.parent / "docker" / RUNTIME_DOCKERFILE_NAME

    def _image_created_at(self) -> datetime | None:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name, "--format", "{{.Created}}"],
                capture_output=True,
                check=True,
                **TEXT_SUBPROCESS_KWARGS,
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
                check=True,
                **TEXT_SUBPROCESS_KWARGS,
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
            [
                "docker",
                "build",
                *self._runtime_build_args(),
                "-t",
                self.image_name,
                "-f",
                str(dockerfile_path),
                str(context),
            ],
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

    def _langfuse_container_base_url(self) -> str | None:
        return self._container_base_url(self.config.agent.langfuse.base_url)

    def _langfuse_is_host_local(self) -> bool:
        langfuse = self.config.agent.langfuse
        return langfuse.enabled and "host.docker.internal" in (
            self._langfuse_container_base_url() or ""
        )

    @staticmethod
    def _prompt_preview(prompt: str, limit: int = 60) -> str:
        normalized = " ".join(prompt.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @staticmethod
    def _container_name(kind: str, session_id: str) -> str:
        safe_session_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id).strip("-")
        return f"teich-{kind}-{safe_session_id}"

    def _register_active_process(self, process: subprocess.Popen[str], container_name: str | None) -> None:
        with self._active_processes_lock:
            self._active_processes[process] = container_name

    def _unregister_active_process(self, process: subprocess.Popen[str]) -> None:
        with self._active_processes_lock:
            self._active_processes.pop(process, None)

    def _register_active_container(self, container_name: str | None) -> None:
        if not container_name:
            return
        with self._active_containers_lock:
            self._active_containers.add(container_name)

    def _unregister_active_container(self, container_name: str | None) -> None:
        if not container_name:
            return
        with self._active_containers_lock:
            self._active_containers.discard(container_name)

    @staticmethod
    def _remove_container(container_name: str | None) -> None:
        if not container_name:
            return
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
            **TEXT_SUBPROCESS_KWARGS,
        )

    def _terminate_process(
        self,
        process: subprocess.Popen[str],
        container_name: str | None,
        *,
        remove_container: bool = True,
    ) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if remove_container:
                self._remove_container(container_name)

    def _terminate_active_processes(self) -> None:
        with self._active_processes_lock:
            active = list(self._active_processes.items())
        for process, container_name in active:
            self._terminate_process(process, container_name)
            self._unregister_active_process(process)
        with self._active_containers_lock:
            active_containers = list(self._active_containers)
        for container_name in active_containers:
            self._remove_container(container_name)
            self._unregister_active_container(container_name)

    @staticmethod
    def _start_container(command: list[str]) -> None:
        try:
            subprocess.run(command, capture_output=True, check=True, **TEXT_SUBPROCESS_KWARGS)
        except FileNotFoundError as exc:
            if shutil.which(command[0]) is None:
                raise RuntimeError(
                    "Docker runtime not available. Ensure Docker is installed and the runtime image can be built."
                ) from exc
            raise RuntimeError(f"Failed to start Docker runtime process: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
            details = stderr.strip() or stdout.strip() or str(exc)
            raise RuntimeError(f"Failed to start Docker runtime container: {details}") from exc

    def _start_tracked_container(self, command: list[str], container_name: str | None) -> None:
        self._register_active_container(container_name)
        try:
            self._start_container(command)
        except BaseException:
            self._remove_container(container_name)
            self._unregister_active_container(container_name)
            raise

    def _cleanup_tracked_container(self, container_name: str | None) -> None:
        self._remove_container(container_name)
        self._unregister_active_container(container_name)

    @staticmethod
    def _resolve_unique_path(directory: Path, file_name: str) -> Path:
        destination = directory / file_name
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

    def _failure_destination(self, trace_path: Path) -> Path:
        return self._resolve_unique_path(self.config.output.failures_dir, trace_path.name)

    def _discard_output_file(self, trace_path: Path) -> None:
        """Move an invalid generated trace out of the clean output directory."""
        try:
            if trace_path.exists() and trace_path.is_file():
                destination = self._failure_destination(trace_path)
                if trace_path.resolve() == destination.resolve():
                    return
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(trace_path), str(destination))
                metadata_path = self._hermes_metadata_path(trace_path)
                if metadata_path.exists() and metadata_path.is_file():
                    metadata_destination = destination.with_suffix(".metadata.json")
                    if metadata_destination.exists():
                        metadata_destination = self._resolve_unique_path(
                            metadata_destination.parent,
                            metadata_destination.name,
                        )
                    shutil.move(str(metadata_path), str(metadata_destination))
        except OSError:
            pass

    def _trace_contains_completed_turns(self, trace_path: Path, turn_prompts: list[str]) -> bool:
        try:
            examples = convert_traces_to_training_data(trace_path)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return False
        return any(_training_example_completed_turns(example, turn_prompts) for example in examples)

    def _accept_trace_if_complete(self, trace_path: Path, turn_prompts: list[str]) -> Path | None:
        if self._trace_contains_completed_turns(trace_path, turn_prompts):
            return trace_path
        self._discard_output_file(trace_path)
        return None

    @staticmethod
    def _hermes_metadata_path(trace_path: Path) -> Path:
        return trace_path.with_suffix(".metadata.json")

    @classmethod
    def _load_hermes_metadata_sidecar(cls, trace_file: Path) -> dict[str, Any] | None:
        metadata_path = cls._hermes_metadata_path(trace_file)
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return metadata if isinstance(metadata, dict) else None

    @staticmethod
    def _hermes_session_payload_to_usage(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_tokens": payload.get("input_tokens"),
            "output_tokens": payload.get("output_tokens"),
            "reasoning_tokens": payload.get("reasoning_tokens"),
            "cached_input_tokens": payload.get("cache_read_tokens"),
            "cacheWrite": payload.get("cache_write_tokens"),
            "total_tokens": payload.get("total_tokens"),
        }

    @classmethod
    def _apply_hermes_session_metrics(cls, metrics: TraceMetrics, payload: dict[str, Any]) -> None:
        provider = payload.get("model_provider") or payload.get("billing_provider")
        model = payload.get("model")
        if isinstance(provider, str) and provider.strip() and not metrics.provider:
            metrics.provider = provider.strip()
        if isinstance(model, str) and model.strip() and not metrics.model:
            metrics.model = model.strip()
        metrics.add_structured_usage(cls._hermes_session_payload_to_usage(payload))
        cost = payload.get("actual_cost_usd")
        if cost is None:
            cost = payload.get("estimated_cost_usd")
        if cost is None:
            cost = payload.get("total_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            metrics.has_cost = True
            metrics.total_cost += float(cost)

    @staticmethod
    def _copy_workspace_snapshot(workspace: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            workspace,
            destination,
            dirs_exist_ok=False,
            ignore_dangling_symlinks=True,
        )

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
                check=True,
                **TEXT_SUBPROCESS_KWARGS,
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
        workspace_root.chmod(0o777)
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

    @staticmethod
    def _read_new_stdout_json_events(
        stdout_handle,
        offset: int,
        buffer: str,
    ) -> tuple[list[dict[str, Any]], int, str]:
        stdout_handle.flush()
        stdout_handle.seek(offset)
        chunk = stdout_handle.read()
        offset = stdout_handle.tell()
        if not chunk:
            return [], offset, buffer

        buffer += chunk
        if buffer.endswith(("\n", "\r")):
            raw_lines = buffer.splitlines()
            buffer = ""
        else:
            raw_lines = buffer.splitlines()
            buffer = raw_lines.pop() if raw_lines else buffer

        events: list[dict[str, Any]] = []
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events, offset, buffer

    def _stdout_progress_update(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[TraceMetrics | None, str | None]:
        return None, None

    @staticmethod
    def _progress_metrics_signature(metrics: TraceMetrics | None) -> tuple[object, ...] | None:
        if metrics is None:
            return None
        return (
            metrics.provider,
            metrics.model,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.reasoning_tokens,
            metrics.cache_read_tokens,
            metrics.cache_write_tokens,
            metrics.total_tokens,
            metrics.est_total_tokens,
            metrics.has_token_usage,
            metrics.total_cost,
            metrics.has_cost,
        )

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
        stdout_offset = 0
        stdout_buffer = ""
        last_stdout_signature: tuple[str | None, tuple[object, ...] | None] | None = None
        last_stdout_metrics_signature: tuple[object, ...] | None = None
        while True:
            return_code = process.poll()
            trace_path: Path | None = None
            metrics: TraceMetrics | None = None
            stdout_events, stdout_offset, stdout_buffer = self._read_new_stdout_json_events(
                stdout_handle,
                stdout_offset,
                stdout_buffer,
            )
            if stdout_events and progress_callback and progress_base:
                stdout_metrics, stdout_details = self._stdout_progress_update(stdout_events)
                stdout_metrics_signature = self._progress_metrics_signature(stdout_metrics)
                if stdout_metrics_signature is not None:
                    if stdout_metrics_signature == last_stdout_metrics_signature:
                        stdout_metrics = None
                    else:
                        last_stdout_metrics_signature = stdout_metrics_signature
                stdout_signature = (
                    stdout_details,
                    stdout_metrics_signature,
                )
                if (stdout_details or stdout_metrics is not None) and stdout_signature != last_stdout_signature:
                    last_stdout_signature = stdout_signature
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
                            details=stdout_details,
                            metrics=stdout_metrics,
                            metrics_delta=stdout_metrics is not None,
                        )
                    )
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

    @staticmethod
    def _openrouter_generation_ids_from_value(value: Any) -> set[str]:
        generation_ids: set[str] = set()
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key in {"id", "generation_id"} and isinstance(nested_value, str) and nested_value.startswith("gen-"):
                    generation_ids.add(nested_value)
                    continue
                if key == "generation_ids" and isinstance(nested_value, list):
                    generation_ids.update(
                        item
                        for item in nested_value
                        if isinstance(item, str) and item.startswith("gen-")
                    )
                    continue
                if isinstance(nested_value, (dict, list)):
                    generation_ids.update(DockerRuntimeRunner._openrouter_generation_ids_from_value(nested_value))
            return generation_ids
        if isinstance(value, list):
            for nested_value in value:
                if isinstance(nested_value, (dict, list)):
                    generation_ids.update(DockerRuntimeRunner._openrouter_generation_ids_from_value(nested_value))
        return generation_ids

    @staticmethod
    def _openrouter_usage_from_generation_data(data: dict[str, Any]) -> dict[str, Any] | None:
        input_tokens = TraceMetrics._int_value(data.get("native_tokens_prompt") or data.get("tokens_prompt"))
        output_tokens = TraceMetrics._int_value(
            data.get("native_tokens_completion") or data.get("tokens_completion")
        )
        reasoning_tokens = TraceMetrics._int_value(data.get("native_tokens_reasoning"))
        prompt_token_details = data.get("prompt_tokens_details")
        if not isinstance(prompt_token_details, dict):
            prompt_token_details = {}
        cache_read_tokens = TraceMetrics._int_value(
            data.get("native_tokens_cached")
            or data.get("cacheRead")
            or data.get("cached_input_tokens")
            or prompt_token_details.get("cached_tokens")
        )
        cache_write_tokens = TraceMetrics._int_value(
            data.get("cacheWrite")
            or prompt_token_details.get("cache_write_tokens")
        )
        total_tokens = input_tokens + output_tokens + reasoning_tokens
        cost = data.get("total_cost")
        if not total_tokens and not cache_read_tokens and not cache_write_tokens and not isinstance(cost, (int, float)):
            return None
        usage: dict[str, Any] = {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "cacheRead": cache_read_tokens,
            "totalTokens": total_tokens,
        }
        if cache_write_tokens:
            usage["cacheWrite"] = cache_write_tokens
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            usage["cost"] = {"total": float(cost)}
        generation_id = data.get("id")
        if isinstance(generation_id, str) and generation_id.strip():
            usage["generation_id"] = generation_id.strip()
        provider_name = data.get("provider_name")
        if isinstance(provider_name, str) and provider_name.strip():
            usage["provider_name"] = provider_name.strip()
        return usage

    def _openrouter_generation_usage(self, generation_id: str) -> dict[str, Any] | None:
        if self.config.api.provider.strip().lower() != "openrouter":
            return None
        api_key = self.config.get_api_key()
        if not api_key:
            return None
        base_url = (self.config.get_base_url() or "https://openrouter.ai/api/v1").rstrip("/")
        endpoint = f"{base_url}/generation?{urlencode({'id': generation_id})}"
        request = Request(
            endpoint,
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            method="GET",
        )
        attempts = OPENROUTER_GENERATION_STATS_ATTEMPTS
        for attempt in range(attempts):
            try:
                with urlopen(
                    request,
                    timeout=min(OPENROUTER_GENERATION_STATS_TIMEOUT_SECONDS, self.config.timeout_seconds),
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
                if attempt < attempts - 1:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                return None
            if not isinstance(payload, dict):
                return None
            data = payload.get("data")
            if isinstance(data, dict):
                usage = self._openrouter_usage_from_generation_data(data)
                if usage is not None:
                    return usage
            if attempt < attempts - 1:
                time.sleep(0.4 * (attempt + 1))
        return None

    def _openrouter_usage_from_generation_ids(self, generation_ids: set[str]) -> dict[str, Any] | None:
        usages = [
            usage
            for generation_id in sorted(generation_ids)
            if (usage := self._openrouter_generation_usage(generation_id)) is not None
        ]
        return ChatRunner._merge_usage_totals(usages)

    def _openrouter_generation_ids_from_trace(self, trace_file: Path) -> set[str]:
        generation_ids: set[str] = set()
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    generation_ids.update(self._openrouter_generation_ids_from_value(event))
        except OSError:
            return set()
        return generation_ids

    @classmethod
    def _summarize_trace_file(cls, trace_file: Path) -> TraceMetrics:
        metrics = TraceMetrics()
        sidecar_metadata = cls._load_hermes_metadata_sidecar(trace_file)
        if sidecar_metadata is not None:
            cls._apply_hermes_session_metrics(metrics, sidecar_metadata)
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

                metadata = event.get("metadata")
                if isinstance(event.get("traces"), list) and isinstance(metadata, dict):
                    cls._apply_hermes_session_metrics(metrics, metadata)
                    continue

                if isinstance(event.get("messages"), list):
                    if isinstance(event.get("source"), str) and event.get("source") == "cli":
                        cls._apply_hermes_session_metrics(metrics, event)
                        continue
                    if isinstance(event.get("provider"), str) and event.get("provider") and not metrics.provider:
                        metrics.provider = event["provider"]
                    metadata = event.get("metadata")
                    if isinstance(metadata, dict):
                        provider = metadata.get("model_provider")
                        model = metadata.get("model")
                        if isinstance(provider, str) and provider.strip() and not metrics.provider:
                            metrics.provider = provider.strip()
                        if isinstance(model, str) and model.strip() and not metrics.model:
                            metrics.model = model.strip()
                        usage = metadata.get("usage")
                        if isinstance(usage, dict):
                            metrics.add_structured_usage(usage)
                    if isinstance(event.get("model"), str) and event.get("model") and not metrics.model:
                        metrics.model = event["model"]
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        metrics.add_structured_usage(usage)
                    continue

                event_type = event.get("type")
                if event_type == "hermes_session_meta":
                    payload = event.get("payload")
                    if isinstance(payload, dict):
                        cls._apply_hermes_session_metrics(metrics, payload)
                    continue

                if event_type == "external_session_meta":
                    payload = event.get("payload")
                    if isinstance(payload, dict):
                        cls._apply_hermes_session_metrics(metrics, payload)
                    continue

                if event_type == "system" and event.get("subtype") == "init":
                    model = event.get("model")
                    if isinstance(model, str) and model.strip() and not metrics.model:
                        metrics.model = model.strip()
                    continue

                if event_type == "assistant":
                    payload = event.get("message")
                    if isinstance(payload, dict):
                        model = payload.get("model")
                        if isinstance(model, str) and model.strip() and not metrics.model:
                            metrics.model = model.strip()
                        usage = payload.get("usage")
                        if isinstance(usage, dict):
                            metrics.add_structured_usage(usage)
                    continue

                if event_type == "result":
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        metrics.add_structured_usage(usage)
                    total_cost = event.get("total_cost_usd")
                    if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
                        metrics.has_cost = True
                        metrics.total_cost += float(total_cost)
                    continue

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

    def _summarize_trace_file_with_provider_stats(self, trace_file: Path) -> TraceMetrics:
        metrics = self._summarize_trace_file(trace_file)
        provider_stats = self._openrouter_usage_from_generation_ids(self._openrouter_generation_ids_from_trace(trace_file))
        if provider_stats is not None:
            metrics.input_tokens = 0
            metrics.output_tokens = 0
            metrics.reasoning_tokens = 0
            metrics.cache_read_tokens = 0
            metrics.cache_write_tokens = 0
            metrics.total_tokens = 0
            metrics.est_total_tokens = 0
            metrics.total_cost = 0.0
            metrics.has_token_usage = False
            metrics.has_cost = False
            metrics.add_structured_usage(provider_stats)
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
        prompt_inputs: list[PromptInput] | None = None,
        resume: bool = False,
    ) -> list[Path]:
        prompt_inputs = prompt_inputs if prompt_inputs is not None else self.config.get_prompt_inputs()
        prompt_inputs = prompt_inputs_for_run(
            prompt_inputs,
            self.config.output.traces_dir,
            resume=resume,
            excluded_dirs=[self.config.output.failures_dir],
        )
        if not prompt_inputs:
            if resume:
                return []
            raise ValueError("No prompts configured")

        total_prompts = len(prompt_inputs)
        worker_count = max(1, min(max_concurrency, total_prompts))
        prompt_queue: queue.Queue[tuple[int, PromptInput]] = queue.Queue()
        for item in enumerate(prompt_inputs, start=1):
            prompt_queue.put(item)
        results_by_index: dict[int, Path] = {}
        errors: list[Exception] = []
        result_lock = threading.Lock()
        stop_event = threading.Event()

        def emit_queued(prompt_index: int, prompt_input: PromptInput) -> None:
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

        for prompt_index, prompt_input in enumerate(prompt_inputs, start=1):
            emit_queued(prompt_index, prompt_input)

        def worker() -> None:
            while True:
                if stop_event.is_set():
                    return
                try:
                    prompt_index, prompt_input = prompt_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = self._run_prompt_task(
                        f"prompt-{prompt_index}",
                        prompt_index,
                        total_prompts,
                        prompt_input,
                        progress_callback,
                    )
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)
                else:
                    with result_lock:
                        results_by_index[prompt_index] = result
                finally:
                    prompt_queue.task_done()

        threads = [
            threading.Thread(target=worker, name=f"teich-prompt-worker-{index}", daemon=True)
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=0.1)
        except KeyboardInterrupt:
            stop_event.set()
            self._terminate_active_processes()
            raise

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
        container_name: str | None = None,
        stdin_text: str | None = None,
        remove_container_on_error: bool = True,
    ) -> None:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_handle, tempfile.TemporaryFile(
            mode="w+", encoding="utf-8"
        ) as stderr_handle:
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE if stdin_text is not None else None,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    **TEXT_SUBPROCESS_KWARGS,
                )
            except FileNotFoundError as exc:
                if shutil.which(command[0]) is None:
                    raise RuntimeError(
                        "Docker runtime not available. Ensure Docker is installed and the runtime image can be built."
                    ) from exc
                raise RuntimeError(f"Failed to start Docker runtime process: {exc}") from exc
            self._register_active_process(process, container_name)
            try:
                if stdin_text is not None and process.stdin is not None:
                    process.stdin.write(stdin_text)
                    process.stdin.close()
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
            except BaseException:
                self._terminate_process(process, container_name, remove_container=remove_container_on_error)
                raise
            finally:
                self._unregister_active_process(process)


class CodexRunner(DockerRuntimeRunner):
    """Manages Docker-based Codex sessions."""

    def __init__(self, config: Config):
        self._broker: CodexTokenBroker | None = None
        self._broker_lock = threading.Lock()
        self._langfuse_plugin_cache: Path | None = None
        self._langfuse_plugin_lock = threading.Lock()
        super().__init__(config)

    def _runtime_image_name(self) -> str:
        if self.config.agent.langfuse.enabled:
            return LANGFUSE_RUNTIME_IMAGE_NAME
        return RUNTIME_IMAGE_NAME

    def _runtime_build_args(self) -> list[str]:
        if self.config.agent.langfuse.enabled:
            return ["--build-arg", "TEICH_INSTALL_LANGFUSE=1"]
        return []

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

    def _ensure_langfuse_plugin_cache(self) -> Path | None:
        """Extract the image-baked Langfuse plugin once per run."""
        if not self.config.agent.langfuse.enabled:
            return None
        if self._langfuse_plugin_cache is None:
            with self._langfuse_plugin_lock:
                if self._langfuse_plugin_cache is None:
                    self._ensure_image()
                    cache_dir = Path(tempfile.mkdtemp(prefix="teich-langfuse-plugin-"))
                    self._extract_langfuse_plugin(cache_dir)
                    self._langfuse_plugin_cache = cache_dir
        return self._langfuse_plugin_cache

    def _extract_langfuse_plugin(self, cache_dir: Path) -> None:
        created = subprocess.run(
            ["docker", "create", self.image_name],
            capture_output=True,
            text=True,
            check=True,
        )
        container_id = created.stdout.strip()
        try:
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container_id}:{LANGFUSE_PLUGIN_STAGE_CODEX_HOME}/plugins",
                    str(cache_dir / "plugins"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                check=False,
            )

    def _install_codex_langfuse_plugin(self, codex_home: Path) -> None:
        cache = self._ensure_langfuse_plugin_cache()
        if cache is None:
            return
        source = cache / "plugins"
        dest = codex_home / "plugins"
        shutil.copytree(source, dest, dirs_exist_ok=True)
        for path in dest.rglob("*"):
            path.chmod(0o777 if path.is_dir() else 0o644)
        dest.chmod(0o777)

    def _prepare_shared_host_auth(self) -> Path:
        """Seed (once) and return the shared project ``auth.json`` snapshot.

        Copies the host Codex login into ``agent.codex.auth_dir`` the first time,
        and re-seeds only when the host file is newer than the snapshot, so a
        token that Codex has already rotated in place is never clobbered by a
        now-stale host file.
        """
        source = self.config.get_codex_host_auth_source()
        if not source.exists():
            raise RuntimeError(
                f"Codex host auth file not found: {source}. Run `codex login` on the host "
                "first, or disable agent.codex.use_host_auth."
            )
        shared_dir = self.config.agent.codex.auth_dir
        shared_dir.mkdir(parents=True, exist_ok=True)
        # The snapshot holds live credentials; keep it out of version control by
        # default rather than relying on the user's project .gitignore.
        gitignore = shared_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n", encoding="utf-8")
        shared_auth = shared_dir / "auth.json"
        if not shared_auth.exists() or source.stat().st_mtime > shared_auth.stat().st_mtime:
            shutil.copyfile(source, shared_auth)
            shared_auth.chmod(0o666)
        return shared_auth

    def _ensure_broker(self) -> CodexTokenBroker | None:
        """Start (once per run) and return the host-side token broker.

        Returns ``None`` when host auth is disabled. The broker seeds itself
        from the ``auth_dir`` snapshot and owns the rotating refresh token for
        the whole run; every container is pointed at it via the refresh-URL
        override and seeded with its own copy.
        """
        if not self.config.agent.codex.use_host_auth:
            return None
        # run_all drives sessions from a thread pool, so guard creation: only one
        # thread may build+start the single shared broker (double-checked lock).
        if self._broker is None:
            with self._broker_lock:
                if self._broker is None:
                    shared_auth = self._prepare_shared_host_auth()
                    broker = CodexTokenBroker(
                        shared_auth,
                        port=self.config.agent.codex.broker_port,
                    )
                    broker.start()
                    self._broker = broker
        return self._broker

    def _write_seeded_codex_auth(self, codex_home: Path, broker: CodexTokenBroker) -> None:
        """Write a per-container ``auth.json`` (real tokens, secret refresh token)."""
        auth_path = codex_home / "auth.json"
        auth_path.write_text(
            json.dumps(broker.seed_auth_json(), indent=2) + "\n",
            encoding="utf-8",
        )
        # 0o666 (matching _write_codex_config's config.toml) so the in-container
        # `codex` user can read/rewrite the bind-mounted file regardless of how
        # its uid maps to the host. This copy holds only a short-lived access
        # token and the per-run broker secret -- never the real refresh token.
        auth_path.chmod(0o666)

    def _write_codex_config(self, codex_home: Path) -> None:
        codex_home.mkdir(parents=True, exist_ok=True)
        codex_home.chmod(0o777)
        lines: list[str] = []
        if self.config.developer_instructions:
            lines.append(
                f"developer_instructions = {self._toml_string(self.config.developer_instructions)}"
            )
        if self.config.model.reasoning_effort:
            lines.append(
                f"model_reasoning_effort = {self._toml_string(self.config.model.reasoning_effort)}"
            )
        if self.config.model.reasoning_summary:
            lines.append(
                f"model_reasoning_summary = {self._toml_string(self.config.model.reasoning_summary)}"
            )
        if self.config.model.service_tier:
            lines.append(
                f"service_tier = {self._toml_string(self.config.model.service_tier)}"
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

        if self.config.agent.langfuse.enabled:
            lines.extend(
                [
                    "",
                    "[features]",
                    "plugin_hooks = true",
                    "",
                    f'[plugins."{LANGFUSE_PLUGIN_ID}"]',
                    "enabled = true",
                ]
            )

        config_text = "\n".join(lines).strip()
        config_path = codex_home / "config.toml"
        config_path.write_text(
            (config_text + "\n") if config_text else "",
            encoding="utf-8",
        )
        config_path.chmod(0o666)

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
        return _normalize_teich_prompt_user_event(normalize_codex_trace_event(event))

    @classmethod
    def _copy_normalized_session_file(cls, source_path: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        events = _read_jsonl_dict_events(source_path)
        if events is None:
            shutil.copyfile(source_path, destination)
            return
        normalized_events = normalize_codex_trace_events(events, normalize_custom_tools=False)
        if normalized_events == events:
            shutil.copyfile(source_path, destination)
            return
        _write_jsonl_dict_events(destination, normalized_events)

    @staticmethod
    def _codex_tool_schema_event(tool: dict[str, Any]) -> dict[str, object] | None:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        schema = {key: deepcopy(value) for key, value in function.items() if key != "name"}
        return {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "response_item",
            "payload": {
                "type": "tool_schema",
                "name": name.strip(),
                "schema": schema,
                "source": "teich",
            },
        }

    @staticmethod
    def _codex_existing_tool_schema_names(events: list[dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for event in events:
            if event.get("type") != "response_item":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "tool_schema":
                continue
            name = payload.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names

    def _append_codex_available_tool_schemas(self, trace_path: Path) -> None:
        if not trace_path.exists():
            return
        events = _read_jsonl_dict_events(trace_path) or []
        existing_names = self._codex_existing_tool_schema_names(events)
        metadata_events: list[dict[str, object]] = []
        for tool in self._configured_tools():
            event = self._codex_tool_schema_event(tool)
            if event is None:
                continue
            payload = event.get("payload")
            name = payload.get("name") if isinstance(payload, dict) else None
            if not isinstance(name, str) or name in existing_names:
                continue
            existing_names.add(name)
            metadata_events.append(event)
        self._append_jsonl_events(trace_path, metadata_events)

    def _finalize_trace_export(self, trace_path: Path, prompt_input: PromptInput | None = None) -> None:
        self._append_codex_available_tool_schemas(trace_path)

    def _codex_base_url_and_proxy_target(self) -> tuple[str | None, str | None]:
        configured_base_url = self.config.get_base_url()
        base_url = configured_base_url
        provider_name = self.config.api.provider
        proxy_target = self._local_provider_proxy_target(provider_name, configured_base_url)
        if configured_base_url and not self._is_oss_local_provider(provider_name):
            base_url = self._container_base_url(configured_base_url)
        return base_url, proxy_target

    def _build_codex_docker_base_command(
        self,
        workspace: Path,
        codex_home: Path,
        container_name: str,
        *,
        detached: bool = False,
    ) -> tuple[list[str], str | None]:
        configured_base_url = self.config.get_base_url()
        api_key = self.config.get_api_key() or ("none" if configured_base_url else "")
        base_url, proxy_target = self._codex_base_url_and_proxy_target()
        provider_name = self.config.api.provider
        provider_env_key = self._provider_env_key(provider_name)
        broker = self._ensure_broker()
        langfuse = self.config.agent.langfuse
        langfuse_base_url = self._langfuse_container_base_url()
        cmd = [
            "docker",
            "run",
            *([] if detached else ["--rm", "-i"]),
            *(["-d"] if detached else []),
            "--name",
            container_name,
            "--user",
            RUNTIME_CONTAINER_USER,
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
        broker_active = broker is not None
        if (
            proxy_target
            or broker_active
            or self._langfuse_is_host_local()
            or (configured_base_url and base_url != configured_base_url)
        ):
            cmd.extend([
                "--add-host",
                "host.docker.internal:host-gateway",
            ])
        if broker is not None:
            # Codex stays native (talks to chatgpt.com directly) but every
            # token refresh is redirected to the host-side broker, which is the
            # sole owner of the rotating refresh token. Codex is seeded with its
            # own auth.json copy in CODEX_HOME (see _write_seeded_codex_auth).
            cmd.extend(["-e", f"CODEX_REFRESH_TOKEN_URL_OVERRIDE={broker.override_url}"])
        if langfuse.enabled:
            cmd.extend(
                [
                    "-e",
                    "TRACE_TO_LANGFUSE=true",
                    "-e",
                    f"LANGFUSE_PUBLIC_KEY={langfuse.public_key}",
                    "-e",
                    f"LANGFUSE_SECRET_KEY={langfuse.secret_key}",
                    "-e",
                    f"LANGFUSE_BASE_URL={langfuse_base_url or ''}",
                ]
            )
        if proxy_target:
            cmd.extend(
                [
                    "-e",
                    f"TEICH_LOCAL_PROVIDER_TARGET={proxy_target}",
                    "-e",
                    f"TEICH_LOCAL_PROVIDER_PORT={self._local_provider_default_port(provider_name)}",
                ]
            )
        if api_key and not broker_active:
            cmd.extend(["-e", f"{provider_env_key}={api_key}"])
        return cmd, proxy_target

    def _build_codex_agent_command(self, resume: bool = False) -> list[str]:
        base_url, _proxy_target = self._codex_base_url_and_proxy_target()
        model = self.config.get_effective_model()
        provider_name = self.config.api.provider
        provider_env_key = self._provider_env_key(provider_name)
        codex_cmd = [
            "codex",
            "--ask-for-approval",
            self.config.model.approval_policy,
            "--sandbox",
            self.config.model.sandbox,
        ]
        if self._is_oss_local_provider(provider_name):
            codex_cmd.extend(
                [
                    "--oss",
                    "--local-provider",
                    self._normalize_oss_local_provider(provider_name),
                ]
            )
        codex_cmd.append("exec")
        if resume:
            codex_cmd.extend(["resume", "--last"])
        codex_cmd.extend(["--model", model])
        codex_cmd.append("--skip-git-repo-check")
        if self.config.agent.langfuse.enabled:
            codex_cmd.append("--dangerously-bypass-hook-trust")
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
            codex_cmd.extend(
                [
                    "--config",
                    f"model_providers.{provider_key}={provider_literal}",
                    "--config",
                    f"model_provider={self._toml_string(provider_key)}",
                ]
            )
        codex_cmd.append("-")
        return codex_cmd

    def _build_codex_command(
        self,
        prompt: str,
        workspace: Path,
        codex_home: Path,
        container_name: str,
        resume: bool = False,
    ) -> list[str]:
        cmd, proxy_target = self._build_codex_docker_base_command(workspace, codex_home, container_name)
        codex_cmd = self._build_codex_agent_command(resume=resume)
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

    def _build_codex_persistent_container_command(
        self,
        workspace: Path,
        codex_home: Path,
        container_name: str,
    ) -> list[str]:
        cmd, proxy_target = self._build_codex_docker_base_command(
            workspace,
            codex_home,
            container_name,
            detached=True,
        )
        cmd.append(self.image_name)
        if proxy_target:
            proxy_script = f"{CODEX_HOME_IN_CONTAINER}/{LOCAL_PROVIDER_PROXY_SCRIPT_NAME}"
            shell_command = (
                f"node {shlex.quote(proxy_script)} >/tmp/local-provider-proxy.log 2>&1 & "
                "exec sleep infinity"
            )
            cmd.extend(["bash", "-lc", shell_command])
        else:
            cmd.extend(["sleep", "infinity"])
        return cmd

    def _build_codex_exec_command(self, container_name: str, resume: bool = False) -> list[str]:
        return [
            "docker",
            "exec",
            "-i",
            "--user",
            RUNTIME_CONTAINER_USER,
            "-w",
            WORKSPACE_IN_CONTAINER,
            container_name,
            *self._build_codex_agent_command(resume=resume),
        ]

    def _latest_session_source_file(
        self,
        codex_home: Path,
        existing_sessions: set[Path],
        started_at: datetime,
    ) -> Path | None:
        session_files = self._list_session_files(codex_home)
        fresh_files = [path for path in session_files if path.resolve() not in existing_sessions]
        if not fresh_files:
            fresh_files = [
                path
                for path in session_files
                if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= started_at
            ]
        if not fresh_files:
            return None
        return max(fresh_files, key=lambda path: path.stat().st_mtime)

    def _extract_session_file(
        self,
        session_id: str,
        codex_home: Path,
        existing_sessions: set[Path],
        started_at: datetime,
    ) -> Path:
        """Extract a session file from the mounted CODEX_HOME and save it to output."""
        source_path = self._latest_session_source_file(codex_home, existing_sessions, started_at)
        if source_path is None:
            raise RuntimeError(f"No session file found for {session_id}")
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
        container_name = self._container_name("codex", session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        failure_trace_saved = False
        failure_trace_checked = False
        existing_sessions: set[Path] = set()

        try:
            self._write_codex_config(codex_home)
            broker = self._ensure_broker()
            if broker is not None:
                self._write_seeded_codex_auth(codex_home, broker)
            self._install_codex_langfuse_plugin(codex_home)
            if self._local_provider_proxy_target(self.config.api.provider, self.config.get_base_url()):
                self._write_local_provider_proxy(codex_home)
            existing_sessions = {path.resolve() for path in self._list_session_files(codex_home)}
            if len(turn_prompts) > 1:
                self._start_tracked_container(
                    self._build_codex_persistent_container_command(
                        workspace,
                        codex_home,
                        container_name,
                    ),
                    container_name,
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                expected_turns = turn_prompts[: turn_index + 1]
                for attempt in range(AGENT_TURN_RETRY_LIMIT + 1):
                    resume = turn_index > 0 or attempt > 0
                    if len(turn_prompts) > 1:
                        cmd = self._build_codex_exec_command(container_name, resume=resume)
                    else:
                        cmd = self._build_codex_command(
                            turn_prompt,
                            workspace,
                            codex_home,
                            container_name,
                            resume=resume,
                        )
                    process_error: subprocess.CalledProcessError | None = None
                    details = ""
                    try:
                        self._run_process(
                            cmd,
                            session_id,
                            started_at,
                            codex_home / "sessions",
                            progress_callback,
                            progress_base,
                            container_name,
                            turn_prompt,
                            remove_container_on_error=len(turn_prompts) <= 1,
                        )
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"Session {session_id[:8]} timed out after {self.config.timeout_seconds}s"
                        )
                    except subprocess.CalledProcessError as exc:
                        process_error = exc
                        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
                        details = stderr.strip() or stdout.strip() or str(exc)
                        if "'type' of tool must be 'function'" in details:
                            details = (
                                f"{details}\nHint: this custom provider endpoint is not fully compatible with Codex's Responses API tool format. "
                                "Use a provider that supports Codex Responses semantics, or use Codex OSS mode with a supported local provider."
                            )

                    source_trace = self._latest_session_source_file(codex_home, existing_sessions, started_at)
                    if source_trace is None and process_error is None:
                        break
                    if source_trace is not None and self._trace_contains_completed_turns(source_trace, expected_turns):
                        break
                    if attempt < AGENT_TURN_RETRY_LIMIT:
                        continue

                    failure_trace_checked = True
                    if turn_index == len(turn_prompts) - 1:
                        try:
                            trace_path = self._extract_session_file(
                                session_id,
                                codex_home,
                                existing_sessions,
                                started_at,
                            )
                        except RuntimeError:
                            trace_path = None
                        if trace_path is not None:
                            accepted_trace = self._accept_trace_if_complete(trace_path, turn_prompts)
                            if accepted_trace is not None:
                                self._finalize_trace_export(trace_path, prompt_input)
                                self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
                                return trace_path
                            failure_trace_saved = True
                    elif source_trace is not None:
                        try:
                            trace_path = self._extract_session_file(
                                session_id,
                                codex_home,
                                existing_sessions,
                                started_at,
                            )
                        except RuntimeError:
                            trace_path = None
                        if trace_path is not None:
                            self._discard_output_file(trace_path)
                            failure_trace_saved = True
                    if process_error is not None:
                        raise RuntimeError(
                            f"Session {session_id[:8]} failed after {AGENT_TURN_RETRY_LIMIT} turn retries: {details}"
                        ) from process_error
                    raise RuntimeError(
                        f"Session {session_id[:8]} stopped before completing turn {turn_index + 1} "
                        f"after {AGENT_TURN_RETRY_LIMIT} retries"
                    )
            trace_path = self._extract_session_file(
                session_id,
                codex_home,
                existing_sessions,
                started_at,
            )
            self._finalize_trace_export(trace_path, prompt_input)
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
            return trace_path
        except BaseException:
            if not failure_trace_saved and not failure_trace_checked:
                try:
                    trace_path = self._extract_session_file(session_id, codex_home, existing_sessions, started_at)
                except RuntimeError:
                    trace_path = None
                if trace_path is not None:
                    self._discard_output_file(trace_path)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._cleanup_tracked_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(codex_home, ignore_errors=True)

    def run_all(
        self,
        max_concurrency: int = 1,
        progress_callback: SessionProgressCallback | None = None,
        prompt_inputs: list[PromptInput] | None = None,
        resume: bool = False,
    ) -> list[Path]:
        try:
            return super().run_all(
                max_concurrency=max_concurrency,
                progress_callback=progress_callback,
                prompt_inputs=prompt_inputs,
                resume=resume,
            )
        finally:
            if self._broker is not None:
                self._broker.stop()
                self._broker = None
            if self._langfuse_plugin_cache is not None:
                shutil.rmtree(self._langfuse_plugin_cache, ignore_errors=True)
                self._langfuse_plugin_cache = None


class ExternalCliRunner(DockerRuntimeRunner):
    """Shared Docker-backed runner for agents that expose a one-shot CLI."""

    provider_name = "external-agent"
    container_kind = "agent"
    home_in_container = "/home/codex/.agent"
    source_name = "external-agent"
    default_model_provider = "external"

    @staticmethod
    def _provider_env_key(provider: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", provider.strip().lower())
        aliases = {
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "claude_code": "ANTHROPIC_API_KEY",
            "claude-code": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "nous": "NOUS_API_KEY",
            "nous_portal": "NOUS_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "xai": "XAI_API_KEY",
            "grok": "XAI_API_KEY",
            "zai": "GLM_API_KEY",
            "z_ai": "GLM_API_KEY",
            "glm": "GLM_API_KEY",
        }
        return aliases.get(normalized, f"{normalized.upper() or 'TEICH'}_API_KEY")

    def _api_env_items(self) -> list[tuple[str, str]]:
        api_key = self.config.get_api_key() or ""
        if not api_key:
            return []
        provider_env_key = self._provider_env_key(self.config.api.provider)
        items = [("TEICH_API_KEY", api_key), (provider_env_key, api_key)]
        return list(dict.fromkeys(items))

    def _base_url_env_items(self) -> list[tuple[str, str]]:
        base_url = self._container_base_url(self.config.get_base_url())
        if not base_url:
            return []
        return [
            ("TEICH_BASE_URL", base_url),
            ("OPENAI_BASE_URL", base_url),
            ("ANTHROPIC_BASE_URL", base_url),
        ]

    def _langfuse_env_items(self) -> list[tuple[str, str]]:
        return []

    def _agent_env_items(self) -> list[tuple[str, str]]:
        """Agent-specific container env vars; overridden per runner."""
        return []

    def _prepare_agent_home(self, home_dir: Path) -> None:
        return

    def _build_external_docker_base_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
        *,
        detached: bool = False,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            *([] if detached else ["--rm"]),
            *(["-d"] if detached else []),
            "--name",
            container_name,
            "--user",
            RUNTIME_CONTAINER_USER,
            "-e",
            "HOME=/home/codex",
            "-e",
            f"HERMES_HOME={HERMES_HOME_IN_CONTAINER}",
            "-v",
            f"{workspace}:{WORKSPACE_IN_CONTAINER}",
            "-v",
            f"{home_dir}:{self.home_in_container}",
            "-w",
            WORKSPACE_IN_CONTAINER,
        ]
        configured_base_url = self.config.get_base_url()
        base_url_is_host_local = bool(
            configured_base_url
            and self._container_base_url(configured_base_url) != configured_base_url
        )
        if base_url_is_host_local or self._langfuse_is_host_local():
            command.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in [
            *self._api_env_items(),
            *self._base_url_env_items(),
            *self._langfuse_env_items(),
            *self._agent_env_items(),
        ]:
            command.extend(["-e", f"{key}={value}"])
        command.append(self.image_name)
        return command

    def _build_shell_command(
        self,
        *,
        continue_session: bool = False,
        resume_session_id: str | None = None,
    ) -> str:
        raise NotImplementedError

    def _wrap_external_shell_command(self, shell_command: str) -> str:
        chmod_targets = " ".join(
            shlex.quote(path)
            for path in (self.home_in_container, WORKSPACE_IN_CONTAINER)
        )
        chmod_command = f"chmod -R a+rwX {chmod_targets} >/dev/null 2>&1 || true"
        return f"set +e; {shell_command}; status=$?; {chmod_command}; exit $status"

    def _build_external_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
        *,
        continue_session: bool = False,
        resume_session_id: str | None = None,
    ) -> list[str]:
        command = self._build_external_docker_base_command(workspace, home_dir, container_name)
        shell_command = self._wrap_external_shell_command(
            self._build_shell_command(
                continue_session=continue_session,
                resume_session_id=resume_session_id,
            )
        )
        command.extend(["bash", "-lc", shell_command])
        return command

    def _build_external_persistent_container_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
    ) -> list[str]:
        command = self._build_external_docker_base_command(workspace, home_dir, container_name, detached=True)
        command.extend(["sleep", "infinity"])
        return command

    def _build_external_exec_command(
        self,
        container_name: str,
        *,
        continue_session: bool = False,
        resume_session_id: str | None = None,
    ) -> list[str]:
        return [
            "docker",
            "exec",
            "--user",
            RUNTIME_CONTAINER_USER,
            "-w",
            WORKSPACE_IN_CONTAINER,
            container_name,
            "bash",
            "-lc",
            self._wrap_external_shell_command(
                self._build_shell_command(
                    continue_session=continue_session,
                    resume_session_id=resume_session_id,
                )
            ),
        ]

    def _run_external_process(self, command: list[str], container_name: str | None) -> tuple[str, str]:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **TEXT_SUBPROCESS_KWARGS,
            )
        except FileNotFoundError as exc:
            if shutil.which(command[0]) is None:
                raise RuntimeError(
                    "Docker runtime not available. Ensure Docker is installed and the runtime image can be built."
                ) from exc
            raise RuntimeError(f"Failed to start Docker runtime process: {exc}") from exc
        self._register_active_process(process, container_name)
        try:
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Session timed out after {self.config.timeout_seconds}s")
            if process.returncode:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    process.args,
                    output=stdout,
                    stderr=stderr,
                )
            return stdout, stderr
        except BaseException:
            self._terminate_process(process, container_name)
            raise
        finally:
            self._unregister_active_process(process)

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

    def _session_meta_event(self, session_id: str, started_at: datetime, workspace: Path) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": session_id,
            "timestamp": started_at.isoformat().replace("+00:00", "Z"),
            "cwd": str(workspace),
            "source": self.source_name,
            "model_provider": self.config.api.provider or self.default_model_provider,
            "model": self.config.get_effective_model(),
        }
        tools = self._configured_tools()
        if tools:
            payload["tools"] = tools
        return {
            "timestamp": started_at.isoformat().replace("+00:00", "Z"),
            "type": "external_session_meta",
            "payload": payload,
        }

    @staticmethod
    def _event_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _events_from_turn_output(
        self,
        prompt: str,
        stdout: str,
        stderr: str,
        *,
        turn_index: int,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = [
            {
                "timestamp": self._event_timestamp(),
                "type": "external_message",
                "role": "user",
                "turn_index": turn_index,
                "content": prompt,
            }
        ]
        if stdout.strip():
            events.append(
                {
                    "timestamp": self._event_timestamp(),
                    "type": "external_message",
                    "role": "assistant",
                    "turn_index": turn_index,
                    "content": stdout.strip(),
                }
            )
        if stderr.strip():
            events.append(
                {
                    "timestamp": self._event_timestamp(),
                    "type": "external_stderr",
                    "turn_index": turn_index,
                    "content": stderr.strip(),
                }
            )
        return events

    def _write_events(self, destination: Path, events: list[dict[str, object]]) -> None:
        with destination.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")

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
        workspace_root, workspace = self._prepare_workspace(session_id, prompt_input, self.container_kind)
        home_dir = Path(tempfile.mkdtemp(prefix=f"{self.container_kind}-home-{session_id}-"))
        home_dir.chmod(0o777)
        self._prepare_agent_home(home_dir)
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name(self.container_kind, session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        destination = self._resolve_output_path(f"{self.source_name}-{session_id}.jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_events(destination, [self._session_meta_event(session_id, started_at, workspace)])
            if len(turn_prompts) > 1:
                self._start_tracked_container(
                    self._build_external_persistent_container_command(workspace, home_dir, container_name),
                    container_name,
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                (workspace / TEICH_PROMPT_FILE_NAME).write_text(turn_prompt, encoding="utf-8")
                (workspace / TEICH_PROMPT_FILE_NAME).chmod(0o666)
                if len(turn_prompts) > 1:
                    command = self._build_external_exec_command(container_name, continue_session=turn_index > 0)
                else:
                    command = self._build_external_command(
                        workspace,
                        home_dir,
                        container_name,
                        continue_session=turn_index > 0,
                    )
                try:
                    stdout, stderr = self._run_external_process(command, container_name)
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                    stdout = exc.output if isinstance(exc.output, str) else (exc.output or "")
                    details = stderr.strip() or stdout.strip()
                    raise RuntimeError(f"Session {session_id[:8]} failed: {details}") from exc
                self._write_events(
                    destination,
                    self._events_from_turn_output(turn_prompt, stdout, stderr, turn_index=turn_index),
                )
                if progress_callback and progress_base:
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
                            trace_path=destination,
                            metrics=self._summarize_trace_file(destination),
                        )
                    )
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(destination))
            return destination
        except BaseException:
            self._discard_output_file(destination)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._cleanup_tracked_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(home_dir, ignore_errors=True)


class ClaudeCodeRunner(ExternalCliRunner):
    """Runs Claude Code in non-interactive stream-json mode."""

    provider_name = "claude-code"
    container_kind = "claude"
    home_in_container = CLAUDE_HOME_IN_CONTAINER
    source_name = "claude-code"
    default_model_provider = "anthropic"

    def _runtime_image_name(self) -> str:
        if self.config.agent.langfuse.enabled:
            return LANGFUSE_RUNTIME_IMAGE_NAME
        return RUNTIME_IMAGE_NAME

    def _runtime_build_args(self) -> list[str]:
        if self.config.agent.langfuse.enabled:
            return ["--build-arg", "TEICH_INSTALL_LANGFUSE=1"]
        return []

    def _langfuse_env_items(self) -> list[tuple[str, str]]:
        langfuse = self.config.agent.langfuse
        if not langfuse.enabled:
            return []
        return [
            ("TRACE_TO_LANGFUSE", "true"),
            ("LANGFUSE_PUBLIC_KEY", langfuse.public_key or ""),
            ("LANGFUSE_SECRET_KEY", langfuse.secret_key or ""),
            ("LANGFUSE_BASE_URL", self._langfuse_container_base_url() or ""),
        ]

    def _agent_env_items(self) -> list[tuple[str, str]]:
        # 0 is meaningful (disables thinking on models that allow it), so only
        # None is "unset".
        if self.config.agent.claude.max_thinking_tokens is None:
            return []
        return [("MAX_THINKING_TOKENS", str(self.config.agent.claude.max_thinking_tokens))]

    def _prepare_agent_home(self, home_dir: Path) -> None:
        settings = self._claude_settings()
        if not settings:
            return
        settings_path = home_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        settings_path.chmod(0o666)

    def _claude_settings(self) -> dict[str, object]:
        """Compose the container's ~/.claude/settings.json from config."""
        settings: dict[str, object] = {}
        if self.config.agent.claude.always_thinking is not None:
            settings["alwaysThinkingEnabled"] = self.config.agent.claude.always_thinking
        if self.config.agent.langfuse.enabled:
            hook = {"hooks": [{"type": "command", "command": CLAUDE_LANGFUSE_HOOK_COMMAND}]}
            settings["hooks"] = {"Stop": [hook], "SessionEnd": [hook]}
        return settings

    @staticmethod
    def _list_native_session_files(home_dir: Path) -> list[Path]:
        projects_dir = home_dir / "projects"
        if not projects_dir.exists():
            return []
        return sorted(path for path in projects_dir.rglob("*.jsonl") if path.is_file())

    def _latest_native_session_file(
        self,
        home_dir: Path,
        existing_sessions: set[Path],
        started_at: datetime,
    ) -> Path | None:
        session_files = self._list_native_session_files(home_dir)
        fresh_files = [path for path in session_files if path.resolve() not in existing_sessions]
        if not fresh_files:
            fresh_files = [
                path
                for path in session_files
                if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= started_at
            ]
        if not fresh_files:
            return None
        return max(fresh_files, key=lambda path: path.stat().st_mtime)

    def _extract_native_session_file(
        self,
        session_id: str,
        home_dir: Path,
        existing_sessions: set[Path],
        started_at: datetime,
    ) -> Path:
        _make_tree_world_writable(home_dir)
        source_path = self._latest_native_session_file(home_dir, existing_sessions, started_at)
        if source_path is None:
            raise RuntimeError(f"No Claude Code native session file found for {session_id}")
        destination = self._resolve_output_path(source_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._copy_normalized_native_session_file(source_path, destination)
        return destination

    @staticmethod
    def _copy_normalized_native_session_file(source_path: Path, destination: Path) -> None:
        events = _read_jsonl_dict_events(source_path)
        if events is None:
            shutil.copy2(source_path, destination)
            return
        normalized_events = normalize_claude_code_trace_events(events)
        if normalized_events == events:
            shutil.copy2(source_path, destination)
            return
        _write_jsonl_dict_events(destination, normalized_events)
        shutil.copystat(source_path, destination)

    def _needs_openrouter_model_proxy(self) -> bool:
        if not self.config.get_base_url():
            return False
        model = self.config.get_effective_model().strip().lower()
        return not (model.startswith("claude-") or model.startswith("anthropic/claude-"))

    def _claude_visible_model(self) -> str:
        if self._needs_openrouter_model_proxy():
            return CLAUDE_OPENROUTER_SURROGATE_MODEL
        return self.config.get_effective_model()

    def _base_url_env_items(self) -> list[tuple[str, str]]:
        if not self._needs_openrouter_model_proxy():
            return super()._base_url_env_items()
        base_url = self._container_base_url(self.config.get_base_url())
        if not base_url:
            return []
        proxy_base_url = f"http://127.0.0.1:{CLAUDE_OPENROUTER_PROXY_PORT}"
        return [
            ("TEICH_BASE_URL", base_url),
            ("OPENAI_BASE_URL", base_url),
            ("ANTHROPIC_BASE_URL", proxy_base_url),
            ("TEICH_CLAUDE_PROXY_TARGET", base_url),
            ("TEICH_CLAUDE_PROXY_TARGET_MODEL", self.config.get_effective_model()),
            ("TEICH_CLAUDE_PROXY_PORT", str(CLAUDE_OPENROUTER_PROXY_PORT)),
        ]

    def _api_env_items(self) -> list[tuple[str, str]]:
        token = self.config.get_claude_oauth_token() if self.config.claude_host_auth_active() else None
        if token:
            # Subscription auth: pass only the long-lived OAuth token, never an
            # API key — ANTHROPIC_API_KEY silently takes precedence over
            # subscription credentials inside Claude Code and would bill the API.
            return [("CLAUDE_CODE_OAUTH_TOKEN", token)]
        api_key = self.config.get_api_key() or ("none" if self.config.get_base_url() else "")
        if not api_key:
            return []
        provider_env_key = self._provider_env_key(self.config.api.provider)
        if self._needs_openrouter_model_proxy():
            items = [
                ("TEICH_API_KEY", api_key),
                ("ANTHROPIC_AUTH_TOKEN", api_key),
                ("ANTHROPIC_API_KEY", ""),
            ]
            if provider_env_key != "ANTHROPIC_API_KEY":
                items.append((provider_env_key, api_key))
            return items
        items = [("TEICH_API_KEY", api_key), ("ANTHROPIC_API_KEY", api_key), (provider_env_key, api_key)]
        return list(dict.fromkeys(items))

    def _permission_mode(self) -> str | None:
        approval_policy = (self.config.model.approval_policy or "").strip().lower()
        if approval_policy == "never":
            return "bypassPermissions"
        if approval_policy in {"on-request", "on_failure", "on-failure"}:
            return "default"
        return None

    @staticmethod
    def _write_openrouter_proxy(home_dir: Path) -> Path:
        home_dir.mkdir(parents=True, exist_ok=True)
        proxy_script = home_dir / CLAUDE_OPENROUTER_PROXY_SCRIPT_NAME
        proxy_script.write_text(CLAUDE_OPENROUTER_PROXY_SCRIPT + "\n", encoding="utf-8")
        proxy_script.chmod(0o755)
        return proxy_script

    def _openrouter_proxy_shell_prefix(self) -> str:
        proxy_script = f"{self.home_in_container}/{CLAUDE_OPENROUTER_PROXY_SCRIPT_NAME}"
        readiness_probe = (
            f"ready=0; for i in {{1..50}}; do "
            f"(: >/dev/tcp/127.0.0.1/{CLAUDE_OPENROUTER_PROXY_PORT}) >/dev/null 2>&1 "
            f"&& ready=1 && break; "
            f"sleep 0.02; "
            f"done; "
            f'[ "$ready" = 1 ] || {{ echo "Claude OpenRouter proxy did not become ready" >&2; exit 1; }}; '
        )
        return f"node {shlex.quote(proxy_script)} >/tmp/claude-openrouter-proxy.log 2>&1 & {readiness_probe}"

    def _claude_passthrough_args(self) -> list[str]:
        """Build model-related flags shared by batch and Studio Claude sessions."""
        args: list[str] = []
        if self.config.model.reasoning_effort:
            args.extend(["--effort", self.config.model.reasoning_effort])
        if fallback_model := self.config.get_claude_fallback_model():
            args.extend(["--fallback-model", fallback_model])
        return args

    def _build_shell_command(
        self,
        *,
        continue_session: bool = False,
        resume_session_id: str | None = None,
    ) -> str:
        claude_command = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self._claude_visible_model(),
        ]
        permission_mode = self._permission_mode()
        if permission_mode:
            claude_command.extend(["--permission-mode", permission_mode])
        claude_command.extend(self._claude_passthrough_args())
        if self.config.developer_instructions:
            claude_command.extend(["--append-system-prompt", self.config.developer_instructions])
        if resume_session_id:
            claude_command.extend(["--resume", resume_session_id])
        elif continue_session:
            claude_command.append("--continue")
        return f"{shlex.join(claude_command)} < {shlex.quote(WORKSPACE_IN_CONTAINER + '/' + TEICH_PROMPT_FILE_NAME)}"

    def _build_external_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
        *,
        continue_session: bool = False,
        resume_session_id: str | None = None,
    ) -> list[str]:
        if not self._needs_openrouter_model_proxy():
            return super()._build_external_command(
                workspace,
                home_dir,
                container_name,
                continue_session=continue_session,
                resume_session_id=resume_session_id,
            )
        self._write_openrouter_proxy(home_dir)
        command = self._build_external_docker_base_command(workspace, home_dir, container_name)
        shell_command = self._wrap_external_shell_command(
            f"{self._openrouter_proxy_shell_prefix()}"
            f"{self._build_shell_command(continue_session=continue_session, resume_session_id=resume_session_id)}"
        )
        command.extend(
            [
                "bash",
                "-lc",
                shell_command,
            ]
        )
        return command

    def _build_external_persistent_container_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
    ) -> list[str]:
        if not self._needs_openrouter_model_proxy():
            return super()._build_external_persistent_container_command(workspace, home_dir, container_name)
        self._write_openrouter_proxy(home_dir)
        command = self._build_external_docker_base_command(workspace, home_dir, container_name, detached=True)
        command.extend(["bash", "-lc", f"{self._openrouter_proxy_shell_prefix()}exec sleep infinity"])
        return command

    @staticmethod
    def _native_session_id_from_file(trace_path: Path) -> str | None:
        events = _read_jsonl_dict_events(trace_path)
        if not events:
            return None
        for event in events:
            session_id = event.get("sessionId")
            if isinstance(session_id, str) and session_id.strip():
                return session_id.strip()
        return None

    def _run_native_process_with_progress(
        self,
        command: list[str],
        container_name: str | None,
        *,
        session_id: str,
        home_dir: Path,
        existing_sessions: set[Path],
        started_at: datetime,
        progress_callback: SessionProgressCallback | None,
        progress_base: SessionProgressUpdate | None,
    ) -> tuple[str, str]:
        process: subprocess.Popen[str] | None = None
        stdout_handle = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
        stderr_handle = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
        try:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    **TEXT_SUBPROCESS_KWARGS,
                )
            except FileNotFoundError as exc:
                if shutil.which(command[0]) is None:
                    raise RuntimeError(
                        "Docker runtime not available. Ensure Docker is installed and the runtime image can be built."
                    ) from exc
                raise RuntimeError(f"Failed to start Docker runtime process: {exc}") from exc

            self._register_active_process(process, container_name)
            deadline = time.monotonic() + self.config.timeout_seconds
            last_signature: tuple[str, int, float] | None = None
            last_heartbeat = 0.0
            while True:
                return_code = process.poll()
                trace_path = self._latest_native_session_file(home_dir, existing_sessions, started_at)
                if trace_path is not None:
                    try:
                        stat = trace_path.stat()
                    except OSError:
                        stat = None
                    if stat is not None:
                        signature = (str(trace_path), stat.st_size, stat.st_mtime)
                        if signature != last_signature:
                            last_signature = signature
                            metrics: TraceMetrics | None
                            try:
                                metrics = self._summarize_trace_file(trace_path)
                            except (OSError, json.JSONDecodeError, ValueError):
                                metrics = None
                            if progress_callback and progress_base:
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
                                        details=f"Claude trace: {trace_path.name}",
                                        metrics=metrics,
                                    )
                                )
                elif progress_callback and progress_base and time.monotonic() - last_heartbeat >= 5.0:
                    last_heartbeat = time.monotonic()
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
                            details="waiting for Claude Code trace",
                        )
                    )

                if return_code is not None:
                    stdout_handle.flush()
                    stderr_handle.flush()
                    stdout_handle.seek(0)
                    stderr_handle.seek(0)
                    stdout = stdout_handle.read()
                    stderr = stderr_handle.read()
                    if return_code:
                        raise subprocess.CalledProcessError(
                            return_code,
                            process.args,
                            output=stdout,
                            stderr=stderr,
                        )
                    return stdout, stderr
                if time.monotonic() >= deadline:
                    process.kill()
                    stdout, stderr = process.communicate()
                    raise RuntimeError(f"Session timed out after {self.config.timeout_seconds}s")
                time.sleep(0.5)
        except BaseException:
            if process is not None:
                self._terminate_process(process, container_name)
            raise
        finally:
            if process is not None:
                self._unregister_active_process(process)
            stdout_handle.close()
            stderr_handle.close()

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
        workspace_root, workspace = self._prepare_workspace(session_id, prompt_input, self.container_kind)
        home_dir = Path(tempfile.mkdtemp(prefix=f"{self.container_kind}-home-{session_id}-"))
        home_dir.chmod(0o777)
        self._prepare_agent_home(home_dir)
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name(self.container_kind, session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        failure_trace_saved = False
        failure_trace_checked = False
        existing_sessions: set[Path] = set()
        native_resume_session_id: str | None = None
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            existing_sessions = {path.resolve() for path in self._list_native_session_files(home_dir)}
            if len(turn_prompts) > 1:
                self._start_tracked_container(
                    self._build_external_persistent_container_command(workspace, home_dir, container_name),
                    container_name,
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                (workspace / TEICH_PROMPT_FILE_NAME).write_text(turn_prompt, encoding="utf-8")
                (workspace / TEICH_PROMPT_FILE_NAME).chmod(0o666)
                expected_turns = turn_prompts[: turn_index + 1]
                for attempt in range(AGENT_TURN_RETRY_LIMIT + 1):
                    continue_session = turn_index > 0 or attempt > 0
                    resume_session_id = native_resume_session_id if continue_session else None
                    if len(turn_prompts) > 1:
                        command = self._build_external_exec_command(
                            container_name,
                            continue_session=continue_session,
                            resume_session_id=resume_session_id,
                        )
                    else:
                        command = self._build_external_command(
                            workspace,
                            home_dir,
                            container_name,
                            continue_session=continue_session,
                            resume_session_id=resume_session_id,
                        )
                    process_error: subprocess.CalledProcessError | None = None
                    details = ""
                    try:
                        self._run_native_process_with_progress(
                            command,
                            container_name,
                            session_id=session_id,
                            home_dir=home_dir,
                            existing_sessions=existing_sessions,
                            started_at=started_at,
                            progress_callback=progress_callback,
                            progress_base=progress_base,
                        )
                    except subprocess.CalledProcessError as exc:
                        process_error = exc
                        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                        stdout = exc.output if isinstance(exc.output, str) else (exc.output or "")
                        details = stderr.strip() or stdout.strip() or str(exc)

                    source_trace = self._latest_native_session_file(home_dir, existing_sessions, started_at)
                    if source_trace is not None:
                        native_resume_session_id = self._native_session_id_from_file(source_trace) or native_resume_session_id
                    if source_trace is not None and self._trace_contains_completed_turns(source_trace, expected_turns):
                        break
                    if attempt < AGENT_TURN_RETRY_LIMIT:
                        continue

                    failure_trace_checked = True
                    if source_trace is not None:
                        try:
                            trace_path = self._extract_native_session_file(
                                session_id,
                                home_dir,
                                existing_sessions,
                                started_at,
                            )
                        except RuntimeError:
                            trace_path = None
                        if trace_path is not None:
                            self._discard_output_file(trace_path)
                            failure_trace_saved = True
                    if process_error is not None:
                        raise RuntimeError(
                            f"Session {session_id[:8]} failed after {AGENT_TURN_RETRY_LIMIT} turn retries: {details}"
                        ) from process_error
                    raise RuntimeError(
                        f"Session {session_id[:8]} stopped before completing turn {turn_index + 1} "
                        f"after {AGENT_TURN_RETRY_LIMIT} retries"
                    )
            trace_path = self._extract_native_session_file(session_id, home_dir, existing_sessions, started_at)
            if not self._trace_contains_completed_turns(trace_path, turn_prompts):
                failure_trace_checked = True
                self._discard_output_file(trace_path)
                failure_trace_saved = True
                raise RuntimeError(f"Session {session_id[:8]} stopped before completing all Claude Code turns")
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
            return trace_path
        except BaseException:
            if not failure_trace_saved and not failure_trace_checked:
                try:
                    trace_path = self._extract_native_session_file(
                        session_id,
                        home_dir,
                        existing_sessions,
                        started_at,
                    )
                except RuntimeError:
                    trace_path = None
                if trace_path is not None:
                    self._discard_output_file(trace_path)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._cleanup_tracked_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(home_dir, ignore_errors=True)

    def _events_from_turn_output(
        self,
        prompt: str,
        stdout: str,
        stderr: str,
        *,
        turn_index: int,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = [
            {
                "timestamp": self._event_timestamp(),
                "type": "external_message",
                "role": "user",
                "turn_index": turn_index,
                "content": prompt,
            }
        ]
        fallback_lines: list[str] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                fallback_lines.append(line)
                continue
            if isinstance(parsed, dict):
                parsed.setdefault("teich_turn_index", turn_index)
                events.append(parsed)
            else:
                fallback_lines.append(line)
        fallback = "\n".join(fallback_lines).strip()
        if fallback:
            events.append(
                {
                    "timestamp": self._event_timestamp(),
                    "type": "external_message",
                    "role": "assistant",
                    "turn_index": turn_index,
                    "content": fallback,
                }
            )
        if stderr.strip():
            events.append(
                {
                    "timestamp": self._event_timestamp(),
                    "type": "external_stderr",
                    "turn_index": turn_index,
                    "content": stderr.strip(),
                }
            )
        return events


class HermesRunner(ExternalCliRunner):
    """Runs Hermes Agent through its non-interactive chat CLI."""

    provider_name = "hermes"
    container_kind = "hermes"
    home_in_container = HERMES_HOME_IN_CONTAINER
    source_name = "hermes-agent"
    default_model_provider = "hermes"

    def _hermes_cli_provider(self) -> str:
        provider = self.config.api.provider.strip().lower()
        if self.config.get_base_url() and provider in {"openai", "custom"}:
            return "custom"
        return self.config.api.provider

    def _build_shell_command(
        self,
        *,
        continue_session: bool = False,
        resume_session_id: str | None = None,
    ) -> str:
        prompt_path = shlex.quote(WORKSPACE_IN_CONTAINER + "/" + TEICH_PROMPT_FILE_NAME)
        hermes_command = [
            "hermes",
            "chat",
            "--provider",
            self._hermes_cli_provider(),
            "--model",
            self.config.get_effective_model(),
            "--toolsets",
            HERMES_DEFAULT_TOOLSETS,
            "--quiet",
            "--yolo",
            "--ignore-user-config",
            "--source",
            "teich",
        ]
        if continue_session:
            hermes_command.append("--continue")
        return f"{shlex.join(hermes_command)} -q \"$(cat {prompt_path})\""

    @staticmethod
    def _hermes_state_db(home_dir: Path) -> Path:
        return home_dir / "state.db"

    @staticmethod
    def _safe_session_file_id(session_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id.strip()) or "session"

    @staticmethod
    def _hermes_timestamp(value: Any) -> str:
        if isinstance(value, (int, float)):
            timestamp = value / 1000 if value > 10_000_000_000 else value
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ExternalCliRunner._event_timestamp()

    @staticmethod
    def _json_or_original(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _sqlite_row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
        return row[key] if key in row.keys() else default

    @staticmethod
    def _parse_hermes_stdout_session_id(stdout: str) -> str | None:
        for line in stdout.splitlines():
            if line.startswith("session_id:"):
                value = line.split(":", maxsplit=1)[1].strip()
                if value:
                    return value
        return None

    def _write_hermes_runtime_config(self, home_dir: Path) -> None:
        if self._hermes_cli_provider().strip().lower() != "custom":
            return
        base_url = self._container_base_url(self.config.get_base_url())
        if not base_url:
            return
        api_key = (self.config.get_api_key() or "").strip()
        if api_key.lower() in {"none", "null", "dummy", "placeholder", "example"}:
            api_key = ""
        model = self.config.get_effective_model()
        custom_provider: dict[str, object] = {
            "name": "teich-custom",
            "base_url": base_url,
            "model": model,
            "api_mode": "chat_completions",
        }
        model_config: dict[str, object] = {
            "default": model,
            "provider": "custom",
            "base_url": base_url,
            "api_mode": "chat_completions",
        }
        if self.config.model.context_length:
            context_length = int(self.config.model.context_length)
            custom_provider["models"] = {model: {"context_length": context_length}}
            model_config["context_length"] = context_length
        if api_key:
            custom_provider["api_key"] = api_key
            model_config["api_key"] = api_key
        hermes_config = {
            "model": model_config,
            "custom_providers": [custom_provider],
        }
        (home_dir / "config.yaml").write_text(json.dumps(hermes_config, indent=2), encoding="utf-8")

    def _write_agents_md(self, workspace: Path) -> None:
        """Inject developer_instructions via AGENTS.md, which Hermes auto-loads.

        Hermes has no system-prompt flag; it auto-injects AGENTS.md from the
        workspace (Teich does not pass --ignore-rules). Append to an existing
        AGENTS.md so a cloned repo's own guidance is preserved.
        """
        instructions = self.config.developer_instructions
        if not instructions or not instructions.strip():
            return
        block = instructions.strip() + "\n"
        agents_md = workspace / "AGENTS.md"
        if agents_md.exists():
            existing = agents_md.read_text(encoding="utf-8").rstrip("\n")
            agents_md.write_text(f"{existing}\n\n{block}", encoding="utf-8")
        else:
            agents_md.write_text(block, encoding="utf-8")

    @staticmethod
    def _decode_hermes_content(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("\x00json:"):
            try:
                return json.loads(value[len("\x00json:"):])
            except json.JSONDecodeError:
                return value
        return value

    def _hermes_session_export(
        self,
        row: sqlite3.Row,
        messages: list[dict[str, object]],
        workspace: Path,
        *,
        partial: bool = False,
    ) -> dict[str, object]:
        session = dict(row)
        input_tokens = self._sqlite_row_get(row, "input_tokens")
        output_tokens = self._sqlite_row_get(row, "output_tokens")
        total_tokens = self._sqlite_row_get(row, "total_tokens")
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        actual_cost = self._sqlite_row_get(row, "actual_cost_usd")
        estimated_cost = self._sqlite_row_get(row, "estimated_cost_usd")
        total_cost = actual_cost if actual_cost is not None else estimated_cost
        if total_cost is None:
            total_cost = self._sqlite_row_get(row, "total_cost")

        session.update(
            {
                "timestamp": self._hermes_timestamp(self._sqlite_row_get(row, "started_at")),
                "cwd": str(workspace),
                "teich_export_status": "partial" if partial else "completed",
                "teich_partial": partial,
                "model_provider": self._hermes_cli_provider() or self.default_model_provider,
                "configured_model_provider": self.config.api.provider or self.default_model_provider,
                "model": self._sqlite_row_get(row, "model") or self.config.get_effective_model(),
                "configured_context_length": self.config.model.context_length,
                "hermes_source": self._sqlite_row_get(row, "source"),
                "toolsets": HERMES_DEFAULT_TOOLSETS.split(","),
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
                "actual_cost_usd": actual_cost,
                "total_cost": total_cost,
                "messages": messages,
            }
        )
        tools = self._configured_tools()
        if tools:
            session["tools"] = tools
        model_config = self._json_or_original(session.get("model_config"))
        if model_config is not None:
            session["model_config"] = model_config
        return session

    def _hermes_trace_row(
        self,
        row: sqlite3.Row,
        messages: list[dict[str, object]],
        workspace: Path,
        *,
        partial: bool = False,
    ) -> dict[str, object]:
        return self._hermes_session_export(row, messages, workspace, partial=partial)

    def _resolve_hermes_trace_path(self, _session_id: str | None = None, *, partial: bool = False) -> Path:
        destination_dir = self.config.output.failures_dir if partial else self.config.output.traces_dir
        return destination_dir / "sessions.jsonl"

    def _hermes_message_export(self, row: sqlite3.Row) -> dict[str, object]:
        message = dict(row)
        message["content"] = self._decode_hermes_content(message.get("content")) or ""
        tool_calls = self._json_or_original(message.get("tool_calls"))
        if tool_calls:
            message["tool_calls"] = tool_calls
        for key in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
            value = self._json_or_original(message.get(key))
            if value is not None and value != "":
                message[key] = value
        return message

    @staticmethod
    def _hermes_external_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _hermes_external_session_payload(self, row: dict[str, object]) -> dict[str, object]:
        payload = {key: value for key, value in row.items() if key != "messages"}
        hermes_source = payload.get("source")
        payload["source"] = self.source_name
        payload.setdefault("toolsets", HERMES_DEFAULT_TOOLSETS.split(","))
        tools = self._configured_tools()
        if tools:
            payload.setdefault("tools", tools)
        if hermes_source is not None and hermes_source != "":
            payload.setdefault("hermes_source", hermes_source)
        return payload

    def _hermes_external_message_event(self, message: dict[str, object]) -> dict[str, object] | None:
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        event: dict[str, object] = {
            "timestamp": self._hermes_timestamp(message.get("timestamp")),
            "type": "external_message",
            "role": role.strip(),
            "content": self._hermes_external_content(message.get("content")),
        }
        passthrough_keys = (
            "tool_call_id",
            "tool_calls",
            "token_count",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
        )
        for key in passthrough_keys:
            value = message.get(key)
            if value is not None and value != "":
                event[key] = value
        tool_name = message.get("tool_name") or message.get("name")
        if isinstance(tool_name, str) and tool_name.strip():
            event["name"] = tool_name.strip()
        return event

    def _hermes_external_trace_events(self, row: dict[str, object]) -> list[dict[str, object]]:
        payload = self._hermes_external_session_payload(row)
        events: list[dict[str, object]] = [
            {
                "timestamp": payload.get("timestamp") or self._event_timestamp(),
                "type": "external_session_meta",
                "payload": payload,
            }
        ]
        messages = row.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                event = self._hermes_external_message_event(message)
                if event is not None:
                    events.append(event)
        return events

    def _export_hermes_state_sessions(
        self,
        home_dir: Path,
        workspace: Path,
        *,
        partial: bool = False,
    ) -> dict[str, Path]:
        rows = self._hermes_state_session_rows(home_dir, workspace, partial=partial)
        return self._write_hermes_state_rows(rows, partial=partial)

    def _export_completed_hermes_state_sessions(
        self,
        home_dir: Path,
        workspace: Path,
        turn_prompts: list[str],
    ) -> dict[str, Path]:
        rows = [
            row
            for row in self._hermes_state_session_rows(home_dir, workspace, partial=False)
            if _training_example_completed_turns(row, turn_prompts)
        ]
        return self._write_hermes_state_rows(rows, partial=False)

    def _write_hermes_state_rows(
        self,
        rows: list[dict[str, object]],
        *,
        partial: bool = False,
    ) -> dict[str, Path]:
        if not rows:
            return {}
        exported: dict[str, Path] = {}
        destination = self._resolve_hermes_trace_path(partial=partial)
        with HERMES_TRACE_WRITE_LOCK:
            destination.parent.mkdir(parents=True, exist_ok=True)
            existing_ids: set[str] = set()
            if destination.exists():
                try:
                    with destination.open("r", encoding="utf-8") as handle:
                        for raw_line in handle:
                            if not raw_line.strip():
                                continue
                            existing = json.loads(raw_line)
                            if isinstance(existing, dict) and existing.get("id"):
                                existing_ids.add(str(existing["id"]))
                except (OSError, json.JSONDecodeError):
                    existing_ids = set()
            with destination.open("a", encoding="utf-8") as handle:
                for row in rows:
                    session_id = str(row.get("id") or "")
                    if not session_id:
                        continue
                    if session_id not in existing_ids:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        existing_ids.add(session_id)
                    exported[session_id] = destination
        return exported

    def _hermes_state_session_rows(
        self,
        home_dir: Path,
        workspace: Path,
        *,
        partial: bool = False,
    ) -> list[dict[str, object]]:
        state_db = self._hermes_state_db(home_dir)
        if not state_db.exists():
            return []
        connection = sqlite3.connect(state_db)
        connection.row_factory = sqlite3.Row
        try:
            sessions = connection.execute(
                "SELECT * FROM sessions WHERE source = ? ORDER BY started_at ASC, id ASC",
                ("cli",),
            ).fetchall()
            rows: list[dict[str, object]] = []
            for session_row in sessions:
                message_rows = connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (str(session_row["id"]),),
                ).fetchall()
                messages = [self._hermes_message_export(message_row) for message_row in message_rows]
                rows.append(self._hermes_trace_row(session_row, messages, workspace, partial=partial))
            return rows
        finally:
            connection.close()

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
        workspace_root, workspace = self._prepare_workspace(session_id, prompt_input, self.container_kind)
        home_dir = Path(tempfile.mkdtemp(prefix=f"{self.container_kind}-home-{session_id}-"))
        home_dir.chmod(0o777)
        self._prepare_agent_home(home_dir)
        container_name = self._container_name(self.container_kind, session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        fallback_destination = self._resolve_hermes_trace_path()
        fallback_destination.parent.mkdir(parents=True, exist_ok=True)
        stdout_parts: list[str] = []
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_hermes_runtime_config(home_dir)
            self._write_agents_md(workspace)
            if len(turn_prompts) > 1:
                self._start_tracked_container(
                    self._build_external_persistent_container_command(workspace, home_dir, container_name),
                    container_name,
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                (workspace / TEICH_PROMPT_FILE_NAME).write_text(turn_prompt, encoding="utf-8")
                (workspace / TEICH_PROMPT_FILE_NAME).chmod(0o666)
                if len(turn_prompts) > 1:
                    command = self._build_external_exec_command(container_name, continue_session=turn_index > 0)
                else:
                    command = self._build_external_command(
                        workspace,
                        home_dir,
                        container_name,
                        continue_session=turn_index > 0,
                    )
                try:
                    stdout, stderr = self._run_external_process(command, container_name)
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                    stdout = exc.output if isinstance(exc.output, str) else (exc.output or "")
                    details = stderr.strip() or stdout.strip()
                    if turn_index == len(turn_prompts) - 1:
                        final_exports = self._export_completed_hermes_state_sessions(home_dir, workspace, turn_prompts)
                        parsed_session_id = self._parse_hermes_stdout_session_id("\n".join([*stdout_parts, stdout]))
                        destination = (
                            final_exports.get(parsed_session_id)
                            or next((path for sid, path in final_exports.items() if sid.startswith(session_id)), None)
                            or next(
                                (path for sid, path in final_exports.items() if self._trace_has_no_parent(path)),
                                None,
                            )
                            or next(iter(final_exports.values()), None)
                        )
                        if destination is not None:
                            self._copy_workspace_snapshot(workspace, self._sandbox_destination(destination))
                            return destination
                    raise RuntimeError(f"Session {session_id[:8]} failed: {details}") from exc
                except RuntimeError:
                    raise
                stdout_parts.append(stdout)
                if progress_callback and progress_base:
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
                            trace_path=fallback_destination if fallback_destination.exists() else None,
                            metrics=TraceMetrics(),
                        )
                    )
            final_exports = self._export_hermes_state_sessions(home_dir, workspace)
            if not final_exports:
                raise RuntimeError(f"Session {session_id[:8]} failed: Hermes did not write any state.db sessions")
            parsed_session_id = self._parse_hermes_stdout_session_id("\n".join(stdout_parts))
            destination = (
                final_exports.get(parsed_session_id)
                or next((path for sid, path in final_exports.items() if sid.startswith(session_id)), None)
                or next(
                    (path for sid, path in final_exports.items() if self._trace_has_no_parent(path)),
                    None,
                )
                or next(iter(final_exports.values()), fallback_destination)
            )
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(destination))
            return destination
        except BaseException:
            try:
                self._export_hermes_state_sessions(home_dir, workspace, partial=True)
            except Exception:
                pass
            raise
        finally:
            if len(turn_prompts) > 1:
                self._cleanup_tracked_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(home_dir, ignore_errors=True)

    @staticmethod
    def _trace_has_no_parent(trace_path: Path) -> bool:
        try:
            with trace_path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
        except OSError:
            return False
        if not first_line.strip():
            return False
        try:
            event = json.loads(first_line)
        except json.JSONDecodeError:
            return False
        if isinstance(event, dict) and isinstance(event.get("metadata"), dict):
            return event["metadata"].get("parent_session_id") in {None, ""}
        if isinstance(event, dict) and "parent_session_id" in event:
            return event.get("parent_session_id") in {None, ""}
        payload = event.get("payload") if isinstance(event, dict) else None
        return isinstance(payload, dict) and payload.get("parent_session_id") in {None, ""}


class ChatRunner(DockerRuntimeRunner):
    """Generates text-only chat datasets from prompt inputs via an OpenAI-compatible API."""

    def __init__(self, config: Config):
        self.config = config
        self.image_name = RUNTIME_IMAGE_NAME

    def _default_base_url(self) -> str:
        provider = self.config.api.provider.strip().lower()
        if provider == "openai":
            return "https://api.openai.com/v1"
        if provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        raise RuntimeError(
            "Chat runner requires api.base_url for providers other than openai or openrouter."
        )

    def _wire_api(self) -> str:
        return self.config.api.wire_api.strip().lower()

    def _api_base_url(self) -> str:
        base_url = self.config.get_base_url() or self._default_base_url()
        return base_url.rstrip("/")

    @staticmethod
    def _chat_system_prompt(prompt_input: PromptInput | None = None) -> str | None:
        if prompt_input is None:
            return None
        if isinstance(prompt_input.system, str) and prompt_input.system.strip():
            return prompt_input.system.strip()
        return None

    def _chat_endpoint(self) -> str:
        if self._wire_api() in {"completions", "chat_completions", "chat-completions", "openai-completions"}:
            return f"{self._api_base_url()}/chat/completions"
        return f"{self._api_base_url()}/responses"

    def _chat_request_body(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        model = self.config.get_effective_model()
        reasoning_effort = self.config.model.reasoning_effort
        conversation = [*(history or []), {"role": "user", "content": prompt}]
        if self._wire_api() in {"completions", "chat_completions", "chat-completions", "openai-completions"}:
            body: dict[str, Any] = {
                "model": model,
                "messages": conversation,
            }
            if isinstance(system_prompt, str) and system_prompt.strip():
                body["messages"] = [{"role": "system", "content": system_prompt.strip()}, *conversation]
            if isinstance(reasoning_effort, str) and reasoning_effort.strip():
                body["reasoning_effort"] = reasoning_effort.strip()
            return body
        body = {
            "model": model,
            "input": conversation if history else prompt,
        }
        if isinstance(system_prompt, str) and system_prompt.strip():
            body["instructions"] = system_prompt.strip()
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            body["reasoning"] = {"effort": reasoning_effort.strip()}
        return body

    def _chat_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "teich",
        }
        api_key = self.config.get_api_key()
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_reasoning_text(payload: Any) -> str | None:
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        if isinstance(payload, dict):
            summary = payload.get("summary")
            if isinstance(summary, list):
                parts = [
                    item.get("text", "").strip()
                    for item in summary
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip()
                ]
                if parts:
                    return "\n\n".join(parts)
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        if isinstance(payload, list):
            parts = [
                ChatRunner._extract_reasoning_text(item)
                for item in payload
            ]
            merged = [part for part in parts if isinstance(part, str) and part.strip()]
            if merged:
                return "\n\n".join(merged)
        return None

    def _parse_chat_response(self, payload: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None, str]:
        model = payload.get("model") if isinstance(payload.get("model"), str) and payload.get("model") else self.config.get_effective_model()
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        if self._wire_api() in {"completions", "chat_completions", "chat-completions", "openai-completions"}:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("Chat completion response did not include any choices.")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if not isinstance(message, dict):
                raise RuntimeError("Chat completion response did not include a valid assistant message.")
            content = self._extract_content_text(message.get("content"))
            thinking = self._extract_reasoning_text(message.get("reasoning"))
            return content, thinking, usage, model

        output_items = payload.get("output")
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "reasoning":
                    reasoning = self._extract_reasoning_text(item.get("summary") or item.get("content"))
                    if isinstance(reasoning, str) and reasoning.strip():
                        reasoning_parts.append(reasoning.strip())
                    continue
                if item_type == "message" and item.get("role") == "assistant":
                    content = self._extract_content_text(item.get("content"))
                    if content:
                        content_parts.append(content)
        if not content_parts:
            output_text = payload.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                content_parts.append(output_text.strip())
        content = "\n\n".join(part for part in content_parts if part).strip()
        thinking = "\n\n".join(part for part in reasoning_parts if part).strip() or None
        return content, thinking, usage, model

    @staticmethod
    def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(usage, dict):
            return None
        input_tokens = TraceMetrics._int_value(usage.get("input") or usage.get("prompt_tokens") or usage.get("input_tokens"))
        output_tokens = TraceMetrics._int_value(usage.get("output") or usage.get("completion_tokens") or usage.get("output_tokens"))
        reasoning_tokens = TraceMetrics._int_value(
            usage.get("reasoning")
            or usage.get("reasoning_tokens")
            or usage.get("reasoning_output_tokens")
            or (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        )
        prompt_token_details = usage.get("prompt_tokens_details")
        if not isinstance(prompt_token_details, dict):
            prompt_token_details = {}
        cache_read_tokens = TraceMetrics._int_value(
            usage.get("cacheRead")
            or usage.get("cached_input_tokens")
            or prompt_token_details.get("cached_tokens")
        )
        cache_write_tokens = TraceMetrics._int_value(
            usage.get("cacheWrite")
            or prompt_token_details.get("cache_write_tokens")
        )
        total_tokens = TraceMetrics._int_value(usage.get("totalTokens") or usage.get("total_tokens"))
        normalized = {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "totalTokens": total_tokens or (input_tokens + output_tokens + reasoning_tokens),
        }
        if cache_read_tokens:
            normalized["cacheRead"] = cache_read_tokens
        if cache_write_tokens:
            normalized["cacheWrite"] = cache_write_tokens
        cost = usage.get("cost")
        if isinstance(cost, dict) and cost.get("total") is not None:
            normalized["cost"] = {"total": TraceMetrics._float_value(cost.get("total"))}
        elif isinstance(cost, (int, float)) and not isinstance(cost, bool):
            normalized["cost"] = {"total": float(cost)}
        for passthrough_key in ("generation_id", "provider_name"):
            passthrough_value = usage.get(passthrough_key)
            if isinstance(passthrough_value, str) and passthrough_value.strip():
                normalized[passthrough_key] = passthrough_value.strip()
        generation_ids = usage.get("generation_ids")
        if isinstance(generation_ids, list):
            normalized["generation_ids"] = [
                generation_id
                for generation_id in generation_ids
                if isinstance(generation_id, str) and generation_id.strip()
            ]
        return normalized

    @staticmethod
    def _chat_api_error_message(error: Any) -> str:
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail") or error.get("code")
            if isinstance(message, str) and message.strip():
                return message.strip()
            return json.dumps(error, ensure_ascii=False)
        if isinstance(error, str) and error.strip():
            return error.strip()
        if error is not None:
            return json.dumps(error, ensure_ascii=False)
        return "unknown API error"

    def _raise_for_chat_api_error(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"Chat request returned unexpected payload type: {type(payload).__name__}.")

        if "error" in payload and payload.get("error") is not None:
            raise RuntimeError(f"Chat request failed: {self._chat_api_error_message(payload.get('error'))}")

        status = payload.get("status")
        if isinstance(status, str) and status.strip().lower() in {"failed", "cancelled", "canceled", "incomplete"}:
            details = payload.get("error") or payload.get("incomplete_details") or payload.get("status_details")
            raise RuntimeError(f"Chat request failed with status {status}: {self._chat_api_error_message(details)}")

        output_items = payload.get("output")
        if isinstance(output_items, list):
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "error":
                    raise RuntimeError(f"Chat request failed: {self._chat_api_error_message(item)}")

        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if "error" in choice and choice.get("error") is not None:
                    raise RuntimeError(f"Chat request failed: {self._chat_api_error_message(choice.get('error'))}")
                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason.strip().lower() in {"error", "content_filter"}:
                    raise RuntimeError(f"Chat request failed with finish_reason {finish_reason}.")

        return payload

    @staticmethod
    def _is_transient_chat_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        transient_markers = (
            "http 408",
            "http 409",
            "http 425",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "rate limit",
            "temporarily",
            "timeout",
            "timed out",
            "overloaded",
            "try again",
            "server error",
            "service unavailable",
            "connection reset",
            "remote end closed connection",
        )
        return any(marker in message for marker in transient_markers)

    def _request_chat_turn_once(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, str | None, dict[str, Any] | None, str]:
        body = self._chat_request_body(prompt, history, system_prompt)
        request = Request(
            self._chat_endpoint(),
            data=json.dumps(body).encode("utf-8"),
            headers=self._chat_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat request failed with HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Chat request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Chat request returned invalid JSON.") from exc
        payload = self._raise_for_chat_api_error(payload)
        content, thinking, usage, model = self._parse_chat_response(payload)
        if not content and not thinking:
            raise RuntimeError("Chat request returned neither assistant content nor thinking.")
        generation_ids = self._openrouter_generation_ids_from_value(payload)
        provider_usage = self._openrouter_usage_from_generation_ids(generation_ids)
        if provider_usage is not None:
            usage = provider_usage
        return content, thinking, usage, model

    def _request_chat_turn(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, str | None, dict[str, Any] | None, str]:
        last_error: RuntimeError | None = None
        for attempt in range(CHAT_REQUEST_MAX_ATTEMPTS):
            try:
                return self._request_chat_turn_once(prompt, history, system_prompt)
            except RuntimeError as exc:
                if not self._is_transient_chat_error(exc) or attempt >= CHAT_REQUEST_MAX_ATTEMPTS - 1:
                    raise
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _request_chat_turn_with_optional_system(
        self,
        prompt: str,
        history: list[dict[str, str]] | None,
        system_prompt: str | None,
    ) -> tuple[str, str | None, dict[str, Any] | None, str]:
        if isinstance(system_prompt, str) and system_prompt.strip():
            return self._request_chat_turn(prompt, history, system_prompt.strip())
        return self._request_chat_turn(prompt, history)

    def _request_chat_completion(self, prompt_input: PromptInput) -> dict[str, Any]:
        system_prompt = self._chat_system_prompt(prompt_input)
        content, thinking, usage, model = self._request_chat_turn_with_optional_system(
            prompt_input.prompt,
            None,
            system_prompt,
        )
        messages = [
            {"role": "user", "content": prompt_input.prompt, "thinking": None},
            {"role": "assistant", "content": content, "thinking": thinking},
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt, "thinking": None})
        row: dict[str, Any] = {
            "messages": messages,
            "prompt": prompt_input.prompt,
            "thinking": thinking,
            "response": content,
            "model": model,
            "provider": self.config.api.provider,
            "usage": self._normalize_usage(usage),
            "metadata": {
                "trace_type": "chat",
                "model_provider": self.config.api.provider,
                "model": model,
            },
        }
        if system_prompt:
            row["system"] = system_prompt
        return row

    @staticmethod
    def _merge_usage_totals(usages: list[dict[str, Any] | None]) -> dict[str, Any] | None:
        normalized_usages = [ChatRunner._normalize_usage(usage) for usage in usages if isinstance(usage, dict)]
        normalized_usages = [usage for usage in normalized_usages if usage is not None]
        if not normalized_usages:
            return None
        totals: dict[str, Any] = {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "totalTokens": 0,
        }
        cache_read_tokens = 0
        cache_write_tokens = 0
        total_cost = 0.0
        has_cost = False
        generation_ids: list[str] = []
        for usage in normalized_usages:
            totals["input"] += TraceMetrics._int_value(usage.get("input"))
            totals["output"] += TraceMetrics._int_value(usage.get("output"))
            totals["reasoning"] += TraceMetrics._int_value(usage.get("reasoning"))
            cache_read_tokens += TraceMetrics._int_value(usage.get("cacheRead"))
            cache_write_tokens += TraceMetrics._int_value(usage.get("cacheWrite"))
            totals["totalTokens"] += TraceMetrics._int_value(usage.get("totalTokens"))
            cost = usage.get("cost")
            if isinstance(cost, dict) and cost.get("total") is not None:
                total_cost += TraceMetrics._float_value(cost.get("total"))
                has_cost = True
            generation_id = usage.get("generation_id")
            if isinstance(generation_id, str) and generation_id.strip():
                generation_ids.append(generation_id.strip())
            usage_generation_ids = usage.get("generation_ids")
            if isinstance(usage_generation_ids, list):
                generation_ids.extend(
                    generation_id.strip()
                    for generation_id in usage_generation_ids
                    if isinstance(generation_id, str) and generation_id.strip()
                )
        if not totals["totalTokens"]:
            totals["totalTokens"] = totals["input"] + totals["output"] + totals["reasoning"]
        if cache_read_tokens:
            totals["cacheRead"] = cache_read_tokens
        if cache_write_tokens:
            totals["cacheWrite"] = cache_write_tokens
        if has_cost:
            totals["cost"] = {"total": total_cost}
        if generation_ids:
            totals["generation_ids"] = list(dict.fromkeys(generation_ids))
        return totals

    @staticmethod
    def _read_chat_training_rows(destination: Path) -> list[dict[str, Any]]:
        if not destination.exists():
            return []
        rows: list[dict[str, Any]] = []
        with destination.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Chat dataset row in {destination} is not a JSON object.")
                rows.append(row)
        return rows

    @staticmethod
    def _write_chat_training_rows(destination: Path, rows: list[dict[str, Any]]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, destination)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _chat_row_base_prompt(row: dict[str, Any]) -> str:
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
        return _prompt_from_training_messages(row.get("messages"))

    @staticmethod
    def _chat_user_prompts_from_messages(messages: Any) -> list[str]:
        if not isinstance(messages, list):
            return []
        user_prompts: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = ChatRunner._extract_content_text(message.get("content"))
            if content:
                user_prompts.append(content)
        return user_prompts

    @classmethod
    def _chat_row_follow_up_prompts(cls, row: dict[str, Any], base_prompt: str) -> list[str]:
        follow_up_prompts = row.get("follow_up_prompts")
        if isinstance(follow_up_prompts, list):
            return [prompt.strip() for prompt in follow_up_prompts if isinstance(prompt, str) and prompt.strip()]

        user_prompts = cls._chat_user_prompts_from_messages(row.get("messages"))
        if len(user_prompts) <= 1:
            return []
        if _prompt_text_completion_key(user_prompts[0]) != _prompt_text_completion_key(base_prompt):
            return []
        return user_prompts[1:]

    @staticmethod
    def _chat_assistant_responses_from_messages(messages: Any) -> list[str]:
        if not isinstance(messages, list):
            return []
        responses: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = ChatRunner._extract_content_text(message.get("content"))
            if content:
                responses.append(content)
        return responses

    @classmethod
    def _chat_row_responses(cls, row: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
        responses = row.get("responses")
        if isinstance(responses, list):
            clean_responses = [response.strip() for response in responses if isinstance(response, str) and response.strip()]
            if clean_responses:
                return clean_responses

        assistant_responses = cls._chat_assistant_responses_from_messages(messages)
        if assistant_responses:
            return assistant_responses

        response = row.get("response")
        if isinstance(response, str) and response.strip():
            return [response.strip()]
        return []

    @classmethod
    def _chat_row_completed_follow_up_prompts(cls, row: dict[str, Any], base_prompt: str) -> list[str]:
        follow_up_prompts = cls._chat_row_follow_up_prompts(row, base_prompt)
        if not follow_up_prompts:
            return []

        messages = row.get("messages")
        if isinstance(messages, list):
            assistant_turns = len(cls._chat_assistant_responses_from_messages(messages))
        else:
            responses = row.get("responses")
            if isinstance(responses, list):
                assistant_turns = len([response for response in responses if isinstance(response, str) and response.strip()])
            elif isinstance(row.get("response"), str) and row.get("response", "").strip():
                assistant_turns = 1 + len(follow_up_prompts)
            else:
                assistant_turns = 0

        completed_follow_up_count = max(0, min(len(follow_up_prompts), assistant_turns - 1))
        return follow_up_prompts[:completed_follow_up_count]

    @staticmethod
    def _prompt_sequence_matches_prefix(prefix: list[str], prompts: list[str]) -> bool:
        if len(prefix) > len(prompts):
            return False
        return [
            _prompt_text_completion_key(prompt)
            for prompt in prefix
        ] == [
            _prompt_text_completion_key(prompt)
            for prompt in prompts[: len(prefix)]
        ]

    @classmethod
    def _chat_row_can_extend(cls, row: dict[str, Any], prompt_input: PromptInput) -> tuple[bool, list[str]]:
        base_prompt = cls._chat_row_base_prompt(row)
        if not base_prompt.strip():
            return False, []
        if _prompt_text_completion_key(base_prompt) != _prompt_text_completion_key(prompt_input.prompt):
            return False, []
        row_system = _system_from_training_example(row)
        prompt_system = prompt_input.system.strip() if isinstance(prompt_input.system, str) and prompt_input.system.strip() else None
        if row_system != prompt_system:
            return False, []
        if not _training_example_has_answer(row):
            return False, []

        completed_follow_ups = cls._chat_row_completed_follow_up_prompts(row, base_prompt)
        if len(completed_follow_ups) >= len(prompt_input.follow_up_prompts):
            return False, completed_follow_ups
        if not cls._prompt_sequence_matches_prefix(completed_follow_ups, prompt_input.follow_up_prompts):
            return False, completed_follow_ups
        return True, completed_follow_ups

    @classmethod
    def _find_chat_row_to_extend(
        cls,
        rows: list[dict[str, Any]],
        prompt_input: PromptInput,
    ) -> tuple[int, dict[str, Any]] | None:
        best: tuple[int, dict[str, Any], int] | None = None
        for index, row in enumerate(rows):
            can_extend, completed_follow_ups = cls._chat_row_can_extend(row, prompt_input)
            if not can_extend:
                continue
            completed_turns = len(completed_follow_ups)
            if best is None or completed_turns > best[2]:
                best = (index, row, completed_turns)
        if best is None:
            return None
        return best[0], best[1]

    @classmethod
    def _chat_row_completion_key(cls, row: dict[str, Any]) -> str | None:
        prompt = cls._chat_row_base_prompt(row)
        if not prompt.strip() or not _training_example_has_answer(row):
            return None
        return _prompt_completion_key(_prompt_input_from_training_example(row, prompt))

    @classmethod
    def _chat_rows_include_completed_prompt(cls, rows: list[dict[str, Any]], prompt_input: PromptInput) -> bool:
        prompt_key = _prompt_completion_key(prompt_input)
        return any(cls._chat_row_completion_key(row) == prompt_key for row in rows)

    def _chat_messages_from_row(self, row: dict[str, Any], prompt_input: PromptInput) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        existing_messages = row.get("messages")
        if isinstance(existing_messages, list):
            for message in existing_messages:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                    continue
                messages.append(dict(message))

        if not any(message.get("role") == "system" for message in messages):
            system_prompt = _system_from_training_example(row) or self._chat_system_prompt(prompt_input)
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt, "thinking": None})

        if not any(message.get("role") == "user" for message in messages):
            messages.append({"role": "user", "content": prompt_input.prompt, "thinking": None})

        if not any(message.get("role") == "assistant" for message in messages):
            response = row.get("response")
            if isinstance(response, str) and response.strip():
                messages.append({"role": "assistant", "content": response.strip(), "thinking": row.get("thinking")})

        return messages

    @staticmethod
    def _chat_history_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = ChatRunner._extract_content_text(message.get("content"))
            if content:
                history.append({"role": role, "content": content})
        return history

    @staticmethod
    def _chat_thinking_parts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
        thinking_parts: list[str] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            thinking = message.get("thinking") or message.get("reasoning_content")
            if isinstance(thinking, str) and thinking.strip():
                thinking_parts.append(thinking.strip())
        return thinking_parts

    def _request_chat_conversation(self, prompt_input: PromptInput) -> dict[str, Any]:
        if not prompt_input.follow_up_prompts:
            return self._request_chat_completion(prompt_input)

        system_prompt = self._chat_system_prompt(prompt_input)
        history: list[dict[str, str]] = []
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt, "thinking": None})
        responses: list[str] = []
        thinking_parts: list[str | None] = []
        usages: list[dict[str, Any] | None] = []
        model = self.config.get_effective_model()

        for prompt in prompt_input.turn_prompts():
            content, thinking, usage, model = self._request_chat_turn_with_optional_system(prompt, history, system_prompt)
            messages.append({"role": "user", "content": prompt, "thinking": None})
            messages.append({"role": "assistant", "content": content, "thinking": thinking})
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": content})
            responses.append(content)
            thinking_parts.append(thinking)
            usages.append(usage)

        thinking_text = "\n\n".join(part for part in thinking_parts if isinstance(part, str) and part.strip()) or None
        row: dict[str, Any] = {
            "messages": messages,
            "prompt": prompt_input.prompt,
            "follow_up_prompts": prompt_input.follow_up_prompts,
            "thinking": thinking_text,
            "response": responses[-1] if responses else "",
            "responses": responses,
            "model": model,
            "provider": self.config.api.provider,
            "usage": self._merge_usage_totals(usages),
            "metadata": {
                "trace_type": "chat",
                "model_provider": self.config.api.provider,
                "model": model,
                "turn_count": len(prompt_input.turn_prompts()),
            },
        }
        if system_prompt:
            row["system"] = system_prompt
        return row

    def _request_chat_conversation_from_existing(
        self,
        prompt_input: PromptInput,
        existing_row: dict[str, Any],
    ) -> dict[str, Any]:
        base_prompt = self._chat_row_base_prompt(existing_row) or prompt_input.prompt
        completed_follow_ups = self._chat_row_completed_follow_up_prompts(existing_row, base_prompt)
        missing_follow_ups = prompt_input.follow_up_prompts[len(completed_follow_ups):]
        messages = self._chat_messages_from_row(existing_row, prompt_input)
        history = self._chat_history_from_messages(messages)
        responses = self._chat_row_responses(existing_row, messages)
        thinking_parts = self._chat_thinking_parts_from_messages(messages)
        usages: list[dict[str, Any] | None] = [
            existing_row.get("usage") if isinstance(existing_row.get("usage"), dict) else None
        ]
        model = existing_row.get("model") if isinstance(existing_row.get("model"), str) and existing_row.get("model") else self.config.get_effective_model()
        system_prompt = _system_from_training_example(existing_row) or self._chat_system_prompt(prompt_input)

        for prompt in missing_follow_ups:
            content, thinking, usage, model = self._request_chat_turn_with_optional_system(prompt, history, system_prompt)
            messages.append({"role": "user", "content": prompt, "thinking": None})
            messages.append({"role": "assistant", "content": content, "thinking": thinking})
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": content})
            responses.append(content)
            if isinstance(thinking, str) and thinking.strip():
                thinking_parts.append(thinking.strip())
            usages.append(usage)

        thinking_text = "\n\n".join(thinking_parts) or None
        metadata = existing_row.get("metadata") if isinstance(existing_row.get("metadata"), dict) else {}
        metadata = {
            **metadata,
            "trace_type": "chat",
            "model_provider": self.config.api.provider,
            "model": model,
            "turn_count": len(prompt_input.turn_prompts()),
        }
        row = {
            "messages": messages,
            "prompt": prompt_input.prompt,
            "follow_up_prompts": prompt_input.follow_up_prompts,
            "thinking": thinking_text,
            "response": responses[-1] if responses else "",
            "responses": responses,
            "model": model,
            "provider": self.config.api.provider,
            "usage": self._merge_usage_totals(usages),
            "metadata": metadata,
        }
        if system_prompt:
            row["system"] = system_prompt
        return row

    def _request_or_extend_chat_conversation(
        self,
        prompt_input: PromptInput,
        destination: Path,
        append_lock: threading.Lock | None,
    ) -> dict[str, Any]:
        training_row = self._request_chat_conversation(prompt_input)
        if append_lock is not None:
            with append_lock:
                rows = self._read_chat_training_rows(destination)
                if not self._chat_rows_include_completed_prompt(rows, prompt_input):
                    self._append_chat_training_row(destination, training_row)
        return training_row

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
            raise RuntimeError("Chat runner does not support mcp_servers.")
        if prompt_input and prompt_input.github_repo:
            raise RuntimeError("Chat runner does not support github_repo prompt inputs.")
        destination = self._resolve_output_path(f"{session_id}.jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        training_row = self._request_chat_conversation(prompt_input or PromptInput(prompt=prompt))
        destination.write_text(json.dumps(training_row, ensure_ascii=False) + "\n", encoding="utf-8")
        return destination

    def _metrics_from_training_row(self, training_row: dict[str, Any]) -> TraceMetrics:
        metrics = TraceMetrics()
        provider = training_row.get("provider")
        if isinstance(provider, str) and provider.strip():
            metrics.provider = provider.strip()
        model = training_row.get("model")
        if isinstance(model, str) and model.strip():
            metrics.model = model.strip()
        usage = training_row.get("usage")
        if isinstance(usage, dict):
            metrics.add_structured_usage(usage)
        metrics.finalize()
        return metrics

    def _run_chat_prompt_task(
        self,
        prompt_id: str,
        prompt_index: int,
        total_prompts: int,
        prompt_input: PromptInput,
        destination: Path,
        progress_callback: SessionProgressCallback | None,
        append_lock: threading.Lock | None = None,
    ) -> tuple[int, dict[str, Any]]:
        session_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        prompt_preview = self._prompt_preview(prompt_input.prompt)
        if progress_callback:
            progress_callback(
                SessionProgressUpdate(
                    prompt_id=prompt_id,
                    prompt_index=prompt_index,
                    total_prompts=total_prompts,
                    prompt=prompt_input.prompt,
                    prompt_preview=prompt_preview,
                    status="running",
                    session_id=session_id,
                    started_at=started_at,
                )
            )
        try:
            if self.config.mcp_servers:
                raise RuntimeError("Chat runner does not support mcp_servers.")
            if prompt_input.github_repo:
                raise RuntimeError("Chat runner does not support github_repo prompt inputs.")
            training_row = self._request_or_extend_chat_conversation(prompt_input, destination, append_lock)
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
                        trace_path=destination,
                        metrics=self._metrics_from_training_row(training_row),
                    )
                )
            return prompt_index, training_row
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

    @staticmethod
    def _append_chat_training_row(destination: Path, training_row: dict[str, Any]) -> None:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(training_row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def run_all(
        self,
        max_concurrency: int = 1,
        progress_callback: SessionProgressCallback | None = None,
        prompt_inputs: list[PromptInput] | None = None,
        resume: bool = False,
    ) -> list[Path]:
        prompt_inputs = prompt_inputs if prompt_inputs is not None else self.config.get_prompt_inputs()
        prompt_inputs = prompt_inputs_for_run(
            prompt_inputs,
            self.config.output.traces_dir,
            resume=resume,
            excluded_dirs=[self.config.output.failures_dir],
        )
        if not prompt_inputs:
            if resume:
                return []
            raise ValueError("No prompts configured")

        destination = self.config.output.traces_dir / "chat.jsonl" if resume else self._resolve_output_path("chat.jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        total_prompts = len(prompt_inputs)
        worker_count = max(1, min(max_concurrency, total_prompts))
        append_lock = threading.Lock()
        prompt_queue: queue.Queue[tuple[int, PromptInput]] = queue.Queue()
        for item in enumerate(prompt_inputs, start=1):
            prompt_queue.put(item)
        errors: list[Exception] = []
        error_lock = threading.Lock()
        stop_event = threading.Event()
        running: dict[int, tuple[PromptInput, datetime, float]] = {}
        running_lock = threading.Lock()

        def emit_queued(prompt_index: int, prompt_input: PromptInput) -> None:
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

        for prompt_index, prompt_input in enumerate(prompt_inputs, start=1):
            emit_queued(prompt_index, prompt_input)

        def worker() -> None:
            while True:
                if stop_event.is_set():
                    return
                try:
                    prompt_index, prompt_input = prompt_queue.get_nowait()
                except queue.Empty:
                    return
                if stop_event.is_set():
                    prompt_queue.task_done()
                    return
                try:
                    with running_lock:
                        running[prompt_index] = (
                            prompt_input,
                            datetime.now(timezone.utc),
                            time.monotonic() + self.config.timeout_seconds,
                        )
                    self._run_chat_prompt_task(
                        f"prompt-{prompt_index}",
                        prompt_index,
                        total_prompts,
                        prompt_input,
                        destination,
                        progress_callback,
                        append_lock,
                    )
                except Exception as exc:
                    with error_lock:
                        errors.append(exc)
                finally:
                    with running_lock:
                        running.pop(prompt_index, None)
                    prompt_queue.task_done()

        threads = [
            threading.Thread(target=worker, name=f"teich-chat-worker-{index}", daemon=True)
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                now = time.monotonic()
                timed_out: tuple[int, PromptInput, datetime] | None = None
                with running_lock:
                    for prompt_index, (prompt_input, started_at, deadline) in running.items():
                        if now >= deadline:
                            timed_out = (prompt_index, prompt_input, started_at)
                            break
                if timed_out is not None:
                    prompt_index, prompt_input, started_at = timed_out
                    error = RuntimeError(
                        f"Chat prompt {prompt_index} timed out after {self.config.timeout_seconds}s: "
                        f"{self._prompt_preview(prompt_input.prompt)}"
                    )
                    with error_lock:
                        errors.append(error)
                    stop_event.set()
                    if progress_callback:
                        progress_callback(
                            SessionProgressUpdate(
                                prompt_id=f"prompt-{prompt_index}",
                                prompt_index=prompt_index,
                                total_prompts=total_prompts,
                                prompt=prompt_input.prompt,
                                prompt_preview=self._prompt_preview(prompt_input.prompt),
                                status="failed",
                                started_at=started_at,
                                finished_at=datetime.now(timezone.utc),
                                error=str(error),
                            )
                        )
                    break
                for thread in threads:
                    thread.join(timeout=0.1)
        except KeyboardInterrupt:
            stop_event.set()
            raise

        if errors:
            raise errors[0]

        return [destination]


class PiRunner(DockerRuntimeRunner):
    """Manages Docker-based Pi sessions."""

    def _runtime_trace_guard_error(self, trace_path: Path) -> str | None:
        # Pi writes native session events incrementally. During generation a tool
        # call can briefly be invalid, fail validation, or be followed by a tool
        # error that the model corrects on the next turn. Do not kill the live
        # container from a partial trace snapshot; final export validation handles
        # truly incomplete sessions.
        return None

    @staticmethod
    def _pi_stdout_message(event: dict[str, Any]) -> dict[str, Any] | None:
        message = event.get("message")
        if isinstance(message, dict):
            return message
        assistant_message_event = event.get("assistantMessageEvent")
        if isinstance(assistant_message_event, dict):
            partial = assistant_message_event.get("partial")
            if isinstance(partial, dict):
                return partial
        return None

    @classmethod
    def _pi_stdout_metrics(cls, message: dict[str, Any] | None) -> TraceMetrics | None:
        if not isinstance(message, dict):
            return None
        metrics = TraceMetrics()
        provider = message.get("provider")
        model = message.get("model")
        if isinstance(provider, str) and provider.strip():
            metrics.provider = provider.strip()
        if isinstance(model, str) and model.strip():
            metrics.model = model.strip()
        usage = message.get("usage")
        if isinstance(usage, dict):
            metrics.add_structured_usage(usage)
            metrics.finalize()
        if not metrics.provider and not metrics.model and not metrics.has_token_usage and not metrics.has_cost:
            return None
        return metrics

    def _stdout_progress_update(self, events: list[dict[str, Any]]) -> tuple[TraceMetrics | None, str | None]:
        details: str | None = None
        metrics: TraceMetrics | None = None
        for event in events:
            event_type = event.get("type")
            message = self._pi_stdout_message(event)
            event_metrics = self._pi_stdout_metrics(message)
            if event_metrics is not None:
                metrics = event_metrics
            if event_type == "session":
                details = "pi session started"
            elif event_type == "message_end" and isinstance(message, dict) and message.get("role") == "user":
                details = "pi prompt sent"
            elif event_type == "message_start" and isinstance(message, dict) and message.get("role") == "assistant":
                details = "pi model request started"
            elif event_type == "message_update":
                details = "pi streaming response"
            elif event_type == "message_end" and isinstance(message, dict) and message.get("role") == "assistant":
                details = "pi assistant response complete"
            elif event_type == "turn_end":
                details = "pi turn complete"
        return metrics, details

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
        if self._normalize_provider(self.config.api.provider) == "openrouter":
            return "openai-completions"
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
        models_config = self._pi_models_config()
        if models_config:
            models_file = agent_dir / "models.json"
            models_file.write_text(
                json.dumps(models_config, indent=2) + "\n",
                encoding="utf-8",
            )

    def _write_pi_project_settings(self, workspace: Path) -> None:
        settings_dir = workspace / ".pi"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_file = settings_dir / "settings.json"
        settings_file.write_text(
            json.dumps(self._project_settings(), indent=2) + "\n",
            encoding="utf-8",
        )
        _make_tree_world_writable(settings_dir)

    @staticmethod
    def _resolve_pi_executable() -> str:
        return "pi"

    def _build_pi_command(
        self,
        prompt: str,
        workspace: Path,
        agent_dir: Path,
        session_dir: Path,
        container_name: str,
        continue_session: bool = False,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--user",
            RUNTIME_CONTAINER_USER,
            "-e",
            "HOME=/home/codex",
            "-e",
            f"PI_CODING_AGENT_DIR={PI_AGENT_DIR_IN_CONTAINER}",
            "-e",
            "PI_OFFLINE=1",
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
        pi_command = self._build_pi_agent_command(continue_session=continue_session)
        command.append(self.image_name)
        command.extend(pi_command)
        return command

    def _build_pi_agent_command(self, continue_session: bool = False) -> list[str]:
        configured_base_url = self.config.get_base_url()
        pi_command = [
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
        if self.config.developer_instructions:
            pi_command.extend(["--append-system-prompt", self.config.developer_instructions])
        if continue_session:
            pi_command.append("--continue")
        prompt_path = shlex.quote(f"{WORKSPACE_IN_CONTAINER}/{TEICH_PROMPT_FILE_NAME}")
        shell_command = f"exec {shlex.join(pi_command)} --print \"$(cat {prompt_path})\""
        return ["sh", "-lc", shell_command]

    def _build_pi_persistent_container_command(
        self,
        workspace: Path,
        agent_dir: Path,
        session_dir: Path,
        container_name: str,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--user",
            RUNTIME_CONTAINER_USER,
            "-e",
            "HOME=/home/codex",
            "-e",
            f"PI_CODING_AGENT_DIR={PI_AGENT_DIR_IN_CONTAINER}",
            "-e",
            "PI_OFFLINE=1",
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
        command.append(self.image_name)
        command.extend(["sleep", "infinity"])
        return command

    def _build_pi_exec_command(self, container_name: str, continue_session: bool = False) -> list[str]:
        return [
            "docker",
            "exec",
            "--user",
            RUNTIME_CONTAINER_USER,
            "-w",
            WORKSPACE_IN_CONTAINER,
            container_name,
            *self._build_pi_agent_command(continue_session=continue_session),
        ]

    @classmethod
    def _pi_assistant_has_content(cls, payload: dict[str, object]) -> bool:
        content = payload.get("content")
        if not isinstance(content, list):
            return False
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str) and block["text"].strip():
                return True
            if block_type == "toolCall":
                tool_id = block.get("id")
                tool_name = block.get("name")
                if isinstance(tool_id, str) and tool_id.strip() and isinstance(tool_name, str) and tool_name.strip():
                    return True
        return False

    def _copy_normalized_session_file(self, source_path: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)

    @staticmethod
    def _prompt_input_system_prompt(prompt_input: PromptInput | None) -> str | None:
        if prompt_input is None or not isinstance(prompt_input.system, str):
            return None
        system_prompt = prompt_input.system.strip()
        return system_prompt or None

    @classmethod
    def _pi_trace_includes_system_prompt(cls, trace_path: Path, system_prompt: str) -> bool:
        expected = system_prompt.strip()
        if not expected:
            return True
        try:
            with trace_path.open("r", encoding="utf-8") as source:
                for raw_line in source:
                    line = raw_line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "custom" and event.get("customType") == PI_SYSTEM_PROMPT_CUSTOM_TYPE:
                        data = event.get("data")
                        recorded = data.get("systemPrompt") if isinstance(data, dict) else None
                        if isinstance(recorded, str) and recorded.strip() == expected:
                            return True
                        continue
                    if event.get("type") != "message":
                        continue
                    payload = event.get("message")
                    if not isinstance(payload, dict) or payload.get("role") not in {"system", "developer"}:
                        continue
                    if cls._pi_first_text(payload.get("content")).strip() == expected:
                        return True
        except (OSError, json.JSONDecodeError):
            return False
        return False

    @staticmethod
    def _pi_first_text(content: Any) -> str:
        if isinstance(content, str):
            return content
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

    def _append_pi_system_prompt_metadata(self, trace_path: Path, prompt_input: PromptInput | None) -> None:
        system_prompt = self._prompt_input_system_prompt(prompt_input)
        if not system_prompt or self._pi_trace_includes_system_prompt(trace_path, system_prompt):
            return
        event = {
            "type": "custom",
            "id": f"teich-system-{uuid.uuid4().hex[:8]}",
            "parentId": None,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "customType": PI_SYSTEM_PROMPT_CUSTOM_TYPE,
            "data": {
                "systemPrompt": system_prompt,
                "source": "teich",
            },
        }
        with trace_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(event, separators=(",", ":")) + "\n")

    @staticmethod
    def _pi_trace_includes_available_tools(trace_path: Path) -> bool:
        try:
            with trace_path.open("r", encoding="utf-8") as source:
                for raw_line in source:
                    line = raw_line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "custom" and event.get("customType") == TEICH_AVAILABLE_TOOLS_CUSTOM_TYPE:
                        data = event.get("data")
                        tools = data.get("tools") if isinstance(data, dict) else None
                        return isinstance(tools, list)
        except (OSError, json.JSONDecodeError):
            return False
        return False

    def _append_pi_available_tools_metadata(self, trace_path: Path) -> None:
        if not trace_path.exists() or self._pi_trace_includes_available_tools(trace_path):
            return
        tools = self._configured_tools()
        if not tools:
            return
        event = {
            "type": "custom",
            "id": f"teich-tools-{uuid.uuid4().hex[:8]}",
            "parentId": None,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "customType": TEICH_AVAILABLE_TOOLS_CUSTOM_TYPE,
            "data": {
                "tools": tools,
                "source": "teich",
            },
        }
        with trace_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _finalize_trace_export(self, trace_path: Path, prompt_input: PromptInput | None = None) -> None:
        self._append_pi_system_prompt_metadata(trace_path, prompt_input)
        self._append_pi_available_tools_metadata(trace_path)

    def _latest_session_source_file(self, session_id: str, session_dir: Path, started_at: datetime) -> Path:
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
        return max(fresh_files, key=lambda path: path.stat().st_mtime)

    def _extract_session_file(self, session_id: str, session_dir: Path, started_at: datetime) -> Path:
        source_path = self._latest_session_source_file(session_id, session_dir, started_at)
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
        _make_tree_world_writable(agent_dir)
        _make_tree_world_writable(session_dir)
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name("pi", session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        failure_trace_saved = False
        failure_trace_checked = False
        try:
            self._write_pi_agent_settings(agent_dir)
            _make_tree_world_writable(agent_dir)
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_pi_project_settings(workspace)
            if len(turn_prompts) > 1:
                self._start_tracked_container(
                    self._build_pi_persistent_container_command(
                        workspace,
                        agent_dir,
                        session_dir,
                        container_name,
                    ),
                    container_name,
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                (workspace / TEICH_PROMPT_FILE_NAME).write_text(turn_prompt, encoding="utf-8")
                expected_turns = turn_prompts[: turn_index + 1]
                for attempt in range(AGENT_TURN_RETRY_LIMIT + 1):
                    continue_session = turn_index > 0 or attempt > 0
                    if len(turn_prompts) > 1:
                        command = self._build_pi_exec_command(container_name, continue_session=continue_session)
                    else:
                        command = self._build_pi_command(
                            turn_prompt,
                            workspace,
                            agent_dir,
                            session_dir,
                            container_name,
                            continue_session=continue_session,
                        )
                    process_error: subprocess.CalledProcessError | None = None
                    details = ""
                    try:
                        self._run_process(
                            command,
                            session_id,
                            started_at,
                            None,
                            progress_callback,
                            progress_base,
                            container_name,
                            remove_container_on_error=len(turn_prompts) <= 1,
                        )
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"Session {session_id[:8]} timed out after {self.config.timeout_seconds}s"
                        )
                    except subprocess.CalledProcessError as exc:
                        process_error = exc
                        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
                        details = stderr.strip() or stdout.strip() or str(exc)

                    try:
                        source_trace = self._latest_session_source_file(session_id, session_dir, started_at)
                    except RuntimeError:
                        source_trace = None

                    if source_trace is not None and self._trace_contains_completed_turns(source_trace, expected_turns):
                        break

                    if attempt < AGENT_TURN_RETRY_LIMIT:
                        continue

                    failure_trace_checked = True
                    if source_trace is not None:
                        try:
                            trace_path = self._extract_session_file(session_id, session_dir, started_at)
                        except RuntimeError:
                            trace_path = None
                        if trace_path is not None:
                            self._discard_output_file(trace_path)
                            failure_trace_saved = True
                    if process_error is not None:
                        raise RuntimeError(
                            f"Session {session_id[:8]} failed after {AGENT_TURN_RETRY_LIMIT} turn retries: {details}"
                        ) from process_error
                    raise RuntimeError(
                        f"Session {session_id[:8]} stopped before completing turn {turn_index + 1} "
                        f"after {AGENT_TURN_RETRY_LIMIT} retries"
                    )
            trace_path = self._extract_session_file(session_id, session_dir, started_at)
            if not self._trace_contains_completed_turns(trace_path, turn_prompts):
                failure_trace_checked = True
                self._discard_output_file(trace_path)
                failure_trace_saved = True
                raise RuntimeError(f"Session {session_id[:8]} stopped before completing all Pi turns")
            self._finalize_trace_export(trace_path, prompt_input)
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
            return trace_path
        except BaseException:
            if not failure_trace_saved and not failure_trace_checked:
                try:
                    trace_path = self._extract_session_file(session_id, session_dir, started_at)
                except RuntimeError:
                    trace_path = None
                if trace_path is not None:
                    self._discard_output_file(trace_path)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._cleanup_tracked_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(agent_dir, ignore_errors=True)
            shutil.rmtree(session_dir, ignore_errors=True)

    def run_all(
        self,
        max_concurrency: int = 1,
        progress_callback: SessionProgressCallback | None = None,
        prompt_inputs: list[PromptInput] | None = None,
        resume: bool = False,
    ) -> list[Path]:
        return super().run_all(
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
            prompt_inputs=prompt_inputs,
            resume=resume,
        )
