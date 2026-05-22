"""Docker-based runners for non-interactive Codex and Pi sessions."""

from __future__ import annotations

from collections.abc import Callable
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

from .converter import convert_trace_to_training_example, normalize_codex_trace_event

RUNTIME_IMAGE_NAME = "teich-runtime:v3"
RUNTIME_DOCKERFILE_NAME = "codex-runtime.Dockerfile"
CODEX_HOME_IN_CONTAINER = "/home/codex/.codex"
CLAUDE_HOME_IN_CONTAINER = "/home/codex/.claude"
HERMES_HOME_IN_CONTAINER = "/home/codex/.hermes"
PI_AGENT_DIR_IN_CONTAINER = "/home/codex/.pi/agent"
PI_SESSIONS_DIR_IN_CONTAINER = "/home/codex/pi-sessions"
WORKSPACE_IN_CONTAINER = "/workspace"
HERMES_DEFAULT_TOOLSETS = "safe,terminal,file,skills,memory,session_search,delegation"
HERMES_AGGREGATE_TRACE_FILE_NAME = "hermes-agent.jsonl"
HERMES_AGGREGATE_WRITE_LOCK = threading.Lock()


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


HERMES_UNIVERSAL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute shell commands in the session workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "background": {"type": "boolean", "default": False},
                    "timeout": {"type": "integer", "minimum": 1},
                    "workdir": {"type": "string"},
                    "pty": {"type": "boolean", "default": False},
                },
                "required": ["command"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process",
            "description": "Inspect or control background processes started by terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "poll", "log", "wait", "kill", "write", "submit"],
                    },
                    "session_id": {"type": "string"},
                    "data": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["action"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file with optional pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1, "default": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 500},
                },
                "required": ["path"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents or find files by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "target": {"type": "string", "enum": ["content", "files"], "default": "content"},
                    "path": {"type": "string", "default": "."},
                    "file_glob": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer", "default": 0},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_only", "count"],
                        "default": "content",
                    },
                    "context": {"type": "integer", "default": 0},
                },
                "required": ["pattern"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write complete file content, replacing existing content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "Apply targeted file edits or multi-file patches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["replace", "patch"], "default": "replace"},
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                    "patch": {"type": "string"},
                },
                "required": ["mode"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_view",
            "description": "Load a Hermes skill by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_manage",
            "description": "Create, update, patch, or inspect Hermes skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory",
            "description": "Store or retrieve durable session memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "session_search",
            "description": "Search previous Hermes session transcripts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
                "required": ["query"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "Spawn an isolated delegated subagent session for a bounded task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                    "subagent_type": {"type": "string"},
                },
                "required": ["prompt"],
                "additionalProperties": True,
            },
        },
    },
]
TEXT_SUBPROCESS_KWARGS = {"text": True, "encoding": "utf-8", "errors": "replace"}
LOCAL_PROVIDER_PROXY_SCRIPT_NAME = "local_provider_proxy.js"
CLAUDE_OPENROUTER_PROXY_SCRIPT_NAME = "claude_openrouter_proxy.js"
CLAUDE_OPENROUTER_PROXY_PORT = 17891
CLAUDE_OPENROUTER_SURROGATE_MODEL = "claude-sonnet-4-6"
OPENROUTER_GENERATION_STATS_ATTEMPTS = 2
OPENROUTER_GENERATION_STATS_TIMEOUT_SECONDS = 3
TEICH_PROMPT_FILE_NAME = ".teich-prompt.txt"
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
        self.cache_read_tokens += self._int_value(usage.get("cacheRead") or usage.get("cached_input_tokens"))
        self.cache_write_tokens += self._int_value(usage.get("cacheWrite"))
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
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


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


def completed_prompt_keys_from_outputs(traces_dir: Path) -> set[str]:
    if not traces_dir.exists():
        return set()
    completed: set[str] = set()
    for path in sorted(traces_dir.rglob("*.jsonl")):
        if not path.is_file():
            continue
        try:
            if "partials" in path.relative_to(traces_dir).parts:
                continue
        except ValueError:
            pass
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
            if isinstance(prompt, str) and prompt.strip() and _training_example_has_answer(example):
                completed.add(_prompt_completion_key(_prompt_input_from_training_example(example, prompt)))
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


def pending_prompt_inputs_for_resume(prompt_inputs: list[PromptInput], traces_dir: Path) -> list[PromptInput]:
    prompt_inputs = unique_prompt_inputs_by_completion_key(prompt_inputs)
    completed = completed_prompt_keys_from_outputs(traces_dir)
    if not completed:
        return prompt_inputs
    return [
        prompt_input
        for prompt_input in prompt_inputs
        if _prompt_completion_key(prompt_input) not in completed
    ]


class DockerRuntimeRunner:
    """Shared Docker runtime used by agent runners."""

    def __init__(self, config: Config):
        self.config = config
        self.image_name = RUNTIME_IMAGE_NAME
        self._active_processes: dict[subprocess.Popen[str], str | None] = {}
        self._active_processes_lock = threading.Lock()
        self._ensure_image()

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
    def _container_name(kind: str, session_id: str) -> str:
        safe_session_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id).strip("-")
        return f"teich-{kind}-{safe_session_id}"

    def _register_active_process(self, process: subprocess.Popen[str], container_name: str | None) -> None:
        with self._active_processes_lock:
            self._active_processes[process] = container_name

    def _unregister_active_process(self, process: subprocess.Popen[str]) -> None:
        with self._active_processes_lock:
            self._active_processes.pop(process, None)

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

    def _terminate_process(self, process: subprocess.Popen[str], container_name: str | None) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            self._remove_container(container_name)

    def _terminate_active_processes(self) -> None:
        with self._active_processes_lock:
            active = list(self._active_processes.items())
        for process, container_name in active:
            self._terminate_process(process, container_name)

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

    def _preserve_partial_session_files(self, session_dir: Path, session_id: str, prefix: str) -> list[Path]:
        if not session_dir.exists():
            return []
        preserved: list[Path] = []
        for source_path in sorted(path for path in session_dir.rglob("*.jsonl") if path.is_file()):
            destination = self._resolve_partial_output_path(f"{prefix}-{session_id}-{source_path.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            preserved.append(destination)
        return preserved

    def _resolve_partial_output_path(self, file_name: str) -> Path:
        destination = self.config.output.traces_dir / "partials" / file_name
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

    def _move_output_to_partials(self, trace_path: Path) -> Path | None:
        if not trace_path.exists():
            return None
        try:
            if "partials" in trace_path.relative_to(self.config.output.traces_dir).parts:
                return trace_path
        except ValueError:
            pass
        destination = self._resolve_partial_output_path(trace_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trace_path), destination)
        return destination

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
        cache_read_tokens = TraceMetrics._int_value(data.get("native_tokens_cached"))
        total_tokens = input_tokens + output_tokens + reasoning_tokens
        cost = data.get("total_cost")
        if not total_tokens and not isinstance(cost, (int, float)):
            return None
        usage: dict[str, Any] = {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "cacheRead": cache_read_tokens,
            "totalTokens": total_tokens,
        }
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
                        provider = payload.get("model_provider")
                        model = payload.get("model")
                        if isinstance(provider, str) and provider.strip() and not metrics.provider:
                            metrics.provider = provider.strip()
                        if isinstance(model, str) and model.strip() and not metrics.model:
                            metrics.model = model.strip()
                        input_tokens = TraceMetrics._int_value(payload.get("input_tokens"))
                        output_tokens = TraceMetrics._int_value(payload.get("output_tokens"))
                        total_tokens = TraceMetrics._int_value(payload.get("total_tokens"))
                        if TraceMetrics._has_any_key(payload, ("input_tokens", "output_tokens", "total_tokens")):
                            metrics.has_token_usage = True
                        metrics.input_tokens += input_tokens
                        metrics.output_tokens += output_tokens
                        metrics.total_tokens += total_tokens or input_tokens + output_tokens
                        cost = payload.get("total_cost")
                        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                            metrics.has_cost = True
                            metrics.total_cost += float(cost)
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
            metrics = self._summarize_trace_file_with_provider_stats(result)
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
        prompt_inputs = unique_prompt_inputs_by_completion_key(prompt_inputs)
        if not prompt_inputs:
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
            while not stop_event.is_set():
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
                    stop_event.set()
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
                self._terminate_process(process, container_name)
                raise
            finally:
                self._unregister_active_process(process)


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
        return _normalize_teich_prompt_user_event(normalize_codex_trace_event(event))

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
        api_key = self.config.get_api_key() or ""
        configured_base_url = self.config.get_base_url()
        base_url, proxy_target = self._codex_base_url_and_proxy_target()
        provider_name = self.config.api.provider
        provider_env_key = self._provider_env_key(provider_name)
        cmd = [
            "docker",
            "run",
            *([] if detached else ["--rm", "-i"]),
            *(["-d"] if detached else []),
            "--name",
            container_name,
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
            "codex",
            "-w",
            WORKSPACE_IN_CONTAINER,
            container_name,
            *self._build_codex_agent_command(resume=resume),
        ]

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
        container_name = self._container_name("codex", session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)

        try:
            self._write_codex_config(codex_home)
            if self._local_provider_proxy_target(self.config.api.provider, self.config.get_base_url()):
                self._write_local_provider_proxy(codex_home)
            existing_sessions = {path.resolve() for path in self._list_session_files(codex_home)}
            if len(turn_prompts) > 1:
                self._start_container(
                    self._build_codex_persistent_container_command(
                        workspace,
                        codex_home,
                        container_name,
                    )
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                if len(turn_prompts) > 1:
                    cmd = self._build_codex_exec_command(container_name, resume=turn_index > 0)
                else:
                    cmd = self._build_codex_command(
                        turn_prompt,
                        workspace,
                        codex_home,
                        container_name,
                        resume=turn_index > 0,
                    )
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
        except BaseException:
            self._preserve_partial_session_files(codex_home / "sessions", session_id, "codex")
            raise
        finally:
            if len(turn_prompts) > 1:
                self._remove_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(codex_home, ignore_errors=True)

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
            "codex",
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
        if configured_base_url and self._container_base_url(configured_base_url) != configured_base_url:
            command.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in [*self._api_env_items(), *self._base_url_env_items()]:
            command.extend(["-e", f"{key}={value}"])
        command.append(self.image_name)
        return command

    def _build_shell_command(self, *, continue_session: bool = False) -> str:
        raise NotImplementedError

    def _build_external_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
        *,
        continue_session: bool = False,
    ) -> list[str]:
        command = self._build_external_docker_base_command(workspace, home_dir, container_name)
        command.extend(["bash", "-lc", self._build_shell_command(continue_session=continue_session)])
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

    def _build_external_exec_command(self, container_name: str, *, continue_session: bool = False) -> list[str]:
        return [
            "docker",
            "exec",
            "--user",
            "codex",
            "-w",
            WORKSPACE_IN_CONTAINER,
            container_name,
            "bash",
            "-lc",
            self._build_shell_command(continue_session=continue_session),
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
        return {
            "timestamp": started_at.isoformat().replace("+00:00", "Z"),
            "type": "external_session_meta",
            "payload": {
                "id": session_id,
                "timestamp": started_at.isoformat().replace("+00:00", "Z"),
                "cwd": str(workspace),
                "source": self.source_name,
                "model_provider": self.config.api.provider or self.default_model_provider,
                "model": self.config.get_effective_model(),
            },
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
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name(self.container_kind, session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        destination = self._resolve_output_path(f"{self.source_name}-{session_id}.jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_events(destination, [self._session_meta_event(session_id, started_at, workspace)])
            if len(turn_prompts) > 1:
                self._start_container(self._build_external_persistent_container_command(workspace, home_dir, container_name))
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
            self._move_output_to_partials(destination)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._remove_container(container_name)
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(home_dir, ignore_errors=True)


class ClaudeCodeRunner(ExternalCliRunner):
    """Runs Claude Code in non-interactive stream-json mode."""

    provider_name = "claude-code"
    container_kind = "claude"
    home_in_container = CLAUDE_HOME_IN_CONTAINER
    source_name = "claude-code"
    default_model_provider = "anthropic"

    @staticmethod
    def _list_native_session_files(home_dir: Path) -> list[Path]:
        projects_dir = home_dir / "projects"
        if not projects_dir.exists():
            return []
        return sorted(path for path in projects_dir.rglob("*.jsonl") if path.is_file())

    def _extract_native_session_file(
        self,
        session_id: str,
        home_dir: Path,
        existing_sessions: set[Path],
        started_at: datetime,
    ) -> Path:
        session_files = self._list_native_session_files(home_dir)
        fresh_files = [path for path in session_files if path.resolve() not in existing_sessions]
        if not fresh_files:
            fresh_files = [
                path
                for path in session_files
                if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= started_at
            ]
        if not fresh_files:
            raise RuntimeError(f"No Claude Code native session file found for {session_id}")
        source_path = max(fresh_files, key=lambda path: path.stat().st_mtime)
        destination = self._resolve_output_path(source_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return destination

    def _needs_openrouter_model_proxy(self) -> bool:
        if self._provider_env_key(self.config.api.provider) != "OPENROUTER_API_KEY":
            return False
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
        api_key = self.config.get_api_key() or ""
        if not api_key:
            return []
        provider_env_key = self._provider_env_key(self.config.api.provider)
        if provider_env_key == "OPENROUTER_API_KEY":
            return [
                ("TEICH_API_KEY", api_key),
                ("ANTHROPIC_AUTH_TOKEN", api_key),
                ("ANTHROPIC_API_KEY", ""),
                ("OPENROUTER_API_KEY", api_key),
            ]
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

    def _build_shell_command(self, *, continue_session: bool = False) -> str:
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
        if continue_session:
            claude_command.append("--continue")
        return f"{shlex.join(claude_command)} < {shlex.quote(WORKSPACE_IN_CONTAINER + '/' + TEICH_PROMPT_FILE_NAME)}"

    def _build_external_command(
        self,
        workspace: Path,
        home_dir: Path,
        container_name: str,
        *,
        continue_session: bool = False,
    ) -> list[str]:
        if not self._needs_openrouter_model_proxy():
            return super()._build_external_command(
                workspace,
                home_dir,
                container_name,
                continue_session=continue_session,
            )
        self._write_openrouter_proxy(home_dir)
        command = self._build_external_docker_base_command(workspace, home_dir, container_name)
        command.extend(
            [
                "bash",
                "-lc",
                f"{self._openrouter_proxy_shell_prefix()}exec {self._build_shell_command(continue_session=continue_session)}",
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
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name(self.container_kind, session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            existing_sessions = {path.resolve() for path in self._list_native_session_files(home_dir)}
            if len(turn_prompts) > 1:
                self._start_container(self._build_external_persistent_container_command(workspace, home_dir, container_name))
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
                    self._run_external_process(command, container_name)
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
                    stdout = exc.output if isinstance(exc.output, str) else (exc.output or "")
                    details = stderr.strip() or stdout.strip()
                    raise RuntimeError(f"Session {session_id[:8]} failed: {details}") from exc
            trace_path = self._extract_native_session_file(session_id, home_dir, existing_sessions, started_at)
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(trace_path))
            return trace_path
        except BaseException:
            self._preserve_partial_session_files(home_dir / "projects", session_id, self.source_name)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._remove_container(container_name)
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

    def _build_shell_command(self, *, continue_session: bool = False) -> str:
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
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
                "actual_cost_usd": actual_cost,
                "total_cost": total_cost,
                "messages": messages,
            }
        )
        model_config = self._json_or_original(session.get("model_config"))
        if model_config is not None:
            session["model_config"] = model_config
        return session

    @staticmethod
    def _hermes_tool_call_xml(tool_call: dict[str, Any]) -> str | None:
        function = tool_call.get("function")
        name = None
        arguments: Any = {}
        if isinstance(function, dict):
            raw_name = function.get("name")
            if isinstance(raw_name, str) and raw_name:
                name = raw_name
            arguments = function.get("arguments") or {}
        raw_name = tool_call.get("name")
        if name is None and isinstance(raw_name, str) and raw_name:
            name = raw_name
        if not name:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {"value": arguments}}
        return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"

    def _hermes_conversation_item(self, message: dict[str, object]) -> dict[str, str] | None:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user":
            return {"from": "human", "value": content}
        if role == "tool":
            payload = {
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name") or message.get("tool_name"),
                "content": content,
            }
            return {"from": "tool", "value": f"<tool_response>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_response>"}
        if role == "assistant":
            parts: list[str] = []
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                parts.append(f"<think>\n{reasoning.strip()}\n</think>")
            if content.strip():
                parts.append(content)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_call_xml = self._hermes_tool_call_xml(tool_call)
                        if tool_call_xml:
                            parts.append(tool_call_xml)
            return {"from": "gpt", "value": "\n".join(parts)}
        return None

    def _hermes_conversation_export(
        self,
        row: sqlite3.Row,
        messages: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        conversation: list[dict[str, str]] = []
        system_prompt = self._sqlite_row_get(row, "system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            conversation.append({"from": "system", "value": system_prompt.strip()})
        for message in messages:
            item = self._hermes_conversation_item(message)
            if item is not None:
                conversation.append(item)
        return conversation

    @staticmethod
    def _hermes_trace_task(conversation: list[dict[str, str]]) -> str:
        for item in conversation:
            if item.get("from") == "human":
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    @staticmethod
    def _hermes_tools_snapshot() -> list[dict[str, Any]]:
        return json.loads(json.dumps(HERMES_UNIVERSAL_TOOLS))

    def _hermes_trace_row(
        self,
        row: sqlite3.Row,
        messages: list[dict[str, object]],
        workspace: Path,
        *,
        partial: bool = False,
    ) -> dict[str, object]:
        traces = self._hermes_conversation_export(row, messages)
        metadata = self._hermes_session_export(row, messages, workspace, partial=partial)
        metadata.pop("messages", None)
        session_id = str(metadata.get("id") or self._sqlite_row_get(row, "id") or "")
        return {
            "id": session_id,
            "task": self._hermes_trace_task(traces),
            "traces": traces,
            "tools": self._hermes_tools_snapshot(),
            "metadata": metadata,
        }

    def _hermes_aggregate_trace_path(self, *, partial: bool = False) -> Path:
        if partial:
            return self.config.output.traces_dir / "partials" / HERMES_AGGREGATE_TRACE_FILE_NAME
        return self.config.output.traces_dir / HERMES_AGGREGATE_TRACE_FILE_NAME

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

    def _export_hermes_state_sessions(
        self,
        home_dir: Path,
        workspace: Path,
        *,
        partial: bool = False,
    ) -> dict[str, Path]:
        state_db = self._hermes_state_db(home_dir)
        if not state_db.exists():
            return {}
        exported: dict[str, Path] = {}
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
            if not rows:
                return {}
            destination = self._hermes_aggregate_trace_path(partial=partial)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with HERMES_AGGREGATE_WRITE_LOCK:
                self._write_events(destination, rows)
            for row in rows:
                session_id = str(row.get("id") or "")
                if session_id:
                    exported[session_id] = destination
        finally:
            connection.close()
        return exported

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
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name(self.container_kind, session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        fallback_destination = self._resolve_output_path(f"{self.source_name}-{session_id}.jsonl")
        fallback_destination.parent.mkdir(parents=True, exist_ok=True)
        stdout_parts: list[str] = []
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_hermes_runtime_config(home_dir)
            if len(turn_prompts) > 1:
                self._start_container(self._build_external_persistent_container_command(workspace, home_dir, container_name))
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
                    self._export_hermes_state_sessions(home_dir, workspace, partial=True)
                    self._move_output_to_partials(fallback_destination)
                    details = stderr.strip() or stdout.strip()
                    raise RuntimeError(f"Session {session_id[:8]} failed: {details}") from exc
                except RuntimeError:
                    self._export_hermes_state_sessions(home_dir, workspace, partial=True)
                    self._move_output_to_partials(fallback_destination)
                    raise
                stdout_parts.append(stdout)
                if not self._hermes_state_db(home_dir).exists():
                    if not fallback_destination.exists():
                        self._write_events(
                            fallback_destination,
                            [self._session_meta_event(session_id, started_at, workspace)],
                        )
                    self._write_events(
                        fallback_destination,
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
                            trace_path=fallback_destination,
                            metrics=self._summarize_trace_file(fallback_destination)
                            if fallback_destination.exists()
                            else TraceMetrics(),
                        )
                    )
            final_exports = self._export_hermes_state_sessions(home_dir, workspace)
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
            if not final_exports and not fallback_destination.exists():
                self._write_events(fallback_destination, [self._session_meta_event(session_id, started_at, workspace)])
                destination = fallback_destination
            self._copy_workspace_snapshot(workspace, self._sandbox_destination(destination))
            return destination
        except BaseException:
            self._move_output_to_partials(fallback_destination)
            raise
        finally:
            if len(turn_prompts) > 1:
                self._remove_container(container_name)
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
        )
        cache_read_tokens = TraceMetrics._int_value(usage.get("cacheRead") or usage.get("cached_input_tokens"))
        cache_write_tokens = TraceMetrics._int_value(usage.get("cacheWrite"))
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

    def _request_chat_turn(
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
        extension: tuple[int, dict[str, Any]] | None = None
        if prompt_input.follow_up_prompts and append_lock is not None and destination.exists():
            with append_lock:
                rows = self._read_chat_training_rows(destination)
                extension = self._find_chat_row_to_extend(rows, prompt_input)

        if extension is None:
            training_row = self._request_chat_conversation(prompt_input)
            if append_lock is not None:
                with append_lock:
                    self._append_chat_training_row(destination, training_row)
            return training_row

        training_row = self._request_chat_conversation_from_existing(prompt_input, extension[1])
        if append_lock is not None:
            with append_lock:
                rows = self._read_chat_training_rows(destination)
                latest_extension = self._find_chat_row_to_extend(rows, prompt_input)
                if latest_extension is not None:
                    rows[latest_extension[0]] = training_row
                    self._write_chat_training_rows(destination, rows)
                elif not self._chat_rows_include_completed_prompt(rows, prompt_input):
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
        prompt_inputs = unique_prompt_inputs_by_completion_key(prompt_inputs)
        if not prompt_inputs:
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
            while not stop_event.is_set():
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
                    stop_event.set()
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
            "codex",
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
            "codex",
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
            "-i",
            "--user",
            "codex",
            "-w",
            WORKSPACE_IN_CONTAINER,
            container_name,
            *self._build_pi_agent_command(continue_session=continue_session),
        ]

    @classmethod
    def _normalize_pi_trace_event(cls, event: dict[str, object]) -> dict[str, object]:
        event = _normalize_teich_prompt_user_event(event)
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
        runtime_errors: list[str] = []
        for event in events:
            if event.get("type") != "message":
                continue
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            stop_reason = payload.get("stopReason")
            error_message = payload.get("errorMessage")
            if stop_reason == "error" or (isinstance(error_message, str) and error_message.strip()):
                if isinstance(error_message, str) and error_message.strip():
                    runtime_errors.append(error_message.strip())
                else:
                    runtime_errors.append("model/provider returned stopReason=error")
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
        if runtime_errors:
            raise RuntimeError(
                "Pi session ended with model/provider error: "
                f"{runtime_errors[0]}. "
                "This trace was not exported because the model/provider did not produce a successful assistant response."
            )
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
        _make_tree_world_writable(agent_dir)
        _make_tree_world_writable(session_dir)
        started_at = datetime.now(timezone.utc)
        container_name = self._container_name("pi", session_id)
        turn_prompts = _agent_turn_prompts(prompt, prompt_input)
        try:
            self._write_pi_agent_settings(agent_dir)
            _make_tree_world_writable(agent_dir)
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_pi_project_settings(workspace)
            if len(turn_prompts) > 1:
                self._start_container(
                    self._build_pi_persistent_container_command(
                        workspace,
                        agent_dir,
                        session_dir,
                        container_name,
                    )
                )
            for turn_index, turn_prompt in enumerate(turn_prompts):
                (workspace / TEICH_PROMPT_FILE_NAME).write_text(turn_prompt, encoding="utf-8")
                if len(turn_prompts) > 1:
                    command = self._build_pi_exec_command(container_name, continue_session=turn_index > 0)
                else:
                    command = self._build_pi_command(
                        turn_prompt,
                        workspace,
                        agent_dir,
                        session_dir,
                        container_name,
                        continue_session=turn_index > 0,
                    )
                try:
                    self._run_process(
                        command,
                        session_id,
                        started_at,
                        session_dir,
                        progress_callback,
                        progress_base,
                        container_name,
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
        except BaseException:
            self._preserve_partial_session_files(session_dir, session_id, "pi")
            raise
        finally:
            if len(turn_prompts) > 1:
                self._remove_container(container_name)
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
