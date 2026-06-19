"""Normalize provider-native trace/stdout events into simple display events for the Studio UI.

Display events are plain dicts:
    {"kind": str, "text": str, "name": str | None, "detail": str | None}

Kinds: user, assistant, thinking, tool_call, tool_result, system, log, status, error.
"""

from __future__ import annotations

import json
from typing import Any

MAX_TEXT_CHARS = 20000
MAX_DETAIL_CHARS = 4000


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters truncated]"


def display_event(kind: str, text: str, name: str | None = None, detail: str | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {"kind": kind, "text": _clip(text)}
    if name:
        event["name"] = name
    if detail:
        event["detail"] = _clip(detail, MAX_DETAIL_CHARS)
    return event


def _content_text(content: Any) -> str:
    """Best-effort extraction of human-readable text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "thinking"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value)
                        break
                    if isinstance(value, list):
                        nested = _content_text(value)
                        if nested.strip():
                            parts.append(nested)
                            break
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def _format_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments
    if isinstance(arguments, (dict, list)):
        try:
            return json.dumps(arguments, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(arguments)
    return "" if arguments is None else str(arguments)


# ---------------------------------------------------------------------------
# Codex session events ({"type": "response_item"|"event_msg", "payload": ...})
# ---------------------------------------------------------------------------

def _summarize_codex(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = event.get("type")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return []
    if event_type == "response_item":
        item_type = payload.get("type")
        if item_type == "message":
            role = payload.get("role")
            text = _content_text(payload.get("content"))
            if not text.strip():
                return []
            if role == "assistant":
                return [display_event("assistant", text)]
            if role == "user":
                # User turns are echoed by the studio itself; skip duplicates.
                return []
            return []
        if item_type == "function_call":
            name = payload.get("name")
            return [
                display_event(
                    "tool_call",
                    _format_arguments(payload.get("arguments")),
                    name=name if isinstance(name, str) else "tool",
                )
            ]
        if item_type == "function_call_output":
            output = payload.get("output")
            if isinstance(output, dict):
                output = output.get("content") or output.get("output") or ""
            text = _content_text(output) if not isinstance(output, str) else output
            return [display_event("tool_result", text or "(no output)")]
        if item_type == "reasoning":
            text = _content_text(payload.get("summary")) or _content_text(payload.get("content"))
            if text.strip():
                return [display_event("thinking", text)]
            return []
        if item_type in {"local_shell_call", "custom_tool_call"}:
            name = payload.get("name") or item_type
            return [display_event("tool_call", _format_arguments(payload.get("arguments") or payload.get("action")), name=str(name))]
        return []
    if event_type == "event_msg":
        msg_type = payload.get("type")
        if msg_type == "agent_message":
            text = payload.get("message")
            if isinstance(text, str) and text.strip():
                return [display_event("assistant", text)]
        elif msg_type == "agent_reasoning":
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                return [display_event("thinking", text)]
        elif msg_type == "task_started":
            return [display_event("status", "Agent started working…")]
        elif msg_type in {"exec_command_begin", "exec_command_end"}:
            command = payload.get("command")
            if msg_type == "exec_command_begin" and command:
                command_text = " ".join(command) if isinstance(command, list) else str(command)
                return [display_event("tool_call", command_text, name="shell")]
        elif msg_type == "error":
            text = payload.get("message")
            if isinstance(text, str) and text.strip():
                return [display_event("error", text)]
        return []
    return []


# ---------------------------------------------------------------------------
# Pi session/stdout events ({"type": "message", "message": {...}})
# ---------------------------------------------------------------------------

def _summarize_pi_content_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get("type")
    if item_type == "text":
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return display_event("assistant", text)
        return None
    if item_type == "thinking":
        text = item.get("thinking")
        if isinstance(text, str) and text.strip():
            return display_event("thinking", text)
        return None
    if item_type in {"toolCall", "tool_call", "tool_use", "toolUse"}:
        name = item.get("name") or item.get("toolName") or "tool"
        return display_event("tool_call", _format_arguments(item.get("arguments") or item.get("input")), name=str(name))
    if item_type in {"toolResult", "tool_result"}:
        return display_event("tool_result", _content_text(item.get("content") or item.get("output")) or "(no output)")
    return None


def _summarize_pi(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event.get("type") != "message":
        return []
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    role = message.get("role")
    content = message.get("content")
    if role == "assistant" and isinstance(content, list):
        events: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                summary = _summarize_pi_content_item(item)
                if summary:
                    events.append(summary)
        return events
    if role in {"toolResult", "tool"}:
        text = _content_text(content)
        if text.strip():
            return [display_event("tool_result", text)]
    return []


# ---------------------------------------------------------------------------
# Claude Code stream-json / native transcript lines
# ---------------------------------------------------------------------------

def _summarize_claude_content_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get("type")
    if item_type == "text":
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return display_event("assistant", text)
        return None
    if item_type == "thinking":
        text = item.get("thinking")
        if isinstance(text, str) and text.strip():
            return display_event("thinking", text)
        return None
    if item_type == "tool_use":
        name = item.get("name") or "tool"
        return display_event("tool_call", _format_arguments(item.get("input")), name=str(name))
    if item_type == "tool_result":
        return display_event("tool_result", _content_text(item.get("content")) or "(no output)")
    return None


def _summarize_claude(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = event.get("type")
    if event_type == "system":
        subtype = event.get("subtype")
        if subtype == "init":
            model = event.get("model")
            return [display_event("status", f"Session ready (model: {model})" if model else "Session ready")]
        return []
    if event_type == "result":
        if event.get("is_error"):
            text = event.get("result") or event.get("error") or "Turn failed"
            return [display_event("error", str(text))]
        return []
    if event_type not in {"assistant", "user"}:
        return []
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    events: list[dict[str, Any]] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            summary = _summarize_claude_content_item(item)
            if summary is not None and not (event_type == "user" and summary["kind"] == "assistant"):
                events.append(summary)
    return events


# ---------------------------------------------------------------------------
# Hermes / teich external trace events
# ---------------------------------------------------------------------------

def _summarize_external(event: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(event.get("messages"), list):
        return summarize_chat_row(event)
    if event.get("type") != "external_message":
        return []
    role = event.get("role")
    content = event.get("content")
    text = _content_text(content)
    events: list[dict[str, Any]] = []
    tool_calls = event.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name") or "tool"
                events.append(display_event("tool_call", _format_arguments(function.get("arguments")), name=str(name)))
    if role == "assistant" and text.strip():
        events.append(display_event("assistant", text))
    elif role == "tool" and text.strip():
        events.append(display_event("tool_result", text, name=str(event.get("name") or "") or None))
    return events


def _summarize_structured_row(event: dict[str, Any]) -> list[dict[str, Any]]:
    return summarize_chat_row(event)


def _summarize_cursor(event: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(event.get("messages"), list):
        return summarize_chat_row(event)
    role = event.get("role")
    message = event.get("message")
    if not isinstance(role, str) or not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    events: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        summary = _summarize_claude_content_item(item)
        if summary is not None and not (role == "user" and summary["kind"] == "assistant"):
            events.append(summary)
    return events


_SUMMARIZERS = {
    "codex": _summarize_codex,
    "pi": _summarize_pi,
    "claude-code": _summarize_claude,
    "hermes": _summarize_external,
    "cursor": _summarize_cursor,
    "chat": _summarize_structured_row,
}


def summarize_event(provider: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one provider-native event into zero or more display events."""
    summarizer = _SUMMARIZERS.get(provider)
    if summarizer is None:
        return []
    try:
        return summarizer(event)
    except Exception:
        return []


def summarize_trace_events(provider: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize a full trace file for read-only preview, including user turns."""
    display: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        user_text = _trace_user_text(provider, event)
        if user_text is not None:
            if user_text.strip():
                display.append(display_event("user", user_text))
            continue
        display.extend(summarize_event(provider, event))
    return display


def _trace_user_text(provider: str, event: dict[str, Any]) -> str | None:
    """Return user-turn text when this trace event is a user message, else None."""
    if provider == "codex":
        if event.get("type") == "response_item" and isinstance(event.get("payload"), dict):
            payload = event["payload"]
            if payload.get("type") == "message" and payload.get("role") == "user":
                return _content_text(payload.get("content"))
        if event.get("type") == "event_msg" and isinstance(event.get("payload"), dict):
            payload = event["payload"]
            if payload.get("type") == "user_message":
                message = payload.get("message")
                return message if isinstance(message, str) else ""
        return None
    if provider == "pi":
        message = event.get("message")
        if event.get("type") == "message" and isinstance(message, dict) and message.get("role") == "user":
            return _content_text(message.get("content"))
        return None
    if provider == "claude-code":
        message = event.get("message")
        if event.get("type") == "user" and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and not any(
                isinstance(item, dict) and item.get("type") == "tool_result" for item in content
            ):
                return _content_text(content)
        return None
    if provider == "cursor":
        message = event.get("message")
        if event.get("role") == "user" and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(item, dict) and item.get("type") == "tool_result" for item in content
            ):
                return None
            return _content_text(content)
        return None
    if provider == "hermes":
        if event.get("type") == "external_message" and event.get("role") == "user":
            return _content_text(event.get("content"))
        return None
    return None


def summarize_chat_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize a chat training row (messages list) for preview."""
    display: list[dict[str, Any]] = []
    messages = row.get("messages")
    if not isinstance(messages, list):
        return display
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        text = content if isinstance(content, str) else _content_text(content)
        thinking = message.get("thinking") or message.get("reasoning_content")
        if role == "system" and text.strip():
            display.append(display_event("system", text))
        elif role == "user" and text.strip():
            display.append(display_event("user", text))
        elif role == "assistant":
            if isinstance(thinking, str) and thinking.strip():
                display.append(display_event("thinking", thinking))
            if text.strip():
                display.append(display_event("assistant", text))
    return display
