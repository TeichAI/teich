from __future__ import annotations

import json
from typing import Any


def first_text_block(content_blocks: Any) -> str:
    if isinstance(content_blocks, str):
        return content_blocks.strip()
    if not isinstance(content_blocks, list):
        return ""

    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {"input_text", "output_text", "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts).strip()


def has_message(messages: list[dict[str, Any]], *, role: str, content: str) -> bool:
    return any(message.get("role") == role and message.get("content") == content for message in messages)


def pi_reasoning_text(content_blocks: Any) -> str | None:
    if not isinstance(content_blocks, list):
        return None

    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict) or block.get("type") != "thinking":
            continue
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            parts.append(thinking.strip())

    result = "\n\n".join(parts).strip()
    return result or None


def tool_result_content_text(payload: dict[str, Any]) -> str:
    return first_text_block(payload.get("content"))


def is_tool_not_found_result(tool_name: str | None, payload: dict[str, Any]) -> bool:
    content = tool_result_content_text(payload).strip()
    if tool_name:
        return content == f"Tool {tool_name} not found"
    return content == "Tool  not found"


def reasoning_summary(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary")
    parts: list[str] = []
    if isinstance(summary, list):
        for item in summary:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    result = "\n\n".join(parts).strip()
    if result:
        return result

    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "reasoning_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    result = "\n\n".join(parts).strip()
    return result or None


def parse_tool_descriptions(text: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    in_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_section:
            if line == "Available tools:":
                in_section = True
            continue
        if not line:
            if descriptions:
                break
            continue
        if not line.startswith("- "):
            if descriptions:
                break
            continue

        name, separator, description = line[2:].partition(":")
        tool_name = name.strip()
        tool_description = description.strip()
        if separator and tool_name and tool_description:
            descriptions[tool_name] = tool_description
    return descriptions


def normalize_json_like_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json_like_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_like_value(item) for item in value]
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return normalize_json_like_value(parsed)


def parse_function_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return normalize_json_like_value(arguments) if arguments is not None else {}

    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        return normalize_json_like_value(json.loads(stripped))
    except json.JSONDecodeError:
        return arguments
