from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_INLINE_THINKING_BLOCK_PATTERN = re.compile(r"<(think|thinking)>(.*?)</\1>", re.DOTALL)


@dataclass(slots=True)
class TrainingExample:
    source_file: Path
    prompt: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "tools": self.tools,
            "metadata": self.metadata,
        }


def _first_text_block(content_blocks: Any) -> str:
    if isinstance(content_blocks, str):
        return content_blocks.strip()
    if not isinstance(content_blocks, list):
        return ""
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts).strip()


def _split_inline_thinking(text: str) -> tuple[str, str | None]:
    reasoning_parts: list[str] = []

    def _remove_match(match: re.Match[str]) -> str:
        reasoning = match.group(2).strip()
        if reasoning:
            reasoning_parts.append(reasoning)
        return ""

    content = _INLINE_THINKING_BLOCK_PATTERN.sub(_remove_match, text).strip()
    reasoning = "\n\n".join(reasoning_parts).strip()
    return content, reasoning or None


def _message_content_and_inline_reasoning(value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        return _split_inline_thinking(value)
    text = _first_text_block(value)
    reasoning = _pi_reasoning_content(value)
    return text, reasoning


def _has_same_system_message(messages: list[dict[str, Any]], content: str) -> bool:
    return any(
        message.get("role") == "system" and message.get("content") == content
        for message in messages
    )


def _pi_reasoning_content(content_blocks: Any) -> str | None:
    if not isinstance(content_blocks, list):
        return None
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "thinking":
            continue
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            parts.append(thinking.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _tool_result_content_text(payload: dict[str, Any]) -> str:
    return _first_text_block(payload.get("content"))


def _function_call_output_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, str):
        return output
    text = _first_text_block(output)
    if text:
        return text
    return str(output or "")


def _is_tool_not_found_result(tool_name: str | None, payload: dict[str, Any]) -> bool:
    content = _tool_result_content_text(payload).strip()
    if tool_name:
        return content == f"Tool {tool_name} not found"
    return content == "Tool  not found"


def _reasoning_summary(payload: dict[str, Any]) -> str | None:
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
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _reasoning_summary_blocks_from_content(payload: dict[str, Any]) -> list[dict[str, str]]:
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


def _custom_tool_arguments(name: str, value: Any) -> dict[str, Any]:
    if name == "apply_patch":
        return {"patch": value if isinstance(value, str) else _normalize_json_like_value(value)}
    return {"input": _normalize_json_like_value(value)}


def _custom_tool_output_value(value: Any) -> Any:
    if not isinstance(value, str):
        return _normalize_json_like_value(value)
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict) and isinstance(parsed.get("output"), str):
        return parsed["output"]
    return _normalize_json_like_value(parsed)


def normalize_codex_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict) or event.get("type") != "response_item":
        return event

    payload_type = payload.get("type")
    if payload_type == "reasoning":
        summary = payload.get("summary")
        if isinstance(summary, list) and any(
            isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
            for item in summary
        ):
            return event
        normalized_payload = dict(payload)
        normalized_payload["summary"] = _reasoning_summary_blocks_from_content(normalized_payload)
        normalized_event = dict(event)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    if payload_type == "custom_tool_call":
        name = payload.get("name")
        call_id = payload.get("call_id")
        if not isinstance(name, str) or not isinstance(call_id, str):
            return event
        normalized_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "input"}
        }
        normalized_payload.update(
            {
                "type": "function_call",
                "name": name,
                "call_id": call_id,
                "arguments": json.dumps(_custom_tool_arguments(name, payload.get("input")), ensure_ascii=False),
            }
        )
        normalized_event = dict(event)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    if payload_type == "custom_tool_call_output":
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            return event
        normalized_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "output"}
        }
        normalized_payload.update(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _custom_tool_output_value(payload.get("output")),
            }
        )
        normalized_event = dict(event)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    return event


def _normalize_json_like_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_like_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_like_value(item) for item in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return _normalize_json_like_value(parsed)


def _parse_function_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return _normalize_json_like_value(arguments) if arguments is not None else {}
    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        return _normalize_json_like_value(json.loads(stripped))
    except json.JSONDecodeError:
        return arguments


def _schema_identity(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)


def _infer_schema_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schemas = [_infer_schema_from_value(item) for item in value]
        schema: dict[str, Any] = {"type": "array"}
        if item_schemas:
            schema["items"] = _merge_schemas(item_schemas)
        return schema
    if isinstance(value, dict):
        return _infer_tool_parameters_schema([value])
    return {}


def _merge_object_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    properties_by_name: dict[str, list[dict[str, Any]]] = {}
    required_sets: list[set[str]] = []
    additional_properties = False
    for schema in schemas:
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, value in properties.items():
                if isinstance(value, dict):
                    properties_by_name.setdefault(name, []).append(value)
        required = schema.get("required")
        if isinstance(required, list):
            required_sets.append({item for item in required if isinstance(item, str)})
        else:
            required_sets.append(set())
        if schema.get("additionalProperties", True) is not False:
            additional_properties = True
    merged: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: _merge_schemas(property_schemas)
            for name, property_schemas in sorted(properties_by_name.items())
        },
        "additionalProperties": additional_properties,
    }
    if required_sets:
        required = sorted(set.intersection(*required_sets))
        if required:
            merged["required"] = required
    return merged


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema in schemas:
        if not schema:
            continue
        identity = _schema_identity(schema)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(schema)
    if not unique:
        return {}
    if len(unique) == 1:
        return unique[0]
    schema_types = {schema.get("type") for schema in unique if isinstance(schema.get("type"), str)}
    if schema_types == {"object"}:
        return _merge_object_schemas(unique)
    if schema_types == {"array"}:
        item_schemas = [schema.get("items") for schema in unique if isinstance(schema.get("items"), dict)]
        merged: dict[str, Any] = {"type": "array"}
        if item_schemas:
            merged["items"] = _merge_schemas(item_schemas)
        return merged
    return {"anyOf": unique}


def _infer_tool_parameters_schema(argument_samples: list[Any]) -> dict[str, Any]:
    dict_samples = [sample for sample in argument_samples if isinstance(sample, dict)]
    if not dict_samples:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    properties: dict[str, dict[str, Any]] = {}
    all_keys = sorted({key for sample in dict_samples for key in sample})
    for key in all_keys:
        observed = [_infer_schema_from_value(sample[key]) for sample in dict_samples if key in sample]
        properties[key] = _merge_schemas(observed)
    required = sorted(set.intersection(*(set(sample.keys()) for sample in dict_samples))) if dict_samples else []
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    return schema


def _parse_tool_descriptions_from_text(text: str) -> dict[str, str]:
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


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    if role == "model":
        return "assistant"
    return role


def _build_tool_entry(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name}
    if isinstance(schema, dict) and schema:
        function.update(schema)
    return {"type": "function", "function": function}


def _detect_trace_type(events: list[dict[str, Any]]) -> str:
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "external_session_meta":
            payload = event.get("payload")
            source = payload.get("source") if isinstance(payload, dict) else None
            if isinstance(source, str) and source.strip().lower() in {"claude", "claude-code", "claude_code"}:
                return "claude_code"
            return "external_agent"
        if event_type in {"assistant", "user", "system", "result"} and isinstance(event.get("session_id"), str):
            return "claude_code"
        if event_type in {"session_meta", "turn_context", "response_item", "event_msg"}:
            return "codex"
        if event_type in {
            "session",
            "message",
            "session_info",
            "model_change",
            "thinking_level_change",
            "compaction",
            "branch_summary",
            "custom",
            "custom_message",
            "label",
        }:
            return "pi"
    return "codex"


def _claude_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif block_type == "tool_result":
            text = block.get("content")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _claude_tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    return _first_text_block(content)


def _convert_claude_code_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_argument_samples: dict[str, list[Any]] = {}
    session_meta: dict[str, Any] = {}
    session_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    total_cost_usd: Any = None
    prompt = ""

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "external_session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                session_meta = payload
                value = payload.get("id")
                if isinstance(value, str) and value.strip():
                    session_id = value.strip()
                value = payload.get("model")
                if isinstance(value, str) and value.strip():
                    model = value.strip()
            continue
        if event_type == "external_message":
            role = event.get("role")
            content = event.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            normalized_role = _normalize_role(role)
            if normalized_role == "user" and content.strip() and not prompt:
                prompt = content.strip()
            messages.append({"role": normalized_role, "content": content})
            continue
        if event_type == "system":
            if isinstance(event.get("session_id"), str):
                session_id = event["session_id"]
            if event.get("subtype") == "init":
                value = event.get("model")
                if isinstance(value, str) and value.strip():
                    model = value.strip()
                tools = event.get("tools")
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, str) and tool.strip():
                            tool_names.add(tool.strip())
                        elif isinstance(tool, dict):
                            name = tool.get("name")
                            if isinstance(name, str) and name.strip():
                                tool_names.add(name.strip())
            continue
        if event_type in {"user", "assistant"} and isinstance(event.get("session_id"), str):
            session_id = event["session_id"]

        if event_type == "user":
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            content_blocks = payload.get("content")
            if isinstance(content_blocks, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content_blocks
            ):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_call_id = block.get("tool_use_id") or block.get("tool_call_id")
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        continue
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": "unknown_tool",
                            "content": _claude_tool_result_text(block),
                        }
                    )
                continue
            content = _claude_text_from_content(content_blocks)
            if content and not prompt:
                prompt = content
            if content:
                messages.append({"role": "user", "content": content})
            continue

        if event_type == "assistant":
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            value = payload.get("model")
            if isinstance(value, str) and value.strip():
                model = value.strip()
            payload_usage = payload.get("usage")
            if isinstance(payload_usage, dict):
                usage = payload_usage
            content_blocks = payload.get("content")
            content = _claude_text_from_content(content_blocks)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    if not tool_call_id or not tool_name:
                        continue
                    arguments = _normalize_json_like_value(block.get("input") or {})
                    tool_names.add(tool_name)
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if content or tool_calls:
                messages.append(message)
            continue

        if event_type == "result":
            value = event.get("session_id")
            if isinstance(value, str) and value.strip():
                session_id = value.strip()
            payload_usage = event.get("usage")
            if isinstance(payload_usage, dict):
                usage = payload_usage
            if event.get("total_cost_usd") is not None:
                total_cost_usd = event.get("total_cost_usd")
            result = event.get("result")
            if isinstance(result, str) and result.strip():
                last_content = messages[-1].get("content") if messages else None
                if last_content != result.strip():
                    messages.append({"role": "assistant", "content": result.strip()})

    tools = [
        _build_tool_entry(name, {"parameters": _infer_tool_parameters_schema(tool_argument_samples.get(name, []))})
        for name in sorted(tool_names)
    ]
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )
    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_id or session_meta.get("id") or trace_file.stem,
        "trace_type": "claude-code",
        "model_provider": session_meta.get("model_provider") or "anthropic",
        "model": model or session_meta.get("model"),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    if usage:
        metadata["usage"] = usage
    if total_cost_usd is not None:
        metadata["total_cost_usd"] = total_cost_usd
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _convert_external_agent_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    session_meta: dict[str, Any] = {}
    prompt = ""
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "external_session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                session_meta = payload
            continue
        if event.get("type") != "external_message":
            continue
        role = event.get("role")
        content = event.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        normalized_role = _normalize_role(role)
        if normalized_role == "user" and content.strip() and not prompt:
            prompt = content.strip()
        if content.strip() or normalized_role == "assistant":
            messages.append({"role": normalized_role, "content": content})
    metadata = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("id") or trace_file.stem,
        "trace_type": session_meta.get("source") or "external-agent",
        "model_provider": session_meta.get("model_provider"),
        "model": session_meta.get("model"),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=[],
        metadata=metadata,
    )


def load_trace_file(trace_file: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _convert_codex_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    pending_reasoning_parts: list[str] = []
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_descriptions: dict[str, str] = {}
    tool_call_names: dict[str, str] = {}
    session_meta: dict[str, Any] = {}
    turn_contexts: list[dict[str, Any]] = []
    prompt = ""

    for event in events:
        if not isinstance(event, dict):
            continue
        event = normalize_codex_trace_event(event)
        event_type = event.get("type")
        payload = event.get("payload")
        if event_type == "session_meta" and isinstance(payload, dict):
            session_meta = payload
            base_instructions = payload.get("base_instructions")
            if isinstance(base_instructions, dict):
                text = base_instructions.get("text")
                if isinstance(text, str) and text.strip() and not _has_same_system_message(messages, text):
                    messages.append({"role": "system", "content": text})
                    tool_descriptions.update(_parse_tool_descriptions_from_text(text))
            continue
        if event_type == "turn_context" and isinstance(payload, dict):
            turn_contexts.append(payload)
            continue
        if event_type != "response_item" or not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        if payload_type == "reasoning":
            reasoning = _reasoning_summary(payload)
            if reasoning:
                pending_reasoning_parts.append(reasoning)
            continue

        if payload_type == "message":
            role = payload.get("role")
            if not isinstance(role, str):
                continue
            normalized_role = _normalize_role(role)
            content = _first_text_block(payload.get("content"))
            if normalized_role == "user" and content and not prompt:
                prompt = content
            message: dict[str, Any] = {
                "role": normalized_role,
                "content": content,
            }
            if normalized_role == "assistant" and pending_reasoning_parts:
                message["reasoning_content"] = "\n\n".join(pending_reasoning_parts)
                pending_reasoning_parts = []
            messages.append(message)
            continue

        if payload_type == "function_call":
            name = payload.get("name")
            call_id = payload.get("call_id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                continue
            tool_names.add(name)
            tool_call_names[call_id] = name
            arguments = _parse_function_arguments(payload.get("arguments"))
            tool_argument_samples.setdefault(name, []).append(arguments)
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
            if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                messages[-1]["tool_calls"].append(tool_call)
            else:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call],
                }
                if pending_reasoning_parts:
                    assistant_message["reasoning_content"] = "\n\n".join(pending_reasoning_parts)
                    pending_reasoning_parts = []
                messages.append(assistant_message)
            continue

        if payload_type == "function_call_output":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            tool_name = tool_call_names.get(call_id)
            if tool_name:
                tool_names.add(tool_name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name or "unknown_tool",
                    "content": _function_call_output_text(payload),
                }
            )
            continue

        if payload_type == "tool_schema":
            name = payload.get("name")
            schema = payload.get("schema")
            if isinstance(name, str) and isinstance(schema, dict):
                tool_names.add(name)
                tool_schemas[name] = schema

    tools = []
    for name in sorted(tool_names):
        schema = dict(tool_schemas.get(name) or {})
        if name in tool_descriptions and "description" not in schema:
            schema["description"] = tool_descriptions[name]
        if "parameters" not in schema:
            schema["parameters"] = _infer_tool_parameters_schema(tool_argument_samples.get(name, []))
        tools.append(_build_tool_entry(name, schema))
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )

    metadata = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("id") or trace_file.stem,
        "trace_type": session_meta.get("source") or "codex",
        "model_provider": session_meta.get("model_provider"),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "turn_count": len(turn_contexts),
    }
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _convert_pi_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_descriptions: dict[str, str] = {}
    session_header: dict[str, Any] = {}
    model_change: dict[str, Any] = {}
    session_names: list[str] = []
    thinking_level: str | None = None
    prompt = ""
    invalid_tool_call_ids: set[str] = set()

    for event in events:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        payload = event.get("message")
        if not isinstance(payload, dict) or payload.get("role") != "toolResult":
            continue
        tool_call_id = payload.get("toolCallId")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        tool_name = payload.get("toolName") if isinstance(payload.get("toolName"), str) else None
        if _is_tool_not_found_result(tool_name, payload):
            invalid_tool_call_ids.add(tool_call_id)

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session":
            session_header = event
            continue
        if event_type == "model_change":
            model_change = event
            continue
        if event_type == "thinking_level_change":
            level = event.get("thinkingLevel")
            if isinstance(level, str) and level.strip():
                thinking_level = level.strip()
            continue
        if event_type == "session_info":
            name = event.get("name")
            if isinstance(name, str) and name.strip():
                session_names.append(name.strip())
            continue
        if event_type != "message":
            continue

        payload = event.get("message")
        if not isinstance(payload, dict):
            continue
        role = payload.get("role")
        if not isinstance(role, str):
            continue

        if role == "toolResult":
            tool_call_id = payload.get("toolCallId")
            if not isinstance(tool_call_id, str):
                continue
            if tool_call_id in invalid_tool_call_ids:
                continue
            tool_name = payload.get("toolName")
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name or "unknown_tool",
                "content": _first_text_block(payload.get("content")),
            }
            if payload.get("isError") is True:
                tool_message["is_error"] = True
            messages.append(tool_message)
            continue

        normalized_role = _normalize_role(role)
        content_blocks = payload.get("content")
        content = _first_text_block(content_blocks)

        if role == "developer" and content:
            tool_descriptions.update(_parse_tool_descriptions_from_text(content))

        if normalized_role == "user":
            if content and not prompt:
                prompt = content
            messages.append({"role": normalized_role, "content": content})
            continue

        message: dict[str, Any] = {
            "role": normalized_role,
            "content": content,
        }
        if normalized_role == "assistant":
            reasoning_content = _pi_reasoning_content(content_blocks)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "toolCall":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    if not tool_call_id or not tool_name or tool_call_id in invalid_tool_call_ids:
                        continue
                    tool_names.add(tool_name)
                    arguments = _parse_function_arguments(block.get("arguments"))
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if not message["content"] and "reasoning_content" not in message and "tool_calls" not in message:
                continue
        elif not content:
            continue
        messages.append(message)

    tools = [
        _build_tool_entry(
            name,
            {
                **({"description": tool_descriptions[name]} if name in tool_descriptions else {}),
                "parameters": _infer_tool_parameters_schema(tool_argument_samples.get(name, [])),
            },
        )
        for name in sorted(tool_names)
    ]
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )

    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_header.get("id") or trace_file.stem,
        "trace_type": "pi",
        "model_provider": model_change.get("provider"),
        "model": model_change.get("modelId"),
        "cwd": session_header.get("cwd"),
        "cli_version": None,
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    if thinking_level:
        metadata["thinking_level"] = thinking_level
    if session_names:
        metadata["session_names"] = session_names
        metadata["session_name"] = session_names[-1]
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _is_structured_training_row(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("messages"), list)


def _normalize_training_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        return None
    normalized_role = _normalize_role(role.strip())
    content, inline_reasoning = _message_content_and_inline_reasoning(message.get("content"))
    normalized: dict[str, Any] = {
        "role": normalized_role,
        "content": content,
    }
    if normalized_role == "assistant":
        reasoning_content = message.get("reasoning_content")
        if not isinstance(reasoning_content, str) or not reasoning_content.strip():
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                reasoning_content = thinking.strip()
        if not isinstance(reasoning_content, str) or not reasoning_content.strip():
            reasoning_content = inline_reasoning
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            normalized["reasoning_content"] = reasoning_content.strip()
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized["tool_calls"] = tool_calls
    if normalized_role == "tool":
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            normalized["tool_call_id"] = tool_call_id
        tool_name = message.get("name")
        if isinstance(tool_name, str) and tool_name:
            normalized["name"] = tool_name
        if message.get("is_error") is True:
            normalized["is_error"] = True
    return normalized


def normalize_training_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    return [
        normalized_message
        for normalized_message in (_normalize_training_message(message) for message in messages)
        if normalized_message is not None
    ]


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    return next(
        (
            message.get("content", "")
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )


def _structured_training_example_from_row(
    source_file: Path,
    row: dict[str, Any],
    row_index: int,
) -> TrainingExample:
    messages = normalize_training_messages(row.get("messages"))
    if not messages:
        system = row.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            messages.append({"role": "user", "content": prompt.strip()})
        response = row.get("response") if isinstance(row.get("response"), str) else ""
        assistant_message: dict[str, Any] = {"role": "assistant", "content": response}
        thinking = row.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            assistant_message["reasoning_content"] = thinking.strip()
        if assistant_message["content"] or "reasoning_content" in assistant_message:
            messages.append(assistant_message)
    tools = row.get("tools") if isinstance(row.get("tools"), list) else []
    prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else _prompt_from_messages(messages)
    metadata = dict(row.get("metadata")) if isinstance(row.get("metadata"), dict) else {}
    if isinstance(row.get("model"), str) and row.get("model"):
        metadata.setdefault("model", row["model"])
    if isinstance(row.get("usage"), dict) and row.get("usage"):
        metadata.setdefault("usage", row["usage"])
    metadata.setdefault("source_file", source_file.name)
    metadata.setdefault("source_line", row_index)
    metadata.setdefault(
        "trace_type",
        "chat" if any(key in row for key in ("system", "thinking", "response", "model")) and not tools else "structured",
    )
    return TrainingExample(
        source_file=source_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def convert_trace_to_training_example(trace_file: Path) -> TrainingExample:
    events = load_trace_file(trace_file)
    if events and all(_is_structured_training_row(event) for event in events):
        if len(events) != 1:
            raise ValueError(
                f"Structured training data file {trace_file} contains {len(events)} rows; use convert_traces_to_training_data or load_traces instead."
            )
        return _structured_training_example_from_row(trace_file, events[0], 1)
    trace_type = _detect_trace_type(events)
    if trace_type == "claude_code":
        return _convert_claude_code_trace_to_training_example(trace_file, events)
    if trace_type == "external_agent":
        return _convert_external_agent_trace_to_training_example(trace_file, events)
    if trace_type == "pi":
        return _convert_pi_trace_to_training_example(trace_file, events)
    return _convert_codex_trace_to_training_example(trace_file, events)


def _jsonl_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*.jsonl")
        if path.is_file() and "partials" not in path.relative_to(source).parts
    )


def _convert_jsonl_file_to_training_rows(jsonl_file: Path) -> list[dict[str, Any]]:
    rows = load_trace_file(jsonl_file)
    if not rows:
        return []
    if all(_is_structured_training_row(row) for row in rows):
        return [
            _structured_training_example_from_row(jsonl_file, row, row_index).to_dict()
            for row_index, row in enumerate(rows, start=1)
        ]
    trace_type = _detect_trace_type(rows)
    if trace_type == "claude_code":
        return [_convert_claude_code_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "external_agent":
        return [_convert_external_agent_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "pi":
        return [_convert_pi_trace_to_training_example(jsonl_file, rows).to_dict()]
    return [_convert_codex_trace_to_training_example(jsonl_file, rows).to_dict()]


def convert_traces_to_training_data(traces_dir: Path | str) -> list[dict[str, Any]]:
    source = Path(traces_dir)
    trace_files = _jsonl_files(source)
    rows: list[dict[str, Any]] = []
    for path in trace_files:
        rows.extend(_convert_jsonl_file_to_training_rows(path))
    return rows
