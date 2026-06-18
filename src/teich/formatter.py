from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset, Features, Json, List, Value, concatenate_datasets
from rich.console import Console

from .converter import normalize_training_messages


_GEMMA_TURN_START_PATTERN = re.compile(r"<\|turn>(model|user|system)\n")
_GEMMA_ASSISTANT_TURN_PREFIX = "<|turn>model\n"
_GEMMA_TURN_END = "<turn|>"
_GEMMA_THOUGHT_PREFIX = "<|channel>thought\n"
_GEMMA_TOOL_RESPONSE_START = "<|tool_response>"
_GEMMA_TOOL_RESPONSE_END = "<tool_response|>"
_TOOL_RESPONSE_DELIMITERS = (
    (_GEMMA_TOOL_RESPONSE_START, _GEMMA_TOOL_RESPONSE_END),
    ("<tool_response>", "</tool_response>"),
)
_TOOL_RESPONSE_START_TAG_PATTERN = re.compile(r"<tool_response(?:\s+[^>]*)?>")
_TOOL_RESPONSE_END_TOKENS = ("</tool_response>", _GEMMA_TOOL_RESPONSE_END)
_ASSISTANT_BLOCK_START_TOKENS = (
    "<|im_start|>assistant\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<start_of_turn>model\n",
    "<|assistant|>\n",
    "<|assistant|>",
    "<assistant>",
    "<|start_of_role|>assistant<|end_of_role|>",
)
_ASSISTANT_BLOCK_END_TOKENS = (
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "</assistant>",
    "</s>",
    "<|end_of_text|>",
)
_REASONING_BLOCK_PATTERNS = (
    re.compile(r"<think>\n.*?</think>\n\n?", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL),
)
_REASONING_START_TOKENS = ("<think>\n", "<think>")
_DATASET_MAP_BATCH_SIZE = 8
TEICH_SUPERVISED_SPANS_COLUMN = "teich_supervised_spans"
DEFAULT_PROVENANCE_COLUMNS = ("source", "metadata", "raw_index", "source_key")
_SPAN_KIND_REASONING = "reasoning"
_SPAN_KIND_FINAL_ANSWER = "final_answer"
_SPAN_KIND_TOOL_CALL = "tool_call"
_SPAN_KIND_TOOL_RESPONSE = "tool_response"
_SPAN_KIND_USER = "user"
_SPAN_KIND_SYSTEM = "system"
_SPAN_KIND_DEVELOPER = "developer"
_MARKER_PREFERRED_DICT_KEYS = ("text", "content", "value", "arguments", "name")
_MARKER_STRUCTURAL_DICT_KEYS = {"type"}
_TEICH_LABEL_PAD_TOKEN_ID = -100
_TEICH_LABEL_PADDING_COLLATOR_NAMES = {
    "DataCollatorForLanguageModeling",
    "DataCollatorWithPadding",
}
_OVERSIZED_POLICIES = {"drop", "trim_followups", "error"}
_OVERSIZED_POLICY_KEEP = "keep"


@dataclass(slots=True)
class RowContextFit:
    fits: bool
    token_length: int
    max_length: int
    row_id: Any = None


@dataclass(slots=True)
class PrepareReport:
    total_rows: int = 0
    formatted_rows: int = 0
    returned_rows: int = 0
    max_token_length: int | None = None
    max_prepared_token_length: int | None = None
    token_lengths: list[dict[str, Any]] = field(default_factory=list)
    kept_rows: list[dict[str, Any]] = field(default_factory=list)
    dropped_rows: list[dict[str, Any]] = field(default_factory=list)
    oversized_rows: list[dict[str, Any]] = field(default_factory=list)
    trimmed_rows: list[dict[str, Any]] = field(default_factory=list)

    def record_token_length(self, row_info: dict[str, Any], token_length: int) -> None:
        entry = {**row_info, "token_length": token_length}
        self.token_lengths.append(entry)
        self.max_token_length = token_length if self.max_token_length is None else max(self.max_token_length, token_length)

    def record_kept_row(self, row_info: dict[str, Any], token_length: int | None) -> None:
        entry = dict(row_info)
        if token_length is not None:
            entry["token_length"] = token_length
            self.max_prepared_token_length = (
                token_length
                if self.max_prepared_token_length is None
                else max(self.max_prepared_token_length, token_length)
            )
        self.kept_rows.append(entry)
        self.formatted_rows += 1

    def record_dropped_row(self, row_info: dict[str, Any], reason: str, token_length: int | None = None) -> None:
        entry = {**row_info, "reason": reason}
        if token_length is not None:
            entry["token_length"] = token_length
        self.dropped_rows.append(entry)

    def record_oversized_row(
        self,
        row_info: dict[str, Any],
        *,
        token_length: int,
        max_length: int,
        policy: str,
        final_token_length: int | None = None,
    ) -> None:
        entry = {**row_info, "token_length": token_length, "max_length": max_length, "policy": policy}
        if final_token_length is not None:
            entry["final_token_length"] = final_token_length
        self.oversized_rows.append(entry)

    def record_trimmed_row(
        self,
        row_info: dict[str, Any],
        *,
        initial_token_length: int,
        final_token_length: int,
        max_length: int,
    ) -> None:
        self.trimmed_rows.append(
            {
                **row_info,
                "initial_token_length": initial_token_length,
                "final_token_length": final_token_length,
                "max_length": max_length,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "formatted_rows": self.formatted_rows,
            "returned_rows": self.returned_rows,
            "max_token_length": self.max_token_length,
            "max_prepared_token_length": self.max_prepared_token_length,
            "token_lengths": self.token_lengths,
            "kept_rows": self.kept_rows,
            "dropped_rows": self.dropped_rows,
            "oversized_rows": self.oversized_rows,
            "trimmed_rows": self.trimmed_rows,
        }


@dataclass(slots=True)
class _RenderedRow:
    text: str
    supervised_spans: list[dict[str, Any]]
    tokenized: tuple[list[int], list[int]] | None
    token_length: int | None


def _resolve_chat_template_renderer(tokenizer: Any, text_tokenizer: Any) -> Any:
    if hasattr(text_tokenizer, "apply_chat_template"):
        return text_tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    raise TypeError("tokenizer must define apply_chat_template directly or via tokenizer.apply_chat_template")


def _resolve_text_tokenizer(tokenizer: Any) -> Any:
    text_tokenizer = getattr(tokenizer, "tokenizer", None)
    if text_tokenizer is None:
        text_tokenizer = tokenizer
    if not callable(text_tokenizer):
        raise TypeError("tokenizer must be callable or expose a callable .tokenizer for text tokenization")
    if not hasattr(text_tokenizer, "decode"):
        raise TypeError("tokenizer must expose decode() directly or via tokenizer.decode()")
    return text_tokenizer


def _validate_chat_template_kwargs(chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = dict(chat_template_kwargs or {})
    reserved = {"add_generation_prompt", "tokenize", "tools"}
    overlap = reserved.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"chat_template_kwargs cannot override reserved apply_chat_template arguments: {names}")
    return kwargs


def _as_text_content_parts(content: Any) -> Any:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [{"type": "text", "text": item} if isinstance(item, str) else item for item in content]
    return content


def _messages_with_text_content_parts(messages: list[dict[str, Any]], *, convert_tool_roles: bool = False) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    changed = False
    for message in messages:
        normalized_message = dict(message)
        if convert_tool_roles and normalized_message.get("role") == "tool":
            normalized_message["role"] = "user"
            name = normalized_message.get("name") or normalized_message.get("tool_call_id") or "tool"
            content = normalized_message.get("content") or ""
            normalized_message["content"] = f"<tool_response name={name!r}>{content}</tool_response>"
            changed = True
        if "content" in normalized_message:
            original_content = normalized_message.get("content")
            normalized_content = _as_text_content_parts(original_content)
            if normalized_content is not original_content:
                changed = True
            normalized_message["content"] = normalized_content
        normalized_messages.append(normalized_message)
    return normalized_messages if changed else messages


def _apply_chat_template_with_gemma_fallback(
    renderer: Any,
    messages: list[dict[str, Any]],
    render_kwargs: dict[str, Any],
) -> Any:
    candidates: list[tuple[list[dict[str, Any]], dict[str, Any]]] = [(messages, render_kwargs)]
    normalized_messages = _messages_with_text_content_parts(messages)
    if normalized_messages is not messages:
        candidates.append((normalized_messages, render_kwargs))
    if "tools" in render_kwargs:
        kwargs_without_tools = dict(render_kwargs)
        kwargs_without_tools.pop("tools", None)
        candidates.append((messages, kwargs_without_tools))
        if normalized_messages is not messages:
            candidates.append((normalized_messages, kwargs_without_tools))
        tool_role_messages = _messages_with_text_content_parts(messages, convert_tool_roles=True)
        if tool_role_messages is not messages:
            candidates.append((tool_role_messages, kwargs_without_tools))

    first_exc: Exception | None = None
    for candidate_messages, candidate_kwargs in candidates:
        try:
            return renderer.apply_chat_template(candidate_messages, **candidate_kwargs)
        except Exception as exc:
            if first_exc is None:
                first_exc = exc
    if first_exc is not None:
        raise first_exc
    return renderer.apply_chat_template(messages, **render_kwargs)


def _render_chat(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> str:
    render_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        **chat_template_kwargs,
    }
    if tools:
        render_kwargs["tools"] = tools
    rendered = _apply_chat_template_with_gemma_fallback(renderer, messages, render_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string")
    return rendered


def _normalize_tool_call_arguments_for_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] | None = None
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call_index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if arguments is None:
                parsed_arguments: Any = {}
            elif isinstance(arguments, str):
                stripped = arguments.strip()
                if not stripped:
                    parsed_arguments = {}
                elif stripped[0] in "{[":
                    try:
                        parsed_arguments = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            else:
                continue
            if normalized_messages is None:
                normalized_messages = deepcopy(messages)
            normalized_function = normalized_messages[message_index]["tool_calls"][tool_call_index]["function"]
            normalized_function["arguments"] = parsed_arguments
    return normalized_messages if normalized_messages is not None else messages


def _render_chat_with_generation_prompt(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> str:
    render_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        **chat_template_kwargs,
    }
    if tools:
        render_kwargs["tools"] = tools
    rendered = _apply_chat_template_with_gemma_fallback(renderer, messages, render_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string")
    return rendered


def _tokenized_length(text_tokenizer: Any, text: str) -> int:
    try:
        encoded = text_tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    except TypeError:
        encoded = text_tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape") and len(input_ids.shape) > 0:
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _tokenize_text_with_offsets(text_tokenizer: Any, text: str) -> tuple[list[int], list[int], list[tuple[int, int]]] | None:
    try:
        encoded = text_tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if input_ids is None or offsets is None:
        return None
    attention_mask = encoded.get("attention_mask")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    normalized_offsets = [tuple(offset) for offset in offsets]
    return list(input_ids), list(attention_mask), normalized_offsets


def _tokenize_trainer_text_with_offsets(
    text_tokenizer: Any,
    text: str,
) -> tuple[list[int], list[int], list[tuple[int, int]]] | None:
    call_variants = (
        ((), {"text": text, "add_special_tokens": False, "return_attention_mask": True, "return_offsets_mapping": True}),
        ((text,), {"add_special_tokens": False, "return_attention_mask": True, "return_offsets_mapping": True}),
    )
    encoded = None
    for args, kwargs in call_variants:
        try:
            encoded = text_tokenizer(*args, **kwargs)
            break
        except TypeError:
            continue
        except (ValueError, NotImplementedError):
            return None
    if encoded is None:
        return None
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if input_ids is None or offsets is None:
        return None
    attention_mask = encoded.get("attention_mask")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    normalized_offsets = [tuple(offset) for offset in offsets]
    return list(input_ids), list(attention_mask), normalized_offsets


def _tokenize_trainer_text(text_tokenizer: Any, text: str) -> tuple[list[int], list[int]] | None:
    call_variants = (
        ((), {"text": text, "add_special_tokens": False, "return_attention_mask": True}),
        ((text,), {"add_special_tokens": False, "return_attention_mask": True}),
    )
    encoded = None
    for args, kwargs in call_variants:
        try:
            encoded = text_tokenizer(*args, **kwargs)
            break
        except TypeError:
            continue
    if encoded is None:
        return None
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return None
    attention_mask = encoded.get("attention_mask")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    return list(input_ids), list(attention_mask)


def _is_assistant_message(message: dict[str, Any]) -> bool:
    return isinstance(message, dict) and message.get("role") in {"assistant", "model"}


def _extract_token_sequence(values: Any) -> list[int] | None:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def _subtract_spans(
    spans: list[tuple[int, int]],
    excluded_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not spans or not excluded_spans:
        return spans
    remaining: list[tuple[int, int]] = []
    excluded_index = 0
    ordered_exclusions = sorted(excluded_spans)
    for start, end in sorted(spans):
        cursor = start
        while excluded_index < len(ordered_exclusions) and ordered_exclusions[excluded_index][1] <= cursor:
            excluded_index += 1
        scan_index = excluded_index
        while scan_index < len(ordered_exclusions):
            excluded_start, excluded_end = ordered_exclusions[scan_index]
            if excluded_start >= end:
                break
            if cursor < excluded_start:
                remaining.append((cursor, min(end, excluded_start)))
            cursor = max(cursor, excluded_end)
            if cursor >= end:
                break
            scan_index += 1
        if cursor < end:
            remaining.append((cursor, end))
    return _merge_spans([(start, end) for start, end in remaining if start < end])


def _find_delimited_spans(text: str, start_token: str, end_token: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(start_token, cursor)
        if start < 0:
            break
        end = text.find(end_token, start + len(start_token))
        if end < 0:
            break
        spans.append((start, end + len(end_token)))
        cursor = end + len(end_token)
    return spans


def _gemma_like_supervised_spans(text: str) -> list[tuple[int, int]]:
    turn_matches = list(_GEMMA_TURN_START_PATTERN.finditer(text))
    if not turn_matches:
        return []
    tool_response_spans = _tool_response_spans(text)
    supervised_spans: list[tuple[int, int]] = []
    for index, match in enumerate(turn_matches):
        if match.group(1) != "model":
            continue
        block_start = match.start()
        block_end = turn_matches[index + 1].start() if index + 1 < len(turn_matches) else len(text)
        turn_end = text.find(_GEMMA_TURN_END, block_start, block_end)
        if turn_end >= 0:
            block_end = turn_end
        supervised_start = block_start + len(_GEMMA_ASSISTANT_TURN_PREFIX)
        if supervised_start < block_end:
            supervised_spans.append((supervised_start, block_end))
    return _subtract_spans(supervised_spans, tool_response_spans)


def _tool_response_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start_token, end_token in _TOOL_RESPONSE_DELIMITERS:
        spans.extend(_find_delimited_spans(text, start_token, end_token))
    cursor = 0
    while True:
        match = _TOOL_RESPONSE_START_TAG_PATTERN.search(text, cursor)
        if match is None:
            break
        end_candidates: list[tuple[int, str]] = []
        for end_token in _TOOL_RESPONSE_END_TOKENS:
            end_start = text.find(end_token, match.end())
            if end_start >= 0:
                end_candidates.append((end_start, end_token))
        if not end_candidates:
            cursor = match.end()
            continue
        end_start, end_token = min(end_candidates, key=lambda item: item[0])
        spans.append((match.start(), end_start + len(end_token)))
        cursor = end_start + len(end_token)
    return _merge_spans(spans)


def _expand_span_to_containing_span(
    span: tuple[int, int],
    candidate_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    start, end = span
    containing_spans = [
        (candidate_start, candidate_end)
        for candidate_start, candidate_end in candidate_spans
        if candidate_start <= start and end <= candidate_end
    ]
    if not containing_spans:
        return span
    return min(containing_spans, key=lambda item: item[1] - item[0])


def _extend_span_to_following_assistant_end(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    candidates: list[tuple[int, str]] = []
    for end_token in _ASSISTANT_BLOCK_END_TOKENS:
        end_start = text.find(end_token, end)
        if end_start >= 0:
            candidates.append((end_start, end_token))
    if not candidates:
        return span
    end_start, end_token = min(candidates, key=lambda item: item[0])
    if text[end:end_start].strip():
        return span
    block = _assistant_block_bounds(text, start, end)
    if block is None or end_start >= block[1]:
        return span
    span_end = end_start + len(end_token)
    while span_end < len(text) and text[span_end] in "\r\n":
        span_end += 1
    return start, span_end


def _marker_dict_keys(value: dict[Any, Any]) -> list[Any]:
    preferred_keys = [key for key in _MARKER_PREFERRED_DICT_KEYS if key in value]
    fallback_keys = [key for key in value if key not in preferred_keys and key not in _MARKER_STRUCTURAL_DICT_KEYS]
    structural_keys = [key for key in value if key not in preferred_keys and key not in fallback_keys]
    return preferred_keys + fallback_keys + structural_keys


def _marker_append_dict_keys(value: dict[Any, Any]) -> list[Any]:
    preferred_keys = [key for key in _MARKER_PREFERRED_DICT_KEYS if key in value]
    fallback_keys = [key for key in value if key not in preferred_keys and key not in _MARKER_STRUCTURAL_DICT_KEYS]
    structural_keys = [key for key in value if key not in preferred_keys and key not in fallback_keys]
    if preferred_keys:
        return preferred_keys + list(reversed(fallback_keys)) + structural_keys
    return list(reversed(fallback_keys)) + structural_keys


def _prepend_marker(value: Any, marker: str) -> tuple[Any, bool]:
    if isinstance(value, str) and value:
        return marker + value, True
    if isinstance(value, list):
        updated = list(value)
        for index, item in enumerate(updated):
            new_item, changed = _prepend_marker(item, marker)
            if changed:
                updated[index] = new_item
                return updated, True
        return value, False
    if isinstance(value, dict):
        updated = dict(value)
        for key in _marker_dict_keys(updated):
            item = updated[key]
            new_item, changed = _prepend_marker(item, marker)
            if changed:
                updated[key] = new_item
                return updated, True
        return value, False
    return value, False


def _append_marker(value: Any, marker: str) -> tuple[Any, bool]:
    if isinstance(value, str) and value:
        return value + marker, True
    if isinstance(value, list):
        updated = list(value)
        for index in range(len(updated) - 1, -1, -1):
            new_item, changed = _append_marker(updated[index], marker)
            if changed:
                updated[index] = new_item
                return updated, True
        return value, False
    if isinstance(value, dict):
        updated = dict(value)
        for key in _marker_append_dict_keys(updated):
            new_item, changed = _append_marker(updated[key], marker)
            if changed:
                updated[key] = new_item
                return updated, True
        return value, False
    return value, False


def _wrap_with_markers(value: Any, start_marker: str, end_marker: str) -> tuple[Any, bool]:
    if isinstance(value, str) and value:
        return start_marker + value + end_marker, True
    updated_value, changed_start = _prepend_marker(value, start_marker)
    if not changed_start:
        return value, False
    updated_value, changed_end = _append_marker(updated_value, end_marker)
    if not changed_end:
        return value, False
    return updated_value, True


def _mark_supervised_messages(
    messages: list[dict[str, Any]],
    *,
    include_context_spans: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    marked_messages = deepcopy(messages)
    markers: list[dict[str, str]] = []
    marker_index = 0

    def mark_value(value: Any, *, kind: str, role: str) -> tuple[Any, bool]:
        nonlocal marker_index
        start_marker = f"\ue000AGD{marker_index}S\ue001"
        end_marker = f"\ue000AGD{marker_index}E\ue001"
        updated_value, changed = _wrap_with_markers(value, start_marker, end_marker)
        if changed:
            markers.append(
                {
                    "start_marker": start_marker,
                    "end_marker": end_marker,
                    "kind": kind,
                    "role": role,
                }
            )
            marker_index += 1
        return updated_value, changed

    for message in marked_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if _is_assistant_message(message):
            reasoning = message.get("reasoning_content")
            updated_reasoning, changed = mark_value(reasoning, kind=_SPAN_KIND_REASONING, role=role)
            if changed:
                message["reasoning_content"] = updated_reasoning
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                updated_name, changed = mark_value(name, kind=_SPAN_KIND_TOOL_CALL, role=role)
                if changed:
                    function["name"] = updated_name
                arguments = function.get("arguments")
                updated_arguments, changed = mark_value(arguments, kind=_SPAN_KIND_TOOL_CALL, role=role)
                if changed:
                    function["arguments"] = updated_arguments
            content = message.get("content")
            updated_content, changed = mark_value(content, kind=_SPAN_KIND_FINAL_ANSWER, role=role)
            if changed:
                message["content"] = updated_content
            continue
        if role == "tool":
            kind = _SPAN_KIND_TOOL_RESPONSE
        elif role == "developer":
            kind = _SPAN_KIND_DEVELOPER
        elif role == "system":
            kind = _SPAN_KIND_SYSTEM
        elif role == "user":
            kind = _SPAN_KIND_USER
        else:
            continue
        if not include_context_spans:
            continue
        content = message.get("content")
        updated_content, changed = mark_value(content, kind=kind, role=role)
        if changed:
            message["content"] = updated_content
    return marked_messages, markers


def _marker_text_variants(marker: str) -> tuple[str, ...]:
    escaped = json.dumps(marker)[1:-1]
    return (marker, escaped) if escaped != marker else (marker,)


def _strip_markers_and_collect_spans(text: str, markers: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]] | None:
    if not markers:
        return text, []
    marker_lookup: dict[str, tuple[str, int]] = {}
    pattern_parts: list[str] = []
    for index, marker in enumerate(markers):
        start_marker = marker["start_marker"]
        end_marker = marker["end_marker"]
        for marker_variant in _marker_text_variants(start_marker):
            marker_lookup[marker_variant] = ("start", index)
            pattern_parts.append(re.escape(marker_variant))
        for marker_variant in _marker_text_variants(end_marker):
            marker_lookup[marker_variant] = ("end", index)
            pattern_parts.append(re.escape(marker_variant))
    pattern = re.compile("|".join(pattern_parts))
    cleaned_parts: list[str] = []
    active_starts: dict[int, list[int]] = {}
    spans: list[dict[str, Any]] = []
    cursor = 0
    cleaned_length = 0
    for match in pattern.finditer(text):
        chunk = text[cursor:match.start()]
        if chunk:
            cleaned_parts.append(chunk)
            cleaned_length += len(chunk)
        marker = match.group(0)
        kind, index = marker_lookup[marker]
        if kind == "start":
            active_starts.setdefault(index, []).append(cleaned_length)
        else:
            starts = active_starts.get(index)
            if not starts:
                cursor = match.end()
                continue
            start = starts.pop()
            if not starts:
                active_starts.pop(index, None)
            if start < cleaned_length:
                marker = markers[index]
                spans.append(
                    {
                        "start": start,
                        "end": cleaned_length,
                        "source_start": start,
                        "source_end": cleaned_length,
                        "kind": marker["kind"],
                        "role": marker["role"],
                    }
                )
        cursor = match.end()
    tail = text[cursor:]
    if tail:
        cleaned_parts.append(tail)
    cleaned_text = "".join(cleaned_parts)
    return cleaned_text, spans


def _range_touches_span_boundary(start: int, end: int, spans: list[dict[str, Any]]) -> bool:
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]
        if start == end and (start == span_start or start == span_end):
            return True
        if span_start <= start and end <= span_end:
            return True
    return False


def _shift_span_for_whitespace_edit(
    span: dict[str, Any],
    start: int,
    end: int,
    replacement_length: int,
) -> dict[str, Any] | None:
    removed_length = end - start
    delta = replacement_length - removed_length
    updated = dict(span)

    def shift_range(range_start: int, range_end: int) -> tuple[int, int] | None:
        if start == end and range_start <= start <= range_end:
            return range_start, range_end + replacement_length
        if end <= range_start:
            return range_start + delta, range_end + delta
        if start >= range_end:
            if start == range_end and delta > 0:
                return range_start, range_end + delta
            return range_start, range_end
        overlap_start = max(start, range_start)
        overlap_end = min(end, range_end)
        overlap = max(0, overlap_end - overlap_start)
        if start < range_start:
            range_start = start
        range_end += replacement_length - overlap
        if range_start >= range_end:
            return None
        return range_start, range_end

    span_start = updated["start"]
    span_end = updated["end"]
    shifted = shift_range(span_start, span_end)
    if shifted is None:
        return None
    updated["start"], updated["end"] = shifted
    source_start = updated.get("source_start")
    source_end = updated.get("source_end")
    if isinstance(source_start, int) and isinstance(source_end, int):
        shifted_source = shift_range(source_start, source_end)
        if shifted_source is None:
            updated.pop("source_start", None)
            updated.pop("source_end", None)
        else:
            updated["source_start"], updated["source_end"] = shifted_source
    return updated


def _reconcile_marker_boundary_whitespace(
    text: str,
    spans: list[dict[str, Any]],
    target_text: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    if text == target_text:
        return text, spans
    edits: list[tuple[int, int, str]] = []

    text_index = 0
    target_index = 0
    while text_index < len(text) or target_index < len(target_text):
        text_space_start = text_index
        while text_index < len(text) and text[text_index].isspace():
            text_index += 1
        target_space_start = target_index
        while target_index < len(target_text) and target_text[target_index].isspace():
            target_index += 1

        removed = text[text_space_start:text_index]
        inserted = target_text[target_space_start:target_index]
        if removed != inserted:
            if not _range_touches_span_boundary(text_space_start, text_index, spans):
                return None
            edits.append((text_space_start, text_index, inserted))

        while (
            text_index < len(text)
            and target_index < len(target_text)
            and not text[text_index].isspace()
            and not target_text[target_index].isspace()
        ):
            if text[text_index] != target_text[target_index]:
                return None
            text_index += 1
            target_index += 1

        if text_index >= len(text) and target_index < len(target_text) and not target_text[target_index].isspace():
            return None
        if target_index >= len(target_text) and text_index < len(text) and not text[text_index].isspace():
            return None
    if not edits:
        return None
    adjusted_text = text
    adjusted_spans = [dict(span) for span in spans]
    for start, end, inserted in reversed(edits):
        adjusted_text = adjusted_text[:start] + inserted + adjusted_text[end:]
        shifted_spans: list[dict[str, Any]] = []
        for span in adjusted_spans:
            shifted = _shift_span_for_whitespace_edit(span, start, end, len(inserted))
            if shifted is not None:
                shifted_spans.append(shifted)
        adjusted_spans = shifted_spans
    if adjusted_text != target_text:
        return None
    return adjusted_text, adjusted_spans


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered_spans = sorted(spans)
    merged_spans: list[tuple[int, int]] = [ordered_spans[0]]
    for start, end in ordered_spans[1:]:
        last_start, last_end = merged_spans[-1]
        if start <= last_end:
            merged_spans[-1] = (last_start, max(last_end, end))
        else:
            merged_spans.append((start, end))
    return merged_spans


def _reasoning_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _REASONING_BLOCK_PATTERNS:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return _merge_spans(spans)


def _tool_call_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start_token, end_token in (("<|tool_call>", "<tool_call|>"), ("<tool_call>", "</tool_call>")):
        spans.extend(_find_delimited_spans(text, start_token, end_token))
    return _merge_spans(spans)


def _expand_span_to_containing_delimiters(
    text: str,
    span: tuple[int, int],
    delimiters: tuple[tuple[str, str], ...],
) -> tuple[int, int]:
    start, end = span
    best: tuple[int, int] | None = None
    for start_token, end_token in delimiters:
        block_start = text.rfind(start_token, 0, start + 1)
        if block_start < 0:
            continue
        end_start = text.find(end_token, end)
        if end_start < 0:
            continue
        block_end = end_start + len(end_token)
        if block_start <= start and end <= block_end:
            candidate = (block_start, block_end)
            if best is None or candidate[1] - candidate[0] < best[1] - best[0]:
                best = candidate
    return best if best is not None else span


def _assistant_prompt_probe_contexts(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    contexts: list[str] = []
    for index, message in enumerate(messages):
        if not _is_assistant_message(message) or index == 0:
            continue
        previous_role = messages[index - 1].get("role") if isinstance(messages[index - 1], dict) else None
        if previous_role == "tool" and "after_tool" not in contexts:
            contexts.append("after_tool")
        elif previous_role == "user" and "after_user" not in contexts:
            contexts.append("after_user")
    if not contexts and any(_is_assistant_message(message) for message in messages):
        contexts.append("after_user")
    return tuple(contexts)


def _build_assistant_prompt_probe_messages(context: str) -> list[dict[str, Any]]:
    if context == "after_tool":
        return [
            {"role": "user", "content": "__AGD_USER__"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "__AGD_REASON__",
                "tool_calls": [
                    {
                        "id": "agd_call_1",
                        "type": "function",
                        "function": {"name": "agd_tool", "arguments": {"command": "__AGD_COMMAND__"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "agd_call_1",
                "name": "agd_tool",
                "content": "__AGD_TOOL_RESPONSE__",
            },
        ]
    return [{"role": "user", "content": "__AGD_USER__"}]


def _serialize_tools_for_cache(tools: list[dict[str, Any]]) -> str:
    try:
        return json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        return repr(tools)


def _infer_assistant_prompt_prefixes(
    renderer: Any,
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    probe_contexts: tuple[str, ...],
) -> tuple[str, ...]:
    prefixes: set[str] = set()
    for context in probe_contexts:
        probe_messages = _build_assistant_prompt_probe_messages(context)
        try:
            base_render = _render_chat(renderer, probe_messages, tools, chat_template_kwargs)
            prompt_render = _render_chat_with_generation_prompt(renderer, probe_messages, tools, chat_template_kwargs)
        except Exception:
            continue
        if not prompt_render.startswith(base_render):
            continue
        prompt_prefix = prompt_render[len(base_render) :]
        if prompt_prefix:
            prefixes.add(prompt_prefix)
    return tuple(sorted(prefixes, key=len, reverse=True))


def _resolve_assistant_prompt_prefixes(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    cache: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    probe_contexts = _assistant_prompt_probe_contexts(messages)
    if not probe_contexts:
        return ()
    cache_key = f"{_serialize_tools_for_cache(tools)}::{','.join(probe_contexts)}"
    prefixes = cache.get(cache_key)
    if prefixes is None:
        prefixes = _infer_assistant_prompt_prefixes(renderer, tools, chat_template_kwargs, probe_contexts)
        cache[cache_key] = prefixes
    return prefixes


def _assistant_block_bounds(text: str, start: int, end: int) -> tuple[int, int] | None:
    block_start = -1
    for token in _ASSISTANT_BLOCK_START_TOKENS:
        token_start = text.rfind(token, 0, start)
        if token_start > block_start:
            block_start = token_start
    if block_start < 0:
        return None
    block_end = -1
    for token in _ASSISTANT_BLOCK_END_TOKENS:
        token_end_start = text.find(token, end)
        if token_end_start >= 0 and (block_end < 0 or token_end_start < block_end):
            block_end = token_end_start + len(token)
    if block_end < 0:
        return None
    while block_end < len(text) and text[block_end] in "\r\n":
        block_end += 1
    return block_start, block_end


def _expand_supervised_spans(
    text: str,
    supervised_spans: list[tuple[int, int]],
    assistant_prompt_prefixes: tuple[str, ...],
    train_on_reasoning: bool,
) -> list[tuple[int, int]]:
    expanded_spans: list[tuple[int, int]] = []
    for start, end in supervised_spans:
        assistant_block = _assistant_block_bounds(text, start, end)
        if assistant_block is None:
            expanded_spans.append((start, end))
            continue
        block_start, block_end = assistant_block
        if not assistant_prompt_prefixes:
            expanded_spans.append((block_start, block_end))
            continue
        block_text = text[block_start:block_end]
        matched_prefix = next((prefix for prefix in assistant_prompt_prefixes if block_text.startswith(prefix)), None)
        fallback_prefix = next((prefix for prefix in _ASSISTANT_BLOCK_START_TOKENS if block_text.startswith(prefix)), None)
        if matched_prefix is not None:
            supervised_prefix_length = len(matched_prefix)
            for reasoning_start_token in _REASONING_START_TOKENS:
                if matched_prefix.endswith(reasoning_start_token):
                    supervised_prefix_length -= len(reasoning_start_token)
                    break
            expanded_spans.append((block_start + supervised_prefix_length, block_end))
            continue
        if fallback_prefix is not None:
            expanded_spans.append((block_start + len(fallback_prefix), block_end))
            continue
        expanded_spans.append((start, end))
    return _merge_spans(expanded_spans)


def _expand_typed_spans(
    text: str,
    spans: list[dict[str, Any]],
    assistant_prompt_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    expanded_spans: list[dict[str, Any]] = []
    for span in spans:
        start = span["start"]
        end = span["end"]
        kind = span.get("kind")
        expanded_start, expanded_end = start, end
        if kind in {_SPAN_KIND_REASONING, _SPAN_KIND_FINAL_ANSWER}:
            expanded = _expand_supervised_spans(
                text,
                [(start, end)],
                assistant_prompt_prefixes,
                train_on_reasoning=True,
            )
            if expanded:
                expanded_start, expanded_end = expanded[0]
            if (expanded_start, expanded_end) == (start, end):
                if kind == _SPAN_KIND_REASONING:
                    expanded_start, expanded_end = _expand_span_to_containing_span(
                        (start, end),
                        _reasoning_spans(text),
                    )
        elif kind == _SPAN_KIND_TOOL_CALL:
            expanded_start, expanded_end = _expand_span_to_containing_span(
                (start, end),
                _tool_call_spans(text),
            )
            expanded_start, expanded_end = _extend_span_to_following_assistant_end(
                text,
                (expanded_start, expanded_end),
            )
        elif kind == _SPAN_KIND_TOOL_RESPONSE:
            expanded_start, expanded_end = _expand_span_to_containing_span((start, end), _tool_response_spans(text))
        updated = dict(span)
        updated["start"] = expanded_start
        updated["end"] = expanded_end
        updated.setdefault("source_start", start)
        updated.setdefault("source_end", end)
        if expanded_start < expanded_end:
            expanded_spans.append(updated)
    return expanded_spans


def _span_kind_enabled(
    kind: str | None,
    *,
    train_on_reasoning: bool,
    train_on_final_answers: bool,
    train_on_tools: bool,
    train_on_user: bool,
    train_on_system: bool,
    train_on_developer: bool,
    train_on_tool_responses: bool,
) -> bool:
    if kind in (None, ""):
        return True
    return {
        _SPAN_KIND_REASONING: train_on_reasoning,
        _SPAN_KIND_FINAL_ANSWER: train_on_final_answers,
        _SPAN_KIND_TOOL_CALL: train_on_tools,
        _SPAN_KIND_USER: train_on_user,
        _SPAN_KIND_SYSTEM: train_on_system,
        _SPAN_KIND_DEVELOPER: train_on_developer,
        _SPAN_KIND_TOOL_RESPONSE: train_on_tool_responses,
    }.get(kind, True)


def _source_spans_for_kind(spans: list[dict[str, Any]], kind: str) -> list[tuple[int, int]]:
    source_spans: list[tuple[int, int]] = []
    for span in spans:
        if span.get("kind") != kind:
            continue
        start = span.get("source_start", span.get("start"))
        end = span.get("source_end", span.get("end"))
        if isinstance(start, int) and isinstance(end, int) and start < end:
            source_spans.append((start, end))
    return _merge_spans(source_spans)


def _select_supervised_spans(
    text: str,
    spans: list[dict[str, Any]],
    *,
    train_on_reasoning: bool,
    train_on_final_answers: bool,
    train_on_tools: bool,
    train_on_user: bool,
    train_on_system: bool,
    train_on_developer: bool,
    train_on_tool_responses: bool,
) -> list[tuple[int, int]]:
    selected = _merge_spans(
        [
            (span["start"], span["end"])
            for span in spans
            if _span_kind_enabled(
                span.get("kind"),
                train_on_reasoning=train_on_reasoning,
                train_on_final_answers=train_on_final_answers,
                train_on_tools=train_on_tools,
                train_on_user=train_on_user,
                train_on_system=train_on_system,
                train_on_developer=train_on_developer,
                train_on_tool_responses=train_on_tool_responses,
            )
        ]
    )
    if not selected:
        return []
    if not train_on_reasoning:
        selected = _subtract_spans(selected, _reasoning_spans(text))
    if not train_on_final_answers:
        selected = _subtract_spans(selected, _source_spans_for_kind(spans, _SPAN_KIND_FINAL_ANSWER))
    if not train_on_tools:
        selected = _subtract_spans(selected, _tool_call_spans(text))
        selected = _subtract_spans(selected, _source_spans_for_kind(spans, _SPAN_KIND_TOOL_CALL))
    if not train_on_tool_responses:
        selected = _subtract_spans(selected, _tool_response_spans(text))
    if not train_on_user:
        selected = _subtract_spans(selected, _source_spans_for_kind(spans, _SPAN_KIND_USER))
    if not train_on_system:
        selected = _subtract_spans(selected, _source_spans_for_kind(spans, _SPAN_KIND_SYSTEM))
    if not train_on_developer:
        selected = _subtract_spans(selected, _source_spans_for_kind(spans, _SPAN_KIND_DEVELOPER))
    return _merge_spans(selected)


def _labels_from_offsets(
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    supervised_spans: list[tuple[int, int]],
) -> list[int]:
    labels: list[int] = []
    span_index = 0
    for token_id, (start, end) in zip(input_ids, offsets):
        if end <= start:
            labels.append(-100)
            continue
        while span_index < len(supervised_spans) and supervised_spans[span_index][1] <= start:
            span_index += 1
        is_supervised = (
            span_index < len(supervised_spans)
            and supervised_spans[span_index][0] <= start
            and end <= supervised_spans[span_index][1]
        )
        labels.append(token_id if is_supervised else -100)
    return labels


def _align_labels_to_input_ids(
    input_ids: list[int],
    full_input_ids: list[int],
    full_labels: list[int],
) -> list[int] | None:
    if input_ids == full_input_ids:
        return full_labels
    if len(input_ids) <= len(full_input_ids) and input_ids == full_input_ids[: len(input_ids)]:
        return full_labels[: len(input_ids)]

    if len(input_ids) >= len(full_input_ids):
        for offset in range(len(input_ids) - len(full_input_ids) + 1):
            end = offset + len(full_input_ids)
            if input_ids[offset:end] == full_input_ids:
                labels = [-100] * len(input_ids)
                labels[offset:end] = full_labels
                return labels

    best: tuple[int, int] | None = None
    max_special_side_tokens = min(8, len(input_ids))
    for prefix_count in range(max_special_side_tokens + 1):
        remaining = len(input_ids) - prefix_count
        if remaining <= 0:
            continue
        for suffix_count in range(min(max_special_side_tokens, remaining - 1) + 1):
            end = len(input_ids) - suffix_count if suffix_count else len(input_ids)
            candidate = input_ids[prefix_count:end]
            if not candidate:
                continue
            if len(candidate) <= len(full_input_ids) and candidate == full_input_ids[: len(candidate)]:
                if best is None or len(candidate) > best[1] - best[0]:
                    best = (prefix_count, end)
    if best is None:
        return None
    start, end = best
    labels = [-100] * len(input_ids)
    labels[start:end] = full_labels[: end - start]
    return labels


def _token_text_and_offsets(text_tokenizer: Any, input_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in input_ids:
        token_text = _decode_token(text_tokenizer, token_id)
        parts.append(token_text)
        offsets.append((cursor, cursor + len(token_text)))
        cursor += len(token_text)
    return "".join(parts), offsets


def _find_next_assistant_start(text: str, cursor: int) -> tuple[int, str] | None:
    matches: list[tuple[int, str]] = []
    for start_token in _ASSISTANT_BLOCK_START_TOKENS:
        start = text.find(start_token, cursor)
        if start >= 0:
            matches.append((start, start_token))
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], -len(item[1])))


def _infer_supervised_spans_from_rendered_text(text: str, *, train_on_reasoning: bool) -> list[tuple[int, int]]:
    supervised_spans = _gemma_like_supervised_spans(text)
    if not supervised_spans:
        cursor = 0
        while True:
            match = _find_next_assistant_start(text, cursor)
            if match is None:
                break
            block_start, start_token = match
            content_start = block_start + len(start_token)
            end_candidates: list[tuple[int, str]] = []
            for end_token in _ASSISTANT_BLOCK_END_TOKENS:
                end_start = text.find(end_token, content_start)
                if end_start >= 0:
                    end_candidates.append((end_start, end_token))
            if end_candidates:
                end_start, end_token = min(end_candidates, key=lambda item: item[0])
                block_end = end_start + len(end_token)
            else:
                next_match = _find_next_assistant_start(text, content_start)
                block_end = next_match[0] if next_match is not None else len(text)
            if content_start < block_end:
                supervised_spans.append((content_start, block_end))
            cursor = max(block_end, content_start + 1)
    supervised_spans = _merge_spans(supervised_spans)
    supervised_spans = _subtract_spans(supervised_spans, _tool_response_spans(text))
    if not train_on_reasoning:
        supervised_spans = _subtract_spans(supervised_spans, _reasoning_spans(text))
    return supervised_spans


def _supervised_text_and_spans(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    strict: bool,
) -> tuple[str, list[dict[str, Any]]]:
    original_text = _render_chat(renderer, messages, tools, chat_template_kwargs)
    marked_messages, markers = _mark_supervised_messages(messages)
    marked_text = _render_chat(renderer, marked_messages, tools, chat_template_kwargs)
    stripped = _strip_markers_and_collect_spans(marked_text, markers)
    del marked_text
    if stripped is None:
        inferred_spans = _infer_supervised_spans_from_rendered_text(
            original_text,
            train_on_reasoning=True,
        )
        if inferred_spans:
            return original_text, _span_dicts(inferred_spans)
        if strict:
            raise ValueError("Unable to collect supervised spans from marker-injected chat template output.")
        return original_text, []
    formatted_text, supervised_spans = stripped
    if formatted_text != original_text:
        reconciled = _reconcile_marker_boundary_whitespace(formatted_text, supervised_spans, original_text)
        if reconciled is not None:
            formatted_text, supervised_spans = reconciled
        else:
            assistant_marked_messages, assistant_markers = _mark_supervised_messages(messages, include_context_spans=False)
            assistant_marked_text = _render_chat(renderer, assistant_marked_messages, tools, chat_template_kwargs)
            assistant_stripped = _strip_markers_and_collect_spans(assistant_marked_text, assistant_markers)
            del assistant_marked_text
            if assistant_stripped is not None:
                if assistant_stripped[0] == original_text:
                    formatted_text, supervised_spans = assistant_stripped
                else:
                    reconciled = _reconcile_marker_boundary_whitespace(
                        assistant_stripped[0],
                        assistant_stripped[1],
                        original_text,
                    )
                    if reconciled is not None:
                        formatted_text, supervised_spans = reconciled
                    elif strict:
                        raise ValueError("Marker-injected chat template output does not match the original rendered chat after marker removal.")
                    else:
                        inferred_spans = _infer_supervised_spans_from_rendered_text(original_text, train_on_reasoning=True)
                        return original_text, _span_dicts(inferred_spans)
            else:
                if strict:
                    raise ValueError("Marker-injected chat template output does not match the original rendered chat after marker removal.")
                inferred_spans = _infer_supervised_spans_from_rendered_text(original_text, train_on_reasoning=True)
                return original_text, _span_dicts(inferred_spans)
    del original_text
    if markers and not supervised_spans:
        inferred_spans = _infer_supervised_spans_from_rendered_text(
            formatted_text,
            train_on_reasoning=True,
        )
        if inferred_spans:
            return formatted_text, _span_dicts(inferred_spans)
    gemma_spans = _gemma_like_supervised_spans(formatted_text)
    if gemma_spans:
        gemma_spans = _subtract_spans(gemma_spans, _tool_call_spans(formatted_text))
        gemma_spans = _subtract_spans(gemma_spans, _tool_response_spans(formatted_text))
        gemma_spans = _subtract_spans(gemma_spans, _reasoning_spans(formatted_text))
        supervised_spans.extend(
            {
                "start": start,
                "end": end,
                "source_start": start,
                "source_end": end,
                "kind": _SPAN_KIND_FINAL_ANSWER,
                "role": "assistant",
            }
            for start, end in gemma_spans
        )
    assistant_prompt_prefixes = _resolve_assistant_prompt_prefixes(
        renderer,
        messages,
        tools,
        chat_template_kwargs,
        assistant_prompt_prefix_cache,
    )
    supervised_spans = _expand_typed_spans(
        formatted_text,
        supervised_spans,
        assistant_prompt_prefixes,
    )
    return formatted_text, supervised_spans


def _span_dicts(spans: list[tuple[int, int]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    span_dicts: list[dict[str, Any]] = []
    for item in spans:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or start >= end:
                continue
            span = {"start": start, "end": end}
            source_start = item.get("source_start", start)
            source_end = item.get("source_end", end)
            if isinstance(source_start, int) and isinstance(source_end, int) and source_start < source_end:
                span["source_start"] = source_start
                span["source_end"] = source_end
            kind = item.get("kind")
            if isinstance(kind, str) and kind:
                span["kind"] = kind
            role = item.get("role")
            if isinstance(role, str) and role:
                span["role"] = role
            span_dicts.append(span)
            continue
        start, end = item
        if start < end:
            span_dicts.append({"start": start, "end": end})
    return span_dicts


def _normalize_span_metadata(value: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for item in value or []:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            kind = item.get("kind")
            role = item.get("role")
            source_start = item.get("source_start", start)
            source_end = item.get("source_end", end)
        else:
            start, end = item
            kind = None
            role = None
            source_start = start
            source_end = end
        if isinstance(start, int) and isinstance(end, int) and start < end:
            span: dict[str, Any] = {"start": start, "end": end}
            if isinstance(source_start, int) and isinstance(source_end, int) and source_start < source_end:
                span["source_start"] = source_start
                span["source_end"] = source_end
            if isinstance(kind, str) and kind:
                span["kind"] = kind
            if isinstance(role, str) and role:
                span["role"] = role
            spans.append(span)
    return spans


def normalize_prepared_dataset_features(dataset: Dataset) -> Dataset:
    """Cast Teich prepared columns to one stable Arrow schema."""
    features = dict(dataset.features)
    if TEICH_SUPERVISED_SPANS_COLUMN in dataset.column_names:
        features[TEICH_SUPERVISED_SPANS_COLUMN] = List(Json())
    if "input_ids" in dataset.column_names:
        features["input_ids"] = List(Value("int32"))
    if "attention_mask" in dataset.column_names:
        features["attention_mask"] = List(Value("int8"))
    if "metadata" in dataset.column_names:
        features["metadata"] = Json()
    if "source" in dataset.column_names:
        features["source"] = Value("string")
    if "source_key" in dataset.column_names:
        features["source_key"] = Value("string")
    if "raw_index" in dataset.column_names:
        features["raw_index"] = Value("int64")
    normalized_features = Features({column: features[column] for column in dataset.column_names})
    if dataset.features == normalized_features:
        return dataset
    return dataset.cast(normalized_features)


def _drop_last_user_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_indexes) < 2:
        return None

    before_last_user = messages[: user_indexes[-1]]
    assistant_indexes = [
        index
        for index, message in enumerate(before_last_user)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistant_indexes:
        return None

    trimmed = before_last_user[: assistant_indexes[-1] + 1]
    if not any(message.get("role") == "user" for message in trimmed):
        return None
    return trimmed


def _normalize_span_dicts(value: Any) -> list[tuple[int, int]]:
    return _merge_spans([(span["start"], span["end"]) for span in _normalize_span_metadata(value)])


def _resolve_oversized_policy(
    oversized_policy: str | None,
    *,
    drop_oversized_examples: bool,
    trim_oversized_followups: bool,
) -> str:
    if oversized_policy is not None:
        if oversized_policy not in _OVERSIZED_POLICIES:
            choices = ", ".join(sorted(_OVERSIZED_POLICIES))
            raise ValueError(f"oversized_policy must be one of: {choices}")
        return oversized_policy
    if not drop_oversized_examples:
        return _OVERSIZED_POLICY_KEEP
    if trim_oversized_followups:
        return "trim_followups"
    return "drop"


def _resolve_preserved_columns(
    dataset: Dataset,
    preserve_columns: bool | Sequence[str] | None,
    *,
    source_key: str | None,
) -> list[str]:
    if preserve_columns is None or preserve_columns is False:
        return []
    if preserve_columns is True:
        candidates = DEFAULT_PROVENANCE_COLUMNS
    else:
        if isinstance(preserve_columns, (str, bytes, bytearray)):
            raise TypeError("preserve_columns must be True or a sequence of non-empty column names.")
        candidates = tuple(preserve_columns)
        if not all(isinstance(column, str) and column for column in candidates):
            raise TypeError("preserve_columns must be True or a sequence of non-empty column names.")
    preserved: list[str] = []
    for column in candidates:
        if column in preserved:
            continue
        if column in dataset.column_names or column == "raw_index" or (column == "source_key" and source_key is not None):
            preserved.append(column)
    return preserved


def _row_identity_from_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return value
    return None


def _row_report_info(
    batch: dict[str, list[Any]],
    batch_index: int,
    raw_index: int,
    *,
    source_key: str | None,
) -> dict[str, Any]:
    row_info: dict[str, Any] = {"raw_index": raw_index}
    for column in ("row_id", "id", "source_key"):
        values = batch.get(column)
        if values is None:
            continue
        value = _row_identity_from_value(values[batch_index])
        if value is not None:
            row_info["row_id"] = value
            break
    if "row_id" not in row_info:
        metadata_values = batch.get("metadata")
        metadata = metadata_values[batch_index] if metadata_values is not None else None
        if isinstance(metadata, dict):
            for key in ("session_id", "id", "source_file"):
                value = _row_identity_from_value(metadata.get(key))
                if value is not None:
                    row_info["row_id"] = value
                    break
    source_values = batch.get("source")
    source_value = _row_identity_from_value(source_values[batch_index]) if source_values is not None else None
    if source_value is not None:
        row_info["source"] = source_value
    source_key_values = batch.get("source_key")
    source_key_value = _row_identity_from_value(source_key_values[batch_index]) if source_key_values is not None else None
    if source_key_value is None:
        source_key_value = source_key
    if source_key_value is not None:
        row_info["source_key"] = source_key_value
    return row_info


def _append_preserved_columns(
    output_batch: dict[str, list[Any]],
    batch: dict[str, list[Any]],
    batch_index: int,
    raw_index: int,
    *,
    preserved_columns: list[str],
    source_key: str | None,
) -> None:
    for column in preserved_columns:
        if column == "raw_index" and column not in batch:
            output_batch[column].append(raw_index)
            continue
        if column == "source_key" and column not in batch:
            output_batch[column].append(source_key or "")
            continue
        output_batch[column].append(batch[column][batch_index])


def _render_training_row(
    *,
    renderer: Any,
    text_tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
    teich_masking: bool,
    tokenize: bool,
    measure_token_length: bool,
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    strict: bool,
) -> _RenderedRow | None:
    if teich_masking:
        text, supervised_spans = _supervised_text_and_spans(
            renderer,
            messages,
            tools,
            template_kwargs,
            assistant_prompt_prefix_cache,
            strict,
        )
        if not supervised_spans:
            return None
    else:
        text = _render_chat(renderer, messages, tools, template_kwargs)
        supervised_spans = []
    tokenized: tuple[list[int], list[int]] | None = None
    if tokenize:
        tokenized = _tokenize_trainer_text(text_tokenizer, text)
        if tokenized is None:
            raise ValueError("prepare_data(tokenize=True) requires a tokenizer that can tokenize text.")
    token_length = None
    if measure_token_length:
        token_length = len(tokenized[0]) if tokenized is not None else _tokenized_length(text_tokenizer, text)
    return _RenderedRow(
        text=text,
        supervised_spans=supervised_spans,
        tokenized=tokenized,
        token_length=token_length,
    )


def row_fits_context(
    row: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    *,
    messages_column: str = "messages",
    tools_column: str = "tools",
    text_column: str = "text",
    return_details: bool = False,
) -> bool | RowContextFit:
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer.")
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    row_id = row.get("row_id") or row.get("id") or row.get("source_key") or row.get("raw_index")
    if isinstance(row.get(text_column), str) and messages_column not in row:
        token_length = _tokenized_length(text_tokenizer, row[text_column])
        result = RowContextFit(
            fits=token_length <= max_length,
            token_length=token_length,
            max_length=max_length,
            row_id=row_id,
        )
        return result if return_details else result.fits
    messages = row.get(messages_column)
    if not isinstance(messages, list):
        raise TypeError(f"Row is missing a list-valued '{messages_column}' column.")
    tools = row.get(tools_column) or []
    if not isinstance(tools, list):
        raise TypeError(f"Row has a non-list '{tools_column}' column.")
    messages = _normalize_tool_call_arguments_for_template(normalize_training_messages(messages))
    renderer = _resolve_chat_template_renderer(tokenizer, text_tokenizer)
    template_kwargs = _validate_chat_template_kwargs(chat_template_kwargs)
    text = _render_chat(renderer, messages, tools, template_kwargs)
    token_length = _tokenized_length(text_tokenizer, text)
    result = RowContextFit(
        fits=token_length <= max_length,
        token_length=token_length,
        max_length=max_length,
        row_id=row_id,
    )
    return result if return_details else result.fits


def format_data(
    dataset: Dataset | Sequence[Dataset],
    tokenizer: Any,
    *,
    messages_column: str = "messages",
    tools_column: str = "tools",
    text_column: str = "text",
    chat_template_kwargs: dict[str, Any] | None = None,
    train_on_reasoning: bool | None = None,
    teich_masking: bool = True,
    max_length: int | None = None,
    oversized_policy: str | None = None,
    drop_oversized_examples: bool = True,
    trim_oversized_followups: bool = False,
    preserve_columns: bool | Sequence[str] | None = None,
    source_key: str | None = None,
    report: PrepareReport | None = None,
    validate_tools: bool = False,
    tokenize: bool = False,
    strict: bool = False,
    verbose: bool = True,
) -> Dataset:
    if isinstance(dataset, Sequence) and not isinstance(dataset, Dataset):
        datasets = list(dataset)
        if not datasets:
            raise ValueError("At least one dataset must be provided to prepare_data.")
        if len(datasets) > 1:
            formatted_datasets: list[Dataset] = []
            for item in datasets:
                if not isinstance(item, Dataset):
                    raise TypeError("prepare_data expects a Dataset or a sequence of Dataset objects.")
                formatted_datasets.append(
                    format_data(
                        item,
                        tokenizer,
                        messages_column=messages_column,
                        tools_column=tools_column,
                        text_column=text_column,
                        chat_template_kwargs=chat_template_kwargs,
                        train_on_reasoning=train_on_reasoning,
                        teich_masking=teich_masking,
                        max_length=max_length,
                        oversized_policy=oversized_policy,
                        drop_oversized_examples=drop_oversized_examples,
                        trim_oversized_followups=trim_oversized_followups,
                        preserve_columns=preserve_columns,
                        source_key=source_key,
                        report=report,
                        validate_tools=validate_tools,
                        tokenize=tokenize,
                        strict=strict,
                        verbose=verbose,
                    )
                )
            return concatenate_datasets(
                [normalize_prepared_dataset_features(formatted) for formatted in formatted_datasets]
            )
        dataset = datasets[0]
    if not isinstance(dataset, Dataset):
        raise TypeError("prepare_data expects a Dataset or a sequence of Dataset objects.")

    template_kwargs = _validate_chat_template_kwargs(chat_template_kwargs)
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    renderer = _resolve_chat_template_renderer(tokenizer, text_tokenizer)
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]] = {}
    effective_max_length = max_length if isinstance(max_length, int) and max_length > 0 else None
    effective_oversized_policy = _resolve_oversized_policy(
        oversized_policy,
        drop_oversized_examples=drop_oversized_examples,
        trim_oversized_followups=trim_oversized_followups,
    )
    measure_token_length = effective_max_length is not None or report is not None
    preserved_columns = _resolve_preserved_columns(dataset, preserve_columns, source_key=source_key)
    dropped_count = 0
    dropped_oversized_count = 0
    trimmed_oversized_count = 0

    if messages_column not in dataset.column_names:
        raise TypeError(f"Dataset is missing required '{messages_column}' column")

    output_columns = [text_column]
    if teich_masking:
        output_columns.append(TEICH_SUPERVISED_SPANS_COLUMN)
    if tokenize:
        output_columns.extend(["input_ids", "attention_mask"])
    output_columns.extend(column for column in preserved_columns if column not in output_columns)

    def _empty_output_batch() -> dict[str, list[Any]]:
        return {column_name: [] for column_name in output_columns}

    def _map_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        nonlocal dropped_count
        nonlocal dropped_oversized_count
        nonlocal trimmed_oversized_count
        batch_size = len(batch[messages_column])
        tools_batch = batch.get(tools_column)
        if tools_batch is None:
            tools_batch = [None] * batch_size
        output_batch = _empty_output_batch()

        for index in range(batch_size):
            raw_index = int(indices[index])
            row_info = _row_report_info(batch, index, raw_index, source_key=source_key)
            if report is not None:
                report.total_rows += 1
            messages = batch[messages_column][index]
            if not isinstance(messages, list):
                raise TypeError(f"Row is missing a list-valued '{messages_column}' column")
            if len(messages) == 0:
                dropped_count += 1
                if report is not None:
                    report.record_dropped_row(row_info, "empty_messages")
                continue
            messages = normalize_training_messages(messages)
            if len(messages) == 0:
                dropped_count += 1
                if report is not None:
                    report.record_dropped_row(row_info, "empty_messages")
                continue
            messages = _normalize_tool_call_arguments_for_template(messages)
            tools = tools_batch[index] or []
            if not isinstance(tools, list):
                raise TypeError(f"Row is missing a list-valued '{tools_column}' column")
            if validate_tools:
                from .tool_schema import validate_tool_calls

                validation = validate_tool_calls(
                    {"messages": messages, "tools": tools},
                    row_id=row_info.get("row_id", row_info["raw_index"]),
                )
                validation.raise_for_errors()
            rendered = _render_training_row(
                renderer=renderer,
                text_tokenizer=text_tokenizer,
                messages=messages,
                tools=tools,
                template_kwargs=template_kwargs,
                teich_masking=teich_masking,
                tokenize=tokenize,
                measure_token_length=measure_token_length,
                assistant_prompt_prefix_cache=assistant_prompt_prefix_cache,
                strict=strict,
            )
            if rendered is None:
                dropped_count += 1
                if report is not None:
                    report.record_dropped_row(row_info, "no_trainable_spans")
                continue
            initial_token_length = rendered.token_length
            if report is not None and initial_token_length is not None:
                report.record_token_length(row_info, initial_token_length)
            if effective_max_length is not None and initial_token_length is not None and initial_token_length > effective_max_length:
                if report is not None:
                    report.record_oversized_row(
                        row_info,
                        token_length=initial_token_length,
                        max_length=effective_max_length,
                        policy=effective_oversized_policy,
                    )
                if effective_oversized_policy == "error":
                    row_label = row_info.get("row_id", row_info["raw_index"])
                    raise ValueError(
                        f"Row {row_label!r} is {initial_token_length} tokens, above max_length={effective_max_length}."
                    )
            if effective_oversized_policy in {"drop", "trim_followups"} and effective_max_length is not None:
                did_trim = False
                if rendered.token_length is None:
                    dropped_oversized_count += 1
                    if report is not None:
                        report.record_dropped_row(row_info, "oversized", initial_token_length)
                    continue
                if rendered.token_length > effective_max_length:
                    while effective_oversized_policy == "trim_followups":
                        trimmed_messages = _drop_last_user_turn(messages)
                        if trimmed_messages is None:
                            break
                        messages = trimmed_messages
                        did_trim = True
                        rendered = _render_training_row(
                            renderer=renderer,
                            text_tokenizer=text_tokenizer,
                            messages=messages,
                            tools=tools,
                            template_kwargs=template_kwargs,
                            teich_masking=teich_masking,
                            tokenize=tokenize,
                            measure_token_length=measure_token_length,
                            assistant_prompt_prefix_cache=assistant_prompt_prefix_cache,
                            strict=strict,
                        )
                        if rendered is None:
                            break
                        if rendered.token_length is not None and rendered.token_length <= effective_max_length:
                            break
                    if rendered is None or rendered.token_length is None or rendered.token_length > effective_max_length:
                        dropped_oversized_count += 1
                        if report is not None:
                            report.record_dropped_row(row_info, "oversized", initial_token_length)
                        continue
                    if did_trim:
                        trimmed_oversized_count += 1
                        if report is not None:
                            report.record_trimmed_row(
                                row_info,
                                initial_token_length=initial_token_length or rendered.token_length,
                                final_token_length=rendered.token_length,
                                max_length=effective_max_length,
                            )
                            report.oversized_rows[-1]["final_token_length"] = rendered.token_length
            output_batch[text_column].append(rendered.text)
            if teich_masking:
                output_batch[TEICH_SUPERVISED_SPANS_COLUMN].append(_span_dicts(rendered.supervised_spans))
            if rendered.tokenized is not None:
                input_ids, attention_mask = rendered.tokenized
                output_batch["input_ids"].append(input_ids)
                output_batch["attention_mask"].append(attention_mask)
            _append_preserved_columns(
                output_batch,
                batch,
                index,
                raw_index,
                preserved_columns=preserved_columns,
                source_key=source_key,
            )
            if report is not None:
                report.record_kept_row(row_info, rendered.token_length)
        return output_batch

    formatted_data = dataset.map(
        _map_batch,
        batched=True,
        with_indices=True,
        batch_size=_DATASET_MAP_BATCH_SIZE,
        remove_columns=dataset.column_names,
        load_from_cache_file=False if report is not None else None,
    )
    if formatted_data.num_rows == 0 and dropped_count > 0:
        if teich_masking:
            raise ValueError("Dataset contains no rows with trainable assistant spans.")
        raise ValueError("Dataset contains no non-empty conversations.")
    if formatted_data.num_rows == 0 and drop_oversized_examples and effective_max_length is not None and dropped_oversized_count > 0:
        raise ValueError(
            f"Dataset contains no conversations that fit within context window of {effective_max_length} tokens."
        )
    if verbose and dropped_count and teich_masking:
        Console().print(f"[yellow]Dropped {dropped_count} rows without trainable assistant spans.[/yellow]")
    if verbose and dropped_oversized_count:
        Console().print(f"[yellow]Dropped {dropped_oversized_count} rows above {effective_max_length} tokens.[/yellow]")
    if verbose and trimmed_oversized_count:
        Console().print(
            f"[yellow]Trimmed follow-up turns from {trimmed_oversized_count} oversized rows to fit {effective_max_length} tokens.[/yellow]"
        )
    return formatted_data


def _mask_tokenized_row(
    row: dict[str, Any],
    text_tokenizer: Any,
    text_column: str,
    *,
    train_on_reasoning: bool,
    train_on_final_answers: bool,
    train_on_tools: bool,
    train_on_user: bool,
    train_on_system: bool,
    train_on_developer: bool,
    train_on_tool_responses: bool,
) -> dict[str, Any] | None:
    input_ids = _extract_token_sequence(row.get("input_ids"))
    if input_ids is None:
        raise TypeError("Trainer dataset row is missing tokenized 'input_ids'.")
    text = row.get(text_column)
    span_metadata = _normalize_span_metadata(row.get(TEICH_SUPERVISED_SPANS_COLUMN))
    if isinstance(text, str) and span_metadata:
        supervised_spans = _select_supervised_spans(
            text,
            span_metadata,
            train_on_reasoning=train_on_reasoning,
            train_on_final_answers=train_on_final_answers,
            train_on_tools=train_on_tools,
            train_on_user=train_on_user,
            train_on_system=train_on_system,
            train_on_developer=train_on_developer,
            train_on_tool_responses=train_on_tool_responses,
        )
        if not supervised_spans:
            return None
        encoded = _tokenize_trainer_text_with_offsets(text_tokenizer, text)
        if encoded is None:
            decoded_text, offsets = _token_text_and_offsets(text_tokenizer, input_ids)
            if decoded_text != text and not text.startswith(decoded_text):
                raise ValueError(
                    "mask_data requires offset mappings when decoded trainer input_ids do not match "
                    "the original Teich-rendered text."
                )
            labels = _labels_from_offsets(input_ids, offsets, supervised_spans)
        else:
            full_input_ids, _, offsets = encoded
            full_labels = _labels_from_offsets(full_input_ids, offsets, supervised_spans)
            labels = _align_labels_to_input_ids(input_ids, full_input_ids, full_labels)
            if labels is None:
                raise ValueError("Trainer tokenized input_ids do not align with the original Teich-rendered text.")
    else:
        text, offsets = _token_text_and_offsets(text_tokenizer, input_ids)
        supervised_spans = _infer_supervised_spans_from_rendered_text(text, train_on_reasoning=train_on_reasoning)
        inferred_metadata = _span_dicts(supervised_spans)
        supervised_spans = _select_supervised_spans(
            text,
            inferred_metadata,
            train_on_reasoning=train_on_reasoning,
            train_on_final_answers=train_on_final_answers,
            train_on_tools=train_on_tools,
            train_on_user=train_on_user,
            train_on_system=train_on_system,
            train_on_developer=train_on_developer,
            train_on_tool_responses=train_on_tool_responses,
        )
        if not supervised_spans:
            return None
        labels = _labels_from_offsets(input_ids, offsets, supervised_spans)
    if all(label == -100 for label in labels):
        raise ValueError("Teich masking produced a fully masked row after trainer tokenization/truncation.")
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def _sequence_length(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    if isinstance(value, list) and value and isinstance(value[0], list):
        return max(len(item) for item in value)
    return None


def _tensor_like_padded_labels(input_ids: Any, labels: list[list[int]], target_length: int, padding_side: str) -> Any | None:
    if not hasattr(input_ids, "new_full") or not hasattr(input_ids, "shape"):
        return None
    padded = input_ids.new_full((len(labels), target_length), _TEICH_LABEL_PAD_TOKEN_ID)
    for index, row_labels in enumerate(labels):
        row_length = min(len(row_labels), target_length)
        if row_length <= 0:
            continue
        values = row_labels[-row_length:] if padding_side == "left" else row_labels[:row_length]
        if padding_side == "left":
            padded[index, target_length - row_length :] = input_ids.new_tensor(values)
        else:
            padded[index, :row_length] = input_ids.new_tensor(values)
    return padded


def _list_padded_labels(labels: list[list[int]], target_length: int, padding_side: str) -> list[list[int]]:
    padded_labels: list[list[int]] = []
    for row_labels in labels:
        row_length = min(len(row_labels), target_length)
        values = row_labels[-row_length:] if padding_side == "left" else row_labels[:row_length]
        padding = [_TEICH_LABEL_PAD_TOKEN_ID] * (target_length - row_length)
        if padding_side == "left":
            padded_labels.append(padding + values)
        else:
            padded_labels.append(values + padding)
    return padded_labels


def _truncate_masked_row(
    masked_row: dict[str, list[int]],
    max_length: int | None,
    truncation_mode: str | None,
) -> dict[str, list[int]]:
    if not isinstance(max_length, int) or max_length <= 0:
        return masked_row
    input_ids = masked_row["input_ids"]
    labels = masked_row["labels"]
    if len(input_ids) <= max_length:
        return masked_row
    if truncation_mode == "keep_end":
        return {
            "input_ids": input_ids[-max_length:],
            "labels": labels[-max_length:],
        }
    return {
        "input_ids": input_ids[:max_length],
        "labels": labels[:max_length],
    }


class _TeichLabelPaddingCollator:
    def __init__(self, base_collator: Any, *, padding_side: str = "right"):
        self.base_collator = base_collator
        self.padding_side = "left" if padding_side == "left" else "right"

    def __call__(self, features: list[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        if not features or "labels" not in features[0]:
            return self.base_collator(features, *args, **kwargs)
        labels = [list(feature["labels"]) for feature in features]
        features_without_labels = []
        for feature in features:
            feature_without_labels = dict(feature)
            feature_without_labels.pop("labels", None)
            features_without_labels.append(feature_without_labels)
        batch = self.base_collator(features_without_labels, *args, **kwargs)
        input_ids = batch.get("input_ids") if isinstance(batch, Mapping) else None
        target_length = _sequence_length(input_ids) or max((len(row_labels) for row_labels in labels), default=0)
        padded_labels = _tensor_like_padded_labels(input_ids, labels, target_length, self.padding_side) if input_ids is not None else None
        batch["labels"] = padded_labels if padded_labels is not None else _list_padded_labels(labels, target_length, self.padding_side)
        return batch


def _should_wrap_label_padding_collator(collator: Any) -> bool:
    if collator is None or isinstance(collator, _TeichLabelPaddingCollator):
        return False
    collator_type = type(collator)
    return collator_type.__module__.startswith("transformers.") and collator_type.__name__ in _TEICH_LABEL_PADDING_COLLATOR_NAMES


def _ensure_label_padding_collator(trainer: Any, text_tokenizer: Any) -> None:
    collator = getattr(trainer, "data_collator", None)
    if not _should_wrap_label_padding_collator(collator):
        return
    padding_side = getattr(text_tokenizer, "padding_side", "right")
    trainer.data_collator = _TeichLabelPaddingCollator(collator, padding_side=padding_side)


def mask_data(
    trainer: Any,
    *,
    tokenizer: Any | None = None,
    text_column: str | None = None,
    train_on_reasoning: bool = True,
    train_on_final_answers: bool = True,
    train_on_tools: bool = True,
    train_on_user: bool = False,
    train_on_system: bool = False,
    train_on_developer: bool = False,
    train_on_tool_responses: bool = False,
    max_supervised_tokens: int | None = None,
    audit: bool = True,
    audit_sample_size: int = 8,
    verbose: bool = True,
) -> Any:
    from .audit import audit_sft_dataset

    text_tokenizer = _resolve_text_tokenizer(tokenizer or getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None))
    trainer_args = getattr(trainer, "args", None)
    dataset_text_field = text_column or getattr(trainer_args, "dataset_text_field", "text")
    trainer_max_length = getattr(trainer_args, "max_length", None)
    truncation_mode = getattr(trainer_args, "truncation_mode", None)
    effective_max_supervised_tokens = (
        max_supervised_tokens
        if isinstance(max_supervised_tokens, int) and max_supervised_tokens > 0
        else trainer_max_length
        if isinstance(trainer_max_length, int) and trainer_max_length > 0
        else None
    )
    if getattr(trainer_args, "packing", False):
        raise ValueError("mask_data does not support packed SFTTrainer datasets because packing merges row boundaries.")

    def _mask_dataset(dataset: Any, dataset_name: str) -> Any:
        if dataset is None:
            return None
        if not isinstance(dataset, Dataset):
            raise TypeError(f"trainer.{dataset_name} must be a datasets.Dataset instance.")
        if "input_ids" not in dataset.column_names and dataset_text_field in dataset.column_names:
            def _tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
                output_batch: dict[str, list[Any]] = {"input_ids": [], "attention_mask": []}
                for text in batch[dataset_text_field]:
                    if not isinstance(text, str):
                        raise TypeError(f"trainer.{dataset_name} has a non-string '{dataset_text_field}' value.")
                    tokenized = _tokenize_trainer_text(text_tokenizer, text)
                    if tokenized is None:
                        raise ValueError(
                            f"trainer.{dataset_name} is missing input_ids, and tokenizer could not tokenize "
                            f"the '{dataset_text_field}' column."
                        )
                    input_ids, attention_mask = tokenized
                    output_batch["input_ids"].append(input_ids)
                    output_batch["attention_mask"].append(attention_mask)
                return output_batch

            dataset = dataset.map(
                _tokenize_batch,
                batched=True,
                batch_size=_DATASET_MAP_BATCH_SIZE,
                desc=f"Tokenizing {dataset_name} for Teich masks",
            )
        missing = {"input_ids"}.difference(dataset.column_names)
        if missing:
            raise ValueError(f"trainer.{dataset_name} is missing required columns for mask_data: {', '.join(sorted(missing))}")
        dropped_supervised_count = 0
        dropped_untrainable_count = 0

        def _empty_output_batch() -> dict[str, list[Any]]:
            return {"input_ids": [], "labels": []}

        def _mask_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            nonlocal dropped_supervised_count
            nonlocal dropped_untrainable_count
            output_batch = _empty_output_batch()
            batch_size = len(batch["input_ids"])
            for index in range(batch_size):
                row = {column_name: batch[column_name][index] for column_name in dataset.column_names}
                masked_row = _mask_tokenized_row(
                    row,
                    text_tokenizer,
                    dataset_text_field,
                    train_on_reasoning=train_on_reasoning,
                    train_on_final_answers=train_on_final_answers,
                    train_on_tools=train_on_tools,
                    train_on_user=train_on_user,
                    train_on_system=train_on_system,
                    train_on_developer=train_on_developer,
                    train_on_tool_responses=train_on_tool_responses,
                )
                if masked_row is None:
                    dropped_untrainable_count += 1
                    continue
                supervised_tokens = sum(1 for label in masked_row["labels"] if label != -100)
                if (
                    effective_max_supervised_tokens is not None
                    and supervised_tokens > effective_max_supervised_tokens
                ):
                    dropped_supervised_count += 1
                    continue
                masked_row = _truncate_masked_row(masked_row, trainer_max_length, truncation_mode)
                if all(label == -100 for label in masked_row["labels"]):
                    raise ValueError("Teich masking produced a fully masked row after trainer max_length truncation.")
                output_batch["input_ids"].append(masked_row["input_ids"])
                output_batch["labels"].append(masked_row["labels"])
            return output_batch

        masked_dataset = dataset.map(
            _mask_batch,
            batched=True,
            batch_size=_DATASET_MAP_BATCH_SIZE,
            desc=f"Applying Teich masks to {dataset_name}",
            remove_columns=dataset.column_names,
        )
        if masked_dataset.num_rows == 0 and dropped_supervised_count > 0:
            raise ValueError(
                f"trainer.{dataset_name} contains no rows at or below max_supervised_tokens={effective_max_supervised_tokens}."
            )
        if masked_dataset.num_rows == 0 and dropped_untrainable_count > 0:
            raise ValueError(
                f"trainer.{dataset_name} contains no rows selected by the Teich masking policy; "
                "Teich masking produced fully masked rows."
            )
        if verbose and dropped_untrainable_count:
            Console().print(f"[yellow]Dropped {dropped_untrainable_count} {dataset_name} rows with no selected training spans.[/yellow]")
        if verbose and dropped_supervised_count:
            Console().print(
                f"[yellow]Dropped {dropped_supervised_count} {dataset_name} rows above "
                f"{effective_max_supervised_tokens} supervised tokens.[/yellow]"
            )
        if audit:
            report = audit_sft_dataset(masked_dataset, text_tokenizer, sample_size=audit_sample_size)
            report.raise_for_errors()
            if verbose and report.warnings:
                console = Console()
                for warning in report.warnings:
                    console.print(f"[yellow]Teich audit warning for {dataset_name}: {warning}[/yellow]")
        return _attach_preview(masked_dataset, text_tokenizer)

    trainer.train_dataset = _mask_dataset(getattr(trainer, "train_dataset", None), "train_dataset")
    eval_dataset = getattr(trainer, "eval_dataset", None)
    if isinstance(eval_dataset, dict):
        trainer.eval_dataset = {name: _mask_dataset(dataset, f"eval_dataset[{name!r}]") for name, dataset in eval_dataset.items()}
    elif eval_dataset is not None:
        trainer.eval_dataset = _mask_dataset(eval_dataset, "eval_dataset")
    _ensure_label_padding_collator(trainer, text_tokenizer)
    return trainer


def _decode_token(text_tokenizer: Any, token_id: int) -> str:
    try:
        return text_tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return text_tokenizer.decode([token_id], skip_special_tokens=False)


def _resolve_effective_max_length(max_length: int | None, text_tokenizer: Any) -> int | None:
    if isinstance(max_length, int) and max_length > 0:
        return max_length
    tokenizer_max_length = getattr(text_tokenizer, "model_max_length", None)
    if not isinstance(tokenizer_max_length, int) or tokenizer_max_length <= 0:
        return None
    if tokenizer_max_length >= 1_000_000_000:
        return None
    return tokenizer_max_length


def _build_preview(text_tokenizer: Any, input_ids: list[int], labels: list[int]) -> str:
    parts: list[str] = []
    masked = False
    for token_id, label in zip(input_ids, labels):
        is_masked = label == -100
        if is_masked and not masked:
            parts.append("\033[31m")
            masked = True
        elif not is_masked and masked:
            parts.append("\033[0m")
            masked = False
        parts.append(_escape_preview_control_sequences(_decode_token(text_tokenizer, token_id)))
    if masked:
        parts.append("\033[0m")
    return "".join(parts)


def _escape_preview_control_sequences(text: str) -> str:
    escaped: list[str] = []
    for character in text:
        codepoint = ord(character)
        if character in {"\n", "\t"}:
            escaped.append(character)
        elif character == "\x1b":
            escaped.append("\\x1b")
        elif codepoint < 32 or codepoint == 127 or 0x80 <= codepoint <= 0x9F:
            escaped.append(f"\\x{codepoint:02x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _attach_preview(training_data: Dataset, text_tokenizer: Any) -> Dataset:
    def preview(index: int = 0) -> str:
        return preview_sft_example(training_data, text_tokenizer, index=index)

    training_data.preview = preview
    return training_data


def preview_sft_example(dataset: Dataset, tokenizer: Any, *, index: int = 0) -> str:
    if dataset.num_rows == 0:
        raise IndexError("Cannot preview an empty dataset")
    if index < 0 or index >= dataset.num_rows:
        raise IndexError(f"Preview index {index} is out of range for dataset of size {dataset.num_rows}")
    row = dataset[index]
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    return _build_preview(text_tokenizer, row["input_ids"], row["labels"])
