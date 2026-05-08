from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import json
import re
from collections.abc import Sequence
from typing import Any

from datasets import Dataset, concatenate_datasets
from rich.console import Console
from rich.table import Table


_GEMMA_TURN_START_PATTERN = re.compile(r"<\|turn>(model|user|system)\n")
_GEMMA_ASSISTANT_TURN_PREFIX = "<|turn>model\n"
_GEMMA_THOUGHT_PREFIX = "<|channel>thought\n"
_GEMMA_TOOL_RESPONSE_START = "<|tool_response>"
_GEMMA_TOOL_RESPONSE_END = "<tool_response|>"
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
    "<turn|>",
)
_REASONING_BLOCK_PATTERNS = (
    re.compile(r"<think>\n.*?</think>\n\n?", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL),
)
_FORMAT_AND_MASK_BATCH_SIZE = 8
TEICH_SUPERVISED_SPANS_COLUMN = "teich_supervised_spans"


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
    rendered = renderer.apply_chat_template(messages, **render_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string")
    return rendered


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
    rendered = renderer.apply_chat_template(messages, **render_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string")
    return rendered


def _tokenize_text(text_tokenizer: Any, text: str) -> tuple[list[int], list[int]]:
    encoded = text_tokenizer(text, add_special_tokens=False, return_attention_mask=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    return list(input_ids), list(attention_mask)


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
        ((), {"text": text, "return_attention_mask": True, "return_offsets_mapping": True}),
        ((text,), {"return_attention_mask": True, "return_offsets_mapping": True}),
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


def _supports_offsets(text_tokenizer: Any) -> bool:
    return _tokenize_text_with_offsets(text_tokenizer, "") is not None


def _initial_prefix_length(
    renderer: Any,
    text_tokenizer: Any,
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> tuple[int, list[int]]:
    try:
        prefix_text = _render_chat(renderer, [], tools, chat_template_kwargs)
    except Exception:
        return 0, []
    prefix_ids, _ = _tokenize_text(text_tokenizer, prefix_text)
    return len(prefix_ids), prefix_ids


def _chat_template_supports_assistant_mask(renderer: Any) -> bool:
    template = getattr(renderer, "chat_template", None)
    if not isinstance(template, str):
        return False
    return bool(re.search(r"\{%-?\s*generation\s*-?%\}", template))


def _chat_template_looks_like_gemma(renderer: Any) -> bool:
    template = getattr(renderer, "chat_template", None)
    if not isinstance(template, str):
        return False
    return "<|turn>model" in template and "<|tool_response>" in template


def _has_user_message(messages: list[dict[str, Any]]) -> bool:
    return any(isinstance(message, dict) and message.get("role") == "user" for message in messages)


def _is_skippable_prefix_exception(exc: Exception) -> bool:
    message = str(exc)
    return "No user query found in messages." in message


def _try_render_prefix(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> str | None:
    if not _has_user_message(messages):
        return None
    try:
        return _render_chat(renderer, messages, tools, chat_template_kwargs)
    except Exception as exc:
        if _is_skippable_prefix_exception(exc):
            return None
        raise


def _is_assistant_message(message: dict[str, Any]) -> bool:
    return isinstance(message, dict) and message.get("role") == "assistant"


def _mask_checkpoints(messages: list[dict[str, Any]]) -> list[tuple[int, bool]]:
    if not messages:
        return []
    checkpoints: list[tuple[int, bool]] = []
    current_supervision = _is_assistant_message(messages[0])
    for index in range(1, len(messages) + 1):
        next_supervision = _is_assistant_message(messages[index]) if index < len(messages) else None
        if next_supervision != current_supervision:
            checkpoints.append((index, current_supervision))
            current_supervision = next_supervision
    return checkpoints


def _extract_token_sequence(values: Any) -> list[int] | None:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def _update_labels_with_diff(
    previous_ids: list[int],
    previous_labels: list[int],
    current_ids: list[int],
    supervise_current_message: bool,
) -> list[int]:
    labels: list[int] = []
    matcher = SequenceMatcher(a=previous_ids, b=current_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            labels.extend(previous_labels[i1:i2])
            continue
        if tag in {"replace", "insert"}:
            if supervise_current_message:
                labels.extend(current_ids[j1:j2])
            else:
                labels.extend([-100] * (j2 - j1))
    return labels


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
    tool_response_spans = _find_delimited_spans(text, _GEMMA_TOOL_RESPONSE_START, _GEMMA_TOOL_RESPONSE_END)
    supervised_spans: list[tuple[int, int]] = []
    for index, match in enumerate(turn_matches):
        if match.group(1) != "model":
            continue
        block_start = match.start()
        block_end = turn_matches[index + 1].start() if index + 1 < len(turn_matches) else len(text)
        supervised_start = block_start + len(_GEMMA_ASSISTANT_TURN_PREFIX)
        if supervised_start < block_end:
            supervised_spans.append((supervised_start, block_end))
    return _subtract_spans(supervised_spans, tool_response_spans)


def _gemma_mask_row(
    renderer: Any,
    text_tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    train_on_reasoning: bool,
    max_length: int | None,
    include_debug_columns: bool,
) -> dict[str, Any] | None:
    if not _chat_template_looks_like_gemma(renderer):
        return None
    if not _supports_offsets(text_tokenizer):
        return None
    formatted_text = _render_chat(renderer, messages, tools, chat_template_kwargs)
    supervised_spans = _gemma_like_supervised_spans(formatted_text)
    if not supervised_spans:
        return None
    if not train_on_reasoning:
        supervised_spans = _subtract_spans(supervised_spans, _reasoning_spans(formatted_text))
        if not supervised_spans:
            return None
    encoded = _tokenize_text_with_offsets(text_tokenizer, formatted_text)
    if encoded is None:
        return None
    input_ids, attention_mask, offsets = encoded
    labels = _labels_from_offsets(input_ids, offsets, supervised_spans)
    assistant_masks = [0 if label == -100 else 1 for label in labels]
    if 1 not in assistant_masks:
        return None
    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_masks = assistant_masks[:max_length]
        labels = labels[:max_length]
    row = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if include_debug_columns:
        row["text"] = formatted_text
        row["assistant_masks"] = assistant_masks
    return row


def _fast_mask_row(
    renderer: Any,
    text_tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    max_length: int | None,
    include_debug_columns: bool,
) -> dict[str, Any] | None:
    if not _chat_template_supports_assistant_mask(renderer):
        return None
    render_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
        **chat_template_kwargs,
    }
    if tools:
        render_kwargs["tools"] = tools
    try:
        processed = renderer.apply_chat_template(messages, **render_kwargs)
    except Exception:
        return None
    if not isinstance(processed, dict):
        return None
    input_ids = _extract_token_sequence(processed.get("input_ids"))
    if input_ids is None:
        return None
    attention_mask = _extract_token_sequence(processed.get("attention_mask"))
    assistant_masks = _extract_token_sequence(processed.get("assistant_masks"))
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    if assistant_masks is None or 1 not in assistant_masks:
        return None
    labels = [token_id if assistant_mask else -100 for token_id, assistant_mask in zip(input_ids, assistant_masks)]
    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_masks = assistant_masks[:max_length]
        labels = labels[:max_length]
    row = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if include_debug_columns:
        row["text"] = _render_chat(renderer, messages, tools, chat_template_kwargs)
        row["assistant_masks"] = assistant_masks
    return row


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
        for key, item in updated.items():
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
        keys = list(updated.keys())
        for key in reversed(keys):
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
    train_on_reasoning: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    marked_messages = deepcopy(messages)
    markers: list[tuple[str, str]] = []
    marker_index = 0

    def mark_value(value: Any) -> tuple[Any, bool]:
        nonlocal marker_index
        start_marker = f"\ue000AGD{marker_index}S\ue001"
        end_marker = f"\ue000AGD{marker_index}E\ue001"
        updated_value, changed = _wrap_with_markers(value, start_marker, end_marker)
        if changed:
            markers.append((start_marker, end_marker))
            marker_index += 1
        return updated_value, changed

    for message in marked_messages:
        if not _is_assistant_message(message):
            continue
        if train_on_reasoning:
            reasoning = message.get("reasoning_content")
            updated_reasoning, changed = mark_value(reasoning)
            if changed:
                message["reasoning_content"] = updated_reasoning
        tool_calls = message.get("tool_calls") or []
        for tool_call in tool_calls:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            updated_name, changed = mark_value(name)
            if changed:
                function["name"] = updated_name
            arguments = function.get("arguments")
            updated_arguments, changed = mark_value(arguments)
            if changed:
                function["arguments"] = updated_arguments
        content = message.get("content")
        updated_content, changed = mark_value(content)
        if changed:
            message["content"] = updated_content
    return marked_messages, markers


def _strip_markers_and_collect_spans(text: str, markers: list[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]] | None:
    if not markers:
        return text, []
    marker_lookup: dict[str, tuple[str, int]] = {}
    pattern_parts: list[str] = []
    for index, (start_marker, end_marker) in enumerate(markers):
        marker_lookup[start_marker] = ("start", index)
        marker_lookup[end_marker] = ("end", index)
        pattern_parts.append(re.escape(start_marker))
        pattern_parts.append(re.escape(end_marker))
    pattern = re.compile("|".join(pattern_parts))
    cleaned_parts: list[str] = []
    active_starts: dict[int, int] = {}
    spans: list[tuple[int, int]] = []
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
            active_starts[index] = cleaned_length
        else:
            start = active_starts.pop(index, None)
            if start is None:
                return None
            if start < cleaned_length:
                spans.append((start, cleaned_length))
        cursor = match.end()
    tail = text[cursor:]
    if tail:
        cleaned_parts.append(tail)
    if active_starts:
        return None
    cleaned_text = "".join(cleaned_parts)
    if not spans:
        return cleaned_text, []
    return cleaned_text, _merge_spans(spans)


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
            expanded_spans.append((block_start + len(matched_prefix), block_end))
            continue
        if fallback_prefix is not None:
            expanded_spans.append((block_start + len(fallback_prefix), block_end))
            continue
        expanded_spans.append((start, end))
    return _merge_spans(expanded_spans)


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
            and supervised_spans[span_index][0] < end
            and start < supervised_spans[span_index][1]
        )
        labels.append(token_id if is_supervised else -100)
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
    if not train_on_reasoning:
        supervised_spans = _subtract_spans(supervised_spans, _reasoning_spans(text))
    return supervised_spans


def _offset_mask_row(
    renderer: Any,
    text_tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    train_on_reasoning: bool,
    max_length: int | None,
    include_debug_columns: bool,
    strict: bool,
) -> dict[str, Any] | None:
    if not _supports_offsets(text_tokenizer):
        return None
    assistant_prompt_prefixes = _resolve_assistant_prompt_prefixes(
        renderer,
        messages,
        tools,
        chat_template_kwargs,
        assistant_prompt_prefix_cache,
    )
    original_text = _render_chat(renderer, messages, tools, chat_template_kwargs)
    marked_messages, markers = _mark_supervised_messages(messages, train_on_reasoning=train_on_reasoning)
    marked_text = _render_chat(renderer, marked_messages, tools, chat_template_kwargs)
    stripped = _strip_markers_and_collect_spans(marked_text, markers)
    if stripped is None:
        if strict:
            raise ValueError("Unable to collect supervised spans from marker-injected chat template output.")
        return None
    formatted_text, supervised_spans = stripped
    if formatted_text != original_text:
        if strict:
            raise ValueError("Marker-injected chat template output does not match the original rendered chat after marker removal.")
        return None
    supervised_spans = _expand_supervised_spans(
        formatted_text,
        supervised_spans,
        assistant_prompt_prefixes,
        train_on_reasoning,
    )
    if not train_on_reasoning:
        supervised_spans = _subtract_spans(supervised_spans, _reasoning_spans(formatted_text))
        if not supervised_spans:
            return None
    encoded = _tokenize_text_with_offsets(text_tokenizer, formatted_text)
    if encoded is None:
        return None
    input_ids, attention_mask, offsets = encoded
    labels = _labels_from_offsets(input_ids, offsets, supervised_spans)
    assistant_masks = [0 if label == -100 else 1 for label in labels]
    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_masks = assistant_masks[:max_length]
        labels = labels[:max_length]
    row = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if include_debug_columns:
        row["text"] = formatted_text
        row["assistant_masks"] = assistant_masks
    return row


def _mask_row(
    row: dict[str, Any],
    renderer: Any,
    text_tokenizer: Any,
    messages_column: str,
    tools_column: str,
    chat_template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    train_on_reasoning: bool,
    max_length: int | None,
    include_debug_columns: bool,
    strict: bool,
) -> dict[str, Any]:
    messages = row.get(messages_column)
    if not isinstance(messages, list):
        raise TypeError(f"Row is missing a list-valued '{messages_column}' column")
    tools = row.get(tools_column) or []
    if not isinstance(tools, list):
        raise TypeError(f"Row is missing a list-valued '{tools_column}' column")

    if train_on_reasoning:
        fast_path = _fast_mask_row(
            renderer,
            text_tokenizer,
            messages,
            tools,
            chat_template_kwargs,
            max_length,
            include_debug_columns,
        )
        if fast_path is not None:
            return fast_path

    gemma_path = _gemma_mask_row(
        renderer,
        text_tokenizer,
        messages,
        tools,
        chat_template_kwargs,
        train_on_reasoning,
        max_length,
        include_debug_columns,
    )
    if gemma_path is not None:
        return gemma_path

    offset_path = _offset_mask_row(
        renderer,
        text_tokenizer,
        messages,
        tools,
        chat_template_kwargs,
        assistant_prompt_prefix_cache,
        train_on_reasoning,
        max_length,
        include_debug_columns,
        strict,
    )
    if offset_path is not None:
        return offset_path

    formatted_text = _render_chat(renderer, messages, tools, chat_template_kwargs)
    input_ids, attention_mask = _tokenize_text(text_tokenizer, formatted_text)
    current_ids: list[int] = []
    labels: list[int] = []

    for index, supervise_current_message in _mask_checkpoints(messages):
        if index == len(messages):
            prefix_ids = input_ids
        else:
            prefix_text = _try_render_prefix(renderer, messages[:index], tools, chat_template_kwargs)
            if prefix_text is None:
                continue
            prefix_ids, _ = _tokenize_text(text_tokenizer, prefix_text)
        labels = _update_labels_with_diff(current_ids, labels, prefix_ids, supervise_current_message)
        current_ids = prefix_ids

    if current_ids != input_ids:
        raise ValueError("Unable to align masked token sequence with final chat template output.")

    assistant_masks = [0 if label == -100 else 1 for label in labels]

    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_masks = assistant_masks[:max_length]
        labels = labels[:max_length]

    row = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if include_debug_columns:
        row["text"] = formatted_text
        row["assistant_masks"] = assistant_masks
    return row


def _supervised_text_and_spans(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    train_on_reasoning: bool,
    strict: bool,
) -> tuple[str, list[tuple[int, int]]]:
    original_text = _render_chat(renderer, messages, tools, chat_template_kwargs)
    marked_messages, markers = _mark_supervised_messages(messages, train_on_reasoning=train_on_reasoning)
    marked_text = _render_chat(renderer, marked_messages, tools, chat_template_kwargs)
    stripped = _strip_markers_and_collect_spans(marked_text, markers)
    if stripped is None:
        raise ValueError("Unable to collect supervised spans from marker-injected chat template output.")
    formatted_text, supervised_spans = stripped
    if formatted_text != original_text:
        if strict:
            raise ValueError("Marker-injected chat template output does not match the original rendered chat after marker removal.")
        return original_text, []
    assistant_prompt_prefixes = _resolve_assistant_prompt_prefixes(
        renderer,
        messages,
        tools,
        chat_template_kwargs,
        assistant_prompt_prefix_cache,
    )
    supervised_spans = _expand_supervised_spans(
        formatted_text,
        supervised_spans,
        assistant_prompt_prefixes,
        train_on_reasoning,
    )
    if not train_on_reasoning:
        supervised_spans = _subtract_spans(supervised_spans, _reasoning_spans(formatted_text))
    return formatted_text, supervised_spans


def _span_dicts(spans: list[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"start": start, "end": end} for start, end in spans if start < end]


def _normalize_span_dicts(value: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for item in value or []:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        else:
            start, end = item
        if isinstance(start, int) and isinstance(end, int) and start < end:
            spans.append((start, end))
    return _merge_spans(spans)


def format_data(
    dataset: Dataset | Sequence[Dataset],
    tokenizer: Any,
    *,
    messages_column: str = "messages",
    tools_column: str = "tools",
    text_column: str = "text",
    chat_template_kwargs: dict[str, Any] | None = None,
    train_on_reasoning: bool = True,
    max_length: int | None = None,
    drop_oversized_examples: bool = True,
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
                        max_length=max_length,
                        drop_oversized_examples=drop_oversized_examples,
                        strict=strict,
                        verbose=verbose,
                    )
                )
            return concatenate_datasets(formatted_datasets)
        dataset = datasets[0]
    if not isinstance(dataset, Dataset):
        raise TypeError("prepare_data expects a Dataset or a sequence of Dataset objects.")

    template_kwargs = _validate_chat_template_kwargs(chat_template_kwargs)
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    renderer = _resolve_chat_template_renderer(tokenizer, text_tokenizer)
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]] = {}
    effective_max_length = max_length if isinstance(max_length, int) and max_length > 0 else None
    dropped_count = 0
    dropped_oversized_count = 0

    if messages_column not in dataset.column_names:
        raise TypeError(f"Dataset is missing required '{messages_column}' column")

    output_columns = [text_column, TEICH_SUPERVISED_SPANS_COLUMN]

    def _empty_output_batch() -> dict[str, list[Any]]:
        return {column_name: [] for column_name in output_columns}

    def _map_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        nonlocal dropped_count
        nonlocal dropped_oversized_count
        batch_size = len(batch[messages_column])
        tools_batch = batch.get(tools_column)
        if tools_batch is None:
            tools_batch = [None] * batch_size
        output_batch = _empty_output_batch()

        for index in range(batch_size):
            messages = batch[messages_column][index]
            if not isinstance(messages, list):
                raise TypeError(f"Row is missing a list-valued '{messages_column}' column")
            if len(messages) == 0:
                dropped_count += 1
                continue
            tools = tools_batch[index] or []
            if not isinstance(tools, list):
                raise TypeError(f"Row is missing a list-valued '{tools_column}' column")
            text, supervised_spans = _supervised_text_and_spans(
                renderer,
                messages,
                tools,
                template_kwargs,
                assistant_prompt_prefix_cache,
                train_on_reasoning,
                strict,
            )
            if not supervised_spans:
                if strict:
                    raise ValueError("No supervised assistant spans were found for a non-empty conversation.")
                dropped_count += 1
                continue
            if drop_oversized_examples and effective_max_length is not None:
                input_ids, _ = _tokenize_text(text_tokenizer, text)
                if len(input_ids) > effective_max_length:
                    dropped_oversized_count += 1
                    continue
            output_batch[text_column].append(text)
            output_batch[TEICH_SUPERVISED_SPANS_COLUMN].append(_span_dicts(supervised_spans))
        return output_batch

    formatted_data = dataset.map(
        _map_batch,
        batched=True,
        batch_size=_FORMAT_AND_MASK_BATCH_SIZE,
        remove_columns=dataset.column_names,
    )
    if formatted_data.num_rows == 0 and drop_oversized_examples and effective_max_length is not None and dropped_oversized_count > 0:
        raise ValueError(
            f"Dataset contains no conversations that fit within context window of {effective_max_length} tokens."
        )
    if verbose and dropped_count:
        Console().print(f"[yellow]Dropped {dropped_count} rows without trainable assistant spans.[/yellow]")
    if verbose and dropped_oversized_count:
        Console().print(f"[yellow]Dropped {dropped_oversized_count} rows above {effective_max_length} tokens.[/yellow]")
    return formatted_data


def _mask_tokenized_row(
    row: dict[str, Any],
    text_tokenizer: Any,
    text_column: str,
    train_on_reasoning: bool,
) -> dict[str, Any]:
    input_ids = _extract_token_sequence(row.get("input_ids"))
    if input_ids is None:
        raise TypeError("Trainer dataset row is missing tokenized 'input_ids'.")
    text = row.get(text_column)
    supervised_spans = _normalize_span_dicts(row.get(TEICH_SUPERVISED_SPANS_COLUMN))
    if isinstance(text, str) and supervised_spans:
        encoded = _tokenize_trainer_text_with_offsets(text_tokenizer, text)
        if encoded is None:
            raise ValueError("mask_data requires a tokenizer that can return offset mappings for text tokenization.")
        full_input_ids, full_attention_mask, offsets = encoded
        full_labels = _labels_from_offsets(full_input_ids, offsets, supervised_spans)
        if input_ids == full_input_ids:
            labels = full_labels
            attention_mask = full_attention_mask
        elif len(input_ids) <= len(full_input_ids) and input_ids == full_input_ids[: len(input_ids)]:
            labels = full_labels[: len(input_ids)]
            attention_mask = full_attention_mask[: len(input_ids)]
        else:
            raise ValueError("Trainer tokenized input_ids do not align with the original Teich-rendered text.")
    else:
        text, offsets = _token_text_and_offsets(text_tokenizer, input_ids)
        supervised_spans = _infer_supervised_spans_from_rendered_text(text, train_on_reasoning=train_on_reasoning)
        labels = _labels_from_offsets(input_ids, offsets, supervised_spans)
        attention_mask = [1] * len(input_ids)
    assistant_masks = [0 if label == -100 else 1 for label in labels]
    if 1 not in assistant_masks:
        raise ValueError("Teich masking produced a fully masked row after trainer tokenization/truncation.")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_masks": assistant_masks,
        "labels": labels,
    }


def mask_data(
    trainer: Any,
    *,
    tokenizer: Any | None = None,
    text_column: str | None = None,
    train_on_reasoning: bool = True,
    audit: bool = True,
    audit_sample_size: int = 8,
) -> Any:
    from .audit import audit_sft_dataset

    text_tokenizer = _resolve_text_tokenizer(tokenizer or getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None))
    trainer_args = getattr(trainer, "args", None)
    dataset_text_field = text_column or getattr(trainer_args, "dataset_text_field", "text")
    if getattr(trainer_args, "packing", False):
        raise ValueError("mask_data does not support packed SFTTrainer datasets because packing merges row boundaries.")

    def _mask_dataset(dataset: Any, dataset_name: str) -> Any:
        if dataset is None:
            return None
        if not isinstance(dataset, Dataset):
            raise TypeError(f"trainer.{dataset_name} must be a datasets.Dataset instance.")
        missing = {"input_ids"}.difference(dataset.column_names)
        if missing:
            raise ValueError(f"trainer.{dataset_name} is missing required columns for mask_data: {', '.join(sorted(missing))}")
        masked_dataset = dataset.map(
            lambda row: _mask_tokenized_row(row, text_tokenizer, dataset_text_field, train_on_reasoning),
            desc=f"Applying Teich masks to {dataset_name}",
        )
        if audit:
            report = audit_sft_dataset(masked_dataset, text_tokenizer, sample_size=audit_sample_size)
            report.raise_for_errors()
        return masked_dataset

    trainer.train_dataset = _mask_dataset(getattr(trainer, "train_dataset", None), "train_dataset")
    eval_dataset = getattr(trainer, "eval_dataset", None)
    if isinstance(eval_dataset, dict):
        trainer.eval_dataset = {name: _mask_dataset(dataset, f"eval_dataset[{name!r}]") for name, dataset in eval_dataset.items()}
    elif eval_dataset is not None:
        trainer.eval_dataset = _mask_dataset(eval_dataset, "eval_dataset")
    return trainer


def _is_prefix(prefix_ids: list[int], full_ids: list[int]) -> bool:
    return len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids


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
        parts.append(_decode_token(text_tokenizer, token_id))
    if masked:
        parts.append("\033[0m")
    return "".join(parts)


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


def format_and_mask(
    dataset: Dataset | Sequence[Dataset],
    tokenizer: Any,
    *,
    messages_column: str = "messages",
    tools_column: str = "tools",
    chat_template_kwargs: dict[str, Any] | None = None,
    train_on_reasoning: bool = True,
    max_length: int | None = None,
    include_debug_columns: bool = False,
    drop_oversized_examples: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> Dataset:
    if isinstance(dataset, Sequence) and not isinstance(dataset, Dataset):
        datasets = list(dataset)
        if not datasets:
            raise ValueError("At least one dataset must be provided to format_and_mask.")
        if len(datasets) > 1:
            text_tokenizer = _resolve_text_tokenizer(tokenizer)
            formatted_datasets: list[Dataset] = []
            for item in datasets:
                if not isinstance(item, Dataset):
                    raise TypeError("format_and_mask expects a Dataset or a sequence of Dataset objects.")
                formatted_datasets.append(
                    format_and_mask(
                        item,
                        tokenizer,
                        messages_column=messages_column,
                        tools_column=tools_column,
                        chat_template_kwargs=chat_template_kwargs,
                        train_on_reasoning=train_on_reasoning,
                        max_length=max_length,
                        include_debug_columns=include_debug_columns,
                        drop_oversized_examples=drop_oversized_examples,
                        strict=strict,
                        verbose=verbose,
                    )
                )
            training_data = concatenate_datasets(formatted_datasets)
            return _attach_preview(training_data, text_tokenizer)
        else:
            dataset = datasets[0]
    if not isinstance(dataset, Dataset):
        raise TypeError("format_and_mask expects a Dataset or a sequence of Dataset objects.")

    template_kwargs = _validate_chat_template_kwargs(chat_template_kwargs)
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    renderer = _resolve_chat_template_renderer(tokenizer, text_tokenizer)
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]] = {}
    effective_max_length = _resolve_effective_max_length(max_length, text_tokenizer)
    saw_non_empty_conversation = False
    empty_dropped_count = 0
    dropped_oversized_examples_count = 0

    if messages_column not in dataset.column_names:
        raise TypeError(f"Dataset is missing required '{messages_column}' column")

    output_columns = ["input_ids", "attention_mask", "labels"]
    if include_debug_columns:
        output_columns = ["text", "input_ids", "attention_mask", "assistant_masks", "labels"]

    def _empty_output_batch() -> dict[str, list[Any]]:
        return {column_name: [] for column_name in output_columns}

    def _map_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        nonlocal saw_non_empty_conversation
        nonlocal empty_dropped_count
        nonlocal dropped_oversized_examples_count

        batch_size = len(batch[messages_column])
        tools_batch = batch.get(tools_column)
        if tools_batch is None:
            tools_batch = [None] * batch_size
        output_batch = _empty_output_batch()

        for index in range(batch_size):
            messages = batch[messages_column][index]
            if not isinstance(messages, list):
                raise TypeError(f"Row is missing a list-valued '{messages_column}' column")
            if len(messages) == 0:
                empty_dropped_count += 1
                continue

            saw_non_empty_conversation = True
            tools = tools_batch[index] or []
            if not isinstance(tools, list):
                raise TypeError(f"Row is missing a list-valued '{tools_column}' column")

            formatted_row = _mask_row(
                {messages_column: messages, tools_column: tools},
                renderer,
                text_tokenizer,
                messages_column,
                tools_column,
                template_kwargs,
                assistant_prompt_prefix_cache,
                train_on_reasoning,
                None if drop_oversized_examples else effective_max_length,
                include_debug_columns,
                strict,
            )

            if drop_oversized_examples and effective_max_length is not None:
                if len(formatted_row["input_ids"]) > effective_max_length:
                    dropped_oversized_examples_count += 1
                    continue

            for column_name in output_columns:
                output_batch[column_name].append(formatted_row[column_name])

        return output_batch

    training_data = dataset.map(
        _map_batch,
        batched=True,
        batch_size=_FORMAT_AND_MASK_BATCH_SIZE,
        writer_batch_size=_FORMAT_AND_MASK_BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc="Filtering conversations" if drop_oversized_examples else "Formatting and masking conversations",
    )

    if not saw_non_empty_conversation:
        raise ValueError("Dataset contains no non-empty conversations to format and mask.")
    if training_data.num_rows == 0 and drop_oversized_examples and effective_max_length is not None and dropped_oversized_examples_count > 0:
        raise ValueError(
            f"Dataset contains no conversations that fit within context window of {effective_max_length} tokens."
        )

    if verbose:
        console = Console()
        table = Table(title="format_and_mask summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        table.add_row("Input rows", str(dataset.num_rows))
        table.add_row("Kept rows", str(training_data.num_rows))
        table.add_row("Empty dropped", str(empty_dropped_count))
        if drop_oversized_examples and effective_max_length is not None:
            table.add_row("Oversized dropped", str(dropped_oversized_examples_count))
        console.print(table)

    return _attach_preview(training_data, text_tokenizer)
