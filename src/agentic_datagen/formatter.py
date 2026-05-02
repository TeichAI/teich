from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import json
import re
from typing import Any

from datasets import Dataset


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


def _fast_mask_row(
    renderer: Any,
    text_tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    max_length: int | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
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
    formatted_text = _render_chat(renderer, messages, tools, chat_template_kwargs)
    labels = [token_id if assistant_mask else -100 for token_id, assistant_mask in zip(input_ids, assistant_masks)]
    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_masks = assistant_masks[:max_length]
        labels = labels[:max_length]
    return (
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_masks": assistant_masks,
            "labels": labels,
        },
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "labels": labels,
        },
    )


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


def _mark_supervised_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
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
    assistant_start_tokens = (
        "<|im_start|>assistant\n",
        "<|assistant|>\n",
        "<|assistant|>",
        "<assistant>",
    )
    assistant_end_tokens = (
        "<|im_end|>",
        "</assistant>",
        "</s>",
    )
    block_start = -1
    for token in assistant_start_tokens:
        token_start = text.rfind(token, 0, start)
        if token_start > block_start:
            block_start = token_start
    if block_start < 0:
        return None
    block_end = -1
    for token in assistant_end_tokens:
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
        if matched_prefix is None:
            expanded_spans.append((start, end))
            continue
        expanded_spans.append((block_start + len(matched_prefix), block_end))
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


def _offset_mask_row(
    renderer: Any,
    text_tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    max_length: int | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not _supports_offsets(text_tokenizer):
        return None
    assistant_prompt_prefixes = _resolve_assistant_prompt_prefixes(
        renderer,
        messages,
        tools,
        chat_template_kwargs,
        assistant_prompt_prefix_cache,
    )
    marked_messages, markers = _mark_supervised_messages(messages)
    marked_text = _render_chat(renderer, marked_messages, tools, chat_template_kwargs)
    stripped = _strip_markers_and_collect_spans(marked_text, markers)
    if stripped is None:
        return None
    formatted_text, supervised_spans = stripped
    supervised_spans = _expand_supervised_spans(formatted_text, supervised_spans, assistant_prompt_prefixes)
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
    return (
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_masks": assistant_masks,
            "labels": labels,
        },
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "labels": labels,
        },
    )


def _mask_row(
    row: dict[str, Any],
    renderer: Any,
    text_tokenizer: Any,
    messages_column: str,
    tools_column: str,
    chat_template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    max_length: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = row.get(messages_column)
    if not isinstance(messages, list):
        raise TypeError(f"Row is missing a list-valued '{messages_column}' column")
    tools = row.get(tools_column) or []
    if not isinstance(tools, list):
        raise TypeError(f"Row is missing a list-valued '{tools_column}' column")

    fast_path = _fast_mask_row(renderer, text_tokenizer, messages, tools, chat_template_kwargs, max_length)
    if fast_path is not None:
        return fast_path

    offset_path = _offset_mask_row(
        renderer,
        text_tokenizer,
        messages,
        tools,
        chat_template_kwargs,
        assistant_prompt_prefix_cache,
        max_length,
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

    return (
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_masks": assistant_masks,
            "labels": labels,
        },
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "labels": labels,
        },
    )


def _is_prefix(prefix_ids: list[int], full_ids: list[int]) -> bool:
    return len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids


def _decode_token(text_tokenizer: Any, token_id: int) -> str:
    try:
        return text_tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return text_tokenizer.decode([token_id], skip_special_tokens=False)


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


def format_and_mask(
    dataset: Dataset,
    tokenizer: Any,
    *,
    messages_column: str = "messages",
    tools_column: str = "tools",
    chat_template_kwargs: dict[str, Any] | None = None,
    max_length: int | None = None,
) -> Dataset:
    template_kwargs = _validate_chat_template_kwargs(chat_template_kwargs)
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    renderer = _resolve_chat_template_renderer(tokenizer, text_tokenizer)
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]] = {}

    def _map_row(row: dict[str, Any]) -> dict[str, Any]:
        masked_row, _ = _mask_row(
            row,
            renderer,
            text_tokenizer,
            messages_column,
            tools_column,
            template_kwargs,
            assistant_prompt_prefix_cache,
            max_length,
        )
        return masked_row

    training_data = dataset.map(
        _map_row,
        remove_columns=dataset.column_names,
        desc="Formatting and masking conversations",
    )

    def preview(index: int = 0) -> str:
        if training_data.num_rows == 0:
            raise IndexError("Cannot preview an empty dataset")
        if index < 0 or index >= training_data.num_rows:
            raise IndexError(f"Preview index {index} is out of range for dataset of size {training_data.num_rows}")
        row = training_data[index]
        return _build_preview(text_tokenizer, row["input_ids"], row["labels"])

    training_data.preview = preview
    return training_data
