from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import pytest

from teich import load_traces, mask_data, prepare_data, preview_sft_example, row_fits_context
from teich.converter import convert_trace_to_training_example
from teich.formatter import _labels_from_offsets, _reconcile_marker_boundary_whitespace, _strip_markers_and_collect_spans


def prepare_and_mask_for_test(
    dataset,
    tokenizer,
    *,
    messages_column="messages",
    tools_column="tools",
    chat_template_kwargs=None,
    train_on_reasoning=True,
    max_length=None,
    include_debug_columns=False,
    drop_oversized_examples=True,
    strict=False,
    verbose=False,
):
    prepared = prepare_data(
        dataset,
        tokenizer,
        messages_column=messages_column,
        tools_column=tools_column,
        chat_template_kwargs=chat_template_kwargs,
        train_on_reasoning=train_on_reasoning,
        max_length=max_length,
        drop_oversized_examples=drop_oversized_examples,
        strict=strict,
        verbose=verbose,
    )
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        args=SimpleNamespace(dataset_text_field="text", max_length=max_length if drop_oversized_examples else None, packing=False),
        tokenizer=tokenizer,
    )
    trainer = mask_data(trainer, tokenizer=tokenizer, train_on_reasoning=train_on_reasoning, audit=False, verbose=False)
    training_data = trainer.train_dataset
    if include_debug_columns:
        rows = []
        for index in range(training_data.num_rows):
            row = dict(training_data[index])
            row["text"] = prepared[index]["text"]
            row["assistant_masks"] = [0 if label == -100 else 1 for label in row["labels"]]
            rows.append(row)
        training_data = Dataset.from_list(rows)
    training_data.preview = lambda index=0: preview_sft_example(training_data, tokenizer, index=index)
    return training_data


class FakeTokenizer:
    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._reverse_vocab: dict[int, str] = {}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        enable_thinking=True,
        preserve_thinking=True,
        **kwargs,
    ):
        if kwargs:
            raise AssertionError(f"Unexpected chat template kwargs: {kwargs}")
        tool_prefix = ""
        if tools:
            tool_names = ",".join(tool["function"]["name"] for tool in tools)
            tool_prefix = f"<tools>{tool_names}</tools>"
        parts: list[str] = [tool_prefix]
        for message in messages:
            role = message["role"]
            segment = f"<{role}>"
            if role == "assistant":
                if enable_thinking and preserve_thinking and message.get("reasoning_content"):
                    segment += f"<think>{message['reasoning_content']}</think>"
                tool_calls = message.get("tool_calls") or []
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    segment += f"<tool_call>{name}</tool_call>"
            if message.get("content"):
                segment += str(message["content"])
            segment += f"</{role}>"
            parts.append(segment)
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "".join(parts)
        if tokenize:
            return self(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True, return_offsets_mapping=False):
        token_ids: list[int] = []
        for token in text:
            token_id = self._vocab.setdefault(token, len(self._vocab) + 1)
            self._reverse_vocab[token_id] = token
            token_ids.append(token_id)
        output = {"input_ids": token_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(token_ids)
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


def test_labels_from_offsets_masks_tokens_that_cross_supervision_boundaries():
    labels = _labels_from_offsets(
        [1, 2, 3],
        [(0, 1), (1, 3), (3, 4)],
        [(2, 4)],
    )

    assert labels == [-100, -100, 3]


def test_prepare_data_normalizes_direct_messages_dataset_with_inline_thinking():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "solve"},
                    {"role": "assistant", "content": "<think>reason it out</think> final answer"},
                ]
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True},
        include_debug_columns=True,
    )
    row = training_data[0]
    rendered = row["text"]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])

    assert "<think>reason it out</think> final answer" not in rendered
    assert "<think>reason it out</think>final answer</assistant>" in supervised_text


class RequiresUserTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user query found in messages.")
        return super().apply_chat_template(messages, **kwargs)


class LimitedFakeTokenizer(FakeTokenizer):
    model_max_length = 60


class CountingTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.render_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.render_count += 1
        return super().apply_chat_template(messages, **kwargs)


class OffsetCountingTokenizer(CountingTokenizer):
    def __call__(self, text, add_special_tokens=False, return_attention_mask=True, return_offsets_mapping=False):
        output = super().__call__(text, add_special_tokens=add_special_tokens, return_attention_mask=return_attention_mask)
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


class TrainerStyleTokenizer(OffsetCountingTokenizer):
    eos_token = ""
    pad_token = "<pad>"
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        if token == self.pad_token:
            return self.pad_token_id
        return self._vocab.setdefault(token, len(self._vocab) + 1)


class NoOffsetTrainerStyleTokenizer(TrainerStyleTokenizer):
    def __call__(self, text=None, add_special_tokens=False, return_attention_mask=True, return_offsets_mapping=False, **kwargs):
        if return_offsets_mapping:
            raise NotImplementedError("offsets unavailable")
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            return_offsets_mapping=False,
            **kwargs,
        )


class SpecialTokenWrappingTokenizer(TrainerStyleTokenizer):
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self):
        super().__init__()
        self.add_special_tokens_calls: list[bool] = []
        self._vocab[self.bos_token] = 10_001
        self._vocab[self.eos_token] = 10_002
        self._reverse_vocab[10_001] = self.bos_token
        self._reverse_vocab[10_002] = self.eos_token

    def __call__(
        self,
        text=None,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=False,
        **kwargs,
    ):
        self.add_special_tokens_calls.append(add_special_tokens)
        output = super().__call__(
            text,
            add_special_tokens=False,
            return_attention_mask=return_attention_mask,
            return_offsets_mapping=return_offsets_mapping,
        )
        if add_special_tokens:
            output["input_ids"] = [self._vocab[self.bos_token], *output["input_ids"], self._vocab[self.eos_token]]
            if return_attention_mask:
                output["attention_mask"] = [1, *output["attention_mask"], 1]
            if return_offsets_mapping:
                output["offset_mapping"] = [(0, 0), *output["offset_mapping"], (0, 0)]
        return output


class LengthFilteringTokenizer(TrainerStyleTokenizer):
    def __init__(self):
        super().__init__()
        self.return_attention_mask_values: list[bool] = []

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True, return_offsets_mapping=False):
        self.return_attention_mask_values.append(return_attention_mask)
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            return_offsets_mapping=return_offsets_mapping,
        )


class QwenLikeOffsetTokenizer(OffsetCountingTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        enable_thinking=True,
        **kwargs,
    ):
        self.render_count += 1
        if kwargs:
            raise AssertionError(f"Unexpected chat template kwargs: {kwargs}")
        parts: list[str] = []
        if tools:
            tool_names = ",".join(tool["function"]["name"] for tool in tools)
            parts.append(f"<|im_start|>system\n<tools>{tool_names}</tools><|im_end|>\n")
        for message in messages:
            role = message["role"]
            if role == "user":
                parts.append(f"<|im_start|>user\n{message.get('content', '')}<|im_end|>\n")
                continue
            if role == "assistant":
                reasoning = message.get("reasoning_content") or ""
                content = str(message.get("content") or "")
                segment = "<|im_start|>assistant\n"
                if enable_thinking:
                    segment += f"<think>\n{reasoning}\n</think>\n\n"
                segment += content
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    segment += "<tool_call>\n"
                    segment += f"<function={function['name']}>\n"
                    for argument_name, argument_value in function.get("arguments", {}).items():
                        segment += f"<parameter={argument_name}>\n{argument_value}\n</parameter>\n"
                    segment += "</function>\n</tool_call>"
                segment += "<|im_end|>\n"
                parts.append(segment)
                continue
            if role == "tool":
                parts.append(
                    "<|im_start|>user\n<tool_response>\n"
                    f"{message.get('content', '')}\n"
                    "</tool_response><|im_end|>\n"
                )
                continue
            raise AssertionError(f"Unexpected role: {role}")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if enable_thinking:
                parts.append("<think>\n")
        rendered = "".join(parts)
        if tokenize:
            return self(rendered)
        return rendered


class GemmaLikeOffsetTokenizer(OffsetCountingTokenizer):
    chat_template = "<|turn>model\n<|tool_response><tool_response|>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        enable_thinking=True,
        **kwargs,
    ):
        self.render_count += 1
        if kwargs:
            raise AssertionError(f"Unexpected chat template kwargs: {kwargs}")
        parts: list[str] = ["<bos>"]
        if tools:
            parts.append("<|turn>system\n")
            for tool in tools:
                parts.append(f"<|tool>{tool['function']['name']}<tool|>")
            parts.append("<turn|>\n")
        for message in messages:
            role = message["role"]
            if role == "system":
                parts.append(f"<|turn>system\n{message.get('content', '')}<turn|>\n")
                continue
            if role == "user":
                parts.append(f"<|turn>user\n{message.get('content', '')}<turn|>\n")
                continue
            if role == "assistant":
                parts.append("<|turn>model\n")
                reasoning = message.get("reasoning_content") or ""
                if enable_thinking and reasoning:
                    parts.append(f"<|channel>thought\n{reasoning}\n<channel|>")
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    arguments = function.get("arguments") or {}
                    argument_parts = [f'{name}:"{value}"' for name, value in arguments.items()]
                    parts.append(
                        f"<|tool_call>call:{function['name']}{{{','.join(argument_parts)}}}<tool_call|>"
                    )
                content = str(message.get("content") or "")
                if content:
                    parts.append(content)
                if not message.get("tool_calls"):
                    parts.append("<turn|>\n")
                continue
            if role == "tool":
                parts.append(
                    f"<|tool_response>response:{message.get('name', 'unknown')}{{value:\"{message.get('content', '')}\"}}<tool_response|>"
                )
                continue
            raise AssertionError(f"Unexpected role: {role}")
        if add_generation_prompt:
            parts.append("<|turn>model\n")
            if not enable_thinking:
                parts.append("<|channel>thought\n<channel|>")
        rendered = "".join(parts)
        if tokenize:
            return self(rendered)
        return rendered


def test_reordered_mapping_markers_preserve_typed_tool_spans():
    class SortedToolArgumentTokenizer(OffsetCountingTokenizer):
        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
            parts: list[str] = []
            for index, message in enumerate(messages):
                role = message["role"]
                if role == "tool":
                    if index > 0:
                        previous = messages[index - 1]
                        tool_name = previous["tool_calls"][0]["function"]["name"]
                    else:
                        tool_name = message.get("name", "tool")
                    parts.append(
                        "<|turn>user\n"
                        f"<|tool_response>{tool_name}:{message.get('content', '')}<tool_response|>"
                        "<turn|>\n"
                    )
                    continue
                turn_role = "model" if role == "assistant" else role
                parts.append(f"<|turn>{turn_role}\n")
                if role == "assistant":
                    for tool_call in message.get("tool_calls") or []:
                        function = tool_call["function"]
                        rendered_arguments = "".join(
                            f"{name}:{function['arguments'][name]};" for name in sorted(function["arguments"])
                        )
                        parts.append(f"<|tool_call>call:{function['name']}{{{rendered_arguments}}}<tool_call|>")
                parts.append(str(message.get("content") or ""))
                parts.append("<turn|>\n")
            if add_generation_prompt:
                parts.append("<|turn>model\n")
            rendered = "".join(parts)
            if tokenize:
                return self(rendered)
            return rendered

    tokenizer = SortedToolArgumentTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "update task"},
                    {
                        "role": "assistant",
                        "content": "done final text",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "TaskCreate",
                                    "arguments": {
                                        "description": "Create studio backend",
                                        "activeForm": "Building studio backend",
                                    },
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "TaskCreate", "content": "SECRET_TOOL_OUTPUT"},
                ],
                "tools": [],
            }
        ]
    )

    prepared = prepare_data(dataset, tokenizer, tokenize=True, strict=True, verbose=False)
    spans = prepared[0]["teich_supervised_spans"]
    assert any(span.get("kind") == "tool_call" for span in spans)
    assert any(span.get("kind") == "final_answer" for span in spans)

    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        args=SimpleNamespace(dataset_text_field="text", max_length=None, packing=False),
        tokenizer=tokenizer,
    )
    trainer = mask_data(
        trainer,
        tokenizer=tokenizer,
        train_on_reasoning=False,
        train_on_final_answers=False,
        train_on_tools=True,
        audit=False,
        verbose=False,
    )

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])

    assert "<|tool_call>call:TaskCreate" in supervised_text
    assert "activeForm:Building studio backend" in supervised_text
    assert "description:Create studio backend" in supervised_text
    assert "done final text" not in supervised_text
    assert "SECRET_TOOL_OUTPUT" not in supervised_text
    assert "done final text" in masked_text
    assert "SECRET_TOOL_OUTPUT" in masked_text


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class FastMaskTokenizer(FakeTokenizer):
    chat_template = "{% generation %}{{ message }}{% endgeneration %}"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )
        if not tokenize:
            return rendered
        encoded = self(rendered)
        assistant_masks = [0] * len(encoded["input_ids"])
        cursor = 0
        if tools:
            tool_names = ",".join(tool["function"]["name"] for tool in tools)
            cursor += len(f"<tools>{tool_names}</tools>")
        for message in messages:
            segment = super().apply_chat_template([message], tokenize=False, tools=[])
            next_cursor = cursor + len(segment)
            if message["role"] == "assistant":
                for index in range(cursor, next_cursor):
                    assistant_masks[index] = 1
            cursor = next_cursor
        if return_dict:
            output = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            if return_assistant_tokens_mask:
                output["assistant_masks"] = assistant_masks
            return output
        return encoded["input_ids"]


class NonPrefixStableTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        if kwargs:
            raise AssertionError(f"Unexpected chat template kwargs: {kwargs}")
        tool_prefix = ""
        if tools:
            tool_names = ",".join(tool["function"]["name"] for tool in tools)
            tool_prefix = f"<tools>{tool_names}</tools>"
        parts: list[str] = [tool_prefix]
        for index, message in enumerate(messages):
            role = message["role"]
            segment = f"<{role}>"
            if role == "assistant":
                if message.get("reasoning_content"):
                    segment += f"<think>{message['reasoning_content']}</think>"
                tool_calls = message.get("tool_calls") or []
                for tool_call in tool_calls:
                    segment += f"<tool_call>{tool_call['function']['name']}</tool_call>"
                if message.get("content"):
                    segment += str(message["content"])
                next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
                if next_role is not None:
                    segment += f"</{role}>"
            else:
                if message.get("content"):
                    segment += str(message["content"])
                segment += f"</{role}>"
            parts.append(segment)
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "".join(parts)
        if tokenize:
            return self(rendered)
        return rendered


class MarkerSensitiveOffsetTokenizer(OffsetCountingTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )
        if "\ue000AGD" in rendered:
            rendered += "<marker-side-effect>"
        if tokenize:
            return self(rendered)
        return rendered


class EmbeddedToolResponseFallbackTokenizer(OffsetCountingTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        rendered_parts = []
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "user":
                rendered_parts.append(f"<user>{message.get('content', '')}</user>")
                continue
            if role == "tool":
                continue
            if role == "assistant":
                rendered_parts.append("<assistant>")
                for tool_call in message.get("tool_calls") or []:
                    rendered_parts.append(f"<tool_call>{tool_call['function']['name']}</tool_call>")
                if index + 1 < len(messages) and messages[index + 1].get("role") == "tool":
                    rendered_parts.append(f"<tool_response>{messages[index + 1].get('content', '')}</tool_response>")
                if message.get("content"):
                    rendered_parts.append(str(message["content"]))
                rendered_parts.append("</assistant>")
                continue
            raise AssertionError(f"Unexpected role: {role}")
        if add_generation_prompt:
            rendered_parts.append("<assistant>")
        rendered = "".join(rendered_parts)
        if "\ue000AGD" in rendered:
            rendered += "<marker-side-effect>"
        if tokenize:
            return self(rendered)
        return rendered


def test_prepare_and_mask_supervises_only_assistant_turns_across_multi_turn_conversation():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "user", "content": "summarize findings"},
                    {"role": "assistant", "content": "Found one Python file."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, include_debug_columns=True)

    assert training_data.num_rows == 1
    row = training_data[0]
    rendered = row["text"]
    assert "<tools>bash</tools>" in rendered
    assert "<system>system rules</system>" in rendered
    assert "<user>first request</user>" in rendered
    assert "<assistant><think>inspect repo</think><tool_call>bash</tool_call></assistant>" in rendered
    assert "<tool>file_a.py</tool>" in rendered
    assert "<assistant>Found one Python file.</assistant>" in rendered

    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert "<think>inspect repo</think><tool_call>bash</tool_call></assistant>" in supervised_text
    assert "Found one Python file.</assistant>" in supervised_text

    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<system>system rules</system>" in masked_text
    assert "<user>first request</user>" in masked_text


def test_prepare_and_mask_returns_compact_training_columns_by_default():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    row = training_data[0]
    assert set(row.keys()) == {"input_ids", "labels"}
    assert len(row["input_ids"]) == len(row["labels"])


def test_prepare_and_mask_accepts_multiple_datasets_and_concatenates_them():
    tokenizer = FakeTokenizer()
    tool_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "use tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
                "metadata": {"trace_type": "codex"},
            }
        ]
    )
    chat_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "be friendly"},
                ],
                "tools": [],
                "metadata": {"trace_type": "chat"},
            }
        ]
    )

    training_data = prepare_and_mask_for_test([tool_dataset, chat_dataset], tokenizer, include_debug_columns=True)

    assert training_data.num_rows == 2
    assert "<tool_call>bash</tool_call>" in training_data[0]["text"]
    assert "<assistant><think>be friendly</think>world</assistant>" in training_data[1]["text"]
    preview = training_data.preview(1)
    assert "\033[31m" in preview
    assert "<user>hello</user>" in preview
    assert "<think>be friendly</think>world</assistant>" in preview


def test_prepare_and_mask_passes_chat_template_kwargs_and_preview_marks_unsupervised_text_red():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "hidden"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": False},
        include_debug_columns=True,
    )

    row = training_data[0]
    assert row["text"] == "<user>hello</user><assistant>world</assistant>"
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "world</assistant>"

    preview = training_data.preview()
    assert "\033[31m" in preview
    assert "<user>hello</user>" in preview
    assert "world</assistant>" in preview


def test_prepare_and_mask_can_exclude_reasoning_from_qwen_style_supervision():
    tokenizer = QwenLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True},
        train_on_reasoning=False,
        include_debug_columns=True,
    )

    row = training_data[0]
    assert row["text"] == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\nthink\n</think>\n\nworld<|im_end|>\n"
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "world<|im_end|>\n"
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "think\n</think>\n\n" in masked_text
    assert "<|im_start|>assistant\n<think>\n" in masked_text


def test_prepare_and_mask_supervises_qwen_reasoning_start_tag_without_assistant_header():
    tokenizer = QwenLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True},
        train_on_reasoning=True,
        include_debug_columns=True,
    )

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>\nthink\n</think>\n\nworld<|im_end|>\n"
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|im_start|>assistant\n" in masked_text
    assert "<think>\n" not in masked_text


def test_prepare_and_mask_can_exclude_reasoning_from_gemma_style_supervision():
    tokenizer = GemmaLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        train_on_reasoning=False,
        include_debug_columns=True,
    )

    row = training_data[0]
    assert row["text"] == "<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\nthink\n<channel|>world<turn|>\n"
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "world"
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|channel>thought\nthink\n<channel|>" in masked_text


def test_prepare_and_mask_falls_back_when_gemma_drops_marker_boundaries_with_thinking_disabled():
    class MarkerDroppingGemmaTokenizer(GemmaLikeOffsetTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            rendered = super().apply_chat_template(*args, **kwargs)
            if kwargs.get("enable_thinking") is False and isinstance(rendered, str) and "\ue000AGD" in rendered:
                for index in range(8):
                    rendered = rendered.replace(f"\ue000AGD{index}E\ue001", "")
            return rendered

    tokenizer = MarkerDroppingGemmaTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        train_on_reasoning=False,
    )

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "world"


def test_prepare_data_rejects_reserved_chat_template_kwargs():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list([{"messages": [], "tools": []}])

    try:
        prepare_and_mask_for_test(dataset, tokenizer, chat_template_kwargs={"tools": []})
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("Expected prepare_data to reject reserved chat_template_kwargs")


def test_prepare_data_renders_text_and_supervised_spans_for_trainer_flow():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    prepared = prepare_data(dataset, tokenizer, verbose=False)

    assert prepared.column_names == ["text", "teich_supervised_spans"]
    assert prepared[0]["text"] == "<user>hello</user><assistant><think>think</think>world</assistant>"
    spans = prepared[0]["teich_supervised_spans"]
    assert {span.get("kind") for span in spans} == {"user", "reasoning", "final_answer"}
    source_text_by_kind = {
        span["kind"]: prepared[0]["text"][span["source_start"] : span["source_end"]]
        for span in spans
    }
    assert source_text_by_kind["user"] == "hello"
    assert source_text_by_kind["reasoning"] == "think"
    assert source_text_by_kind["final_answer"] == "world"


def test_row_fits_context_returns_bool_or_details_for_structured_rows():
    tokenizer = TrainerStyleTokenizer()
    row = {
        "id": "fit-row",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        "tools": [],
    }

    details = row_fits_context(row, tokenizer, 100, return_details=True)

    assert details.fits is True
    assert details.row_id == "fit-row"
    assert details.token_length == len("<user>hello</user><assistant>world</assistant>")
    assert row_fits_context(row, tokenizer, 10) is False


def test_mask_data_applies_policy_flags_to_typed_prepare_spans():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "run command"},
                    {
                        "role": "assistant",
                        "content": "final answer",
                        "reasoning_content": "private reasoning",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "tool output"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)

    def supervised_text(**mask_kwargs):
        trainer = SimpleNamespace(
            train_dataset=prepared,
            eval_dataset=None,
            tokenizer=tokenizer,
            args=SimpleNamespace(dataset_text_field="text", packing=False),
        )
        trainer = mask_data(trainer, tokenizer=tokenizer, audit=False, verbose=False, **mask_kwargs)
        row = trainer.train_dataset[0]
        return tokenizer.decode([token for token in row["labels"] if token != -100])

    default_text = supervised_text()
    assert "private reasoning" in default_text
    assert "<tool_call>bash</tool_call>" in default_text
    assert "final answer" in default_text
    assert "system rules" not in default_text
    assert "run command" not in default_text
    assert "tool output" not in default_text

    no_reasoning = supervised_text(train_on_reasoning=False)
    assert "private reasoning" not in no_reasoning
    assert "<tool_call>bash</tool_call>" in no_reasoning
    assert "final answer" in no_reasoning

    no_tools = supervised_text(train_on_tools=False)
    assert "<tool_call>bash</tool_call>" not in no_tools
    assert "private reasoning" in no_tools
    assert "final answer" in no_tools

    no_final = supervised_text(train_on_final_answers=False)
    assert "final answer" not in no_final
    assert "private reasoning" in no_final
    assert "<tool_call>bash</tool_call>" in no_final

    non_model_text = supervised_text(
        train_on_reasoning=False,
        train_on_final_answers=False,
        train_on_tools=False,
        train_on_user=True,
        train_on_system=True,
        train_on_tool_responses=True,
    )
    assert "system rules" in non_model_text
    assert "run command" in non_model_text
    assert "tool output" in non_model_text
    assert "private reasoning" not in non_model_text
    assert "final answer" not in non_model_text
    assert "<tool_call>bash</tool_call>" not in non_model_text


def test_converted_trace_prepares_and_masks_reasoning_text_tool_order(tmp_path: Path):
    trace_file = tmp_path / "codex-order.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Create a file"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": '{"cmd":"touch app.py"}',
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need to create the file first."}],
            },
        },
        {"type": "event_msg", "payload": {"type": "agent_message"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I'll create it now."}],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    example = convert_trace_to_training_example(trace_file).to_dict()

    class OrderedToolTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=False,
            add_generation_prompt=False,
            tools=None,
            enable_thinking=True,
            preserve_thinking=True,
            **kwargs,
        ):
            parts: list[str] = []
            for message in messages:
                segment = f"<{message['role']}>"
                if message["role"] == "assistant" and enable_thinking and preserve_thinking:
                    reasoning = message.get("reasoning_content")
                    if reasoning:
                        segment += f"<think>{reasoning}</think>"
                if message.get("content"):
                    segment += str(message["content"])
                if message["role"] == "assistant":
                    for tool_call in message.get("tool_calls") or []:
                        segment += f"<tool_call>{tool_call['function']['name']}</tool_call>"
                segment += f"</{message['role']}>"
                parts.append(segment)
            if add_generation_prompt:
                parts.append("<assistant>")
            rendered = "".join(parts)
            if tokenize:
                return self(rendered)
            return rendered

    tokenizer = OrderedToolTokenizer()

    training_data = prepare_and_mask_for_test(
        Dataset.from_list([example]),
        tokenizer,
        include_debug_columns=True,
    )
    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])

    assert (
        row["text"].index("Need to create the file first.")
        < row["text"].index("I'll create it now.")
        < row["text"].index("<tool_call>exec_command</tool_call>")
    )
    assert (
        supervised_text.index("Need to create the file first.")
        < supervised_text.index("I'll create it now.")
        < supervised_text.index("<tool_call>exec_command</tool_call>")
    )


def test_prepare_data_can_return_plain_rendered_text_without_teich_masking():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "question without assistant"},
                ],
                "tools": [],
            },
        ]
    )

    prepared = prepare_data(dataset, tokenizer, teich_masking=False, verbose=False)

    assert prepared.column_names == ["text"]
    assert prepared.num_rows == 2
    assert prepared[0]["text"] == "<system>system rules</system><user>hello</user><assistant><think>think</think>world</assistant>"
    assert prepared[1]["text"] == "<user>question without assistant</user>"


def test_prepare_data_plain_text_mode_can_still_tokenize_and_filter_by_length():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            },
        ]
    )

    prepared = prepare_data(
        dataset,
        tokenizer,
        teich_masking=False,
        tokenize=True,
        max_length=60,
        verbose=False,
    )

    assert prepared.column_names == ["text", "input_ids", "attention_mask"]
    assert prepared.num_rows == 1
    assert prepared[0]["text"] == "<user>hello</user><assistant>world</assistant>"
    assert prepared[0]["input_ids"]
    assert prepared[0]["attention_mask"] == [1] * len(prepared[0]["input_ids"])


def test_prepare_data_can_emit_tokenized_rows_for_trainer_skip_prepare_flow():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    prepared = prepare_data(dataset, tokenizer, tokenize=True, verbose=False)
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    assert set(prepared.column_names) == {"text", "teich_supervised_spans", "input_ids", "attention_mask"}
    row = trainer.train_dataset[0]
    assert set(trainer.train_dataset.column_names) == {"input_ids", "labels"}
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>think</think>world</assistant>"


def test_mask_data_can_use_decoded_token_offsets_when_tokenizer_lacks_offsets():
    tokenizer = NoOffsetTrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, tokenize=True, verbose=False)
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True, verbose=False)

    supervised_text = tokenizer.decode([token for token in trainer.train_dataset[0]["labels"] if token != -100])
    assert supervised_text == "world</assistant>"


def test_tokenized_prepare_data_survives_unsloth_style_skip_prepare_columns():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    prepared = prepare_data(dataset, tokenizer, tokenize=True, verbose=False)
    # Unsloth skips text tokenization when input_ids already exist. The important
    # contract is that setup must not go through the text-tokenize/remove-columns path.
    trainer_dataset = prepared
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>think</think>world</assistant>"


def test_text_only_trainer_column_drop_loses_teich_spans_and_uses_fallback():
    class MarkerlessTokenizer(TrainerStyleTokenizer):
        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
            rendered = "|".join(str(message.get("content") or "") for message in messages)
            if tokenize:
                return self(rendered)
            return rendered

    tokenizer = MarkerlessTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    # Simulate the destructive standard text tokenization path used by optimized
    # trainers: only input_ids survive, so Teich's exact spans are gone.
    trainer_dataset = prepared.map(
        lambda row: {"input_ids": tokenizer(text=row["text"])["input_ids"]},
        remove_columns=prepared.column_names,
    )
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    with pytest.raises(ValueError, match="fully masked"):
        mask_data(trainer, audit=False)


def test_prepare_data_filters_oversized_rows_without_returning_tokens():
    tokenizer = LengthFilteringTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "ok"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            },
        ]
    )

    prepared = prepare_data(dataset, tokenizer, max_length=60, verbose=False)

    assert prepared.num_rows == 1
    assert prepared.column_names == ["text", "teich_supervised_spans"]
    assert "input_ids" not in prepared.column_names
    assert "attention_mask" not in prepared.column_names
    assert "labels" not in prepared.column_names
    assert "ok</assistant>" in prepared[0]["text"]
    assert tokenizer.return_attention_mask_values
    assert set(tokenizer.return_attention_mask_values) == {False}


def test_prepare_data_filters_oversized_rows_with_tokenized_mode():
    tokenizer = LengthFilteringTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "ok"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            },
        ]
    )

    prepared = prepare_data(dataset, tokenizer, max_length=60, tokenize=True, verbose=False)

    assert prepared.num_rows == 1
    assert set(prepared.column_names) == {"text", "teich_supervised_spans", "input_ids", "attention_mask"}
    assert "ok</assistant>" in prepared[0]["text"]


def test_prepare_data_trims_oversized_followups_before_dropping():
    tokenizer = LengthFilteringTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "follow up " + ("x" * 80)},
                    {"role": "assistant", "content": "later " + ("y" * 80)},
                ],
                "tools": [],
            },
        ]
    )

    prepared = prepare_data(
        dataset,
        tokenizer,
        max_length=60,
        trim_oversized_followups=True,
        tokenize=True,
        verbose=False,
    )

    assert prepared.num_rows == 1
    assert "ok</assistant>" in prepared[0]["text"]
    assert "follow up" not in prepared[0]["text"]
    assert "later" not in prepared[0]["text"]
    assert len(prepared[0]["input_ids"]) <= 60


def test_prepare_data_only_trims_rows_with_multiple_user_turns():
    tokenizer = LengthFilteringTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "single"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            },
        ]
    )

    with pytest.raises(ValueError, match="context window"):
        prepare_data(
            dataset,
            tokenizer,
            max_length=60,
            trim_oversized_followups=True,
            verbose=False,
        )


def test_prepare_data_keeps_oversized_rows_when_drop_is_disabled_even_with_trim_enabled():
    tokenizer = LengthFilteringTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "follow up"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            },
        ]
    )

    prepared = prepare_data(
        dataset,
        tokenizer,
        max_length=60,
        drop_oversized_examples=False,
        trim_oversized_followups=True,
        verbose=False,
    )

    assert prepared.num_rows == 1
    assert "follow up" in prepared[0]["text"]


def test_prepare_data_can_keep_oversized_rows_for_trainer_truncation():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            }
        ]
    )

    prepared = prepare_data(
        dataset,
        tokenizer,
        max_length=40,
        drop_oversized_examples=False,
        verbose=False,
    )

    assert prepared.num_rows == 1
    assert prepared.column_names == ["text", "teich_supervised_spans"]


def test_prepare_data_accepts_mixed_sources_and_concatenates_chat_and_tool_rows(tmp_path):
    tokenizer = FakeTokenizer()
    chat_file = tmp_path / "chat.jsonl"
    chat_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "friendly"},
                ],
                "prompt": "hello",
                "response": "world",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tool_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "check files",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    prepared = prepare_data([chat_file, tool_dataset], tokenizer, split=None, verbose=False)

    assert prepared.num_rows == 2
    assert prepared.column_names == ["text", "teich_supervised_spans"]
    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert any("<assistant><think>friendly</think>world</assistant>" in text for text in texts)
    assert any("<tools>bash</tools>" in text and "<tool_call>bash</tool_call>" in text for text in texts)


def test_mask_data_applies_teich_labels_after_trainer_tokenization():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    trainer_dataset = prepared.map(lambda row: {"input_ids": tokenizer(text=row["text"])["input_ids"]})
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    row = trainer.train_dataset[0]
    assert set(trainer.train_dataset.column_names) == {"input_ids", "labels"}
    preview = trainer.train_dataset.preview()
    assert "\033[31m" in preview
    assert "<system>system rules</system>" in preview
    assert "<think>inspect repo</think><tool_call>bash</tool_call></assistant>" in preview
    assert "done</assistant>" in preview
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>inspect repo</think><tool_call>bash</tool_call></assistant>done</assistant>"
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert "<system>system rules</system>" in masked_text
    assert "<user>first request</user>" in masked_text
    assert "<tool>file_a.py</tool>" in masked_text


def test_mask_data_aligns_trainer_added_special_tokens_to_teich_spans():
    tokenizer = SpecialTokenWrappingTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    trainer_dataset = prepared.map(lambda row: {"input_ids": tokenizer(text=row["text"])["input_ids"]})
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert supervised_text == "<think>think</think>world</assistant>"
    assert "<bos>" in masked_text
    assert "<eos>" in masked_text


def test_preview_escapes_embedded_ansi_without_breaking_mask_colors():
    tokenizer = FakeTokenizer()
    text = "<user>\x1b[31;1mRED\x1b[0m</user><assistant>OK</assistant>"
    input_ids = tokenizer(text)["input_ids"]
    assistant_start = text.index("<assistant>")
    labels = [-100 if index < assistant_start else token_id for index, token_id in enumerate(input_ids)]
    dataset = Dataset.from_list([{"input_ids": input_ids, "labels": labels}])

    preview = preview_sft_example(dataset, tokenizer)

    assert preview == "\x1b[31m<user>\\x1b[31;1mRED\\x1b[0m</user>\x1b[0m<assistant>OK</assistant>"
    assert preview.count("\x1b[0m") == 1
    assert "\x1b[31;1m" not in preview


def test_teich_tokenization_does_not_add_special_tokens_and_masks_trainer_specials():
    tokenizer = SpecialTokenWrappingTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    prepare_data(dataset, tokenizer, tokenize=True, verbose=False)
    assert tokenizer.add_special_tokens_calls
    assert all(call is False for call in tokenizer.add_special_tokens_calls)

    tokenizer.add_special_tokens_calls.clear()
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    trainer_dataset = prepared.map(lambda row: {"input_ids": tokenizer(text=row["text"])["input_ids"]})
    assert any(call is True for call in tokenizer.add_special_tokens_calls)

    tokenizer.add_special_tokens_calls.clear()
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True, verbose=False)

    row = trainer.train_dataset[0]
    assert all(call is False for call in tokenizer.add_special_tokens_calls)
    assert row["input_ids"][0] == tokenizer._vocab[tokenizer.bos_token]
    assert row["input_ids"][-1] == tokenizer._vocab[tokenizer.eos_token]
    assert row["labels"][0] == -100
    assert row["labels"][-1] == -100
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>think</think>world</assistant>"


def test_mask_data_tokenizes_prepared_text_when_trainer_has_not_tokenized():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    row = trainer.train_dataset[0]
    assert set(trainer.train_dataset.column_names) == {"input_ids", "labels"}
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>inspect repo</think><tool_call>bash</tool_call></assistant>done</assistant>"


def test_mask_data_wraps_standard_transformers_collator_to_pad_labels():
    class DataCollatorWithPadding:
        def __init__(self):
            self.seen_labels = False

        def __call__(self, features):
            self.seen_labels = any("labels" in feature for feature in features)
            max_length = max(len(feature["input_ids"]) for feature in features)
            return {
                "input_ids": [
                    feature["input_ids"] + [0] * (max_length - len(feature["input_ids"]))
                    for feature in features
                ]
            }

    DataCollatorWithPadding.__module__ = "transformers.data.data_collator"

    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}], "tools": []},
            {
                "messages": [
                    {"role": "user", "content": "longer"},
                    {"role": "assistant", "content": "a much longer answer"},
                ],
                "tools": [],
            },
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    collator = DataCollatorWithPadding()
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=collator,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)
    batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[1]])

    assert trainer.data_collator is not collator
    assert collator.seen_labels is False
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert len(batch["labels"][0]) == len(batch["labels"][1]) == len(batch["input_ids"][0])
    assert batch["labels"][0][-1] == -100
    assert any(label != -100 for label in batch["labels"][0])
    assert any(label != -100 for label in batch["labels"][1])


def test_mask_data_does_not_wrap_custom_collator():
    class CustomCollator:
        def __call__(self, features):
            return {"features": features}

    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], "tools": []}
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    collator = CustomCollator()
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=collator,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    assert trainer.data_collator is collator


def test_mask_data_can_drop_rows_with_too_many_supervised_tokens():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": "ok"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "long"},
                    {"role": "assistant", "content": "x" * 100},
                ],
                "tools": [],
            },
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    trainer_dataset = prepared.map(lambda row: {"input_ids": tokenizer(text=row["text"])["input_ids"]})
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=30, truncation_mode="keep_end"),
    )

    trainer = mask_data(trainer, audit=True, verbose=False)

    assert trainer.train_dataset.num_rows == 1
    assert "ok</assistant>" in trainer.train_dataset.preview()
    assert "x" * 100 not in trainer.train_dataset.preview()


def test_mask_data_explicit_supervised_token_limit_overrides_trainer_max_length():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "long"},
                    {"role": "assistant", "content": "x" * 20},
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, verbose=False)
    trainer_dataset = prepared.map(lambda row: {"input_ids": tokenizer(text=row["text"])["input_ids"]})
    trainer = SimpleNamespace(
        train_dataset=trainer_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=5, truncation_mode="keep_end"),
    )

    trainer = mask_data(trainer, max_supervised_tokens=50, audit=True, verbose=False)

    assert trainer.train_dataset.num_rows == 1


def test_mask_data_applies_trainer_keep_end_truncation_before_audit():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "x" * 80},
                    {"role": "assistant", "content": "final answer"},
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, drop_oversized_examples=False, verbose=False)
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=40, truncation_mode="keep_end"),
    )

    trainer = mask_data(trainer, audit=True, verbose=False)

    row = trainer.train_dataset[0]
    assert len(row["input_ids"]) == 40
    assert len(row["labels"]) == 40
    assert "final answer</assistant>" in trainer.train_dataset.preview()


def test_mask_data_rejects_truncation_that_removes_all_supervised_tokens():
    tokenizer = TrainerStyleTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "x" * 80},
                    {"role": "assistant", "content": "final answer"},
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(dataset, tokenizer, drop_oversized_examples=False, verbose=False)
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=40),
    )

    with pytest.raises(ValueError, match="trainer max_length truncation"):
        mask_data(trainer, audit=False, verbose=False)


def test_mask_data_can_fallback_when_trainer_drops_text_columns():
    tokenizer = QwenLikeOffsetTokenizer()
    rendered = (
        "<|im_start|>user\nfirst request<|im_end|>\n"
        "<|im_start|>assistant\n<think>\ninspect repo\n</think>\n\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nfile_a.py\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\nfinal answer<|im_end|>\n"
    )
    trainer = SimpleNamespace(
        train_dataset=Dataset.from_list([{"input_ids": tokenizer(text=rendered)["input_ids"]}]),
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True)

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert "<think>\ninspect repo\n</think>" in supervised_text
    assert "<tool_call>" in supervised_text
    assert "final answer<|im_end|>" in supervised_text
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert "<|im_start|>user\nfirst request" in masked_text
    assert "<tool_response>" in masked_text


def test_mask_data_fallback_masks_attributed_gemma_tool_responses():
    tokenizer = FakeTokenizer()
    rendered = (
        "<bos><|turn>user\nrun tests<turn|>\n"
        "<|turn>model\n"
        '<|tool_call>call:Bash{command:"pytest"}<tool_call|>'
        "<tool_response name='Bash'>ERROR tests/test_checker.py</tool_response>"
        "done<turn|>\n"
    )
    trainer = SimpleNamespace(
        train_dataset=Dataset.from_list([{"input_ids": tokenizer(text=rendered)["input_ids"]}]),
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False),
    )

    trainer = mask_data(trainer, audit=True, verbose=False)

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert '<|tool_call>call:Bash{command:"pytest"}<tool_call|>' in supervised_text
    assert "done" in supervised_text
    assert "ERROR tests/test_checker.py" not in supervised_text
    assert "<tool_response name='Bash'>" not in supervised_text
    assert "ERROR tests/test_checker.py" in masked_text


def test_mask_data_rejects_packing_because_row_boundaries_are_required():
    tokenizer = TrainerStyleTokenizer()
    trainer = SimpleNamespace(
        train_dataset=Dataset.from_list([]),
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=True),
    )

    with pytest.raises(ValueError, match="packed"):
        mask_data(trainer, audit=False)


def test_prepare_and_mask_supports_processor_objects_with_nested_text_tokenizer():
    processor = FakeProcessor()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, processor, include_debug_columns=True)

    row = training_data[0]
    assert row["text"] == "<user>hello</user><assistant><think>think</think>world</assistant>"
    supervised_text = processor.tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>think</think>world</assistant>"
    assert "\033[31m" in training_data.preview()


def test_prepare_and_mask_uses_fast_assistant_mask_path_when_supported():
    tokenizer = FastMaskTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, include_debug_columns=True)

    row = training_data[0]
    assert len(row["assistant_masks"]) == len(row["input_ids"])
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>think</think>world</assistant>"


def test_prepare_and_mask_handles_non_prefix_stable_templates_around_tool_turns():
    tokenizer = NonPrefixStableTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "think",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert "<think>think</think><tool_call>bash</tool_call>" in supervised_text
    assert "done" in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<tool>file_a.py</tool>" in masked_text


def test_prepare_and_mask_skips_unrenderable_prefixes_before_first_user_message():
    tokenizer = RequiresUserTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>think</think>world</assistant>"
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<system>system rules</system>" in masked_text
    assert "<user>hello</user>" in masked_text


def test_prepare_and_mask_renders_only_supervision_checkpoints_in_fallback():
    tokenizer = CountingTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "first request"},
                    {"role": "assistant", "content": "draft answer", "reasoning_content": "think"},
                    {"role": "tool", "content": "tool output"},
                    {"role": "user", "content": "follow up"},
                    {"role": "assistant", "content": "final answer"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    assert training_data.num_rows == 1
    assert tokenizer.render_count == 4


def test_prepare_and_mask_uses_single_render_offset_mask_path_when_offsets_are_available():
    tokenizer = OffsetCountingTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>inspect repo</think><tool_call>bash</tool_call></assistant>done</assistant>"
    assert "<tool_call>" in supervised_text
    assert "</think>" in supervised_text
    assert "<assistant>" not in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<system>system rules</system>" in masked_text
    assert "<user>first request</user>" in masked_text
    assert "<tool>file_a.py</tool>" in masked_text
    assert tokenizer.render_count == 6


def test_prepare_and_mask_masks_qwen_assistant_header_but_supervises_reasoning_start_tag():
    tokenizer = QwenLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, chat_template_kwargs={"enable_thinking": True})

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text.startswith("<think>\ninspect repo\n</think>\n\n<tool_call>\n<function=bash>")
    assert "<|im_start|>assistant\n" not in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|im_start|>assistant\n" in masked_text
    assert "<think>\n" not in masked_text
    assert "first request" in masked_text
    assert tokenizer.render_count == 4


def test_prepare_and_mask_masks_empty_qwen_thinking_after_tool_response():
    tokenizer = QwenLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "make screenshot"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "first call",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "PowerShell", "arguments": {"command": "make png"}},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "PowerShell",
                        "content": (
                            "Exit code 1\n"
                            "Get-Item: Cannot find path "
                            "'C:\\Users\\user1\\AppData\\Local\\Temp\\earth_test.png' "
                            "because it does not exist."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "PowerShell", "arguments": {"command": "retry"}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "PowerShell",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, chat_template_kwargs={"enable_thinking": True})

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    masked_section = (
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>"
    )
    assert masked_section in masked_text
    assert masked_section not in supervised_text
    assert "<tool_call>\n<function=PowerShell>\n<parameter=command>\nretry" in supervised_text
    assert "<think>\n\n</think>" not in supervised_text


def test_prepare_and_mask_uses_gemma_structured_mask_path():
    tokenizer = GemmaLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert "<|channel>thought\ninspect repo\n<channel|>" in supervised_text
    assert '<|tool_call>call:bash{command:"ls"}<tool_call|>' in supervised_text
    assert "done" in supervised_text
    assert "response:bash" not in supervised_text
    assert "file_a.py" not in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "file_a.py" in masked_text
    assert tokenizer.render_count >= 1


class GraniteLikeOffsetTokenizer(OffsetCountingTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        self.render_count += 1
        if kwargs:
            raise AssertionError(f"Unexpected chat template kwargs: {kwargs}")
        parts: list[str] = []
        if tools:
            parts.append("<|start_of_role|>system<|end_of_role|>tools<|end_of_text|>\n")
        for message in messages:
            role = message["role"]
            if role == "user":
                parts.append(f"<|start_of_role|>user<|end_of_role|>{message.get('content', '')}<|end_of_text|>\n")
                continue
            if role == "assistant":
                parts.append("<|start_of_role|>assistant<|end_of_role|>")
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    parts.append('<tool_call>{"name": "')
                    parts.append(function["name"])
                    parts.append('", "arguments": {"command": "')
                    parts.append(function.get("arguments", {}).get("command", ""))
                    parts.append('"}}</tool_call>')
                if message.get("content"):
                    parts.append(str(message["content"]))
                parts.append("<|end_of_text|>\n")
                continue
            if role == "tool":
                parts.append(
                    f"<|start_of_role|>user<|end_of_role|><tool_response>{message.get('content', '')}</tool_response><|end_of_text|>\n"
                )
                continue
            raise AssertionError(f"Unexpected role: {role}")
        if add_generation_prompt:
            parts.append("<|start_of_role|>assistant<|end_of_role|>")
        rendered = "".join(parts)
        if tokenize:
            return self(rendered)
        return rendered


class QwenMismatchOffsetTokenizer(QwenLikeOffsetTokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt=False, enable_thinking=True, **kwargs):
        rendered = super().apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        if add_generation_prompt:
            return rendered
        return rendered.replace("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n")


def test_prepare_and_mask_expands_granite_style_assistant_blocks():
    tokenizer = GraniteLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "List files"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "dir"}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert '<tool_call>{"name": "bash", "arguments": {"command": "dir"}}</tool_call><|end_of_text|>\n' in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "List files" in masked_text


def test_prepare_and_mask_falls_back_to_assistant_header_when_qwen_prefix_probe_mismatches():
    tokenizer = QwenMismatchOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "first request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, chat_template_kwargs={"enable_thinking": True})

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text.startswith("inspect repo\n</think>\n\n<tool_call>\n<function=bash>")
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|im_start|>assistant\n" in masked_text


def test_prepare_and_mask_skips_rows_with_empty_message_lists():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {"messages": [], "tools": []},
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            },
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    assert training_data.num_rows == 1
    assert set(training_data[0].keys()) == {"input_ids", "labels"}


def test_prepare_and_mask_drops_oversized_examples_by_default():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "x" * 80},
                ],
                "tools": [],
            },
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, max_length=60)

    assert training_data.num_rows == 1
    assert len(training_data[0]["input_ids"]) < 60


def test_prepare_and_mask_truncates_oversized_examples_when_drop_is_disabled():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "x" * 80},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, max_length=60, drop_oversized_examples=False)

    assert training_data.num_rows == 1
    row = training_data[0]
    assert len(row["input_ids"]) > 60
    assert len(row["labels"]) > 60


def test_prepare_and_mask_does_not_use_tokenizer_model_max_length_without_explicit_max_length():
    tokenizer = LimitedFakeTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "x" * 80},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer)

    assert training_data.num_rows == 1


def test_prepare_and_mask_strict_drops_rows_with_no_trainable_spans_after_reasoning_exclusion():
    tokenizer = QwenLikeOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "think only"},
                    {"role": "assistant", "content": "", "reasoning_content": "private reasoning"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "answer"},
                    {"role": "assistant", "content": "visible answer", "reasoning_content": "private reasoning"},
                ],
                "tools": [],
            },
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True},
        train_on_reasoning=False,
        strict=True,
    )

    assert training_data.num_rows == 1
    supervised_text = tokenizer.decode([token for token in training_data[0]["labels"] if token != -100])
    assert "visible answer" in supervised_text
    assert "private reasoning" not in supervised_text


def test_prepare_and_mask_raises_when_all_rows_have_empty_message_lists():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {"messages": [], "tools": []},
            {"messages": [], "tools": []},
        ]
    )

    with pytest.raises(ValueError, match="missing|required|empty|no rows|no non-empty"):
        prepare_and_mask_for_test(dataset, tokenizer)


def test_prepare_and_mask_strict_rejects_marker_render_mismatch():
    tokenizer = MarkerSensitiveOffsetTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world", "reasoning_content": "think"},
                ],
                "tools": [],
            }
        ]
    )

    with pytest.raises(ValueError, match="Marker-injected chat template output"):
        prepare_and_mask_for_test(dataset, tokenizer, strict=True)


def test_prepare_and_mask_fallback_masks_embedded_tool_responses():
    tokenizer = EmbeddedToolResponseFallbackTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "SECRET_TOOL_OUTPUT"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(dataset, tokenizer, strict=False)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert "<tool_call>bash</tool_call>" in supervised_text
    assert "done</assistant>" in supervised_text
    assert "SECRET_TOOL_OUTPUT" not in supervised_text
    assert "SECRET_TOOL_OUTPUT" in masked_text


def _real_template_tool_call_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "list files"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "run shell",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        },
                    }
                ],
            }
        ]
    )


class RealJinjaChatTemplateTokenizer(OffsetCountingTokenizer):
    def __init__(self, template_path: Path, jinja2_module):
        super().__init__()
        self._template = jinja2_module.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
            template_path.read_text(encoding="utf-8")
        )

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        def raise_exception(message):
            raise ValueError(message)

        rendered = self._template.render(
            messages=messages,
            tools=tools or [],
            bos_token="<bos>",
            add_generation_prompt=add_generation_prompt,
            raise_exception=raise_exception,
            add_vision_id=False,
            **kwargs,
        )
        if tokenize:
            return self(rendered)
        return rendered


_REAL_TEMPLATE_COMPATIBILITY_CASES = [
    pytest.param(
        {
            "name": "qwen3.6-thinking-on",
            "template_path": "qwen3.6_chat_template.jinja",
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
            "train_on_reasoning": True,
            "expected_supervised_substrings": ["inspect repo", "<function=bash>", "ls", "done"],
            "forbidden_supervised_substrings": ["file_a.py", "<tool_response>"],
        },
        id="qwen3.6-thinking-on",
    ),
    pytest.param(
        {
            "name": "qwen3.6-thinking-off-no-reasoning-labels",
            "template_path": "qwen3.6_chat_template.jinja",
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
            "train_on_reasoning": False,
            "expected_supervised_substrings": ["<function=bash>", "ls", "done"],
            "forbidden_supervised_substrings": ["inspect repo", "<tool_response>", "file_a.py"],
        },
        id="qwen3.6-thinking-off-no-reasoning-labels",
    ),
    pytest.param(
        {
            "name": "qwen3.5-hyphen-thinking-on",
            "template_path": "qwen3.5-chat-template.jinja",
            "chat_template_kwargs": {"enable_thinking": True},
            "train_on_reasoning": True,
            "expected_supervised_substrings": ["inspect repo", "<function=bash>", "ls", "done"],
            "forbidden_supervised_substrings": ["file_a.py", "<tool_response>"],
        },
        id="qwen3.5-hyphen-thinking-on",
    ),
    pytest.param(
        {
            "name": "qwen3.5-hyphen-thinking-off-no-reasoning-labels",
            "template_path": "qwen3.5-chat-template.jinja",
            "chat_template_kwargs": {"enable_thinking": False},
            "train_on_reasoning": False,
            "expected_supervised_substrings": ["<function=bash>", "ls", "done"],
            "forbidden_supervised_substrings": ["inspect repo", "<tool_response>", "file_a.py"],
        },
        id="qwen3.5-hyphen-thinking-off-no-reasoning-labels",
    ),
    pytest.param(
        {
            "name": "qwen3.6-hyphen-thinking-on",
            "template_path": "qwen3.6-chat-template.jinja",
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
            "train_on_reasoning": True,
            "expected_supervised_substrings": ["inspect repo", "<function=bash>", "ls", "done"],
            "forbidden_supervised_substrings": ["file_a.py", "<tool_response>"],
        },
        id="qwen3.6-hyphen-thinking-on",
    ),
    pytest.param(
        {
            "name": "qwen3.6-hyphen-thinking-off-no-reasoning-labels",
            "template_path": "qwen3.6-chat-template.jinja",
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
            "train_on_reasoning": False,
            "expected_supervised_substrings": ["<function=bash>", "ls", "done"],
            "forbidden_supervised_substrings": ["inspect repo", "<tool_response>", "file_a.py"],
        },
        id="qwen3.6-hyphen-thinking-off-no-reasoning-labels",
    ),
    pytest.param(
        {
            "name": "gemma4-thinking-off-no-reasoning-labels",
            "template_path": "gemma-4-chat-template.jinja",
            "chat_template_kwargs": {"enable_thinking": False},
            "train_on_reasoning": False,
            "expected_supervised_substrings": ["<|tool_call>call:bash", '"command":', "ls", "done"],
            "forbidden_supervised_substrings": ["inspect repo", "response:bash", "file_a.py"],
        },
        id="gemma4-thinking-off-no-reasoning-labels",
    ),
    pytest.param(
        {
            "name": "new-gemma4-thinking-on-preserve",
            "template_path": "new_gemma_4_template.jinja",
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
            "train_on_reasoning": True,
            "expected_supervised_substrings": ["inspect repo", "<|tool_call>call:bash", "command:", "<|\"|>ls<|\"|>", "done"],
            "forbidden_supervised_substrings": ["response:bash", "file_a.py"],
        },
        id="new-gemma4-thinking-on-preserve",
    ),
    pytest.param(
        {
            "name": "new-gemma4-thinking-off-no-reasoning-labels",
            "template_path": "new_gemma_4_template.jinja",
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
            "train_on_reasoning": False,
            "expected_supervised_substrings": ["<|tool_call>call:bash", "command:", "<|\"|>ls<|\"|>", "done"],
            "forbidden_supervised_substrings": ["inspect repo", "response:bash", "file_a.py"],
        },
        id="new-gemma4-thinking-off-no-reasoning-labels",
    ),
]


@pytest.mark.parametrize("case", _REAL_TEMPLATE_COMPATIBILITY_CASES)
def test_prepare_and_mask_supports_real_chat_template_file(case):
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path(case["template_path"])
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    training_data = prepare_and_mask_for_test(
        _real_template_tool_call_dataset(),
        tokenizer,
        chat_template_kwargs=case["chat_template_kwargs"],
        train_on_reasoning=case["train_on_reasoning"],
        strict=True,
    )

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text
    for substring in case["expected_supervised_substrings"]:
        assert substring in supervised_text, case["name"]
    for substring in case["forbidden_supervised_substrings"]:
        assert substring not in supervised_text, case["name"]


def test_prepare_and_mask_actual_gemma4_keeps_reasoning_on_earlier_tool_turns():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("gemma-4-chat-template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "earlier reasoning",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "SECRET_TOOL_OUTPUT"},
                    {"role": "user", "content": "summarize"},
                    {"role": "assistant", "content": "done", "reasoning_content": "final reasoning"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "run shell",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True},
        train_on_reasoning=True,
        strict=True,
        include_debug_columns=True,
    )

    row = training_data[0]
    assert "earlier reasoning" in row["text"]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert "earlier reasoning" in supervised_text
    assert "<|tool_call>call:bash" in supervised_text
    assert "final reasoning" in supervised_text
    assert "SECRET_TOOL_OUTPUT" not in supervised_text
    assert "SECRET_TOOL_OUTPUT" in masked_text


def test_actual_gemma4_template_keeps_tool_responses_out_of_model_turns_and_uses_json_arguments():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("gemma-4-chat-template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    rendered = tokenizer.apply_chat_template(
        _real_template_tool_call_dataset()[0]["messages"],
        tools=_real_template_tool_call_dataset()[0]["tools"],
        enable_thinking=True,
    )

    assert '<|tool_call>call:bash{"command": "ls"}<tool_call|><turn|>\n<|turn>user\n<|tool_response>' in rendered
    assert "<|tool_response>" not in rendered.split("<|turn>model\n", 2)[1].split("<turn|>", 1)[0]
    assert "<|turn>user\n<|tool_response>" in rendered
    assert "<|turn>model\ndone<turn|>" in rendered


def test_new_gemma4_template_uses_native_inline_tool_call_and_response_format():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("new_gemma_4_template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    rendered = tokenizer.apply_chat_template(
        _real_template_tool_call_dataset()[0]["messages"],
        tools=_real_template_tool_call_dataset()[0]["tools"],
        enable_thinking=True,
        preserve_thinking=True,
    )

    assert '<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>' in rendered
    assert '<|tool_response>response:bash{value:<|"|>file_a.py<|"|>}<tool_response|>done<turn|>' in rendered
    assert "<|turn>user\n<|tool_response>" not in rendered


def test_new_gemma4_template_preserve_thinking_controls_history_reasoning():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("new_gemma_4_template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    messages = [
        {"role": "developer", "content": "Use tools carefully."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "earlier reasoning",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "ls"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "SECRET_TOOL_OUTPUT"},
        {"role": "user", "content": "summarize"},
        {"role": "assistant", "content": "done", "reasoning_content": "final reasoning"},
    ]

    history_stripped = tokenizer.apply_chat_template(
        messages,
        tools=_real_template_tool_call_dataset()[0]["tools"],
        enable_thinking=True,
        preserve_thinking=False,
    )
    history_preserved = tokenizer.apply_chat_template(
        messages,
        tools=_real_template_tool_call_dataset()[0]["tools"],
        enable_thinking=True,
        preserve_thinking=True,
    )

    assert "Use tools carefully.\nBe concise." in history_stripped
    assert "earlier reasoning" not in history_stripped
    assert "final reasoning" in history_stripped
    assert "earlier reasoning" in history_preserved
    assert "final reasoning" in history_preserved


def test_new_gemma4_template_trains_write_tool_call_arguments_not_tool_outputs():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("new_gemma_4_template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    rust_test = (
        "#[test]\n"
        "fn test_timeout_when_lock_held_forever() {\n"
        '    let temp = tempfile::NamedTempFile::new().expect("tempfile");\n'
        "    assert!(true);\n"
        "}\n"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "write file",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["content", "path"],
                },
            },
        },
        _real_template_tool_call_dataset()[0]["tools"][0],
    ]
    messages = [
        {"role": "user", "content": "write a rust test"},
        {
            "role": "assistant",
            "content": "I will write the integration test first.",
            "tool_calls": [
                {
                    "id": "call_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": {"content": rust_test, "path": "/workspace/tests/integration_test.rs"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_write",
            "name": "write",
            "content": "Successfully wrote 1518 bytes to /workspace/tests/integration_test.rs",
        },
        {
            "role": "assistant",
            "content": "Now let me verify everything compiles and tests pass.",
            "tool_calls": [
                {
                    "id": "call_bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "cd /workspace && cargo build 2>&1"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bash",
            "name": "bash",
            "content": "/bin/bash: line 1: cargo: command not found\n\nCommand exited with code 127",
        },
        {"role": "assistant", "content": "cargo is not available in this environment."},
    ]
    prepared = prepare_data(
        Dataset.from_list([{"messages": messages, "tools": tools}]),
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        tokenize=True,
        strict=True,
        verbose=False,
    )
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        args=SimpleNamespace(dataset_text_field="text", max_length=None, packing=False),
        tokenizer=tokenizer,
    )
    trainer = mask_data(
        trainer,
        tokenizer=tokenizer,
        train_on_reasoning=True,
        train_on_final_answers=True,
        train_on_tools=True,
        audit=False,
        verbose=False,
    )

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])

    assert "#[test]" in supervised_text
    assert "I will write the integration test first." in supervised_text
    assert "/workspace/tests/integration_test.rs" in supervised_text
    assert "cd /workspace && cargo build 2>&1" in supervised_text
    assert "<|tool_call>" not in masked_text
    assert "Successfully wrote 1518 bytes" not in supervised_text
    assert "cargo: command not found" not in supervised_text
    assert "#[test]" not in masked_text
    assert "Successfully wrote 1518 bytes" in masked_text
    assert "cargo: command not found" in masked_text


def test_latest_gemma_template_trains_repeated_tool_call_protocol_not_outputs():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("gemma-template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "write file",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["content", "path"],
                },
            },
        },
        _real_template_tool_call_dataset()[0]["tools"][0],
    ]
    messages = [
        {"role": "user", "content": "write a file"},
        {
            "role": "assistant",
            "content": "I will write first.",
            "reasoning_content": "plan tool call",
            "tool_calls": [
                {
                    "id": "call_write",
                    "type": "function",
                    "function": {"name": "write", "arguments": {"content": "hello", "path": "/tmp/a.txt"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_write", "name": "write", "content": "wrote file"},
        {
            "role": "assistant",
            "content": "Now verify.",
            "reasoning_content": "need test",
            "tool_calls": [
                {
                    "id": "call_bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "cat /tmp/a.txt"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_bash", "name": "bash", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]
    prepared = prepare_data(
        Dataset.from_list([{"messages": messages, "tools": tools}]),
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        tokenize=True,
        strict=True,
        verbose=False,
    )
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        args=SimpleNamespace(dataset_text_field="text", max_length=None, packing=False),
        tokenizer=tokenizer,
    )
    trainer = mask_data(
        trainer,
        tokenizer=tokenizer,
        train_on_reasoning=True,
        train_on_final_answers=True,
        train_on_tools=True,
        audit=False,
        verbose=False,
    )

    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])

    assert "<|channel>thought\nplan tool call\n<channel|>" in supervised_text
    assert "<|channel>thought\nneed test\n<channel|>" in supervised_text
    assert "<|tool_call>call:write" in supervised_text
    assert "<|tool_call>call:bash" in supervised_text
    assert 'path:<|"|>/tmp/a.txt<|"|>' in supervised_text
    assert 'command:<|"|>cat /tmp/a.txt<|"|>' in supervised_text
    assert "I will write first." in supervised_text
    assert "Now verify." in supervised_text
    assert "done" in supervised_text
    assert "wrote file" not in supervised_text
    assert 'response:bash{value:<|"|>hello<|"|>}' not in supervised_text
    assert "<|channel>thought\nneed test\n<channel|>" not in masked_text
    assert "<|tool_call>call:bash" not in masked_text
    assert "wrote file" in masked_text
    assert 'response:bash{value:<|"|>hello<|"|>}' in masked_text


def test_new_gemma4_template_strict_markers_tolerate_trimmed_embedded_thinking():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("new_gemma_4_template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Analyze the issue."},
                    {
                        "role": "assistant",
                        "content": "<thinking>Need to inspect the issue first.</thinking>",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": {"command": "gh issue view 1123 --json title,body"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "bash",
                        "content": '{"title":"config: maximum time to wait for a retry"}',
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "<thinking>The issue is a feature request.</thinking>\n\n"
                            "The issue requests a configurable maximum retry delay."
                        ),
                    },
                ],
                "tools": _real_template_tool_call_dataset()[0]["tools"],
            }
        ]
    )

    prepared = prepare_data(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        strict=True,
        verbose=False,
    )

    rendered = prepared[0]["text"]
    supervised_text = "".join(
        rendered[span["start"]:span["end"]]
        for span in prepared[0]["teich_supervised_spans"]
        if span.get("kind") in {"final_answer", "tool_call"}
    )

    assert "\n\nThe issue requests" not in rendered
    assert "The issue requests a configurable maximum retry delay." in supervised_text
    assert "gh issue view 1123 --json title,body" in supervised_text


def test_marker_boundary_whitespace_reconciliation_handles_insertions_and_deletions():
    spans = [{"start": 5, "end": 12, "kind": "final_answer", "role": "assistant"}]
    deleted = _reconcile_marker_boundary_whitespace("start\n\nanswer", spans, "startanswer")
    inserted = _reconcile_marker_boundary_whitespace("startanswer", spans, "start\n\nanswer")

    assert deleted is not None
    assert deleted[0] == "startanswer"
    assert deleted[1][0]["start"] == 5
    assert deleted[1][0]["end"] == 10

    assert inserted is not None
    assert inserted[0] == "start\n\nanswer"
    assert inserted[1][0]["start"] == 5
    assert inserted[1][0]["end"] == 14


def test_strip_markers_collects_json_escaped_marker_variants():
    marker = {
        "start_marker": "\ue000AGD0S\ue001",
        "end_marker": "\ue000AGD0E\ue001",
        "kind": "tool_call",
        "role": "assistant",
    }
    escaped_start = json.dumps(marker["start_marker"])[1:-1]
    escaped_end = json.dumps(marker["end_marker"])[1:-1]

    stripped = _strip_markers_and_collect_spans(
        f'<parameter=content>\n{{"name":"{escaped_start}teich-studio{escaped_end}"}}\n</parameter>',
        [marker],
    )

    assert stripped is not None
    text, spans = stripped
    assert text == '<parameter=content>\n{"name":"teich-studio"}\n</parameter>'
    assert spans == [
        {
            "start": 29,
            "end": 41,
            "source_start": 29,
            "source_end": 41,
            "kind": "tool_call",
            "role": "assistant",
        }
    ]


def test_actual_qwen_template_receives_normalized_mapping_tool_arguments():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("qwen3.6_chat_template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "list files"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect repo",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                            }
                        ],
                    },
                ],
                "tools": _real_template_tool_call_dataset()[0]["tools"],
            }
        ]
    )

    prepared = prepare_data(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        strict=True,
        verbose=False,
    )

    rendered = prepared[0]["text"]
    assert "<function=bash>" in rendered
    assert "<parameter=command>\nls\n</parameter>" in rendered
    assert "<function=bash>\n</function>" not in rendered


def test_codex_trace_conversion_to_qwen_template_keeps_tool_arguments_parseable(tmp_path: Path):
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("qwen3.6_chat_template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    trace_file = tmp_path / "trace.jsonl"
    events = [
        {
            "type": "session_meta",
            "payload": {
                "base_instructions": {
                    "text": "You are a coding agent.\n\nAvailable tools:\n- bash: Execute shell commands\n"
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "List files"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "bash",
                "call_id": "call_1",
                "arguments": '{"command":"ls -la"}',
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "output_text", "text": "file_a.py"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Found file_a.py."}],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = load_traces(tmp_path, split=None)
    prepared = prepare_data(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        strict=True,
        verbose=False,
    )

    rendered = prepared[0]["text"]
    assert "<function=bash>" in rendered
    assert "<parameter=command>\nls -la\n</parameter>" in rendered
    assert "<function=bash>\n</function>" not in rendered
    assert "<tool_response>\nfile_a.py\n</tool_response>" in rendered


def test_prepare_and_mask_drops_untrainable_rows_with_actual_gemma4_template_under_strict_mode():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("gemma-4-chat-template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "think only"},
                    {"role": "assistant", "content": "", "reasoning_content": "private reasoning"},
                ],
                "tools": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "answer"},
                    {"role": "assistant", "content": "visible answer", "reasoning_content": "private reasoning"},
                ],
                "tools": [],
            },
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        train_on_reasoning=False,
        strict=True,
    )

    assert training_data.num_rows == 1
    supervised_text = tokenizer.decode([token for token in training_data[0]["labels"] if token != -100])
    assert "visible answer" in supervised_text
    assert "private reasoning" not in supervised_text


def test_prepare_and_mask_supervises_typed_text_content_parts_with_actual_gemma4_template():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("gemma-4-chat-template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "question"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "visible answer"}],
                        "reasoning_content": "private reasoning",
                    },
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        train_on_reasoning=False,
        strict=True,
    )

    supervised_text = tokenizer.decode([token for token in training_data[0]["labels"] if token != -100])
    assert "visible answer" in supervised_text
    assert "private reasoning" not in supervised_text
    assert "text" not in supervised_text


def test_prepare_and_mask_supervises_model_role_with_actual_gemma4_template():
    jinja2 = pytest.importorskip("jinja2")
    template_path = Path("gemma-4-chat-template.jinja")
    if not template_path.exists():
        pytest.skip(f"{template_path} is not available")

    tokenizer = RealJinjaChatTemplateTokenizer(template_path, jinja2)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "model", "content": "visible answer", "reasoning_content": "private reasoning"},
                ],
                "tools": [],
            }
        ]
    )

    training_data = prepare_and_mask_for_test(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        train_on_reasoning=False,
        strict=True,
    )

    supervised_text = tokenizer.decode([token for token in training_data[0]["labels"] if token != -100])
    assert "visible answer" in supervised_text
    assert "private reasoning" not in supervised_text


def test_prepare_data_falls_back_for_gemma4_text_parts_and_tool_roles():
    class Gemma4StrictProcessor(FakeTokenizer):
        def _content_text(self, content):
            if not isinstance(content, list):
                raise TypeError("Gemma 4 expects typed text content parts")
            return "".join(part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text")

        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
            if tools is not None:
                raise TypeError("Gemma 4 template does not accept tools")
            parts = []
            for message in messages:
                role = message["role"]
                if role == "tool":
                    raise ValueError("Gemma 4 template does not support tool role")
                content = self._content_text(message.get("content", []))
                if role == "assistant":
                    for tool_call in message.get("tool_calls") or []:
                        content += f"<tool_call>{tool_call['function']['name']}</tool_call>"
                parts.append(f"<{role}>{content}</{role}>")
            if add_generation_prompt:
                parts.append("<assistant>")
            rendered = "".join(parts)
            if tokenize:
                return self(rendered)
            return rendered

    tokenizer = Gemma4StrictProcessor()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "file_a.py"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                        },
                    }
                ],
            }
        ]
    )

    prepared = prepare_data(dataset, tokenizer, verbose=False)

    assert prepared.num_rows == 1
    assert "<tool_response" in prepared[0]["text"]
    assert "<tool_call>bash</tool_call>" in prepared[0]["text"]
    assert prepared[0]["teich_supervised_spans"]
