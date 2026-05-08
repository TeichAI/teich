from __future__ import annotations

import json
from types import SimpleNamespace

from datasets import Dataset
import pytest

from teich import format_and_mask, mask_data, prepare_data


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

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
        token_ids: list[int] = []
        for token in text:
            token_id = self._vocab.setdefault(token, len(self._vocab) + 1)
            self._reverse_vocab[token_id] = token
            token_ids.append(token_id)
        output = {"input_ids": token_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(token_ids)
        return output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


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


def test_format_and_mask_supervises_only_assistant_turns_across_multi_turn_conversation():
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

    training_data = format_and_mask(dataset, tokenizer, include_debug_columns=True)

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
    assert supervised_text == "<assistant><think>inspect repo</think><tool_call>bash</tool_call></assistant><assistant>Found one Python file.</assistant>"

    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<system>system rules</system>" in masked_text
    assert "<user>first request</user>" in masked_text


def test_format_and_mask_returns_compact_training_columns_by_default():
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

    training_data = format_and_mask(dataset, tokenizer)

    row = training_data[0]
    assert set(row.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])


def test_format_and_mask_accepts_multiple_datasets_and_concatenates_them():
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

    training_data = format_and_mask([tool_dataset, chat_dataset], tokenizer, include_debug_columns=True)

    assert training_data.num_rows == 2
    assert "<tool_call>bash</tool_call>" in training_data[0]["text"]
    assert "<assistant><think>be friendly</think>world</assistant>" in training_data[1]["text"]
    preview = training_data.preview(1)
    assert "\033[31m" in preview
    assert "<user>hello</user>" in preview
    assert "<assistant><think>be friendly</think>world</assistant>" in preview


def test_format_and_mask_passes_chat_template_kwargs_and_preview_marks_unsupervised_text_red():
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

    training_data = format_and_mask(
        dataset,
        tokenizer,
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": False},
        include_debug_columns=True,
    )

    row = training_data[0]
    assert row["text"] == "<user>hello</user><assistant>world</assistant>"
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<assistant>world</assistant>"

    preview = training_data.preview()
    assert "\033[31m" in preview
    assert "<user>hello</user>" in preview
    assert "<assistant>world</assistant>" in preview


def test_format_and_mask_can_exclude_reasoning_from_qwen_style_supervision():
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

    training_data = format_and_mask(
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


def test_format_and_mask_can_exclude_reasoning_from_gemma_style_supervision():
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

    training_data = format_and_mask(
        dataset,
        tokenizer,
        train_on_reasoning=False,
        include_debug_columns=True,
    )

    row = training_data[0]
    assert row["text"] == "<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\nthink\n<channel|>world<turn|>\n"
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "world<turn|>\n"
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|channel>thought\nthink\n<channel|>" in masked_text


def test_format_and_mask_rejects_reserved_chat_template_kwargs():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list([{"messages": [], "tools": []}])

    try:
        format_and_mask(dataset, tokenizer, chat_template_kwargs={"tools": []})
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("Expected format_and_mask to reject reserved chat_template_kwargs")


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
    supervised_text = "".join(prepared[0]["text"][span["start"] : span["end"]] for span in spans)
    assert supervised_text == "<think>think</think>world</assistant>"


def test_prepare_data_filters_oversized_rows_without_returning_tokens():
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
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<think>inspect repo</think><tool_call>bash</tool_call></assistant>done</assistant>"
    masked_text = tokenizer.decode([token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100])
    assert "<system>system rules</system>" in masked_text
    assert "<user>first request</user>" in masked_text
    assert "<tool>file_a.py</tool>" in masked_text


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


def test_format_and_mask_supports_processor_objects_with_nested_text_tokenizer():
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

    training_data = format_and_mask(dataset, processor, include_debug_columns=True)

    row = training_data[0]
    assert row["text"] == "<user>hello</user><assistant><think>think</think>world</assistant>"
    supervised_text = processor.tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<assistant><think>think</think>world</assistant>"
    assert "\033[31m" in training_data.preview()


def test_format_and_mask_uses_fast_assistant_mask_path_when_supported():
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

    training_data = format_and_mask(dataset, tokenizer, include_debug_columns=True)

    row = training_data[0]
    assert row["assistant_masks"] == [0] * len("<user>hello</user>") + [1] * len("<assistant><think>think</think>world</assistant>")
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<assistant><think>think</think>world</assistant>"


def test_format_and_mask_handles_non_prefix_stable_templates_around_tool_turns():
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

    training_data = format_and_mask(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert "<assistant><think>think</think><tool_call>bash</tool_call>" in supervised_text
    assert "<assistant>done" in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<tool>file_a.py</tool>" in masked_text


def test_format_and_mask_skips_unrenderable_prefixes_before_first_user_message():
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

    training_data = format_and_mask(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<assistant><think>think</think>world</assistant>"
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<system>system rules</system>" in masked_text
    assert "<user>hello</user>" in masked_text


def test_format_and_mask_renders_only_supervision_checkpoints_in_fallback():
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

    training_data = format_and_mask(dataset, tokenizer)

    assert training_data.num_rows == 1
    assert tokenizer.render_count == 4


def test_format_and_mask_uses_single_render_offset_mask_path_when_offsets_are_available():
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

    training_data = format_and_mask(dataset, tokenizer)

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


def test_format_and_mask_masks_qwen_generation_prompt_prefix_but_supervises_generated_wrappers():
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

    training_data = format_and_mask(dataset, tokenizer, chat_template_kwargs={"enable_thinking": True})

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text.startswith("inspect repo\n</think>\n\n<tool_call>\n<function=bash>")
    assert "<|im_start|>assistant\n<think>\n" not in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|im_start|>assistant\n<think>\n" in masked_text
    assert "first request" in masked_text
    assert tokenizer.render_count == 4


def test_format_and_mask_uses_gemma_structured_mask_path():
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

    training_data = format_and_mask(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text.startswith('<|channel>thought\ninspect repo\n<channel|><|tool_call>call:bash{command:"ls"}<tool_call|>')
    assert supervised_text.endswith('done<turn|>\n')
    assert "response:bash" not in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "file_a.py" in masked_text
    assert tokenizer.render_count == 1


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


def test_format_and_mask_expands_granite_style_assistant_blocks():
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

    training_data = format_and_mask(dataset, tokenizer)

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert '<tool_call>{"name": "bash", "arguments": {"command": "dir"}}</tool_call><|end_of_text|>\n' in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "List files" in masked_text


def test_format_and_mask_falls_back_to_assistant_header_when_qwen_prefix_probe_mismatches():
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

    training_data = format_and_mask(dataset, tokenizer, chat_template_kwargs={"enable_thinking": True})

    row = training_data[0]
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text.startswith("inspect repo\n</think>\n\n<tool_call>\n<function=bash>")
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<|im_start|>assistant\n" in masked_text


def test_format_and_mask_skips_rows_with_empty_message_lists():
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

    training_data = format_and_mask(dataset, tokenizer)

    assert training_data.num_rows == 1
    assert set(training_data[0].keys()) == {"input_ids", "attention_mask", "labels"}


def test_format_and_mask_drops_oversized_examples_by_default():
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

    training_data = format_and_mask(dataset, tokenizer, max_length=60)

    assert training_data.num_rows == 1
    assert len(training_data[0]["input_ids"]) < 60


def test_format_and_mask_truncates_oversized_examples_when_drop_is_disabled():
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

    training_data = format_and_mask(dataset, tokenizer, max_length=60, drop_oversized_examples=False)

    assert training_data.num_rows == 1
    row = training_data[0]
    assert len(row["input_ids"]) == 60
    assert len(row["attention_mask"]) == 60
    assert len(row["labels"]) == 60


def test_format_and_mask_uses_tokenizer_model_max_length_when_dropping_oversized_examples():
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

    with pytest.raises(ValueError, match="fit within context window of 60 tokens"):
        format_and_mask(dataset, tokenizer)


def test_format_and_mask_raises_when_all_rows_have_empty_message_lists():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list(
        [
            {"messages": [], "tools": []},
            {"messages": [], "tools": []},
        ]
    )

    with pytest.raises(ValueError, match="no non-empty conversations"):
        format_and_mask(dataset, tokenizer)


def test_format_and_mask_strict_rejects_marker_render_mismatch():
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
        format_and_mask(dataset, tokenizer, strict=True)
