from __future__ import annotations

from datasets import Dataset

from agentic_datagen import format_and_mask


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

    training_data = format_and_mask(dataset, tokenizer)

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
    assert "<tool>file_a.py</tool>" in masked_text

    assert row["assistant_masks"] == [0 if label == -100 else 1 for label in row["labels"]]


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
    )

    row = training_data[0]
    assert row["text"] == "<user>hello</user><assistant>world</assistant>"
    supervised_text = tokenizer.decode([token for token in row["labels"] if token != -100])
    assert supervised_text == "<assistant>world</assistant>"

    preview = training_data.preview()
    assert "\033[31m" in preview
    assert "<user>hello</user>" in preview
    assert "<assistant>world</assistant>" in preview


def test_format_and_mask_rejects_reserved_chat_template_kwargs():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list([{"messages": [], "tools": []}])

    try:
        format_and_mask(dataset, tokenizer, chat_template_kwargs={"tools": []})
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("Expected format_and_mask to reject reserved chat_template_kwargs")


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

    training_data = format_and_mask(dataset, processor)

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

    training_data = format_and_mask(dataset, tokenizer)

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
    assert supervised_text == "<assistant><think>inspect repo</think><tool_call>bash</tool_call></assistant><assistant>done</assistant>"
    assert "<tool_call>" in supervised_text
    assert "</think>" in supervised_text
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100]
    )
    assert "<system>system rules</system>" in masked_text
    assert "<user>first request</user>" in masked_text
    assert "<tool>file_a.py</tool>" in masked_text
    assert tokenizer.render_count == 1
