import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from teich import load_traces, trace_is_complete
from teich.loader import _dataset_from_rows


def _write_codex_trace(trace_file: Path, prompt: str = "List files") -> None:
    events = [
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-session-1",
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
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "bash",
                "call_id": "call_1",
                "arguments": '{"command":"ls"}',
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_load_traces_from_local_split_directory(tmp_path: Path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    _write_codex_trace(train_dir / "trace.jsonl")

    dataset = load_traces(tmp_path, split="train")

    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 1
    row = dataset[0]
    assert row["prompt"] == "List files"
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    tools_by_name = {tool["function"]["name"]: tool for tool in row["tools"]}
    assert "bash" in tools_by_name
    assert tools_by_name["bash"]["function"]["description"] == "Execute shell commands"


def test_load_traces_reads_nested_jsonl_files_and_skips_non_data_dirs(tmp_path: Path):
    train_dir = tmp_path / "train"
    nested_dir = train_dir / "nested"
    partials_dir = train_dir / "partials"
    failures_dir = train_dir / "failures"
    nested_dir.mkdir(parents=True)
    partials_dir.mkdir(parents=True)
    failures_dir.mkdir(parents=True)
    _write_codex_trace(nested_dir / "trace.jsonl", prompt="Nested prompt")
    _write_codex_trace(partials_dir / "partial.jsonl", prompt="Partial prompt")
    _write_codex_trace(failures_dir / "failed.jsonl", prompt="Failed prompt")

    dataset = load_traces(tmp_path, split="train")

    assert dataset.num_rows == 1
    assert dataset[0]["prompt"] == "Nested prompt"


def test_load_traces_downloads_dataset_repo_and_converts_split(tmp_path: Path):
    repo_dir = tmp_path / "downloaded-repo"
    split_dir = repo_dir / "train"
    split_dir.mkdir(parents=True)
    _write_codex_trace(split_dir / "remote-trace.jsonl", prompt="Inspect repo")

    with patch("teich.loader.snapshot_download", return_value=str(repo_dir)) as mock_download:
        dataset = load_traces("armand0e/ag-datagen-v2-test", split="train", revision="main")

    mock_download.assert_called_once_with(
        repo_id="armand0e/ag-datagen-v2-test",
        repo_type="dataset",
        revision="main",
        token=None,
        cache_dir=None,
        local_dir=None,
        allow_patterns=["*.jsonl", "**/*.jsonl", "README.md", "tools.json", "**/tools.json"],
    )
    assert dataset.num_rows == 1
    assert dataset[0]["prompt"] == "Inspect repo"


def test_load_traces_forwards_hf_token_alias_to_snapshot_download(tmp_path: Path):
    repo_dir = tmp_path / "downloaded-repo"
    split_dir = repo_dir / "train"
    split_dir.mkdir(parents=True)
    _write_codex_trace(split_dir / "remote-trace.jsonl", prompt="Inspect repo")

    with patch("teich.loader.snapshot_download", return_value=str(repo_dir)) as mock_download:
        dataset = load_traces("armand0e/ag-datagen-v2-test", split="train", hf_token="hf-test")

    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["token"] == "hf-test"
    assert dataset.num_rows == 1


def test_load_traces_rejects_conflicting_token_aliases():
    with pytest.raises(ValueError, match="token or hf_token"):
        load_traces("armand0e/ag-datagen-v2-test", token="hf-one", hf_token="hf-two")


def test_load_traces_applies_tools_snapshot_embedded_in_readme(tmp_path: Path):
    dataset_file = tmp_path / "chat.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Open page", "thinking": None},
                    {"role": "assistant", "content": "Done", "thinking": None},
                ],
                "prompt": "Open page",
                "response": "Done",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_open",
                "description": "Open a browser page",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    (tmp_path / "README.md").write_text(
        "<details>\n"
        "<summary>Training-ready tool schema snapshot</summary>\n\n"
        "```json\n"
        + json.dumps(tools, indent=2)
        + "\n```\n"
        "</details>\n",
        encoding="utf-8",
    )

    dataset = load_traces(tmp_path)

    assert dataset.num_rows == 1
    assert dataset[0]["tools"][0]["function"]["name"] == "browser_open"


def test_load_traces_raises_when_no_trace_files_exist(tmp_path: Path):
    with pytest.raises(ValueError, match="No trace files found"):
        load_traces(tmp_path, split="train")


def test_load_traces_supports_mixed_nested_tool_argument_types(tmp_path: Path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    trace_file = train_dir / "trace.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Apply edits"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "editor",
                "call_id": "call_1",
                "arguments": json.dumps(
                    {
                        "path": "/workspace/a.txt",
                        "edits": json.dumps([{"oldText": "a", "newText": "b"}]),
                        "content": "plain text",
                    }
                ),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "editor",
                "call_id": "call_2",
                "arguments": json.dumps(
                    {
                        "path": "/workspace/b.txt",
                        "edits": [{"oldText": "x", "newText": "y"}],
                        "content": [{"type": "text", "text": "structured"}],
                    }
                ),
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    dataset = load_traces(tmp_path, split="train")

    tool_calls = dataset[0]["messages"][-1]["tool_calls"]
    assert tool_calls[0]["function"]["arguments"]["edits"] == [{"oldText": "a", "newText": "b"}]
    assert tool_calls[0]["function"]["arguments"]["content"] == "plain text"
    assert tool_calls[1]["function"]["arguments"]["content"] == [{"type": "text", "text": "structured"}]


def test_load_traces_max_examples_shuffles_trace_rows_deterministically(tmp_path: Path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    for index in range(4):
        _write_codex_trace(train_dir / f"trace-{index}.jsonl", prompt=f"Prompt {index}")

    full_dataset = load_traces(tmp_path, split="train")
    limited_a = load_traces(tmp_path, split="train", max_examples=2)
    limited_b = load_traces(tmp_path, split="train", max_examples=2)
    expected = full_dataset.shuffle(seed=3407).select(range(2))

    limited_prompts_a = [row["prompt"] for row in limited_a]
    limited_prompts_b = [row["prompt"] for row in limited_b]
    expected_prompts = [row["prompt"] for row in expected]

    assert limited_a.num_rows == 2
    assert limited_prompts_a == limited_prompts_b
    assert limited_prompts_a == expected_prompts


def test_load_traces_loads_structured_chat_dataset_file(tmp_path: Path):
    dataset_file = tmp_path / "chat.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant",
                                "thinking": None,
                                "timestamp": "2026-05-18T00:00:01.000Z",
                            },
                            {
                                "role": "user",
                                "content": "Hello",
                                "thinking": None,
                                "timestamp": "2026-05-18T00:00:02.000Z",
                            },
                            {"role": "assistant", "content": "Hi!", "thinking": "I should greet the user."},
                        ],
                        "system": "You are a helpful assistant",
                        "prompt": "Hello",
                        "thinking": "I should greet the user.",
                        "response": "Hi!",
                        "model": "gpt-4.1-mini",
                        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_traces(dataset_file)

    assert dataset.num_rows == 1
    row = dataset[0]
    assert row["prompt"] == "Hello"
    assert row["messages"][2]["role"] == "assistant"
    assert row["messages"][2]["reasoning_content"] == "I should greet the user."
    assert row["tools"] == []
    assert row["metadata"]["trace_type"] == "chat"
    assert row["metadata"]["model"] == "gpt-4.1-mini"
    assert row["metadata"]["usage"]["prompt_tokens"] == 4
    assert row["metadata"]["first_message_timestamp"] == "2026-05-18T00:00:02.000Z"


def test_load_traces_normalizes_separate_assistant_thinking_field(tmp_path: Path):
    dataset_file = tmp_path / "opus-reasoning.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Explain API gateways."},
                    {
                        "role": "assistant",
                        "content": "# API Gateway Architecture",
                        "thinking": "Plan routing, auth, rate limiting, and observability.",
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_traces(dataset_file)

    assistant = dataset[0]["messages"][-1]
    assert assistant["content"] == "# API Gateway Architecture"
    assert assistant["reasoning_content"] == "Plan routing, auth, rate limiting, and observability."


def test_load_traces_splits_inline_think_blocks_from_assistant_content(tmp_path: Path):
    dataset_file = tmp_path / "claude-reasoning.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Find the rectangle dimensions."},
                    {
                        "role": "assistant",
                        "content": (
                            "<think>Let width be w and length be 3w.</think> "
                            "# Answer\n\nThe width is 3sqrt(2)."
                        ),
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_traces(dataset_file)

    assistant = dataset[0]["messages"][-1]
    assert assistant["content"] == "# Answer\n\nThe width is 3sqrt(2)."
    assert assistant["reasoning_content"] == "Let width be w and length be 3w."


def test_load_traces_normalizes_structured_model_role_to_assistant(tmp_path: Path):
    dataset_file = tmp_path / "gemma-chat.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "model", "content": "Hi"},
                ],
                "prompt": "Hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_traces(dataset_file)

    assert dataset[0]["messages"][-1]["role"] == "assistant"


def test_load_traces_max_examples_shuffles_structured_rows_deterministically(tmp_path: Path):
    dataset_file = tmp_path / "chat.jsonl"
    dataset_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant", "thinking": None},
                        {"role": "user", "content": f"Hello {index}", "thinking": None},
                        {"role": "assistant", "content": f"Hi {index}!", "thinking": f"Greet user {index}."},
                    ],
                    "system": "You are a helpful assistant",
                    "prompt": f"Hello {index}",
                    "thinking": f"Greet user {index}.",
                    "response": f"Hi {index}!",
                    "model": "gpt-4.1-mini",
                }
            )
            for index in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    full_dataset = load_traces(dataset_file)
    limited_a = load_traces(dataset_file, max_examples=3)
    limited_b = load_traces(dataset_file, max_examples=3)
    expected = full_dataset.shuffle(seed=3407).select(range(3))

    limited_prompts_a = [row["prompt"] for row in limited_a]
    limited_prompts_b = [row["prompt"] for row in limited_b]
    expected_prompts = [row["prompt"] for row in expected]

    assert limited_a.num_rows == 3
    assert limited_prompts_a == limited_prompts_b
    assert limited_prompts_a == expected_prompts


def test_load_traces_rejects_negative_max_examples(tmp_path: Path):
    with pytest.raises(ValueError, match="max_examples must be non-negative"):
        load_traces(tmp_path, split="train", max_examples=-1)


def test_load_traces_filters_rows_without_assistant_training_signal(tmp_path: Path):
    dataset_file = tmp_path / "mixed.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": ""},
                        ],
                        "prompt": "Hello",
                        "response": "",
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ],
                        "prompt": "Hello",
                        "response": "Hi",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_traces(dataset_file)

    assert dataset.num_rows == 1
    assert dataset[0]["messages"][-1]["content"] == "Hi"


def test_load_traces_raises_when_all_rows_lack_assistant_training_signal(tmp_path: Path):
    dataset_file = tmp_path / "empty.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": ""},
                ],
                "prompt": "Hello",
                "response": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assistant training signal"):
        load_traces(dataset_file)


def test_load_traces_drops_rows_ending_on_tool_result_by_default(tmp_path: Path):
    dataset_file = tmp_path / "tool-final.jsonl"
    incomplete_row = {
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "ls"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "bash", "content": "README.md"},
        ],
        "prompt": "List files",
    }
    complete_row = {
        "messages": [
            *incomplete_row["messages"],
            {"role": "assistant", "content": "Found README.md."},
        ],
        "prompt": "List files",
    }
    dataset_file.write_text(
        "\n".join(json.dumps(row) for row in [incomplete_row, complete_row]) + "\n",
        encoding="utf-8",
    )

    dataset = load_traces(dataset_file)
    unfiltered = load_traces(dataset_file, drop_incomplete_traces=False)

    assert trace_is_complete(incomplete_row) is False
    assert trace_is_complete(complete_row) is True
    assert dataset.num_rows == 1
    assert dataset[0]["messages"][-1]["role"] == "assistant"
    assert unfiltered.num_rows == 2


def test_dataset_from_rows_falls_back_when_on_mixed_types_is_unsupported():
    rows = [
        {
            "prompt": "Apply edits",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": {
                                    "content": "plain text",
                                    "edits": [{"oldText": "a", "newText": "b"}],
                                }
                            }
                        },
                        {
                            "function": {
                                "arguments": {
                                    "content": [{"type": "text", "text": "structured"}],
                                }
                            }
                        },
                    ],
                }
            ],
            "tools": [],
            "metadata": {"source": "test"},
        }
    ]

    original_from_list = Dataset.from_list

    def _compat_from_list(data, *args, **kwargs):
        if "on_mixed_types" in kwargs:
            raise TypeError("Dataset.from_list() got an unexpected keyword argument 'on_mixed_types'")
        return original_from_list(data, *args, **kwargs)

    with patch.object(Dataset, "from_list", side_effect=_compat_from_list):
        dataset = _dataset_from_rows(rows)

    tool_calls = dataset[0]["messages"][0]["tool_calls"]
    assert tool_calls[0]["function"]["arguments"]["content"] == "plain text"
    assert tool_calls[0]["function"]["arguments"]["edits"] == [{"oldText": "a", "newText": "b"}]
    assert tool_calls[1]["function"]["arguments"]["content"] == [{"type": "text", "text": "structured"}]
