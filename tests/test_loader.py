import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from teich import load_traces


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
    assert row["tools"][0]["function"]["name"] == "bash"


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
        allow_patterns=["*.jsonl", "**/*.jsonl"],
    )
    assert dataset.num_rows == 1
    assert dataset[0]["prompt"] == "Inspect repo"


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
