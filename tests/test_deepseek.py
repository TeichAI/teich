"""Synthetic DeepSeek Harness logs based on the public dsh-session/dsh-llm types."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import zstandard
from typer.testing import CliRunner

from teich import detect_trace_type, load_traces
from teich.cli import app
from teich.converter import convert_trace_to_training_example, convert_traces_to_training_data


@pytest.fixture
def events():
    path = Path(__file__).resolve().parents[1] / "examples/example_deepseek_session.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_trace(path, events):
    data = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    if path.suffix in {".zst", ".zstd"}:
        data = zstandard.ZstdCompressor().compress(data)
    path.write_bytes(data)
    return path


def test_deepseek_native_messages_and_metadata(tmp_path, events):
    assert detect_trace_type(events) == "deepseek_harness"
    example = convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))

    assert example.prompt == "Read the example file."
    assert example.messages[-1]["content"] == "The file says: Hello, world."
    assert example.messages == [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Read the example file."},
        {
            "role": "assistant",
            "content": "I'll read it.",
            "reasoning_content": "I need the file contents.",
            "tool_calls": [{"id": "read-1", "type": "function", "function": {
                "name": "read_file", "arguments": {"path": "example.txt"},
            }}],
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "read-1", "content": "Hello, world."},
        {"role": "assistant", "content": "The file says: Hello, world."},
    ]
    assert example.tools == [{"type": "function", "function": events[3]["data"]["header"]["tools"][0]}]
    assert example.metadata["trace_type"] == "deepseek_harness"
    assert example.metadata["session_id"] == "deepseek-example"
    assert example.metadata["model"] == "example-model"
    assert example.metadata["model_provider"] == "deepseek"
    assert example.metadata["cwd"] == "/workspace/example"
    timestamp = datetime.fromisoformat(example.metadata["first_message_timestamp"].replace("Z", "+00:00"))
    assert timestamp == datetime.fromtimestamp(events[2]["time"] / 1000, tz=timezone.utc)


@pytest.mark.parametrize("version", [0, 1, 2])
def test_deepseek_header_versions(tmp_path, events, version):
    events[0]["version"] = version
    if version == 2:
        events[0].update(isSeeded=False, delegationDepth=0)
        for event in events:
            if event["type"] == "assistant/message":
                event["data"]["stream"] = []
    assert detect_trace_type(events) == "deepseek_harness"
    example = convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))
    assert example.messages[-1]["content"] == "The file says: Hello, world."


@pytest.mark.parametrize("header,expected", [
    ({"type": "session", "id": "pi", "version": 3}, "pi"),
    ({"type": "session", "id": "openclaw", "version": 3, "cwd": "/home/user/.openclaw/workspace"}, "openclaw"),
    ({"messages": [{"role": "user", "content": "hello"}]}, None),
])
def test_deepseek_detection_preserves_other_formats(header, expected):
    assert detect_trace_type([header]) == expected


def test_deepseek_ignores_log_only_message_and_chunk_events(tmp_path, events):
    hidden = json.loads(json.dumps(events[8]))
    hidden.pop("surfaceOp")
    hidden["seq"] = 9
    hidden["data"]["message"]["content"] = [{"type": "text", "text": "Do not train on this log-only message."}]
    events.append(hidden)
    events.append({"type": "text-chunks", "seq0": 10, "time0": 1788739201800,
                   "data": {"turn": 1, "step": 0, "index": 0, "dt": [1], "texts": ["Partial", " stream"]}})
    example = convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))
    assert len(example.messages) == 5
    assert example.messages[-1]["content"] == "The file says: Hello, world."


def test_deepseek_latest_request_header_is_full_snapshot(tmp_path, events):
    events.append({"type": "request/header", "seq": 9, "time": 1788739201800, "data": {
        "reason": "change", "header": {"config": {"provider": "alternate", "model": "new-model"}},
    }})
    example = convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))
    assert all(message["role"] != "system" for message in example.messages)
    assert example.metadata["model"] == "new-model"
    assert example.metadata["model_provider"] == "alternate"
    assert [tool["function"]["name"] for tool in example.tools] == ["read_file"]
    assert example.tools[0]["function"]["parameters"]["properties"]["path"] == {"type": "string"}


def test_deepseek_compaction_replaces_sequence_range(tmp_path, events):
    events.extend([
        {"type": "user/message", "seq": 9, "time": 1788739201800,
         "surfaceOp": {"op": "replace", "start": 1, "end": 6}, "sourceEventSeqs": [1, 4, 6],
         "data": {"id": "summary", "role": "user", "source": {"kind": "plugin", "plugin": "compaction"},
                  "content": [{"type": "text", "text": "Summary: the file was read."}]}},
        {"type": "user/message", "seq": 10, "time": 1788739201900,
         "surfaceOp": {"op": "replace", "start": 9, "end": 7}, "sourceEventSeqs": [9, 7],
         "data": {"id": "summary-2", "role": "user", "source": {"kind": "plugin", "plugin": "compaction"},
                  "content": [{"type": "text", "text": "Summary: the user received the file contents."}]}},
    ])
    example = convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))
    assert example.messages == [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Summary: the user received the file contents."},
    ]


@pytest.mark.parametrize("start,end", [(0, 6), (6, 1), (1, 100)])
def test_deepseek_rejects_invalid_surface_replacement(tmp_path, events, start, end):
    events[8]["surfaceOp"] = {"op": "replace", "start": start, "end": end}
    with pytest.raises(ValueError, match="(?i)(surface|replace|range)"):
        convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zstd", ".jsonl.zst"])
def test_deepseek_local_loader_and_cli(tmp_path, events, suffix):
    source = tmp_path / "traces"
    source.mkdir()
    trace = write_trace(source / ("session" + suffix), events)
    assert convert_trace_to_training_example(trace).prompt == "Read the example file."
    rows = convert_traces_to_training_data(source)
    assert len(rows) == 1
    assert rows[0]["metadata"]["trace_type"] == "deepseek_harness"
    dataset = load_traces(source)
    assert len(dataset) == 1
    assert dataset[0]["messages"][-1]["content"] == "The file says: Hello, world."
    output = tmp_path / "converted.jsonl"
    result = CliRunner().invoke(app, ["convert", str(source), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["metadata"]["trace_type"] == "deepseek_harness"


def test_deepseek_huggingface_download_includes_compressed_logs(tmp_path, events):
    write_trace(tmp_path / "session.jsonl.zstd", events)
    with patch("teich.loader.snapshot_download", return_value=str(tmp_path)) as download:
        dataset = load_traces("example/deepseek-traces")
    patterns = download.call_args.kwargs["allow_patterns"]
    assert {"*.jsonl.zstd", "**/*.jsonl.zstd", "*.jsonl.zst", "**/*.jsonl.zst"} <= set(patterns)
    assert len(dataset) == 1
    assert dataset[0]["metadata"]["trace_type"] == "deepseek_harness"


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zstd"])
def test_deepseek_directory_uses_latest_migration_only(tmp_path, events, suffix):
    original = write_trace(tmp_path / ("session" + suffix), events)
    for version in [1, 2]:
        events[0]["version"] = version
        events[8]["data"]["message"]["content"][0]["text"] = f"Version {version} answer."
        write_trace(tmp_path / (f"session.v{version}" + suffix), events)
    rows = convert_traces_to_training_data(tmp_path)
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "Version 2 answer."
    assert convert_trace_to_training_example(original).messages[-1]["content"] == "The file says: Hello, world."


@pytest.mark.parametrize("filenames", [
    ("session.v1.jsonl", "session.v2.jsonl"),
    ("session.v1.jsonl.zstd", "session.v2.jsonl.zstd"),
    ("session.jsonl", "session.jsonl.zst"),
])
def test_deepseek_directory_preserves_distinct_sessions(tmp_path, events, filenames):
    for index, filename in enumerate(filenames):
        events[0].update(id=f"session-{index}", version=index + 1 if ".v" in filename else 0)
        events[8]["data"]["message"]["content"][0]["text"] = f"Answer {index}."
        write_trace(tmp_path / filename, events)

    rows = convert_traces_to_training_data(tmp_path)

    assert {row["metadata"]["session_id"]: row["messages"][-1]["content"] for row in rows} == {
        "session-0": "Answer 0.", "session-1": "Answer 1.",
    }


@pytest.mark.parametrize("is_error", [True, False, None])
def test_deepseek_tool_error_status_survives_conversion(tmp_path, events, is_error):
    result = events[7]["data"]["message"]["content"][0]
    if is_error is None:
        result.pop("isError")
    else:
        result["isError"] = is_error
    raw_trace = write_trace(tmp_path / "session.jsonl", events)

    row = convert_trace_to_training_example(raw_trace).to_dict()
    expected = {"role": "tool", "name": "read_file", "tool_call_id": "read-1", "content": "Hello, world."}
    if is_error is True:
        expected["is_error"] = True
    assert row["messages"][3] == expected

    converted = write_trace(tmp_path / "converted.jsonl", [row])
    assert load_traces(converted)[0]["messages"][3] == expected


def test_deepseek_duplicate_plain_and_compressed_version_is_ambiguous(tmp_path, events):
    write_trace(tmp_path / "session.jsonl", events)
    write_trace(tmp_path / "session.jsonl.zstd", events)
    with pytest.raises(ValueError, match="(?i)(ambiguous|duplicate|multiple)"):
        convert_traces_to_training_data(tmp_path)


def test_deepseek_tolerant_directory_loading_deduplicates_migrations(tmp_path, events):
    write_trace(tmp_path / "session.jsonl", events)
    events[0]["version"] = 2
    migrated = write_trace(tmp_path / "session.v2.jsonl", events)
    migrated.write_text("{broken json\n" + migrated.read_text(encoding="utf-8"), encoding="utf-8")

    rows = convert_traces_to_training_data(tmp_path, skip_invalid_lines=True)

    assert len(rows) == 1
    assert rows[0]["metadata"]["source_file"] == "session.v2.jsonl"


@pytest.mark.parametrize("block_type", ["image", "file"])
def test_deepseek_unresolved_attachments_fail_explicitly(tmp_path, events, block_type):
    events[2]["data"]["content"].append({"type": block_type, "attachment": {"id": "synthetic-attachment"}})
    with pytest.raises(ValueError, match="(?i)(unsupported|attachment)"):
        convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))


def test_deepseek_invalid_json_line_fails(tmp_path, events):
    trace = write_trace(tmp_path / "session.jsonl", events)
    with trace.open("a", encoding="utf-8") as stream:
        stream.write("{broken json\n")
    with pytest.raises(ValueError):
        convert_trace_to_training_example(trace)


def test_deepseek_reads_concatenated_zstd_frames(tmp_path, events):
    trace = tmp_path / "session.jsonl.zstd"
    frames = []
    for batch in [events[:5], events[5:]]:
        data = ("\n".join(json.dumps(event) for event in batch) + "\n").encode()
        frames.append(zstandard.ZstdCompressor().compress(data))
    trace.write_bytes(b"".join(frames))
    example = convert_trace_to_training_example(trace)
    assert example.prompt == "Read the example file."
    assert example.messages[-1]["content"] == "The file says: Hello, world."


def test_deepseek_directory_can_skip_invalid_first_line(tmp_path, events):
    trace = write_trace(tmp_path / "session.jsonl", events)
    trace.write_bytes(b"{broken json\n" + trace.read_bytes())
    rows = convert_traces_to_training_data(tmp_path, skip_invalid_lines=True)
    assert len(rows) == 1
    assert rows[0]["metadata"]["trace_type"] == "deepseek_harness"
    assert rows[0]["messages"][-1]["content"] == "The file says: Hello, world."


@pytest.mark.parametrize("arguments,expected", [
    ('{"path":"config.json","content":"{\\"enabled\\":true}"}',
     {"path": "config.json", "content": '{"enabled":true}'}),
    ('{"path":', '{"path":'),
])
def test_deepseek_tool_arguments_preserve_literal_values(tmp_path, events, arguments, expected):
    call = events[5]["data"]["message"]["content"][-1]
    call.update(name="write_file", arguments=arguments)
    events[6]["data"].update(name="write_file", arguments=arguments)
    example = convert_trace_to_training_example(write_trace(tmp_path / "session.jsonl", events))
    function = example.messages[2]["tool_calls"][0]["function"]
    assert function == {"name": "write_file", "arguments": expected}
