"""Tests for the Teich Studio backend (project state, events, API)."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from teich.cli import _configure_studio_event_loop_policy
from teich.config import Config
from teich.extract import CURSOR_EXTRACTION_NOTICE
from teich.runner import ClaudeCodeRunner, SessionProgressUpdate
from teich.studio.events import summarize_chat_row, summarize_event, summarize_trace_events
from teich.studio.generation import RUNNER_CLASSES, GenerationJob
from teich.studio.interactive import InteractiveSession
from teich.studio.project import ProjectState
from teich.studio import server as server_module
from teich.studio.server import (
    _settle_terminal_tasks,
    _wait_for_terminal_session_ready,
    create_app,
    detect_trace_provider,
)


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as test_client:
        test_client.project_dir = tmp_path
        yield test_client


# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------

def test_ensure_initialized_creates_files(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "prompts.jsonl").exists()


def test_write_config_merges_and_validates(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    merged = state.write_config_data({"model": {"model": "test/model"}, "max_concurrency": 4})
    assert merged["model"]["model"] == "test/model"
    assert merged["max_concurrency"] == 4
    on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert on_disk["model"]["model"] == "test/model"
    # untouched section preserved
    assert on_disk["agent"]["provider"]


def test_write_config_rejects_invalid(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    with pytest.raises(Exception):
        state.write_config_data({"max_concurrency": 0})


def test_prompts_round_trip(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    rows = [
        {"prompt": "Build a CLI"},
        {"prompt": "Fix the bug", "system": "Be terse", "follow_up_prompts": ["Add tests"]},
    ]
    state.write_prompts(rows)
    loaded = state.read_prompts()
    assert loaded[0] == {"prompt": "Build a CLI"}
    assert loaded[1]["follow_up_prompts"] == ["Add tests"]


def test_read_prompts_supports_csv_prompt_file(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    (tmp_path / "prompts.csv").write_text(
        "prompt,system,github_repo\n"
        "Build the feature,Be concise,owner/repo\n",
        encoding="utf-8",
    )
    state.write_config_data({"prompts_file": "prompts.csv"})

    assert state.read_prompts() == [
        {"prompt": "Build the feature", "system": "Be concise", "github_repo": "owner/repo"}
    ]


def test_write_prompts_migrates_non_jsonl_prompt_file(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    (tmp_path / "prompts.csv").write_text("prompt\nold prompt\n", encoding="utf-8")
    state.write_config_data({"prompts_file": "prompts.csv"})

    path = state.write_prompts([{"prompt": "new prompt"}])

    assert path == tmp_path / "prompts.jsonl"
    assert json.loads(path.read_text(encoding="utf-8").strip()) == {"prompt": "new prompt"}
    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert config["prompts_file"] == "prompts.jsonl"


def test_import_prompts_append_and_replace(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    state.write_prompts([{"prompt": "one"}])
    state.import_prompts_text('{"prompt": "two"}\n"three"\n', replace=False)
    assert [row["prompt"] for row in state.read_prompts()] == ["one", "two", "three"]
    state.import_prompts_text('{"prompt": "only"}', replace=True)
    assert [row["prompt"] for row in state.read_prompts()] == ["only"]


def test_import_prompts_rejects_non_jsonl_upload_filename(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()

    state.import_prompts_text('{"prompt": "accepted"}', replace=True, filename="prompts.ndjson")

    assert [row["prompt"] for row in state.read_prompts()] == ["accepted"]
    with pytest.raises(ValueError, match="JSONL or NDJSON"):
        state.import_prompts_text('{"prompt": "rejected"}', replace=True, filename="prompts.txt")


# ---------------------------------------------------------------------------
# Event summarizers
# ---------------------------------------------------------------------------

def test_summarize_codex_events():
    assistant = {
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    }
    tool = {
        "type": "response_item",
        "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"},
    }
    assert summarize_event("codex", assistant)[0]["kind"] == "assistant"
    tool_events = summarize_event("codex", tool)
    assert tool_events[0]["kind"] == "tool_call"
    assert tool_events[0]["name"] == "exec_command"


def test_summarize_claude_stream_json():
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "done"},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ]
        },
    }
    kinds = [e["kind"] for e in summarize_event("claude-code", event)]
    assert kinds == ["thinking", "assistant", "tool_call"]


def test_summarize_hermes_external():
    event = {
        "type": "external_message",
        "role": "assistant",
        "content": "answer",
        "tool_calls": [{"function": {"name": "skill_manage", "arguments": "{}"}}],
    }
    kinds = [e["kind"] for e in summarize_event("hermes", event)]
    assert kinds == ["tool_call", "assistant"]


def test_summarize_trace_events_includes_user_turns():
    events = [
        {"type": "external_session_meta", "payload": {}},
        {"type": "external_message", "role": "user", "content": "question"},
        {"type": "external_message", "role": "assistant", "content": "answer"},
    ]
    display = summarize_trace_events("hermes", events)
    assert [e["kind"] for e in display] == ["user", "assistant"]


def test_summarize_codex_trace_includes_response_item_user_turns():
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "build a CLI"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
    ]

    display = summarize_trace_events("codex", events)

    assert [(event["kind"], event["text"]) for event in display] == [
        ("user", "build a CLI"),
        ("assistant", "done"),
    ]


def test_summarize_chat_row():
    row = {
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "thinking": "greeting"},
        ]
    }
    kinds = [e["kind"] for e in summarize_chat_row(row)]
    assert kinds == ["system", "user", "thinking", "assistant"]


def test_detect_trace_provider():
    assert detect_trace_provider([{"type": "session_meta", "payload": {}}]) == "codex"
    assert detect_trace_provider([{"type": "session", "id": "x"}]) == "pi"
    assert detect_trace_provider([{"type": "external_session_meta"}]) == "hermes"
    assert detect_trace_provider([{"type": "user", "sessionId": "abc"}]) == "claude-code"
    assert detect_trace_provider([{"source": "cli", "hermes_source": "cli", "messages": []}]) == "hermes"
    assert detect_trace_provider([{"metadata": {"trace_type": "cursor"}, "messages": []}]) == "cursor"
    assert detect_trace_provider([{"messages": []}]) == "chat"


def test_summarize_structured_cursor_and_native_hermes_rows():
    row = {
        "messages": [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": "done"},
        ]
    }

    assert [event["kind"] for event in summarize_trace_events("cursor", [row])] == ["user", "assistant"]
    assert [event["kind"] for event in summarize_trace_events("hermes", [{"source": "cli", **row}])] == [
        "user",
        "assistant",
    ]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

def test_status_endpoint(client):
    payload = client.get("/api/status").json()
    assert payload["config_exists"] is True
    assert payload["prompts_count"] == 0
    assert {p["id"] for p in payload["providers"]} == {"pi", "codex", "claude-code", "hermes", "chat"}


def test_status_counts_inline_config_prompts(client):
    response = client.put(
        "/api/config",
        json={"config": {"prompts_file": None, "prompts": [{"prompt": "inline prompt"}]}},
    )
    assert response.status_code == 200

    payload = client.get("/api/status").json()

    assert payload["prompts_count"] == 1


def test_status_reports_config_error_for_invalid_config(client):
    (client.project_dir / "config.yaml").write_text("not a mapping\n", encoding="utf-8")

    response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["config_error"]
    assert payload["prompts_count"] == -1
    assert payload["prompts_file"].endswith("prompts.jsonl")


def test_config_endpoints(client):
    config = client.get("/api/config").json()["config"]
    assert config["agent"]["provider"]
    response = client.put("/api/config", json={"config": {"model": {"model": "acme/model-1"}}})
    assert response.status_code == 200
    assert response.json()["config"]["model"]["model"] == "acme/model-1"
    bad = client.put("/api/config", json={"config": {"max_concurrency": -1}})
    assert bad.status_code == 400


def test_config_rejects_chat_with_direct_anthropic_api(client):
    response = client.put(
        "/api/config",
        json={
            "config": {
                "agent": {"provider": "chat"},
                "api": {"provider": "anthropic", "base_url": "https://api.anthropic.com"},
            }
        },
    )

    assert response.status_code == 400
    assert "OpenAI-compatible" in response.json()["detail"]
    assert client.get("/api/config").json()["config"]["agent"]["provider"] != "chat"

    custom_direct = client.put(
        "/api/config",
        json={
            "config": {
                "agent": {"provider": "chat"},
                "api": {"provider": "custom", "base_url": "https://api.anthropic.com/"},
            }
        },
    )
    assert custom_direct.status_code == 400
    assert "OpenAI-compatible" in custom_direct.json()["detail"]


def test_session_override_rejects_chat_with_direct_anthropic_api(client):
    response = client.put(
        "/api/config",
        json={
            "config": {
                "agent": {"provider": "claude-code"},
                "api": {"provider": "anthropic", "base_url": "https://api.anthropic.com"},
            }
        },
    )
    assert response.status_code == 200

    session = client.post("/api/sessions", json={"provider": "chat"})

    assert session.status_code == 400
    assert "OpenAI-compatible" in session.json()["detail"]


def test_prompts_endpoints(client):
    response = client.put(
        "/api/prompts",
        json={"prompts": [{"prompt": "hello world", "follow_up_prompts": ["again"]}]},
    )
    assert response.status_code == 200
    prompts = client.get("/api/prompts").json()["prompts"]
    assert prompts[0]["prompt"] == "hello world"

    imported = client.post(
        "/api/prompts/import",
        json={"text": '{"prompt": "uploaded"}', "replace": False, "filename": "prompts.jsonl"},
    )
    assert imported.status_code == 200
    assert len(imported.json()["prompts"]) == 2

    invalid = client.post("/api/prompts/import", json={"text": "{not json", "replace": False})
    assert invalid.status_code == 400
    unsupported = client.post(
        "/api/prompts/import",
        json={"text": '{"prompt": "uploaded"}', "replace": False, "filename": "prompts.txt"},
    )
    assert unsupported.status_code == 400
    assert "JSONL or NDJSON" in unsupported.json()["detail"]


def test_trace_listing_and_preview(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "hermes-agent-test.jsonl"
    events = [
        {"type": "external_session_meta", "payload": {"id": "x"}},
        {"type": "external_message", "role": "user", "content": "do the thing"},
        {"type": "external_message", "role": "assistant", "content": "did the thing"},
    ]
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (output_dir / "failures").mkdir()
    (output_dir / "failures" / "bad.jsonl").write_text("{}\n", encoding="utf-8")

    listing = client.get("/api/traces").json()["traces"]
    assert [t["name"] for t in listing] == ["hermes-agent-test.jsonl"]

    preview = client.get("/api/traces/preview", params={"name": "hermes-agent-test.jsonl"}).json()
    assert preview["provider"] == "hermes"
    assert [e["kind"] for e in preview["display"]] == ["user", "assistant"]

    missing = client.get("/api/traces/preview", params={"name": "nope.jsonl"})
    assert missing.status_code == 404
    escape = client.get("/api/traces/preview", params={"name": "../config.yaml"})
    assert escape.status_code in {400, 404}


def test_dataset_preview_endpoint_returns_rows_features(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "hermes-agent-test.jsonl"
    events = [
        {"type": "external_session_meta", "payload": {"id": "x", "model": "test/model"}},
        {"type": "external_message", "role": "user", "content": "build a preview"},
        {"type": "external_message", "role": "assistant", "content": "preview built"},
    ]
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text("---\nconfigs: []\n---\n# Demo\n", encoding="utf-8")
    client.put("/api/config", json={"config": {"publish": {"repo_id": "TeichAI/demo-dataset"}}})

    response = client.get("/api/dataset-preview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hf_embed_url"] == "https://huggingface.co/datasets/TeichAI/demo-dataset/embed/viewer"
    assert payload["dataset"]["num_rows"] == 1
    assert {feature["name"] for feature in payload["dataset"]["features"]} >= {"messages", "metadata"}
    assert payload["dataset"]["rows"][0]["preview"]["prompt"] == "build a preview"
    assert "trace_previews" not in payload
    assert payload["readme"].startswith("---")

    filtered = client.get("/api/dataset-preview", params={"search": "missing"}).json()
    assert filtered["dataset"]["num_rows"] == 0


def test_dataset_preview_tolerates_malformed_jsonl_and_counts_user_turns_and_tool_calls(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    valid_trace = output_dir / "codex-good.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list files"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "bash",
                "call_id": "call-1",
                "arguments": {"command": "ls"},
            },
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call-1", "output": "README.md"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
    ]
    valid_trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    (output_dir / "bad.jsonl").write_text("not json\n", encoding="utf-8")

    response = client.get("/api/dataset-preview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["errors"] == []
    assert payload["dataset"]["num_rows"] == 1
    preview = payload["dataset"]["rows"][0]["preview"]
    assert preview["message_count"] == 1
    assert preview["tool_count"] == 1


def test_dataset_upload_endpoint_generates_readme_and_uploads_with_env_token(client, monkeypatch):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "prompt": "hello",
        "response": "hi",
        "model": "gpt-test",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "dataset_tool",
                    "description": "Tool recovered from the dataset row",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
    }
    (output_dir / "chat.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf-env-token")

    with patch("teich.studio.server.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/studio-dataset"
        mock_api_cls.return_value = mock_api

        response = client.post(
            "/api/dataset-preview/upload",
            json={"repo_id": "armand0e/studio-dataset"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_id"] == "armand0e/studio-dataset"
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Use this dataset as `armand0e/studio-dataset`" in readme
    assert "dataset = load_traces('armand0e/studio-dataset')" not in readme
    assert "run `teich convert`" in readme
    assert "https://github.com/TeichAI/teich/blob/main/docs/training.md" in readme
    assert '"name": "dataset_tool"' in readme
    assert '"description": "Tool recovered from the dataset row"' in readme
    mock_api_cls.assert_called_once_with(token="hf-env-token")
    mock_api.create_repo.assert_called_once_with(
        repo_id="armand0e/studio-dataset",
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    mock_api.delete_file.assert_called_once_with(
        path_in_repo="tools.json",
        repo_id="armand0e/studio-dataset",
        repo_type="dataset",
        commit_message="Remove legacy teich tools snapshot",
    )
    mock_api.upload_large_folder.assert_called_once_with(
        repo_id="armand0e/studio-dataset",
        folder_path=str(output_dir),
        repo_type="dataset",
        private=False,
        ignore_patterns=["partials/**", "failures/**", "README.md", "tools.json"],
    )
    mock_api.upload_folder.assert_called_once_with(
        folder_path=str(output_dir),
        repo_id="armand0e/studio-dataset",
        repo_type="dataset",
        commit_message="Upload teich dataset metadata",
        allow_patterns=["README.md"],
        ignore_patterns=["partials/**", "failures/**"],
    )
    preview = client.get("/api/dataset-preview").json()
    assert preview["repo_id"] == "armand0e/studio-dataset"
    assert preview["hf_embed_url"] == "https://huggingface.co/datasets/armand0e/studio-dataset/embed/viewer"


def test_dataset_upload_preserves_extracted_cursor_readme_metadata(client, monkeypatch):
    config_response = client.put(
        "/api/config",
        json={"config": {"agent": {"provider": "pi"}, "model": {"model": "wrong-default-model"}}},
    )
    assert config_response.status_code == 200
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    row = {
        "messages": [
            {"role": "user", "content": "inspect cursor session"},
            {"role": "assistant", "content": "done"},
        ],
        "prompt": "inspect cursor session",
        "metadata": {
            "trace_type": "cursor",
            "source": "cursor",
            "cursor_table": "composerData:test",
        },
    }
    (output_dir / "cursor-session.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf-env-token")

    with patch("teich.studio.server.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/cursor-dataset"
        mock_api_cls.return_value = mock_api

        response = client.post(
            "/api/dataset-preview/upload",
            json={"repo_id": "armand0e/cursor-dataset"},
        )

    assert response.status_code == 200
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert '- "cursor"' in readme
    assert '- "pi"' not in readme
    assert '- "wrong-default-model"' not in readme
    assert "Model metadata:" not in readme
    assert "teich extract cursor --out data" in readme


def test_dataset_row_delete_removes_single_backing_trace(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "hermes-agent-test.jsonl"
    events = [
        {"type": "external_session_meta", "payload": {"id": "x"}},
        {"type": "external_message", "role": "user", "content": "delete me"},
        {"type": "external_message", "role": "assistant", "content": "deleted"},
    ]
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    response = client.delete("/api/dataset-preview/rows/0")

    assert response.status_code == 200
    assert response.json()["mode"] == "file"
    assert not trace.exists()


def test_dataset_row_update_rewrites_single_raw_trace_as_structured_row(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "hermes-agent-test.jsonl"
    events = [
        {"type": "external_session_meta", "payload": {"id": "x"}},
        {"type": "external_message", "role": "user", "content": "old prompt"},
        {"type": "external_message", "role": "assistant", "content": "old answer"},
    ]
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    updated = {
        "prompt": "new prompt",
        "messages": [
            {"role": "user", "content": "new prompt"},
            {"role": "assistant", "content": "new answer"},
        ],
        "tools": [],
        "metadata": {"source_file": "hermes-agent-test.jsonl"},
    }

    response = client.put("/api/dataset-preview/rows/0", json={"row": updated})

    assert response.status_code == 200
    assert response.json()["mode"] == "file"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows == [updated]
    preview = client.get("/api/dataset-preview").json()
    assert preview["dataset"]["rows"][0]["preview"]["prompt"] == "new prompt"


def test_dataset_row_update_replaces_structured_jsonl_line(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "rows.jsonl"
    original_rows = [
        {
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
            "tools": [],
        },
        {
            "messages": [
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "tools": [],
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in original_rows) + "\n", encoding="utf-8")
    updated = {
        "messages": [
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "updated"},
        ],
        "tools": [],
        "metadata": {"source_file": "rows.jsonl", "source_line": 2},
    }

    response = client.put("/api/dataset-preview/rows/1", json={"row": updated})

    assert response.status_code == 200
    assert response.json()["mode"] == "line"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == original_rows[0]
    assert rows[1] == updated


def test_dataset_row_edit_and_delete_refuse_malformed_backing_jsonl(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "rows.jsonl"
    original_row = {
        "messages": [
            {"role": "user", "content": "keep me"},
            {"role": "assistant", "content": "kept"},
        ],
        "tools": [],
    }
    original_text = json.dumps(original_row) + "\nnot json\n"
    trace.write_text(original_text, encoding="utf-8")
    updated = {
        "messages": [
            {"role": "user", "content": "changed"},
            {"role": "assistant", "content": "changed"},
        ],
        "tools": [],
        "metadata": {"source_file": "rows.jsonl", "source_line": 1},
    }

    preview = client.get("/api/dataset-preview")

    assert preview.status_code == 200
    assert preview.json()["dataset"]["num_rows"] == 1

    update_response = client.put("/api/dataset-preview/rows/0", json={"row": updated})
    delete_response = client.delete("/api/dataset-preview/rows/0")

    assert update_response.status_code == 400
    assert "Cannot edit malformed JSONL file rows.jsonl at line 2" in update_response.json()["detail"]
    assert delete_response.status_code == 400
    assert "Cannot edit malformed JSONL file rows.jsonl at line 2" in delete_response.json()["detail"]
    assert trace.read_text(encoding="utf-8") == original_text


def _wait_for_extract_job(client) -> dict:
    delay = threading.Event()
    for _ in range(200):
        job = client.get("/api/extract").json()["job"]
        if job and job["status"] in {"completed", "failed"}:
            return job
        delay.wait(timeout=0.01)
    raise AssertionError("extraction job did not finish")


def test_extract_endpoint_accepts_provider_home_and_anonymizes(client):
    source_home = client.project_dir / ".codex"
    sessions_dir = source_home / "sessions"
    sessions_dir.mkdir(parents=True)
    trace = sessions_dir / "trace.jsonl"
    events = [
        {"type": "session_meta", "payload": {"model": "claude-fable-5"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "email arin@company.ai"}],
            },
        },
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    response = client.post(
        "/api/extract",
        json={
            "provider": "codex",
            "output": "staged",
            "sessions_dirs": [str(source_home)],
            "model": "fable-5",
        },
    )

    assert response.status_code == 200
    job = _wait_for_extract_job(client)
    assert job["status"] == "completed"
    assert job["result_files"] == ["trace.jsonl"]
    assert job["result_rows"] == 2
    output_trace = client.project_dir / "staged" / "trace.jsonl"
    assert output_trace.exists()
    text = output_trace.read_text(encoding="utf-8")
    assert "arin@company.ai" not in text
    readme = (client.project_dir / "staged" / "README.md").read_text(encoding="utf-8")
    assert '- "codex"' in readme
    assert '- "pi"' not in readme
    assert '- "fable-5"' not in readme
    assert "Model metadata:" not in readme
    assert "teich extract codex --out data" in readme
    listing = client.get("/api/traces").json()["traces"]
    assert [trace["name"] for trace in listing] == ["trace.jsonl"]


def test_extract_endpoint_can_skip_anonymization(client):
    sessions_dir = client.project_dir / ".codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "raw.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"model": "test-model", "email": "raw@company.ai"}}) + "\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/extract",
        json={
            "provider": "codex",
            "output": "raw-staged",
            "sessions_dirs": [str(sessions_dir)],
            "skip_anonymize": True,
        },
    )

    assert response.status_code == 200
    job = _wait_for_extract_job(client)
    assert job["status"] == "completed"
    assert "raw@company.ai" in (client.project_dir / "raw-staged" / "raw.jsonl").read_text(encoding="utf-8")


def test_extract_endpoint_warns_cursor_may_take_a_while(client):
    source = client.project_dir / "state.vscdb"
    source.write_text("", encoding="utf-8")

    def fake_extract(provider, *, output_dir, sources=None, model_filter=None, clear_destination=False, progress=None, anonymize=False):
        assert provider == "cursor"
        assert sources == [source]
        assert model_filter is None
        assert clear_destination is True
        assert progress is not None
        progress({"kind": "extract_progress", "text": "Scanning Cursor database..."})
        output_dir.mkdir(parents=True, exist_ok=True)
        trace = output_dir / "cursor-session.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ],
                    "metadata": {"trace_type": "cursor"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(source_paths=[source], copied_files=[trace], count=1)

    with patch("teich.studio.extraction.extract_local_sessions", side_effect=fake_extract):
        response = client.post(
            "/api/extract",
            json={
                "provider": "cursor",
                "output": "cursor-staged",
                "sessions_dirs": [str(source)],
                "skip_anonymize": True,
            },
        )

    assert response.status_code == 200
    job = _wait_for_extract_job(client)
    assert job["status"] == "completed"
    events = client.app.state.extraction.current().events.snapshot()
    assert any(
        event.get("kind") == "extract_warning" and event.get("text") == CURSOR_EXTRACTION_NOTICE
        for event in events
    )
    assert any(
        event.get("kind") == "extract_progress" and event.get("text") == "Scanning Cursor database..."
        for event in events
    )


def test_extract_sources_rejects_unknown_provider(client):
    response = client.get("/api/extract/sources", params={"provider": "unknown"})

    assert response.status_code == 400
    assert "Provider must be one of" in response.json()["detail"]


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Teich Studio" in response.text


def test_session_endpoints_validation(client):
    missing = client.get("/api/sessions/does-not-exist")
    assert missing.status_code == 404
    assert client.get("/api/sessions").json()["sessions"] == []


def test_chat_session_discard_rejected_while_turn_running(tmp_path):
    config = Config(
        agent={"provider": "chat"},
        model={"model": "test/model"},
        api={"provider": "openai", "base_url": "https://api.openai.com/v1"},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    session = InteractiveSession(config)
    session.status = "running"
    session._busy = True

    with pytest.raises(RuntimeError, match="current turn"):
        session.discard()

    assert session.status == "running"


def test_claude_terminal_forwards_effort_and_fallback_model(tmp_path):
    config = Config(
        agent={"provider": "claude-code", "claude": {"fallback_model": ["sonnet", "haiku"]}},
        model={"model": "claude-opus-4-8", "reasoning_effort": "high"},
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    session = InteractiveSession(config)
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        session._runner = ClaudeCodeRunner(config)

    command = session._native_cli_command()

    assert command == [
        "claude",
        "--model",
        "claude-opus-4-8",
        "--permission-mode",
        "bypassPermissions",
        "--effort",
        "high",
        "--fallback-model",
        "sonnet,haiku",
    ]


def test_generation_stop_prevents_later_prompt_starts(tmp_path, monkeypatch):
    first_started = threading.Event()
    stop_called = threading.Event()
    launched: list[int] = []

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run_all(self, *, max_concurrency, progress_callback, prompt_inputs, resume):
            for prompt_index, prompt_input in enumerate(prompt_inputs, start=1):
                progress_callback(
                    SessionProgressUpdate(
                        prompt_id=f"prompt-{prompt_index}",
                        prompt_index=prompt_index,
                        total_prompts=len(prompt_inputs),
                        prompt=prompt_input.prompt,
                        prompt_preview=prompt_input.prompt,
                        status="running",
                    )
                )
                launched.append(prompt_index)
                if prompt_index == 1:
                    first_started.set()
                    assert stop_called.wait(timeout=2)
            return []

        def _terminate_active_processes(self):
            stop_called.set()

    monkeypatch.setitem(RUNNER_CLASSES, "chat", FakeRunner)
    config = Config(
        agent={"provider": "chat"},
        model={"model": "test/model"},
        api={"provider": "openai", "base_url": "https://api.openai.com/v1"},
        prompts=[{"prompt": "one"}, {"prompt": "two"}],
        prompts_file=None,
        output={"traces_dir": tmp_path / "output", "sandbox_dir": tmp_path / "sandbox"},
    )
    job = GenerationJob(config)

    job.start()
    assert first_started.wait(timeout=2)
    job.stop()
    wait_for_worker = threading.Event()
    for _ in range(200):
        if job.finished_at is not None:
            break
        wait_for_worker.wait(timeout=0.01)

    assert job.finished_at is not None
    assert launched == [1]
    assert job.status == "stopped"


def test_studio_uses_selector_event_loop_policy_on_windows():
    class DummyPolicy:
        pass

    class DummyAsyncio:
        WindowsSelectorEventLoopPolicy = DummyPolicy

        def __init__(self) -> None:
            self.policy = None

        def set_event_loop_policy(self, policy) -> None:
            self.policy = policy

    asyncio_module = DummyAsyncio()

    assert _configure_studio_event_loop_policy(platform="win32", asyncio_module=asyncio_module)
    assert isinstance(asyncio_module.policy, DummyPolicy)


def test_studio_event_loop_policy_noop_off_windows():
    class DummyAsyncio:
        def set_event_loop_policy(self, policy) -> None:
            raise AssertionError("event loop policy should not be changed off Windows")

    assert not _configure_studio_event_loop_policy(platform="linux", asyncio_module=DummyAsyncio())


def test_terminal_task_cleanup_suppresses_disconnects():
    async def run() -> None:
        async def disconnect() -> None:
            raise WebSocketDisconnect(code=1005)

        async def wait_forever() -> None:
            await asyncio.sleep(60)

        done_task = asyncio.create_task(disconnect())
        pending_task = asyncio.create_task(wait_forever())
        await asyncio.sleep(0)

        await _settle_terminal_tasks({done_task}, {pending_task})

        assert pending_task.cancelled()

    asyncio.run(run())


def test_terminal_task_cleanup_propagates_unexpected_errors():
    async def run() -> None:
        async def fail() -> None:
            raise ValueError("boom")

        async def wait_forever() -> None:
            await asyncio.sleep(60)

        done_task = asyncio.create_task(fail())
        pending_task = asyncio.create_task(wait_forever())
        await asyncio.sleep(0)

        with pytest.raises(ValueError, match="boom"):
            await _settle_terminal_tasks({done_task}, {pending_task})

        assert pending_task.cancelled()

    asyncio.run(run())


def test_terminal_wait_keeps_socket_open_until_session_ready(monkeypatch):
    class DummyWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = []

        async def send_json(self, message: dict[str, str]) -> None:
            self.messages.append(message)

    class DummySession:
        status = "starting"

    websocket = DummyWebSocket()
    session = DummySession()
    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 3:
            session.status = "ready"

    monkeypatch.setattr(server_module.asyncio, "sleep", fake_sleep)

    asyncio.run(
        _wait_for_terminal_session_ready(
            websocket,
            session,
            notice_seconds=0,
            sleep_seconds=0.01,
        )
    )

    assert sleep_calls == 3
    assert websocket.messages
    assert websocket.messages[0]["type"] == "status"
