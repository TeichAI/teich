import json
import os
import sqlite3
from pathlib import Path
import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from teich import anonymize as anonymize_module
from teich.converter import convert_traces_to_training_data
from teich import extract as extract_module
from teich.extract import CURSOR_EXTRACTION_NOTICE, default_session_sources
from teich.cli import app

runner = CliRunner()
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_cli_output(output: str) -> str:
    return ANSI_ESCAPE_RE.sub("", output)


def _read_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_extracted_jsonl_rows(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_dir.glob("*.jsonl")):
        rows.extend(_read_jsonl_rows(path))
    return rows


def _write_minimal_hermes_state_db(state_db: Path, *, session_id: str = "session/1", model: str = "test-model") -> None:
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at TEXT, model TEXT)")
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)"
        )
        connection.execute(
            "INSERT INTO sessions (id, source, started_at, model) VALUES (?, ?, ?, ?)",
            (session_id, "cli", "2026-06-13T00:00:00Z", model),
        )
        connection.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, "user", "hello", "2026-06-13T00:00:01Z"),
        )
        connection.commit()
    finally:
        connection.close()


def _cursor_rich_text(value: str) -> str:
    children = []
    if value:
        children.append(
            {
                "detail": 0,
                "format": 0,
                "mode": "normal",
                "style": "",
                "text": value,
                "type": "text",
                "version": 1,
            }
        )
    return json.dumps(
        {
            "root": {
                "children": [
                    {
                        "children": children,
                        "direction": None,
                        "format": "",
                        "indent": 0,
                        "type": "paragraph",
                        "version": 1,
                    }
                ],
                "direction": None,
                "format": "",
                "indent": 0,
                "type": "root",
                "version": 1,
            }
        }
    )


def _write_minimal_cursor_state_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("CREATE TABLE composerSessions (id TEXT PRIMARY KEY, data TEXT)")
        connection.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "workbench.panel.aichat.view.aichat.chatdata",
                json.dumps(
                    {
                        "model": "cursor-fable-5",
                        "messages": [
                            {"role": "user", "content": _cursor_rich_text("")},
                            {"role": "user", "content": _cursor_rich_text("open chat prompt")},
                            {
                                "role": "assistant",
                                "content": "chat answer",
                                "toolCalls": [
                                    {
                                        "id": "call-read",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": {"path": "README.md"},
                                        },
                                    }
                                ],
                            },
                        ],
                        "tools": [
                            {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO composerSessions (id, data) VALUES (?, ?)",
            (
                "composer-1",
                json.dumps(
                    {
                        "modelName": "cursor-fable-5",
                        "composer": {
                            "messages": [
                                {"type": "human", "text": "composer prompt"},
                                {"type": "assistant", "text": "composer answer"},
                                {
                                    "role": "tool",
                                    "content": "tool output",
                                    "tool_call_id": "call-read",
                                    "name": "read_file",
                                },
                            ]
                        },
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "agentKv:blob:internal-state",
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "internal agent prompt"},
                        ]
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_cursor_image_state_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "workbench.panel.aichat.view.aichat.chatdata",
                json.dumps(
                    {
                        "model": "cursor-vision-1",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "make me a website like this"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "abc123",
                                        },
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": "data:image/jpeg;base64,def456",
                                    },
                                ],
                            },
                            {"role": "assistant", "content": "I'll build a similar site."},
                        ],
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_archived_cursor_state_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "composerData:archived-composer",
                json.dumps(
                    {
                        "composerId": "archived-composer",
                        "status": "completed",
                        "text": "archived prompt",
                        "fullConversationHeadersOnly": [
                            {"bubbleId": "user-bubble", "type": 1},
                            {"bubbleId": "assistant-bubble", "type": 2},
                            {"bubbleId": "tool-bubble", "type": 2},
                            {"bubbleId": "final-bubble", "type": 2},
                        ],
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "bubbleId:archived-composer:user-bubble",
                json.dumps(
                    {
                        "bubbleId": "user-bubble",
                        "type": 1,
                        "richText": _cursor_rich_text(""),
                        "modelInfo": {"modelName": "cursor-fable-5"},
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "bubbleId:archived-composer:assistant-bubble",
                json.dumps(
                    {
                        "bubbleId": "assistant-bubble",
                        "type": 2,
                        "text": "archived answer",
                        "thinking": {"text": "archived reasoning"},
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "bubbleId:archived-composer:tool-bubble",
                json.dumps(
                    {
                        "bubbleId": "tool-bubble",
                        "type": 2,
                        "toolFormerData": {
                            "name": "read_file",
                            "toolCallId": "call-read",
                            "rawArgs": '{"path":"README.md"}',
                            "result": "file contents",
                            "status": "completed",
                        },
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "bubbleId:archived-composer:final-bubble",
                json.dumps(
                    {
                        "bubbleId": "final-bubble",
                        "type": 2,
                        "text": "final response",
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_id_keyed_cursor_state_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE cursorState (id TEXT PRIMARY KEY, data TEXT, largeBlob TEXT)")
        composer_id = "id-keyed-composer"
        rows = [
            (
                f"composerData:{composer_id}",
                {
                    "composerId": composer_id,
                    "fullConversationHeadersOnly": [
                        {"bubbleId": "user", "type": 1},
                        {"bubbleId": "assistant", "type": 2},
                    ],
                    "selectedModel": "cursor-fable-5",
                },
            ),
            (
                f"bubbleId:{composer_id}:user",
                {
                    "bubbleId": "user",
                    "type": 1,
                    "richText": _cursor_rich_text("id-keyed prompt"),
                },
            ),
            (
                f"bubbleId:{composer_id}:assistant",
                {
                    "bubbleId": "assistant",
                    "type": 2,
                    "text": "id-keyed answer",
                },
            ),
            (
                f"bubbleId:{composer_id}:unused",
                {
                    "bubbleId": "unused",
                    "type": 2,
                    "text": "unused answer",
                },
            ),
        ]
        for key, value in rows:
            connection.execute(
                "INSERT INTO cursorState (id, data, largeBlob) VALUES (?, ?, ?)",
                (key, json.dumps(value), "x" * 10000),
            )
        connection.commit()
    finally:
        connection.close()


def _write_cursor_escaped_rich_text_model_info_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE cursorData (key TEXT PRIMARY KEY, value TEXT)")
        escaped_empty_editor_state = _cursor_rich_text("").replace('"', '\\"')
        connection.execute(
            "INSERT INTO cursorData (key, value) VALUES (?, ?)",
            (
                "workbench.panel.aichat.view.aichat.chatdata",
                json.dumps(
                    {
                        "modelInfo": {
                            "name": "claude-opus-4.5",
                            "provider": "anthropic",
                        },
                        "prompt": "actual cursor prompt",
                        "messages": [
                            {
                                "role": "user",
                                "content": escaped_empty_editor_state,
                            },
                            {
                                "role": "assistant",
                                "content": "actual cursor answer",
                            },
                        ],
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_cursor_rich_text_only_selected_model_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE cursorData (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO cursorData (key, value) VALUES (?, ?)",
            (
                "workbench.panel.aichat.view.aichat.chatdata.richTextOnly",
                json.dumps(
                    {
                        "selectedModelId": "anthropic/claude-opus-4.5",
                        "messages": [
                            {
                                "role": "user",
                                "richText": _cursor_rich_text("rich text only prompt"),
                            },
                            {
                                "role": "assistant",
                                "richText": _cursor_rich_text("rich text only answer"),
                            },
                        ],
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_unsafe_cursor_state_db(state_db: Path) -> None:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")

        def insert(key: str, value: dict[str, object]) -> None:
            connection.execute("INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)", (key, json.dumps(value)))

        def insert_composer(composer_id: str, bubble_ids: list[str]) -> None:
            insert(
                f"composerData:{composer_id}",
                {
                    "composerId": composer_id,
                    "status": "completed",
                    "fullConversationHeadersOnly": [
                        {"bubbleId": bubble_id, "type": 1 if bubble_id.startswith("user") else 2}
                        for bubble_id in bubble_ids
                    ],
                },
            )

        shared_bubbles = {
            "user-1": {"bubbleId": "user-1", "type": 1, "text": "first prompt"},
            "assistant-1": {"bubbleId": "assistant-1", "type": 2, "text": "first answer"},
            "user-2": {"bubbleId": "user-2", "type": 1, "text": "second prompt"},
            "assistant-2": {"bubbleId": "assistant-2", "type": 2, "text": "second answer"},
            "leading-assistant": {"bubbleId": "leading-assistant", "type": 2, "text": "orphaned start"},
            "trailing-user": {"bubbleId": "trailing-user", "type": 1, "text": "dangling user"},
            "no-user-assistant": {"bubbleId": "no-user-assistant", "type": 2, "text": "no user"},
        }
        for composer_id in ("safe-full", "safe-duplicate"):
            insert_composer(
                composer_id,
                [
                    "leading-assistant",
                    "user-1",
                    "assistant-1",
                    "user-2",
                    "assistant-2",
                    "trailing-user",
                ],
            )
            for bubble_id, bubble in shared_bubbles.items():
                insert(f"bubbleId:{composer_id}:{bubble_id}", bubble)

        insert_composer("contained", ["user-2", "assistant-2"])
        insert("bubbleId:contained:user-2", shared_bubbles["user-2"])
        insert("bubbleId:contained:assistant-2", shared_bubbles["assistant-2"])

        insert_composer("no-user", ["no-user-assistant"])
        insert("bubbleId:no-user:no-user-assistant", shared_bubbles["no-user-assistant"])
        connection.commit()
    finally:
        connection.close()


def test_extract_help_covers_providers_options_defaults_and_examples():
    root_help = runner.invoke(app, ["--help"])

    assert root_help.exit_code == 0
    root_output = _plain_cli_output(root_help.output)
    assert "Extract existing local agent sessions" in root_output
    assert "teich extract claude --model fable-5 --out data" in root_output

    group_help = runner.invoke(app, ["extract", "--help"])

    assert group_help.exit_code == 0
    group_output = _plain_cli_output(group_help.output)
    assert "PROVIDER" in group_output
    for provider in ("claude", "codex", "cursor", "pi", "hermes"):
        assert provider in group_output
    assert "--sessions-dir" in group_output
    assert "--output,--out" in group_output
    assert "--model" in group_output
    assert "--no-anon" in group_output
    assert "--no-anonymize" in group_output
    assert ".claude" in group_output
    assert ".claude/projects" in group_output
    assert ".codex" in group_output
    assert ".codex/sessions" in group_output
    assert ".pi" in group_output
    assert ".pi/agent/sessions" in group_output
    assert ".pi/sessions" in group_output
    assert ".hermes" in group_output
    assert ".hermes/state.db" in group_output
    assert "Cursor/User/workspaceStorage" in group_output
    assert "Cursor/User/globalStorage/state.vscdb" in group_output
    assert "CLAUDE_CONFIG_DIR/projects" in group_output
    assert "CODEX_HOME/sessions" in group_output
    assert "HERMES_STATE_DB" in group_output
    assert "CURSOR_WORKSPACE_STORAGE" in group_output
    assert "fable-5" in group_output
    assert "teich convert data --out teich-training.jsonl" in group_output
    assert "standalone OpenAI-style JSONL rows" in group_output

    for provider in ("claude", "codex", "cursor", "hermes", "pi"):
        result = runner.invoke(app, ["extract", provider, "--help"])

        assert result.exit_code == 0
        output = _plain_cli_output(result.output)
        assert "--output" in output
        assert "--out" in output
        assert "--sessions-dir" in output
        assert "--model" in output
        assert "--no-anon" in output
        assert "--no-anonymize" in output
        assert ".claude" in output
        assert ".codex" in output
        assert ".pi" in output
        assert ".hermes" in output
        assert "Cursor" in output


def test_cursor_default_sources_follow_current_runtime_not_windows_env(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    linux_db = home / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    linux_db.parent.mkdir(parents=True)
    linux_db.write_text("", encoding="utf-8")

    windows_profile = tmp_path / "WindowsUser"
    windows_roaming = windows_profile / "AppData" / "Roaming"
    windows_db = windows_roaming / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    windows_db.parent.mkdir(parents=True)
    windows_db.write_text("", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(windows_roaming))
    monkeypatch.setenv("USERPROFILE", str(windows_profile))

    with patch("teich.extract._running_in_wsl", return_value=True):
        wsl_sources = default_session_sources("cursor", home=home)

    assert linux_db in wsl_sources
    assert windows_db not in wsl_sources

    with patch("teich.extract._running_in_wsl", return_value=False):
        native_sources = default_session_sources("cursor", home=home)

    assert linux_db in native_sources
    assert windows_db in native_sources


def test_extract_codex_from_explicit_sessions_dir(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "session"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(output_dir)],
        input="n\n",
    )

    assert result.exit_code == 0
    extracted = output_dir / "session.jsonl"
    assert extracted.exists()
    assert (output_dir / "README.md").exists()
    assert "Extracted 1 codex trace" in result.output
    assert "Automatically scrambled 0 API keys, 0 email addresses, and 0 username references" in result.output
    assert "Data was written to" in result.output


def test_extract_hermes_from_explicit_state_db(tmp_path: Path):
    state_db = tmp_path / "state.db"
    _write_minimal_hermes_state_db(state_db)
    connection = sqlite3.connect(state_db)
    try:
        connection.execute(
            "INSERT INTO sessions (id, source, started_at, model) VALUES (?, ?, ?, ?)",
            ("session/2", "cli", "2026-06-13T00:00:02Z", "test-model"),
        )
        connection.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("session/2", "user", "second", "2026-06-13T00:00:03Z"),
        )
        connection.commit()
    finally:
        connection.close()

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        ["extract", "hermes", "--sessions-dir", str(state_db), "--output", str(output_dir)],
        input="n\n",
    )

    assert result.exit_code == 0
    extracted_files = sorted(output_dir.glob("hermes-session-*.jsonl"))
    assert [path.name for path in extracted_files] == ["hermes-session-1.jsonl", "hermes-session-2.jsonl"]
    assert all(len(_read_jsonl_rows(path)) == 1 for path in extracted_files)
    rows = _read_extracted_jsonl_rows(output_dir)
    rows_by_id = {row["id"]: row for row in rows}
    assert rows_by_id["session/1"]["source"] == "cli"
    assert rows_by_id["session/1"]["hermes_source"] == "cli"
    assert rows_by_id["session/1"]["messages"][0]["role"] == "user"
    assert rows_by_id["session/1"]["messages"][0]["content"] == "hello"
    assert rows_by_id["session/2"]["messages"][0]["content"] == "second"
    converted = convert_traces_to_training_data(output_dir)
    assert {row["metadata"]["session_id"] for row in converted} == {"session/1", "session/2"}
    assert (output_dir / "README.md").exists()


def test_extract_cursor_from_workspace_state_db(tmp_path: Path):
    state_db = tmp_path / "workspaceStorage" / "workspace-hash" / "state.vscdb"
    _write_minimal_cursor_state_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db.parent.parent),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    extracted_files = sorted(output_dir.glob("cursor-*.jsonl"))
    assert len(extracted_files) == 2
    assert all(_read_jsonl_rows(path)[0]["type"] == "cursor_session_meta" for path in extracted_files)
    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 2
    rows_by_table = {row["metadata"]["cursor_table"]: row for row in converted}
    assert set(rows_by_table) == {"ItemTable", "composerSessions"}
    assert {row["metadata"]["cursor_scope"] for row in converted} == {"workspace"}
    assert {row["metadata"]["cursor_workspace_id"] for row in converted} == {"workspace-hash"}
    assert rows_by_table["composerSessions"]["prompt"] == "composer prompt"
    assert rows_by_table["ItemTable"]["prompt"] == "open chat prompt"
    assert [message["content"] for message in rows_by_table["ItemTable"]["messages"] if message["role"] == "user"] == [
        "open chat prompt"
    ]
    assert '"root"' not in rows_by_table["ItemTable"]["prompt"]
    assert "read_file" in {tool["function"]["name"] for tool in rows_by_table["ItemTable"]["tools"]}
    assert rows_by_table["ItemTable"]["messages"][1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert {row["prompt"] for row in converted} == {"composer prompt", "open chat prompt"}
    assert {row["metadata"]["trace_type"] for row in converted} == {"cursor"}


def test_extract_cursor_preserves_native_project_transcripts_and_tools(tmp_path: Path):
    projects_dir = tmp_path / ".cursor" / "projects"
    project_dir = projects_dir / "c-Users-test-project"
    transcript = project_dir / "agent-transcripts" / "session-1" / "session-1.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript_events = [
        {
            "role": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "native prompt"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
                    },
                ]
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "text": "native thinking"},
                    {"type": "tool_use", "id": "call-shell", "name": "Shell", "input": {"command": "ls"}},
                    {"type": "text", "text": "native answer"},
                ]
            },
        },
        {
            "role": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "call-shell", "content": "files"}]},
        },
        {"type": "turn_ended", "status": "success"},
    ]
    transcript.write_text("\n".join(json.dumps(event) for event in transcript_events) + "\n", encoding="utf-8")
    tools_dir = project_dir / "mcps" / "user-shell" / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "Shell.json").write_text(
        json.dumps(
            {
                "name": "Shell",
                "description": "Run shell commands.",
                "arguments": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(projects_dir),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    copied_transcript = output_dir / "c-Users-test-project" / "agent-transcripts" / "session-1" / "session-1.jsonl"
    assert copied_transcript.exists()
    events = _read_jsonl_rows(copied_transcript)
    assert events[0]["type"] == "cursor_session_meta"
    assert events[0]["project"] == "c-Users-test-project"
    assert events[1]["type"] == "cursor_available_tools"
    assert events[2:] == transcript_events

    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert row["prompt"] == "native prompt"
    assert [message["role"] for message in row["messages"]] == ["user", "assistant", "tool"]
    assert row["messages"][0]["content"] == [
        {"type": "text", "text": "native prompt"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
    ]
    assert row["messages"][1]["content"] == "native answer"
    assert row["messages"][1]["reasoning_content"] == "native thinking"
    assert row["messages"][1]["tool_calls"][0]["function"]["name"] == "Shell"
    assert row["messages"][2]["content"] == "files"
    tools_by_name = {tool["function"]["name"]: tool for tool in row["tools"]}
    assert {"Shell", "read_file", "run_terminal_cmd", "edit_file", "codebase_search"}.issubset(tools_by_name)
    assert tools_by_name["Shell"]["function"]["parameters"]["required"] == ["command"]


def test_extract_cursor_preserves_recovered_db_image_blocks(tmp_path: Path):
    state_db = tmp_path / "workspaceStorage" / "workspace-images" / "state.vscdb"
    _write_cursor_image_state_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    events = _read_extracted_jsonl_rows(output_dir)
    user_event = next(event for event in events if event.get("role") == "user")
    assert user_event["message"]["content"] == [
        {"type": "text", "text": "make me a website like this"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,def456"},
    ]

    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert row["prompt"] == "make me a website like this"
    assert row["messages"][0]["content"] == user_event["message"]["content"]
    assert row["messages"][1] == {"role": "assistant", "content": "I'll build a similar site."}


def test_extract_cursor_prefers_recovered_db_sessions_over_redacted_native_transcripts(tmp_path: Path):
    state_db = tmp_path / "workspaceStorage" / "workspace-hash" / "state.vscdb"
    _write_minimal_cursor_state_db(state_db)
    transcript = (
        tmp_path
        / ".cursor"
        / "projects"
        / "c-Users-test-project"
        / "agent-transcripts"
        / "redacted-session"
        / "redacted-session.jsonl"
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "role": "user",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"root":{"children":[{"children":[],"type":"paragraph"}]}}',
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(tmp_path),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in output_dir.glob("cursor-*.jsonl")) == [
        "cursor-composer-1.jsonl",
        "cursor-workbench.panel.aichat.view.aichat.chatdata.jsonl",
    ]
    assert not (output_dir / ".cursor" / "projects").exists()
    converted = convert_traces_to_training_data(output_dir)
    assert {row["prompt"] for row in converted} == {"composer prompt", "open chat prompt"}
    assert '{"root"' not in json.dumps(converted)


def test_extract_cursor_reconstructs_archived_composer_data(tmp_path: Path):
    state_db = tmp_path / "globalStorage" / "state.vscdb"
    _write_archived_cursor_state_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    events = _read_extracted_jsonl_rows(output_dir)
    assert events[0]["type"] == "cursor_session_meta"
    assert events[0]["cursor_storage_kind"] == "composerData"
    assert events[0]["cursor_scope"] == "global"
    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert row["prompt"] == "archived prompt"
    assert row["messages"][0]["content"] == "archived prompt"
    assert '"root"' not in row["prompt"]
    assert row["metadata"]["model"] == "cursor-fable-5"
    assert [message["role"] for message in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert row["messages"][1]["reasoning_content"] == "archived reasoning"
    assert row["messages"][1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert row["messages"][2]["tool_call_id"] == "call-read"
    assert row["messages"][2]["content"] == "file contents"
    assert row["messages"][3]["content"] == "final response"
    tools_by_name = {tool["function"]["name"]: tool for tool in row["tools"]}
    assert "run_terminal_cmd" in tools_by_name
    assert tools_by_name["read_file"]["function"]["parameters"]["properties"]["path"] == {"type": "string"}
    assert tools_by_name["read_file"]["function"]["parameters"]["required"] == ["path"]

    assert any(
        message.get("role") == "assistant" and message.get("content") == "final response"
        for message in row["messages"]
    )


def test_extract_cursor_reconstructs_id_keyed_composer_tables(tmp_path: Path):
    state_db = tmp_path / "workspaceStorage" / "workspace-id" / "state.vscdb"
    _write_id_keyed_cursor_state_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    events = _read_extracted_jsonl_rows(output_dir)
    assert events[0]["type"] == "cursor_session_meta"
    assert events[0]["cursor_storage_kind"] == "composerData"
    assert events[0]["cursor_workspace_id"] == "workspace-id"
    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert row["prompt"] == "id-keyed prompt"
    assert [message["content"] for message in row["messages"]] == ["id-keyed prompt", "id-keyed answer"]
    assert row["metadata"]["model"] == "cursor-fable-5"
    assert "unused answer" not in json.dumps(row)


def test_cursor_state_db_directory_scan_returns_all_dbs_sorted_by_recency(tmp_path: Path):
    workspace_storage = tmp_path / "workspaceStorage"
    expected_newest = workspace_storage / "workspace-newest" / "state.vscdb"
    db_count = 15
    for index in range(db_count):
        state_db = workspace_storage / f"workspace-{index:03d}" / "state.vscdb"
        state_db.parent.mkdir(parents=True)
        state_db.write_text("", encoding="utf-8")
        os_time = 1_000_000 + index
        os.utime(state_db, (os_time, os_time))
    expected_newest.parent.mkdir(parents=True)
    expected_newest.write_text("", encoding="utf-8")
    os.utime(expected_newest, (2_000_000, 2_000_000))

    found = extract_module._cursor_state_dbs(workspace_storage)

    assert len(found) == db_count + 1
    assert found[0] == expected_newest
    assert found[-1] == workspace_storage / "workspace-000" / "state.vscdb"


def test_extract_cursor_uses_fallback_prompt_for_escaped_empty_editor_state_and_nested_model(
    tmp_path: Path,
):
    state_db = tmp_path / "workspaceStorage" / "workspace-model-info" / "state.vscdb"
    _write_cursor_escaped_rich_text_model_info_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db),
            "--output",
            str(output_dir),
            "--model",
            "claude",
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert row["metadata"]["model"] == "claude-opus-4.5"
    assert row["prompt"] == "actual cursor prompt"
    assert row["messages"][0] == {"role": "user", "content": "actual cursor prompt"}
    assert row["messages"][1] == {"role": "assistant", "content": "actual cursor answer"}
    assert '"root"' not in json.dumps(row["messages"])


def test_extract_cursor_detects_rich_text_only_messages_and_selected_model_id(tmp_path: Path):
    state_db = tmp_path / "workspaceStorage" / "workspace-rich-text-only" / "state.vscdb"
    _write_cursor_rich_text_only_selected_model_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db),
            "--output",
            str(output_dir),
            "--model",
            "claude",
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert row["metadata"]["model"] == "anthropic/claude-opus-4.5"
    assert row["prompt"] == "rich text only prompt"
    assert row["messages"][0] == {"role": "user", "content": "rich text only prompt"}
    assert row["messages"][1] == {"role": "assistant", "content": "rich text only answer"}


def test_extract_cursor_sanitizes_training_unsafe_archived_rows(tmp_path: Path):
    state_db = tmp_path / "globalStorage" / "state.vscdb"
    _write_unsafe_cursor_state_db(state_db)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "cursor",
            "--sessions-dir",
            str(state_db),
            "--output",
            str(output_dir),
            "--no-anon",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    converted = convert_traces_to_training_data(output_dir)
    assert len(converted) == 1
    row = converted[0]
    assert [message["role"] for message in row["messages"]] == ["user", "assistant", "user", "assistant"]
    assert [message["content"] for message in row["messages"]] == [
        "first prompt",
        "first answer",
        "second prompt",
        "second answer",
    ]
    assert row["prompt"] == "first prompt"
    tool_names = {tool["function"]["name"] for tool in row["tools"]}
    assert {"read_file", "run_terminal_cmd", "edit_file"}.issubset(tool_names)

    assert row["metadata"]["trace_type"] == "cursor"


def test_extract_accepts_agent_root_or_native_session_store(tmp_path: Path):
    claude_root = tmp_path / ".claude"
    claude_projects = claude_root / "projects" / "-repo"
    claude_projects.mkdir(parents=True)
    (claude_projects / "session.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-fable-5", "content": []}})
        + "\n",
        encoding="utf-8",
    )

    codex_root = tmp_path / ".codex"
    codex_sessions = codex_root / "sessions"
    codex_sessions.mkdir(parents=True)
    (codex_sessions / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )

    pi_root = tmp_path / ".pi"
    pi_sessions = pi_root / "agent" / "sessions"
    pi_sessions.mkdir(parents=True)
    (pi_sessions / "session.jsonl").write_text(
        json.dumps({"type": "session_info", "modelId": "pi-fable-5"}) + "\n",
        encoding="utf-8",
    )

    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir()
    _write_minimal_hermes_state_db(hermes_root / "state.db", model="hermes-fable-5")

    cases = [
        ("claude-root", "claude", claude_root, "session.jsonl"),
        ("claude-projects", "claude", claude_root / "projects", "session.jsonl"),
        ("codex-root", "codex", codex_root, "session.jsonl"),
        ("codex-sessions", "codex", codex_sessions, "session.jsonl"),
        ("pi-root", "pi", pi_root, "session.jsonl"),
        ("pi-sessions", "pi", pi_sessions, "session.jsonl"),
        ("hermes-root", "hermes", hermes_root, "hermes-session-1.jsonl"),
        ("hermes-state-db", "hermes", hermes_root / "state.db", "hermes-session-1.jsonl"),
    ]
    for label, provider, source, expected_file in cases:
        output_dir = tmp_path / f"out-{label}"
        result = runner.invoke(
            app,
            ["extract", provider, "--sessions-dir", str(source), "--output", str(output_dir)],
            input="n\n",
        )

        assert result.exit_code == 0, result.output
        assert (output_dir / expected_file).exists()
        assert "Extracted 1" in result.output


def test_extract_defaults_to_data_folder_and_anonymizes_before_prompt(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    original_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    (sessions_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "test-model"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"hello from /home/alice/project alice@example.com {original_key}",
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["extract", "codex", "--sessions-dir", str(sessions_dir)], input="n\n")
        data_dir = Path("data")
        text = (data_dir / "session.jsonl").read_text(encoding="utf-8")

    assert result.exit_code == 0
    assert "Extracted 1 codex trace" in result.output
    assert "Automatically scrambled 1 API keys, 1 email addresses, and 1 username references" in result.output
    assert "Data was written to data" in result.output
    assert "Would you like to upload to Hugging Face?" in result.output
    assert "/home/user1/project" in text
    assert "redacted-user1@example.com" in text
    assert "redacted_api_key_" in text
    assert "sk-or-v1-" not in text
    assert original_key not in text
    assert "alice@example.com" not in text


def test_extract_no_anon_skips_automatic_anonymization(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    original_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    (sessions_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "test-model"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"hello from /home/alice/project alice@example.com {original_key}",
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(output_dir), "--no-anon"],
        input="n\n",
    )

    assert result.exit_code == 0
    text = (output_dir / "session.jsonl").read_text(encoding="utf-8")
    assert "Skipped anonymization because --no-anon was passed" in result.output
    assert "Automatically scrambled" not in result.output
    assert "/home/alice/project" in text
    assert "alice@example.com" in text
    assert original_key in text


def test_extract_model_filter_for_codex_claude_cursor_pi_and_hermes(tmp_path: Path):
    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    pi_dir = tmp_path / "pi"
    codex_dir.mkdir()
    claude_dir.mkdir()
    pi_dir.mkdir()
    (codex_dir / "fable.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-fable", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    (codex_dir / "opus.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-opus", "model": "codex-opus"}}) + "\n",
        encoding="utf-8",
    )
    (claude_dir / "fable.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-fable-5", "content": []}})
        + "\n",
        encoding="utf-8",
    )
    (claude_dir / "opus.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-4-8", "content": []}})
        + "\n",
        encoding="utf-8",
    )
    (pi_dir / "fable.jsonl").write_text(
        json.dumps({"type": "session_info", "modelId": "pi-fable-5"}) + "\n",
        encoding="utf-8",
    )
    (pi_dir / "opus.jsonl").write_text(
        json.dumps({"type": "session_info", "modelId": "pi-opus"}) + "\n",
        encoding="utf-8",
    )
    state_db = tmp_path / "state.db"
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at TEXT, model TEXT)")
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)"
        )
        for session_id, model in (("hermes-fable", "hermes-fable-5"), ("hermes-opus", "hermes-opus")):
            connection.execute(
                "INSERT INTO sessions (id, source, started_at, model) VALUES (?, ?, ?, ?)",
                (session_id, "cli", "2026-06-13T00:00:00Z", model),
            )
            connection.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "user", "hello", "2026-06-13T00:00:01Z"),
            )
        connection.commit()
    finally:
        connection.close()

    cursor_db = tmp_path / "cursor" / "globalStorage" / "state.vscdb"
    cursor_db.parent.mkdir(parents=True)
    connection = sqlite3.connect(cursor_db)
    try:
        connection.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
        for composer_id, model in (("cursor-fable", "cursor-fable-5"), ("cursor-opus", "cursor-opus")):
            connection.execute(
                "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                (
                    f"composerData:{composer_id}",
                    json.dumps(
                        {
                            "composerId": composer_id,
                            "status": "completed",
                            "fullConversationHeadersOnly": [
                                {"bubbleId": "user", "type": 1},
                                {"bubbleId": "assistant", "type": 2},
                            ],
                        }
                    ),
                ),
            )
            connection.execute(
                "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                (
                    f"bubbleId:{composer_id}:user",
                    json.dumps({"bubbleId": "user", "type": 1, "text": "hello"}),
                ),
            )
            connection.execute(
                "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                (
                    f"bubbleId:{composer_id}:assistant",
                    json.dumps(
                        {
                            "bubbleId": "assistant",
                            "type": 2,
                            "text": "answer",
                            "modelInfo": {"modelName": model},
                        }
                    ),
                ),
            )
        connection.commit()
    finally:
        connection.close()

    cases = [
        ("codex", codex_dir, "fable.jsonl"),
        ("claude", claude_dir, "fable.jsonl"),
        ("cursor", cursor_db, "cursor-fable.jsonl"),
        ("pi", pi_dir, "fable.jsonl"),
        ("hermes", state_db, "hermes-fable.jsonl"),
    ]
    for provider, source, expected_file in cases:
        output_dir = tmp_path / f"out-{provider}"
        result = runner.invoke(
            app,
            [
                "extract",
                provider,
                "--sessions-dir",
                str(source),
                "--output",
                str(output_dir),
                "--model",
                "fable-5",
            ],
            input="n\n",
        )

        assert result.exit_code == 0, result.output
        assert f"Extracted 1 {provider} trace with fable-5" in result.output
        if provider == "cursor":
            assert CURSOR_EXTRACTION_NOTICE in result.output
        else:
            assert CURSOR_EXTRACTION_NOTICE not in result.output
        assert (output_dir / expected_file).exists()
        assert len(list(output_dir.glob("*.jsonl"))) == 1


def test_extract_claude_preserves_raw_order_and_only_anonymizes_inline(tmp_path: Path):
    sessions_dir = tmp_path / "claude"
    sessions_dir.mkdir()
    source_file = sessions_dir / "session.jsonl"
    secret_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    raw_lines = [
        (
            '{"type":"last-prompt","sessionId":"claude-session",'
            f'"lastPrompt":"Email alice@example.com key {secret_key} path /home/alice/project"}}\n'
        ),
        '{ "type": "mode", "mode": "normal", "sessionId": "claude-session" }\n',
        '{"type":"permission-mode","permissionMode":"bypassPermissions","sessionId":"claude-session"}\n',
        '{"type":"file-history-snapshot","sessionId":"claude-session","leafUuid":"leaf-1"}\n',
        (
            '{"type":"user","timestamp":"2026-06-10T00:00:01.000Z","sessionId":"claude-session",'
            '"uuid":"user-1","message":{"role":"user","content":"Build app"}}\n'
        ),
        (
            '{"type":"assistant","timestamp":"2026-06-10T00:00:02.000Z","sessionId":"claude-session",'
            '"parentUuid":"user-1","message":{"id":"msg-1","role":"assistant","model":"claude-fable-5",'
            '"content":[{"type":"text","text":"I will build it."}]}}\n'
        ),
        '{"type":"mode","mode":"normal","sessionId":"claude-session"}\n',
        '{"type":"permission-mode","permissionMode":"auto","sessionId":"claude-session"}\n',
    ]
    source_file.write_text("".join(raw_lines), encoding="utf-8")
    source_examples = convert_traces_to_training_data(source_file)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "claude",
            "--sessions-dir",
            str(sessions_dir),
            "--output",
            str(output_dir),
            "--model",
            "claude-fable-5",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Automatically scrambled 1 API keys, 1 email addresses, and 1 username references" in result.output
    extracted_file = output_dir / "session.jsonl"
    extracted_lines = extracted_file.read_text(encoding="utf-8").splitlines(keepends=True)
    assert [json.loads(line)["type"] for line in extracted_lines] == [
        "last-prompt",
        "mode",
        "permission-mode",
        "file-history-snapshot",
        "user",
        "assistant",
        "mode",
        "permission-mode",
    ]
    assert extracted_lines[1:] == raw_lines[1:]
    assert "alice@example.com" not in extracted_lines[0]
    assert secret_key not in extracted_lines[0]
    assert "/home/alice" not in extracted_lines[0]
    assert "redacted-user1@example.com" in extracted_lines[0]
    assert "redacted_api_key_" in extracted_lines[0]
    assert "sk-or-v1-" not in extracted_lines[0]
    assert "/home/user1/project" in extracted_lines[0]
    assert convert_traces_to_training_data(extracted_file) == source_examples


def test_extract_refreshes_flat_output_before_writing(tmp_path: Path):
    sessions_dir = tmp_path / "codex"
    sessions_dir.mkdir()
    (sessions_dir / "fable.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-fable", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True)
    stale_file = output_dir / "old-opus.jsonl"
    stale_file.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-opus", "model": "codex-opus"}}) + "\n",
        encoding="utf-8",
    )
    stale_legacy_file = output_dir / "codex" / "old-nested-opus.jsonl"
    stale_legacy_file.parent.mkdir(parents=True)
    stale_legacy_file.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-nested-opus", "model": "codex-opus"}}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text("old staged path: /home/stale/project\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "extract",
            "codex",
            "--sessions-dir",
            str(sessions_dir),
            "--output",
            str(output_dir),
            "--model",
            "fable-5",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Automatically scrambled 0 API keys, 0 email addresses, and 0 username references" in result.output
    assert not stale_file.exists()
    assert not stale_legacy_file.exists()
    assert not stale_legacy_file.parent.exists()
    assert (output_dir / "fable.jsonl").exists()
    assert len(list(output_dir.glob("*.jsonl"))) == 1


def test_extract_can_upload_staged_anonymized_output_to_huggingface(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data"

    with patch("teich.cli.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/fable-traces"
        mock_api_cls.return_value = mock_api

        result = runner.invoke(
            app,
            [
                "extract",
                "codex",
                "--sessions-dir",
                str(sessions_dir),
                "--output",
                str(output_dir),
                "--model",
                "fable-5",
            ],
            input="y\narmand0e/fable-traces\n",
            env={"HF_TOKEN": "hf-test-token"},
        )

    assert result.exit_code == 0
    assert "Published dataset: https://huggingface.co/datasets/armand0e/fable-traces" in result.output
    mock_api_cls.assert_called_once_with(token="hf-test-token")
    mock_api.create_repo.assert_called_once_with(
        repo_id="armand0e/fable-traces",
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    mock_api.upload_large_folder.assert_called_once_with(
        repo_id="armand0e/fable-traces",
        folder_path=str(output_dir),
        repo_type="dataset",
        private=False,
        ignore_patterns=["partials/**", "failures/**", "README.md", "tools.json"],
    )
    mock_api.upload_folder.assert_called_once_with(
        folder_path=str(output_dir),
        repo_id="armand0e/fable-traces",
        repo_type="dataset",
        commit_message="Upload teich dataset metadata",
        allow_patterns=["README.md"],
        ignore_patterns=["partials/**", "failures/**"],
    )
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "armand0e/fable-traces" in readme
    assert 'path: "**/*.jsonl"' in readme
    assert '- "codex"' in readme
    assert '- "pi"' not in readme
    assert '- "fable-5"' not in readme
    assert '- "codex-fable-5"' not in readme
    assert "Model metadata:" not in readme
    assert "teich extract codex --out data" in readme


def test_extract_prompts_for_hf_token_when_env_token_is_missing(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data"

    with patch("teich.cli.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/fable-traces"
        mock_api_cls.return_value = mock_api

        result = runner.invoke(
            app,
            [
                "extract",
                "codex",
                "--sessions-dir",
                str(sessions_dir),
                "--output",
                str(output_dir),
                "--model",
                "fable-5",
            ],
            input="y\narmand0e/fable-traces\nhf-prompted-token\n",
            env={"HF_TOKEN": "", "HUGGINGFACE_HUB_TOKEN": "", "TEICH_HF_TOKEN": ""},
        )

    assert result.exit_code == 0
    assert "HF_TOKEN" in result.output
    mock_api_cls.assert_called_once_with(token="hf-prompted-token")


def test_extract_rejects_blank_huggingface_upload_inputs(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )

    with patch("teich.cli.HfApi") as mock_api_cls:
        blank_repo = runner.invoke(
            app,
            ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(tmp_path / "blank-repo")],
            input="y\n   \n",
            env={"HF_TOKEN": "hf-test-token"},
        )

    assert blank_repo.exit_code == 1
    assert "Hugging Face dataset repo id is required" in blank_repo.output
    mock_api_cls.assert_not_called()

    with patch("teich.cli.HfApi") as mock_api_cls:
        blank_token = runner.invoke(
            app,
            ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(tmp_path / "blank-token")],
            input="y\narmand0e/fable-traces\n   \n",
            env={"HF_TOKEN": "", "HUGGINGFACE_HUB_TOKEN": "", "TEICH_HF_TOKEN": ""},
        )

    assert blank_token.exit_code == 1
    assert "HF_TOKEN is required" in blank_token.output
    mock_api_cls.assert_not_called()


def test_anonymize_replaces_emails_keys_and_home_usernames_consistently(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    original_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "path": "/home/alice/project",
                "windows_path": "C:\\Users\\alice\\Documents\\repo",
                "encoded_project_path": "projects/-home-alice-Documents-repo/session.jsonl\n-home-alice",
                "listing": "drwxrwxr-x 2 alice alice 4096 Jun 11 18:40 -home-alice",
                "email": "alice@example.com",
                "message": f"email alice@example.com key {original_key}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "alice" not in text
    assert "example.com" in text
    assert "redacted-user1@example.com" in text
    assert text.count("redacted-user1@example.com") == 2
    assert "/home/user1/project" in text
    assert "C:\\\\Users\\\\user1\\\\Documents" in text
    assert "projects/-home-user1-Documents-repo/session.jsonl" in text
    assert "-home-user1" in text
    assert "user1 user1 4096" in text
    assert "redacted_api_key_" in text
    assert "sk-or-v1-" not in text
    assert original_key not in text
    assert "email=2" in result.output
    assert "username=7" in result.output
    assert "api_key=1" in result.output


def test_anonymize_parallel_worker_count_caps_windows(monkeypatch):
    monkeypatch.setattr(anonymize_module.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(anonymize_module.sys, "platform", "win32")

    assert anonymize_module._process_worker_count(100) == 61


def test_anonymize_jsonl_preserves_valid_json_after_escaped_path_replacements(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    original_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    row = {
        "path": r"C:\Users\alice\Documents\repo",
        "message": f"Email alice@example.com and use {original_key}",
    }
    (input_dir / "trace.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_dir = tmp_path / "anonymized"

    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert "alice" not in text
    assert parsed["path"].startswith(r"C:\Users\user")
    assert "redacted-user1@example.com" in parsed["message"]
    assert original_key not in parsed["message"]


def test_anonymize_generalizes_to_synthetic_secret_and_reference_matrix(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    stripe_key = "sk_" + "live_" + "abcdefghijklmnopqrstuvwxyz123456"
    twilio_key = "S" + "K" + "0123456789abcdef0123456789abcdef"
    raw_values = {
        "email": "dev.team+alerts@company.io",
        "sk_ant": "sk-ant-api03-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
        "sk_proj": "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
        "sk": "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK123456",
        "cursor_synthetic": "sk-Mmu5OJR5NQoTJOGj6z6cHPraZ35yuZfK",
        "hf": "hf_abcdefghijklmnopqrstuvwxyz123456",
        "github_pat": "github_pat_abcdefghijklmnopqrstuvwxyz_1234567890",
        "gho": "gho_abcdefghijklmnopqrstuvwxyz123456",
        "glpat": "glpat-abcdefghijklmnopqrstuvwxyz123456",
        "linear": "lin_api_abcdefghijklmnopqrstuvwxyz123456",
        "npm": "npm_abcdefghijklmnopqrstuvwxyz123456",
        "pypi": "pypi-abcdefghijklmnopqrstuvwxyz123456",
        "stripe": stripe_key,
        "xoxb": "xoxb-abcdefghijklmnopqrstuvwxyz-123456",
        "google": "AIzaabcdefghijklmnopqrstuvwxyz123456",
        "aws": "AKIAIOSFODNN7EXAMPLE",
        "twilio": twilio_key,
        "sendgrid": "SG.abcdefghijklmnopqrstuv.ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghi",
        "jwt": (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkphbmUifQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "bearer": "Bearer abcdefghijklmnopqrstuvwxyz1234567890",
        "env_secret": "PROJECT_SECRET=abcdefghijklmnopqrstuvwxyz1234567890",
    }
    image_data = "aGVsbG8=" * 40
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": "\n".join(
                    [
                        f"Contact {raw_values['email']} and then {raw_values['email']} again.",
                        "/home/zoe/app /Users/zoe/app D:\\Users\\zoe\\Documents\\app",
                        "projects/-Users-zoe-Documents-app/session.jsonl",
                        "drwxr-xr-x 2 zoe zoe 4096 Jun 13 09:00 app",
                        "Public path stays /Users/Public/Documents/shared",
                        " ".join(raw_values[key] for key in raw_values if key != "email"),
                        "Read @src/main.ts and @docs/*.md before editing.",
                        "Install @scope/package and keep @media, @keyframes, @pytest.mark, @app.get.",
                        "Remote git@gitlab.com:example/repo.git and https://medium.com/@someone/post stay.",
                        "Keep code expression process.env.PROJECT_SECRET intact.",
                    ]
                ),
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    for value in raw_values.values():
        assert value not in text
    assert "zoe" not in text
    assert text.count("@example.com") == 2
    assert "/home/user" in text
    assert "/Users/user" in text
    assert "D:\\\\Users\\\\user" in text
    assert "-Users-user" in text
    assert "/Users/Public/Documents/shared" in text
    assert text.count("redacted_api_key_") >= 9
    assert "sk-ant-api03-" not in text
    assert "sk-proj-" not in text
    assert "sk-" not in text
    assert "hf_" not in text
    assert "github_pat_" not in text
    assert "gho_" not in text
    assert "glpat-" not in text
    assert "lin_api_" not in text
    assert "npm_" not in text
    assert "pypi-" not in text
    assert "sk_live_" not in text
    assert "xoxb-" not in text
    assert "AIza" not in text
    assert "AKIA" not in text
    assert twilio_key not in text
    assert "SG." not in text
    assert "redacted_jwt_" in text
    assert "eyJ" not in text
    assert "Bearer redacted_" in text
    assert "PROJECT_SECRET=redacted_" in text
    assert "process.env.PROJECT_SECRET" in text
    assert "sk-Mmu5OJR5NQoTJOGj6z6cHPraZ35yuZfK" not in text
    assert image_data in text
    assert "redacted_base64" not in text
    assert "@src/main.ts" in text
    assert "@docs/*.md" in text
    assert "@scope/package" in text
    assert "@media" in text
    assert "@keyframes" in text
    assert "@pytest.mark" in text
    assert "@app.get" in text
    assert "git@gitlab.com:example/repo.git" in text
    assert "https://medium.com/@someone/post" in text


def test_anonymize_scrubs_supabase_database_and_structured_secret_surfaces(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    supabase_anon = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIn0."
        "signature1234567890abcdef"
    )
    service_role = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJyb2xlIjoic2VydmljZV9yb2xlIiwiaXNzIjoic3VwYWJhc2UifQ."
        "servicerolesignature1234567890"
    )
    raw_secrets = {
        "postgres_password": "pg-password-abcdefghijklmnopqrstuvwxyz",
        "jwt_secret": "jwt-secret-abcdefghijklmnopqrstuvwxyz",
        "dashboard_password": "dashboard-password-abcdefghijklmnopqrstuvwxyz",
        "secret_key_base": "PHOENIX_SECRET_KEY_BASE_RANDOM_abcdefghijklmnopqrstuvwxyz",
        "vault_enc_key": "RANDOM_ENCRYPTION_KEY_32_CHARS_abcdef",
        "database_password": "database-pass-abcdefghijklmnopqrstuvwxyz",
        "mongodb_password": "mongo-pass-abcdefghijklmnopqrstuvwxyz",
        "redis_password": "redis-pass-abcdefghijklmnopqrstuvwxyz",
        "query_token": "query-token-abcdefghijklmnopqrstuvwxyz",
        "azure_account_key": "azurestorageaccountkeyabcdefghijklmnopqrstuvwxyz123456",
        "json_password": "json-password-abcdefghijklmnopqrstuvwxyz",
        "json_api_key": "json-api-key-abcdefghijklmnopqrstuvwxyz",
        "aws_secret": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "private_key": (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCabcdefghijklmnopqrstuvwxyz\n"
            "-----END PRIVATE KEY-----"
        ),
    }
    env_text = "\n".join(
        [
            "# Supabase-style generated env",
            f"POSTGRES_PASSWORD={raw_secrets['postgres_password']}",
            f"JWT_SECRET={raw_secrets['jwt_secret']}",
            f"ANON_KEY={supabase_anon}",
            f"SERVICE_ROLE_KEY={service_role}",
            "DASHBOARD_USERNAME=supabase",
            f"DASHBOARD_PASSWORD={raw_secrets['dashboard_password']}",
            f"SECRET_KEY_BASE={raw_secrets['secret_key_base']}",
            f"VAULT_ENC_KEY={raw_secrets['vault_enc_key']}",
            "POSTGRES_HOST=db",
            "POSTGRES_DB=postgres",
            "POSTGRES_PORT=5432",
            "NEXT_PUBLIC_SUPABASE_URL=https://demo.supabase.co",
            f"NEXT_PUBLIC_SUPABASE_ANON_KEY={supabase_anon}",
            f"DATABASE_URL=postgresql://postgres:{raw_secrets['database_password']}@db:5432/postgres",
            f"MONGODB_URI=mongodb+srv://admin:{raw_secrets['mongodb_password']}@cluster.mongodb.net/app",
            f"REDIS_URL=redis://:{raw_secrets['redis_password']}@redis:6379/0",
            f"AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=acct;AccountKey={raw_secrets['azure_account_key']};EndpointSuffix=core.windows.net",
            "JWT_EXPIRY=3600",
            "PROCEDURAL_WARMUP_TOKENS=4096",
        ]
    )
    direct_urls = "\n".join(
        [
            f"direct postgres postgresql://app:{raw_secrets['database_password']}@db.example.com/app",
            f"direct mongodb mongodb://root:{raw_secrets['mongodb_password']}@mongo.example.com/app",
            f"https://api.example.com/data?access_token={raw_secrets['query_token']}&page=1",
        ]
    )
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": env_text + "\n" + direct_urls,
                "config": {
                    "password": raw_secrets["json_password"],
                    "apiKey": raw_secrets["json_api_key"],
                    "aws_secret_access_key": raw_secrets["aws_secret"],
                    "privateKey": raw_secrets["private_key"],
                    "supportsLocalAgentJwt": "abcdefghijklmnopqrstuvwxyz1234567890",
                    "maxTokens": 4096,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    for value in [*raw_secrets.values(), supabase_anon, service_role]:
        assert value not in text
    assert "eyJ" not in text
    assert "redacted_jwt_" in text
    assert "redacted_secret_" in text
    assert "redacted_private_key_" in text
    assert "postgresql://app:redacted_secret_" in text
    assert "mongodb://root:redacted_secret_" in text
    assert "access_token=redacted_secret_" in text
    assert "POSTGRES_HOST=db" in text
    assert "POSTGRES_DB=postgres" in text
    assert "POSTGRES_PORT=5432" in text
    assert "NEXT_PUBLIC_SUPABASE_URL=https://demo.supabase.co" in text
    assert "JWT_EXPIRY=3600" in text
    assert "PROCEDURAL_WARMUP_TOKENS=4096" in text
    assert "supportsLocalAgentJwt" in text
    assert "abcdefghijklmnopqrstuvwxyz1234567890" in text
    assert "maxTokens" in text
    assert "api_key=" in result.output


def test_anonymize_keeps_secret_replacements_consistent_per_trace(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    key = "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK"
    email = "operator@example.net"
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "first": f"{email} {key} /home/operator/app",
                "second": f"{email} {key} /Users/operator/app",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    emails = re.findall(r"redacted-user\d+@example\.com", text)
    keys = re.findall(r"redacted_api_key_[A-Za-z0-9]{16}", text)
    paths = re.findall(r"/(?:home|Users)/(user\d+)/app", text)
    assert len(set(emails)) == 1
    assert len(set(keys)) == 1
    assert len(set(paths)) == 1
    assert key not in text
    assert email not in text
    assert "operator" not in text


def test_anonymize_scrubs_env_style_keys_without_scrubbing_tokenizer_terms(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": "\n".join(
                    [
                        'CONTEXT7_API_KEY: "ctx7sk-ba627c37-3cea-45c2-8808-720374cce2a3"',
                        'HF_TOKEN="abcdefghijklmnopqrstuvwxyz1234567890"',
                        "export const GROQ_API_KEY =",
                        '  process.env.GROQ_API_KEY ?? "gsk_TESTabcdefghijklmnop";',
                        'tokenizer = "AutoTokenizer.from_pretrained"',
                        'PROCEDURAL_WARMUP_TOKENS = "abcdefghijklmnopqrstuvwxyz1234567890"',
                        'adapter.supportsLocalAgentJwt = "abcdefghijklmnopqrstuvwxyz1234567890"',
                    ]
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "ctx7sk-ba627c37-3cea-45c2-8808-720374cce2a3" not in text
    assert "abcdefghijklmnopqrstuvwxyz1234567890" in text
    assert "HF_TOKEN=\\\"redacted_" in text
    assert "redacted_api_key_" in text
    assert "ctx7sk-" not in text
    assert "process.env.GROQ_API_KEY" in text
    assert "gsk_TESTabcdefghijklmnop" not in text
    assert "gsk_" not in text
    assert "AutoTokenizer.from_pretrained" in text
    assert "PROCEDURAL_WARMUP_TOKENS" in text
    assert "adapter.supportsLocalAgentJwt" in text
    assert "api_key=3" in result.output


def test_anonymize_preserves_base64_media_payloads_without_touching_metadata(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    image_data = "aGVsbG8=" * 40
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    row = json.loads((output_dir / "trace.jsonl").read_text(encoding="utf-8"))
    source = row["content"][0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert source["data"] == image_data
    assert image_data in json.dumps(row)
    assert "image_data" not in result.output


def test_anonymize_does_not_treat_systemd_units_as_emails(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": "\n".join(
                    [
                        "Failed to start org.gnome.Shell@x11.service",
                        "Read `@AGENT.md` before editing",
                        "Also inspect @README.md and -@plan.md",
                        "Read https://medium.com/@ahmed.soliman/why-anthropics",
                        "origin git@github.com:CompactAIOfficial/MythosMini.git",
                        "package source ssh://user3@example.com/user/repo",
                        "Keep repo@feature.module and package@v1.2.3 as code references",
                        "Keep user@example.test because it is a reserved test-domain reference",
                        "contact 'lane@example.com' for help",
                    ]
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "org.gnome.Shell@x11.service" in text
    assert "`@AGENT.md`" in text
    assert "@README.md" in text
    assert "-@plan.md" in text
    assert "https://medium.com/@ahmed.soliman/why-anthropics" in text
    assert "git@github.com:CompactAIOfficial/MythosMini.git" in text
    assert "ssh://user3@example.com/user/repo" in text
    assert "repo@feature.module" in text
    assert "package@v1.2.3" in text
    assert "user@example.test" in text
    assert "lane@example.com" not in text
    assert "redacted-user1@example.com" in text
    assert "email=1" in result.output


def test_pool_upload_is_reserved_until_backend_exists(tmp_path: Path):
    result = runner.invoke(app, ["pool", "upload", str(tmp_path)])

    assert result.exit_code == 1
    assert "not wired to a deployed pool backend yet" in result.output
