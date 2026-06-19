import json
from pathlib import Path

import pytest

from teich import detect_trace_type
from teich.converter import (
    convert_trace_to_training_example,
    convert_traces_to_training_data,
    normalize_claude_code_trace_events,
    normalize_codex_trace_events,
)


def _read_jsonl_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("example_file", "expected_trace_type"),
    [
        ("example_claude_code.jsonl", "claude_code"),
        ("example_codex_session.jsonl", "codex"),
        ("example_droid_session.jsonl", "droid"),
        ("example_hermes_session.jsonl", "hermes"),
        ("example_pi_session.jsonl", "pi"),
        ("example_cursor_session.jsonl", "cursor"),
        ("example_openclaw_session.jsonl", "openclaw"),
    ],
)
def test_detect_trace_type_matches_checked_in_examples(example_file: str, expected_trace_type: str):
    example_path = Path(__file__).resolve().parent.parent / "examples" / example_file

    assert detect_trace_type(_read_jsonl_events(example_path)) == expected_trace_type


def test_detect_trace_type_returns_known_trace_type():
    cases = [
        ([{"type": "session_meta", "payload": {"id": "codex-session"}}], "codex"),
        (
            [{"type": "user", "session_id": "claude-session", "message": {"role": "user", "content": "hello"}}],
            "claude_code",
        ),
        ([{"type": "session", "id": "pi-session"}], "pi"),
        (
            [{"id": "hermes-session", "source": "cli", "messages": [{"role": "user", "content": "hello"}]}],
            "hermes",
        ),
        (
            [
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [],
                    "metadata": {"trace_type": "cursor", "cursor_storage_kind": "composerData"},
                }
            ],
            "cursor",
        ),
        (
            [
                {
                    "type": "session",
                    "version": 3,
                    "id": "openclaw-session",
                    "cwd": "/Users/calebfahlgren/.openclaw/workspace",
                }
            ],
            "openclaw",
        ),
        ([{"type": "external_session_meta", "payload": {"source": "hermes-agent"}}], "hermes"),
        ([{"type": "external_session_meta", "payload": {"source": "custom-agent"}}], "external_agent"),
        (
            [
                {"type": "session_start", "id": "droid-session", "version": 2, "cwd": "/workspace/project"},
                {"type": "message", "id": "message-1", "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
            ],
            "droid",
        ),
    ]

    for events, expected_trace_type in cases:
        assert detect_trace_type(events) == expected_trace_type


def test_detect_trace_type_returns_none_for_non_agent_jsonl():
    assert detect_trace_type([{"text": "hello"}, {"text": "world"}]) is None


def test_detect_trace_type_does_not_confuse_generic_chat_tools_for_cursor():
    events = [
        {
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
            ],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
        }
    ]

    assert detect_trace_type(events) is None


def test_detect_trace_type_finds_openclaw_session_header_after_earlier_event():
    events = [
        {"type": "custom", "customType": "teich-system-prompt", "data": {"systemPrompt": "OpenClaw prompt"}},
        {
            "type": "session",
            "version": 3,
            "id": "openclaw-session",
            "cwd": "/Users/calebfahlgren/.openclaw/workspace",
        },
        {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
    ]

    assert detect_trace_type(events) == "openclaw"


def test_convert_structured_cursor_rows_preserves_messages_tools_and_trace_type(tmp_path: Path):
    trace_file = tmp_path / "cursor-session.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "Inspect app.js"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"app.js\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "contents"},
            {"role": "assistant", "content": "Done"},
        ],
        "prompt": "Inspect app.js",
        "response": "Done",
        "model": "claude-fable-5",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "metadata": {"trace_type": "cursor", "cursor_storage_kind": "composerData"},
    }
    trace_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    examples = convert_traces_to_training_data(trace_file)

    assert detect_trace_type(events) == "cursor"
    assert len(examples) == 1
    assert examples[0]["metadata"]["trace_type"] == "cursor"
    assert examples[0]["metadata"]["model"] == "claude-fable-5"
    assert examples[0]["messages"] == row["messages"]
    tools_by_name = {tool["function"]["name"]: tool for tool in examples[0]["tools"]}
    assert {"read_file", "run_terminal_cmd", "edit_file", "codebase_search"}.issubset(tools_by_name)
    assert tools_by_name["read_file"]["function"]["parameters"]["required"] == ["path"]
    assert tools_by_name["read_file"]["function"]["description"] == "Read file contents from the workspace."


def test_convert_cursor_session_events_preserves_messages_tools_and_trace_type(tmp_path: Path):
    trace_file = tmp_path / "cursor-session.jsonl"
    events = [
        {
            "type": "cursor_session_meta",
            "source": "cursor",
            "session_id": "session-1",
            "model": "claude-opus-4.5",
            "cursor_scope": "project",
        },
        {
            "type": "cursor_available_tools",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Shell",
                        "description": "Run shell commands.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
        },
        {"role": "user", "message": {"content": [{"type": "text", "text": "List files"}]}},
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "text": "Need a directory listing."},
                    {"type": "tool_use", "id": "call-shell", "name": "Shell", "input": {"command": "ls"}},
                ]
            },
        },
        {
            "role": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "call-shell", "content": "README.md\nsrc"}]
            },
        },
        {"role": "assistant", "message": {"content": [{"type": "text", "text": "Found README.md and src."}]}},
        {"type": "turn_ended", "status": "success"},
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    assert detect_trace_type(events) == "cursor"
    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "List files"
    assert example.metadata["trace_type"] == "cursor"
    assert example.metadata["session_id"] == "session-1"
    assert example.metadata["model"] == "claude-opus-4.5"
    assert [message["role"] for message in example.messages] == ["user", "assistant", "tool", "assistant"]
    assert example.messages[1]["reasoning_content"] == "Need a directory listing."
    assert example.messages[1]["tool_calls"][0]["function"]["name"] == "Shell"
    assert example.messages[2]["tool_call_id"] == "call-shell"
    assert example.messages[2]["content"] == "README.md\nsrc"
    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"Shell", "read_file", "run_terminal_cmd", "edit_file", "codebase_search"}.issubset(tools_by_name)
    assert tools_by_name["Shell"]["function"]["parameters"]["required"] == ["command"]


def test_convert_structured_provider_rows_seed_provider_builtin_tools(tmp_path: Path):
    trace_file = tmp_path / "provider-rows.jsonl"
    expectations = {
        "codex": {"exec_command", "update_plan"},
        "cursor": {"read_file", "run_terminal_cmd"},
        "claude-code": {"Bash", "Read", "Edit"},
        "droid": {"Read", "Execute", "TodoWrite"},
        "hermes": {"terminal", "read_file"},
        "pi": {"bash", "read", "write", "edit"},
        "openclaw": {"exec", "process", "message", "sessions_spawn"},
    }
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Run {trace_type}"},
                {"role": "assistant", "content": "Done"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": f"{trace_type.replace('-', '_')}_extra",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "metadata": {"trace_type": trace_type},
        }
        for trace_type in expectations
    ]
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "Plain chat"},
                {"role": "assistant", "content": "Done"},
            ],
            "metadata": {"trace_type": "chat"},
        }
    )
    trace_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = convert_traces_to_training_data(trace_file)

    examples_by_type = {example["metadata"]["trace_type"]: example for example in examples}
    for trace_type, required_names in expectations.items():
        names = {tool["function"]["name"] for tool in examples_by_type[trace_type]["tools"]}
        assert required_names.issubset(names)
        assert f"{trace_type.replace('-', '_')}_extra" in names
    assert examples_by_type["chat"]["tools"] == []


def test_convert_openclaw_trace_uses_distinct_type_with_shared_event_envelope(tmp_path: Path):
    trace_file = tmp_path / "openclaw-session.jsonl"
    events = [
        {
            "type": "session",
            "version": 3,
            "id": "0f853abd-d578-4a01-b37f-504880057fe4",
            "timestamp": "2026-02-16T23:40:43.787Z",
            "cwd": "/Users/calebfahlgren/.openclaw/workspace",
        },
        {
            "type": "model_change",
            "id": "d11668ab",
            "provider": "anthropic",
            "modelId": "claude-opus-4-6",
        },
        {
            "type": "thinking_level_change",
            "id": "1196999b",
            "thinkingLevel": "low",
        },
        {
            "type": "custom",
            "id": "teich-system-1",
            "customType": "teich-system-prompt",
            "data": {"systemPrompt": "Pi runner system prompt should not contaminate OpenClaw."},
        },
        {
            "type": "custom",
            "id": "teich-tools-1",
            "customType": "teich-available-tools",
            "data": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "openclaw_extra_tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
        },
        {
            "type": "message",
            "id": "user-1",
            "timestamp": "2026-02-16T23:40:43.700Z",
            "message": {
                "role": "user",
                "timestamp": 1771285243795,
                "content": [{"type": "text", "text": "Hi"}],
            },
        },
        {
            "type": "message",
            "id": "assistant-1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "\n\n"},
                    {"type": "thinking", "thinking": "First session, BOOTSTRAP.md exists. Let me follow it."},
                    {"type": "text", "text": "OpenClaw response."},
                ],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "openclaw"
    assert example.metadata["session_id"] == "0f853abd-d578-4a01-b37f-504880057fe4"
    assert example.metadata["model_provider"] == "anthropic"
    assert example.metadata["model"] == "claude-opus-4-6"
    assert example.metadata["cwd"] == "/Users/calebfahlgren/.openclaw/workspace"
    assert example.metadata["thinking_level"] == "low"
    assert example.metadata["first_message_timestamp"] == "2026-02-16T23:40:43.795000Z"
    assert "system_prompt" not in example.metadata
    assert example.prompt == "Hi"
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"read", "write", "edit", "exec", "process", "message", "sessions_spawn"}.issubset(tool_names)
    assert "openclaw_extra_tool" in tool_names
    assert example.messages == [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "OpenClaw response.",
            "reasoning_content": "First session, BOOTSTRAP.md exists. Let me follow it.",
        },
    ]


def test_convert_pi_trace_ignores_malformed_tool_calls(tmp_path: Path):
    trace_file = tmp_path / "pi-bad-tools.jsonl"
    events = [
        {
            "type": "session",
            "id": "pi-session-2",
            "version": 3,
            "cwd": "/workspace/project",
        },
        {
            "type": "model_change",
            "id": "model-2",
            "modelId": "minimax/minimax-m2.7",
        },
        {
            "type": "message",
            "id": "user-1",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Write a file"}],
            },
        },
        {
            "type": "message",
            "id": "assistant-1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "call_write", "name": "write", "arguments": {}},
                    {"type": "toolCall", "id": "call_content", "name": "content", "arguments": {}},
                    {"type": "toolCall", "id": "call_path", "name": "path", "arguments": {}},
                ],
            },
        },
        {
            "type": "message",
            "id": "tool-1",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_write",
                "toolName": "write",
                "content": [{"type": "text", "text": "Validation failed for tool \"write\""}],
                "isError": True,
            },
        },
        {
            "type": "message",
            "id": "tool-2",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_content",
                "toolName": "content",
                "content": [{"type": "text", "text": "Tool content not found"}],
                "isError": True,
            },
        },
        {
            "type": "message",
            "id": "tool-3",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_path",
                "toolName": "path",
                "content": [{"type": "text", "text": "Tool path not found"}],
                "isError": True,
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"bash", "read", "write", "edit"}.issubset(tool_names)
    assert all(message.get("name") != "content" for message in example.messages if message.get("role") == "tool")
    assert all(message.get("name") != "path" for message in example.messages if message.get("role") == "tool")


def test_convert_codex_trace_uses_tool_descriptions_from_base_instructions(tmp_path: Path):
    trace_file = tmp_path / "codex-trace.jsonl"
    events = [
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-session-1",
                "base_instructions": {
                    "text": "You are a coding agent.\n\nAvailable tools:\n- bash: Execute shell commands\n- read: Read file contents\n"
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
                "arguments": '{"command":"ls"}',
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"bash", "exec_command", "apply_patch", "update_plan", "view_image"}.issubset(tools_by_name)
    assert tools_by_name["bash"]["function"]["description"] == "Execute shell commands"
    assert tools_by_name["bash"]["function"]["parameters"]["required"] == ["command"]


def test_convert_pi_trace_uses_teich_eof_system_prompt_metadata(tmp_path: Path):
    trace_file = tmp_path / "pi-trace.jsonl"
    events = [
        {"type": "session", "id": "pi-session-1"},
        {
            "type": "message",
            "id": "user-1",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Build the app"}],
            },
        },
        {
            "type": "message",
            "id": "assistant-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
        {
            "type": "custom",
            "id": "teich-system-1",
            "customType": "teich-system-prompt",
            "data": {"systemPrompt": "Use the prompt-level system.", "source": "teich"},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[0] == {"role": "system", "content": "Use the prompt-level system."}
    assert example.messages[1] == {"role": "user", "content": "Build the app"}
    assert example.metadata["system_prompt"] == "Use the prompt-level system."


def test_convert_pi_trace_uses_teich_available_tools_without_tool_calls(tmp_path: Path):
    trace_file = tmp_path / "pi-tools.jsonl"
    events = [
        {"type": "session", "id": "pi-session-1"},
        {
            "type": "message",
            "id": "user-1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
        },
        {
            "type": "message",
            "id": "assistant-1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
        },
        {
            "type": "custom",
            "id": "teich-tools-1",
            "customType": "teich-available-tools",
            "data": {
                "source": "teich",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run shell commands.",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        },
                    }
                ],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[-1] == {"role": "assistant", "content": "Hi."}
    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"bash", "read", "write", "edit"}.issubset(tools_by_name)
    assert tools_by_name["bash"]["function"]["description"] == "Run shell commands."
    assert tools_by_name["bash"]["function"]["parameters"]["required"] == ["command"]


def test_convert_codex_trace_keeps_system_prompt_and_tools_without_tool_calls(tmp_path: Path):
    trace_file = tmp_path / "codex-tools.jsonl"
    events = [
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-session-1",
                "source": "exec",
                "model_provider": "openrouter",
                "base_instructions": {"text": "You are a careful coding agent."},
            },
        },
        {
            "type": "turn_context",
            "payload": {
                "turn_id": "turn-1",
                "model": "google/gemini-3.1-flash-lite",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-05-13T06:03:00.000Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hi"}],
            },
        },
        {
            "type": "event_msg",
            "timestamp": "2026-05-13T06:03:06.000Z",
            "payload": {"type": "user_message", "message": "Say hi"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "tool_schema",
                "name": "exec_command",
                "schema": {
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[0] == {"role": "system", "content": "You are a careful coding agent."}
    assert example.metadata["trace_type"] == "codex"
    assert example.metadata["source"] == "exec"
    assert example.metadata["model_provider"] == "openrouter"
    assert example.metadata["model"] == "google/gemini-3.1-flash-lite"
    assert example.metadata["system_prompt"] == "You are a careful coding agent."
    assert example.metadata["first_message_timestamp"] == "2026-05-13T06:03:06.000Z"
    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"bash", "exec_command", "apply_patch", "update_plan", "view_image"}.issubset(tools_by_name)
    assert tools_by_name["exec_command"]["function"]["parameters"]["required"] == ["cmd"]


def test_convert_codex_trace_normalizes_custom_tool_calls(tmp_path: Path):
    trace_file = tmp_path / "codex-custom-tool.jsonl"
    events = [
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-session-1",
                "base_instructions": {"text": "You are a coding agent."},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Patch the app"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "status": "completed",
                "name": "apply_patch",
                "call_id": "call_patch",
                "input": "*** Begin Patch\n*** Add File: app.py\n+print('hi')\n*** End Patch\n",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call_patch",
                "output": json.dumps({"output": "Success. Updated the following files:\nA app.py\n", "metadata": {"exit_code": 0}}),
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "Patch the app"
    assert example.messages[-2]["tool_calls"] == [
        {
            "id": "call_patch",
            "type": "function",
            "function": {
                "name": "apply_patch",
                "arguments": {"patch": "*** Begin Patch\n*** Add File: app.py\n+print('hi')\n*** End Patch\n"},
            },
        }
    ]
    assert example.messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_patch",
        "name": "apply_patch",
        "content": "Success. Updated the following files:\nA app.py\n",
    }
    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"bash", "exec_command", "apply_patch", "update_plan", "view_image"}.issubset(tools_by_name)
    assert tools_by_name["apply_patch"]["function"]["parameters"]["required"] == ["patch"]


def test_convert_claude_code_stream_json_trace(tmp_path: Path):
    trace_file = tmp_path / "claude-code.jsonl"
    events = [
        {
            "type": "external_session_meta",
            "payload": {
                "id": "teich-session",
                "source": "claude-code",
                "model_provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "cwd": "/workspace",
            },
        },
        {
            "type": "external_message",
            "role": "user",
            "content": "Inspect the project",
            "timestamp": "2026-05-14T00:00:01.000Z",
        },
        {
            "type": "system",
            "subtype": "init",
            "session_id": "claude-session",
            "model": "claude-sonnet-4-6",
            "tools": ["Bash", "Edit"],
        },
        {
            "type": "assistant",
            "session_id": "claude-session",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "I'll inspect the files."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
        {
            "type": "user",
            "session_id": "claude-session",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md\nsrc"},
                ],
            },
        },
        {
            "type": "result",
            "session_id": "claude-session",
            "result": "Done.",
            "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
            "total_cost_usd": 0.01,
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "claude-code"
    assert example.metadata["session_id"] == "claude-session"
    assert example.metadata["model_provider"] == "anthropic"
    assert example.metadata["first_message_timestamp"] == "2026-05-14T00:00:01.000Z"
    assert example.prompt == "Inspect the project"
    assert example.messages[0] == {"role": "user", "content": "Inspect the project"}
    assert example.messages[1]["role"] == "assistant"
    assert example.messages[1]["content"] == "I'll inspect the files."
    assert example.messages[1]["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": {"command": "ls"}},
        }
    ]
    assert example.messages[2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "name": "unknown_tool",
        "content": "README.md\nsrc",
    }
    assert example.messages[3] == {"role": "assistant", "content": "Done."}
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"Bash", "Edit", "TodoWrite"}.issubset(tool_names)
    bash_tool = next(tool for tool in example.tools if tool["function"]["name"] == "Bash")
    assert bash_tool["function"]["parameters"]["properties"]["command"]["type"] == "string"
    todo_tool = next(tool for tool in example.tools if tool["function"]["name"] == "TodoWrite")
    assert todo_tool["function"]["parameters"]["properties"]["todos"]["type"] == "array"


def test_convert_claude_code_trace_ignores_tool_result_timestamp_for_first_message(tmp_path: Path):
    trace_file = tmp_path / "claude-code-tool-result-timestamp.jsonl"
    events = [
        {
            "type": "user",
            "session_id": "claude-session",
            "message": {
                "role": "user",
                "content": "Inspect the project",
            },
        },
        {
            "type": "assistant",
            "session_id": "claude-session",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
            },
        },
        {
            "type": "user",
            "session_id": "claude-session",
            "timestamp": "2026-05-14T00:00:02.000Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md\nsrc"},
                ],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "Inspect the project"
    assert "first_message_timestamp" not in example.metadata
    assert example.messages[2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "name": "unknown_tool",
        "content": "README.md\nsrc",
    }


def test_convert_droid_trace(tmp_path: Path):
    trace_file = tmp_path / "droid.jsonl"
    events = [
        {
            "type": "session_start",
            "id": "droid-session",
            "title": "inspect the project",
            "sessionTitle": "Inspect project files",
            "owner": "caleb",
            "version": 2,
            "cwd": "/workspace/project",
        },
        {
            "type": "message",
            "id": "message-1",
            "timestamp": "2026-06-02T18:55:30.274Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Inspect the project"}],
            },
            "parentId": None,
        },
        {
            "type": "message",
            "id": "message-2",
            "timestamp": "2026-06-02T18:55:35.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should list the files first.",
                        "signature": "reasoning_content",
                        "signatureProvider": "generic-chat-completion-api",
                        "durationMs": 1200,
                    },
                    {"type": "text", "text": "I'll list the files."},
                    {"type": "tool_use", "id": "LS_0", "name": "LS", "input": {"directory_path": "/workspace/project"}},
                ],
                "chatCompletionReasoningField": "reasoning_content",
                "chatCompletionReasoningContent": "I should list the files first.",
            },
            "parentId": "message-1",
        },
        {
            "type": "message",
            "id": "message-3",
            "timestamp": "2026-06-02T18:55:36.000Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "LS_0", "is_error": False, "content": "README.md\nsrc"},
                ],
            },
            "parentId": "message-2",
        },
        {
            "type": "message",
            "id": "message-4",
            "timestamp": "2026-06-02T18:55:40.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "The project contains README.md and src."}],
            },
            "parentId": "message-3",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "droid"
    assert example.metadata["session_id"] == "droid-session"
    assert example.metadata["cwd"] == "/workspace/project"
    assert example.metadata["title"] == "Inspect project files"
    assert example.metadata["turn_count"] == 1
    assert example.metadata["first_message_timestamp"] == "2026-06-02T18:55:30.274Z"
    assert example.prompt == "Inspect the project"
    assert example.messages[0] == {"role": "user", "content": "Inspect the project"}
    assert example.messages[1]["role"] == "assistant"
    assert example.messages[1]["content"] == "I'll list the files."
    assert example.messages[1]["reasoning_content"] == "I should list the files first."
    assert example.messages[1]["tool_calls"] == [
        {
            "id": "LS_0",
            "type": "function",
            "function": {"name": "LS", "arguments": {"directory_path": "/workspace/project"}},
        }
    ]
    assert example.messages[2] == {
        "role": "tool",
        "tool_call_id": "LS_0",
        "name": "LS",
        "content": "README.md\nsrc",
    }
    assert example.messages[3] == {"role": "assistant", "content": "The project contains README.md and src."}
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert "LS" in tool_names


def test_convert_droid_trace_reads_settings_sidecar(tmp_path: Path):
    trace_file = tmp_path / "droid-settings.jsonl"
    events = [
        {"type": "session_start", "id": "droid-session", "version": 2, "cwd": "/workspace/project"},
        {
            "type": "message",
            "id": "message-1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
        },
        {
            "type": "message",
            "id": "message-2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    settings = {
        "model": "custom:Kimi-K2.6-[HF-Router]-0",
        "reasoningEffort": "high",
        "autonomyLevel": "medium",
        "providerLock": "generic-chat-completion-api",
        "tokenUsage": {
            "inputTokens": 202809,
            "outputTokens": 29390,
            "cacheCreationTokens": 0,
            "cacheReadTokens": 4843520,
            "thinkingTokens": 0,
            "factoryCredits": 0,
        },
    }
    (tmp_path / "droid-settings.settings.json").write_text(json.dumps(settings), encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["model"] == "custom:Kimi-K2.6-[HF-Router]-0"
    assert example.metadata["model_provider"] == "generic-chat-completion-api"
    assert example.metadata["usage"] == settings["tokenUsage"]
    assert example.metadata["reasoning_effort"] == "high"
    assert example.metadata["autonomy_level"] == "medium"


def test_convert_droid_trace_without_settings_sidecar(tmp_path: Path):
    trace_file = tmp_path / "droid-no-settings.jsonl"
    events = [
        {"type": "session_start", "id": "droid-session", "version": 2, "cwd": "/workspace/project"},
        {
            "type": "message",
            "id": "message-1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["model"] is None
    assert example.metadata["model_provider"] == "factory"
    assert "usage" not in example.metadata


def test_convert_droid_trace_prefers_user_authored_prompt(tmp_path: Path):
    trace_file = tmp_path / "droid-prompt.jsonl"
    events = [
        {"type": "session_start", "id": "droid-session", "version": 2, "cwd": "/workspace/project"},
        {
            "type": "message",
            "id": "message-1",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "<system-reminder>Injected environment context.</system-reminder>"}],
                "visibility": "llm_only",
            },
        },
        {
            "type": "message",
            "id": "message-2",
            "message": {"role": "user", "content": [{"type": "text", "text": "Summarize the README"}]},
        },
        {
            "type": "message",
            "id": "message-3",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "Summarize the README"
    assert example.messages[0]["content"].startswith("<system-reminder>")
    assert example.messages[1] == {"role": "user", "content": "Summarize the README"}


def test_convert_droid_trace_skips_user_only_and_state_events(tmp_path: Path):
    trace_file = tmp_path / "droid-edge-cases.jsonl"
    events = [
        {"type": "session_start", "id": "droid-session", "version": 2, "cwd": "/workspace/project"},
        {
            "type": "message",
            "id": "message-1",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "No active subscription found."}],
                "visibility": "user_only",
            },
        },
        {
            "type": "message",
            "id": "message-2",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this screenshot"},
                    {"type": "image", "source": {"type": "base64", "data": "abc123"}},
                ],
                "visibility": "llm_only",
            },
        },
        {
            "type": "todo_state",
            "id": "todo-1",
            "todos": {"todos": "1. [in_progress] Describe the screenshot"},
            "messageIndex": 1,
        },
        {
            "type": "compaction_state",
            "id": "compaction-1",
            "summaryText": "USER: earlier conversation summary",
        },
        {
            "type": "message",
            "id": "message-3",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "It shows a terminal."}]},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "Describe this screenshot"
    assert example.messages == [
        {"role": "user", "content": "Describe this screenshot"},
        {"role": "assistant", "content": "It shows a terminal."},
    ]


def test_convert_droid_trace_includes_builtin_tools(tmp_path: Path):
    trace_file = tmp_path / "droid-tools.jsonl"
    events = [
        {"type": "session_start", "id": "droid-session", "version": 2, "cwd": "/workspace/project"},
        {
            "type": "message",
            "id": "message-1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
        },
        {
            "type": "message",
            "id": "message-2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"Read", "Edit", "Create", "Execute", "LS", "Glob", "Grep", "TodoWrite", "FetchUrl", "Skill"}.issubset(tool_names)
    read_tool = next(tool for tool in example.tools if tool["function"]["name"] == "Read")
    assert read_tool["function"]["parameters"]["properties"]["file_path"]["type"] == "string"
    assert read_tool["function"]["parameters"]["required"] == ["file_path"]
    glob_tool = next(tool for tool in example.tools if tool["function"]["name"] == "Glob")
    assert glob_tool["function"]["parameters"]["properties"]["patterns"]["type"] == "array"


def test_convert_claude_code_keeps_init_tools_without_tool_calls(tmp_path: Path):
    trace_file = tmp_path / "claude-no-tools.jsonl"
    events = [
        {"type": "system", "subtype": "init", "session_id": "claude-session", "model": "claude-sonnet", "tools": ["Bash", "Edit"]},
        {"type": "user", "session_id": "claude-session", "message": {"role": "user", "content": "Say hi"}},
        {
            "type": "assistant",
            "session_id": "claude-session",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi."}],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[-1] == {"role": "assistant", "content": "Hi."}
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"Bash", "Edit"}.issubset(tool_names)


def test_convert_claude_code_marks_native_api_error_message(tmp_path: Path):
    trace_file = tmp_path / "claude-code-api-error.jsonl"
    events = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Build app"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "error": "api_error",
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "content": [
                    {
                        "type": "text",
                        "text": 'API Error: ZlibError fetching "http://127.0.0.1:17891/v1/messages?beta=true".',
                    }
                ],
            },
            "uuid": "assistant-uuid",
            "parentUuid": "user-uuid",
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[1]["content"].startswith("API Error: ZlibError")
    assert example.messages[1]["teich_provider_error"] == "api_error"


def test_convert_claude_code_filters_local_commands_but_keeps_goal(tmp_path: Path):
    trace_file = tmp_path / "claude-code-local-commands.jsonl"
    events = [
        {
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-06-09T19:39:49.192Z",
            "message": {
                "role": "user",
                "content": "<local-command-caveat>Local command output follows.</local-command-caveat>",
            },
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "timestamp": "2026-06-09T19:39:49.192Z",
            "message": {
                "role": "user",
                "content": (
                    "<command-name>/model</command-name>\n"
                    "<command-message>model</command-message>\n"
                    "<command-args></command-args>"
                ),
            },
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "timestamp": "2026-06-09T19:39:49.192Z",
            "message": {
                "role": "user",
                "content": "<local-command-stdout>Set model to Fable 5.</local-command-stdout>",
            },
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "timestamp": "2026-06-09T19:40:12.388Z",
            "message": {
                "role": "user",
                "content": (
                    "<command-name>/goal</command-name>\n"
                    "<command-message>goal</command-message>\n"
                    "<command-args>create a realistic Boeing 747</command-args>"
                ),
            },
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "timestamp": "2026-06-09T19:40:12.388Z",
            "message": {
                "role": "user",
                "content": "<local-command-stdout>Goal set: create a realistic Boeing 747</local-command-stdout>",
            },
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-06-09T19:40:12.388Z",
            "message": {
                "role": "user",
                "content": "A session-scoped Stop hook is now active.",
            },
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "timestamp": "2026-06-09T19:40:23.229Z",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "Goal acknowledged."}],
            },
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "create a realistic Boeing 747"
    assert example.metadata["first_message_timestamp"] == "2026-06-09T19:40:12.388Z"
    assert example.messages == [
        {"role": "user", "content": "create a realistic Boeing 747"},
        {"role": "assistant", "content": "Goal acknowledged."},
    ]
    serialized = json.dumps(example.messages)
    assert "/model" not in serialized
    assert "<command-name>" not in serialized
    assert "<local-command-" not in serialized


def test_convert_claude_code_drops_synthetic_session_limit_but_keeps_continuation(tmp_path: Path):
    trace_file = tmp_path / "claude-code-session-limit.jsonl"
    events = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Build app"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "Created the initial files."}],
            },
            "uuid": "assistant-uuid",
            "parentUuid": "user-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "error": "rate_limit",
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "content": [
                    {
                        "type": "text",
                        "text": "You've hit your session limit · resets 1am (America/New_York)",
                    }
                ],
            },
            "uuid": "session-limit-uuid",
            "parentUuid": "assistant-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "finish up"},
            "uuid": "continuation-uuid",
            "parentUuid": "session-limit-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "Finished the remaining work."}],
            },
            "uuid": "final-assistant-uuid",
            "parentUuid": "continuation-uuid",
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages == [
        {"role": "user", "content": "Build app"},
        {"role": "assistant", "content": "Created the initial files."},
        {"role": "user", "content": "finish up"},
        {"role": "assistant", "content": "Finished the remaining work."},
    ]
    assert example.metadata["turn_count"] == 2
    serialized = json.dumps(example.messages)
    assert "session limit" not in serialized
    assert "teich_provider_error" not in serialized


def test_convert_claude_code_drops_synthetic_no_response_requested(tmp_path: Path):
    trace_file = tmp_path / "claude-code-no-response-requested.jsonl"
    events = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Build the studio"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "I started the implementation."}],
            },
            "uuid": "assistant-uuid",
            "parentUuid": "user-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "isApiErrorMessage": False,
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [{"type": "text", "text": "No response requested."}],
            },
            "uuid": "cancel-uuid",
            "parentUuid": "assistant-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Actually, use the native terminal view."},
            "uuid": "continuation-uuid",
            "parentUuid": "cancel-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "I switched the design."}],
            },
            "uuid": "final-assistant-uuid",
            "parentUuid": "continuation-uuid",
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages == [
        {"role": "user", "content": "Build the studio"},
        {"role": "assistant", "content": "I started the implementation."},
        {"role": "user", "content": "Actually, use the native terminal view."},
        {"role": "assistant", "content": "I switched the design."},
    ]
    assert example.metadata["turn_count"] == 2
    assert "No response requested" not in json.dumps(example.messages)


def test_convert_claude_code_preserves_native_context_and_desktop_tool_schemas(tmp_path: Path):
    trace_file = tmp_path / "claude-code-native-context.jsonl"
    events = [
        {
            "type": "mode",
            "mode": "normal",
            "sessionId": "claude-session",
        },
        {
            "type": "permission-mode",
            "permissionMode": "bypassPermissions",
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Build app"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
            "entrypoint": "claude-desktop",
            "cwd": "C:\\Users\\aranr\\project",
            "version": "2.1.175",
            "gitBranch": "main",
        },
        {
            "type": "attachment",
            "attachment": {
                "type": "deferred_tools_delta",
                "addedNames": [
                    "TaskCreate",
                    "mcp__Claude_Preview__preview_start",
                    "mcp__computer-use__screenshot",
                ],
            },
            "sessionId": "claude-session",
        },
        {
            "type": "attachment",
            "attachment": {
                "type": "skill_listing",
                "content": "- verify: Verify the app in a real preview.",
            },
            "sessionId": "claude-session",
        },
        {
            "type": "attachment",
            "attachment": {
                "type": "mcp_instructions_delta",
                "addedNames": ["Claude_Preview"],
                "addedBlocks": ["## Claude Preview\nUse preview tools to inspect browser apps."],
            },
            "sessionId": "claude-session",
        },
        {
            "type": "system",
            "subtype": "informational",
            "content": "Auto mode lets Claude handle permission prompts automatically.",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "TaskCreate",
                        "input": {
                            "subject": "Build app",
                            "description": "Create the app shell.",
                            "activeForm": "Building app",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "mcp__Claude_Preview__preview_start",
                        "input": {"name": "app"},
                    },
                ],
            },
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Task created"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": '{"serverId":"abc"}'},
                ],
            },
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "Started."}],
            },
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert [message["role"] for message in example.messages[:4]] == ["user", "system", "assistant", "tool"]
    system_context = example.messages[1]["content"]
    assert example.messages[1]["masked"] is True
    assert "Claude Code deferred tools available through ToolSearch" in system_context
    assert "Claude Code available skills" in system_context
    assert "Use preview tools to inspect browser apps." in system_context
    assert "Auto mode lets Claude handle permission prompts automatically." in system_context
    assert example.metadata["system_prompt"] == system_context
    assert example.metadata["entrypoint"] == "claude-desktop"
    assert example.metadata["cwd"] == "C:\\Users\\aranr\\project"
    assert example.metadata["cli_version"] == "2.1.175"
    assert example.metadata["git_branch"] == "main"
    assert example.metadata["mode"] == "normal"
    assert example.metadata["permission_mode"] == "bypassPermissions"
    assert example.metadata["claude_deferred_tools"] == [
        "TaskCreate",
        "mcp__Claude_Preview__preview_start",
        "mcp__computer-use__screenshot",
    ]
    assert example.metadata["claude_mcp_instruction_names"] == ["Claude_Preview"]

    tools = {tool["function"]["name"]: tool["function"] for tool in example.tools}
    assert tools["TaskCreate"]["parameters"]["required"] == ["subject", "description", "activeForm"]
    assert tools["mcp__Claude_Preview__preview_start"]["parameters"]["required"] == ["name"]
    assert tools["mcp__computer-use__screenshot"]["parameters"]["properties"]["application"] == {"type": "string"}
    assert tools["TaskCreate"]["description"].startswith("Create a tracked task")
    assert tools["mcp__Claude_Preview__preview_start"]["description"].startswith("Start a Claude Preview")
    assert tools["mcp__computer-use__screenshot"]["description"].startswith("Call the computer use MCP tool")


def test_convert_claude_code_preserves_queued_command_prompt(tmp_path: Path):
    trace_file = tmp_path / "claude-code-queued-command.jsonl"
    events = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Build the animation"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "npm test"}}],
            },
            "uuid": "assistant-uuid",
            "parentUuid": "user-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "tests passed"}],
            },
            "uuid": "tool-result-uuid",
            "parentUuid": "assistant-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "attachment",
            "attachment": {
                "type": "queued_command",
                "prompt": "dont add any button just wait 2 seconds on page load and start the animation",
                "commandMode": "prompt",
            },
            "parentUuid": "tool-result-uuid",
            "uuid": "queued-command-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I removed the button and added the timer."}],
            },
            "uuid": "final-assistant-uuid",
            "parentUuid": "queued-command-uuid",
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages == [
        {"role": "user", "content": "Build the animation"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": {"command": "npm test"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "name": "unknown_tool", "content": "tests passed"},
        {
            "role": "user",
            "content": "dont add any button just wait 2 seconds on page load and start the animation",
        },
        {"role": "assistant", "content": "I removed the button and added the timer."},
    ]
    assert example.metadata["turn_count"] == 2


def test_convert_native_claude_code_transcript_with_camel_session_id(tmp_path: Path):
    trace_file = tmp_path / "native-claude.jsonl"
    events = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Inspect the project"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "I'll inspect the files."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            "uuid": "assistant-uuid",
            "parentUuid": "user-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md\nsrc"},
                ],
            },
            "uuid": "tool-result-uuid",
            "parentUuid": "assistant-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "result",
            "sessionId": "claude-session",
            "result": "Done.",
            "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
            "total_cost_usd": 0.01,
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "claude-code"
    assert example.metadata["session_id"] == "claude-session"
    assert example.prompt == "Inspect the project"
    assert example.messages[0] == {"role": "user", "content": "Inspect the project"}
    assert example.messages[1]["tool_calls"][0]["function"] == {"name": "Bash", "arguments": {"command": "ls"}}
    assert example.messages[2]["role"] == "tool"
    assert example.messages[3] == {"role": "assistant", "content": "Done."}
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"Bash", "Read", "Edit", "TodoWrite"}.issubset(tool_names)


def test_convert_native_claude_code_orders_fragmented_assistant_turn_semantically(tmp_path: Path):
    trace_file = tmp_path / "native-fragmented-claude.jsonl"
    events = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Update the server"},
            "uuid": "user-uuid",
            "parentUuid": None,
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "I'll update the timeout handling."},
                ],
            },
            "uuid": "assistant-text-uuid",
            "parentUuid": "user-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The timeout should be configurable and covered by tests.",
                    },
                ],
            },
            "uuid": "assistant-thinking-uuid",
            "parentUuid": "assistant-text-uuid",
            "sessionId": "claude-session",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": {"file_path": "server.py"}},
                ],
            },
            "uuid": "assistant-tool-uuid",
            "parentUuid": "assistant-thinking-uuid",
            "sessionId": "claude-session",
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert len(example.messages) == 2
    assert example.messages[0] == {"role": "user", "content": "Update the server"}
    assert example.messages[1]["role"] == "assistant"
    assert example.messages[1]["content"] == "I'll update the timeout handling."
    assert example.messages[1]["reasoning_content"] == "The timeout should be configurable and covered by tests."
    assert example.messages[1]["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "Edit", "arguments": {"file_path": "server.py"}},
        }
    ]


def test_convert_teich_hermes_external_trace(tmp_path: Path):
    trace_file = tmp_path / "hermes.jsonl"
    events = [
        {
            "type": "external_session_meta",
            "payload": {
                "id": "hermes-session",
                "source": "hermes-agent",
                "model_provider": "hermes",
                "model": "qwen/qwen3-coder",
            },
        },
        {
            "type": "external_message",
            "role": "user",
            "content": "Build a CLI",
            "timestamp": "2026-05-15T00:00:01.000Z",
        },
        {"type": "external_message", "role": "assistant", "content": "Built it."},
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "hermes"
    assert example.metadata["source"] == "hermes-agent"
    assert example.metadata["session_id"] == "hermes-session"
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"read_file", "write_file", "terminal", "patch"}.issubset(tool_names)
    assert example.metadata["first_message_timestamp"] == "2026-05-15T00:00:01.000Z"
    assert example.prompt == "Build a CLI"
    assert example.messages == [
        {"role": "user", "content": "Build a CLI"},
        {"role": "assistant", "content": "Built it."},
    ]


def test_convert_teich_hermes_external_trace_uses_meta_tools_without_tool_calls(tmp_path: Path):
    trace_file = tmp_path / "hermes-tools.jsonl"
    events = [
        {
            "type": "external_session_meta",
            "payload": {
                "id": "hermes-session",
                "source": "hermes-agent",
                "model_provider": "hermes",
                "model": "qwen/qwen3-coder",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "delegate_task",
                            "description": "Spawn a delegated subagent.",
                            "parameters": {
                                "type": "object",
                                "properties": {"goal": {"type": "string"}},
                                "required": ["goal"],
                            },
                        },
                    }
                ],
            },
        },
        {"type": "external_message", "role": "user", "content": "Plan it"},
        {"type": "external_message", "role": "assistant", "content": "Done."},
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "hermes"
    assert example.messages[-1] == {"role": "assistant", "content": "Done."}
    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"delegate_task", "read_file", "write_file", "terminal", "patch"}.issubset(tools_by_name)
    assert tools_by_name["delegate_task"]["function"]["parameters"]["required"] == ["goal"]


def test_convert_teich_hermes_external_trace_preserves_tool_calls_and_parent_metadata(tmp_path: Path):
    trace_file = tmp_path / "hermes-child.jsonl"
    events = [
        {
            "type": "external_session_meta",
            "payload": {
                "id": "child-session",
                "source": "hermes-agent",
                "model_provider": "hermes",
                "model": "minimax/minimax-m2.5:free",
                "parent_session_id": "parent-session",
                "tool_call_count": 1,
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_read_tokens": 3,
                "reasoning_tokens": 2,
                "total_tokens": 22,
                "estimated_cost_usd": 0.001,
                "billing_provider": "openrouter",
                "billing_base_url": "https://openrouter.ai/api/v1",
                "system_prompt": "Use the delegated task contract.",
            },
        },
        {"type": "external_message", "role": "user", "content": "Delegate a task"},
        {
            "type": "external_message",
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "delegate_task",
                        "arguments": {"prompt": "sub task"},
                    },
                }
            ],
        },
        {
            "type": "external_message",
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "delegate_task",
            "content": "subagent smoke ok",
        },
        {"type": "external_message", "role": "assistant", "content": "Done.", "reasoning_content": "Checked output."},
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "hermes"
    assert example.metadata["parent_session_id"] == "parent-session"
    assert example.metadata["tool_call_count"] == 1
    assert example.metadata["input_tokens"] == 12
    assert example.metadata["cache_read_tokens"] == 3
    assert example.metadata["reasoning_tokens"] == 2
    assert example.metadata["total_tokens"] == 22
    assert example.metadata["estimated_cost_usd"] == 0.001
    assert example.metadata["billing_provider"] == "openrouter"
    assert example.metadata["billing_base_url"] == "https://openrouter.ai/api/v1"
    assert example.metadata["system_prompt"] == "Use the delegated task contract."
    assert example.messages[1]["tool_calls"][0]["function"] == {
        "name": "delegate_task",
        "arguments": {"prompt": "sub task"},
    }
    assert example.messages[2] == {
        "role": "tool",
        "content": "subagent smoke ok",
        "tool_call_id": "call_1",
        "name": "delegate_task",
    }
    assert example.messages[3]["reasoning_content"] == "Checked output."
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"delegate_task", "read_file", "write_file", "terminal", "patch"}.issubset(tool_names)


def test_convert_hermes_native_trace_preserves_raw_messages_and_metadata(tmp_path: Path):
    trace_file = tmp_path / "hermes-agent.jsonl"
    metadata = {
        "id": "child-session",
        "source": "cli",
        "teich_export_status": "completed",
        "teich_partial": False,
        "model_provider": "openrouter",
        "configured_model_provider": "openrouter",
        "model": "minimax/minimax-m2.5:free",
        "parent_session_id": "parent-session",
        "tool_call_count": 1,
        "input_tokens": 12,
        "output_tokens": 5,
        "cache_read_tokens": 3,
        "reasoning_tokens": 2,
        "total_tokens": 22,
        "estimated_cost_usd": 0.001,
        "billing_provider": "openrouter",
        "billing_base_url": "https://openrouter.ai/api/v1",
        "system_prompt": "Use the delegated task contract.",
    }
    conversation = [
        {"from": "system", "value": "Use the delegated task contract."},
        {"from": "human", "value": "Delegate a task", "timestamp": "2026-05-16T00:00:01.000Z"},
        {
            "from": "gpt",
            "value": '<think>\nNeed isolated context.\n</think>\n<tool_call>\n{"name": "delegate_task", "arguments": {"prompt": "sub task"}}\n</tool_call>',
        },
        {
            "from": "tool",
            "value": '<tool_response>\n{"tool_call_id": "call_1", "name": "delegate_task", "content": "subagent smoke ok"}\n</tool_response>',
        },
        {"from": "gpt", "value": "<think>\nChecked output.\n</think>\nDone."},
    ]
    row = {
        "id": "child-session",
        "task": "Delegate a task",
        "traces": conversation,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "delegate_task",
                    "description": "Spawn an isolated delegated subagent session.",
                    "parameters": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                },
            }
        ],
        "metadata": metadata,
    }
    trace_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "hermes"
    assert example.metadata["teich_export_status"] == "completed"
    assert example.metadata["teich_partial"] is False
    assert example.metadata["session_id"] == "child-session"
    assert example.metadata["parent_session_id"] == "parent-session"
    assert example.metadata["model_provider"] == "openrouter"
    assert example.metadata["configured_model_provider"] == "openrouter"
    assert example.metadata["billing_provider"] == "openrouter"
    assert example.metadata["billing_base_url"] == "https://openrouter.ai/api/v1"
    assert example.metadata["estimated_cost_usd"] == 0.001
    assert example.metadata["system_prompt"] == "Use the delegated task contract."
    assert example.metadata["first_message_timestamp"] == "2026-05-16T00:00:01.000Z"
    assert example.prompt == "Delegate a task"
    assert example.messages[2]["tool_calls"][0]["function"] == {
        "name": "delegate_task",
        "arguments": {"prompt": "sub task"},
    }
    assert example.messages[2]["reasoning_content"] == "Need isolated context."
    assert example.messages[3] == {
        "role": "tool",
        "content": "subagent smoke ok",
        "tool_call_id": "call_1",
        "name": "delegate_task",
    }
    assert example.messages[4]["reasoning_content"] == "Checked output."
    tools_by_name = {tool["function"]["name"]: tool for tool in example.tools}
    assert {"delegate_task", "read_file", "write_file", "terminal", "patch"}.issubset(tools_by_name)
    assert tools_by_name["delegate_task"]["function"]["parameters"]["required"] == ["prompt"]


def test_convert_hermes_aggregate_jsonl_returns_one_training_row_per_trace(tmp_path: Path):
    trace_file = tmp_path / "hermes-agent.jsonl"
    rows = [
        {
            "id": "session-1",
            "task": "first prompt",
            "traces": [
                {"from": "human", "value": "first prompt"},
                {"from": "gpt", "value": "first answer"},
            ],
            "tools": [],
            "metadata": {"id": "session-1", "source": "cli", "model_provider": "custom", "model": "Opus-Agent"},
        },
        {
            "id": "session-2",
            "task": "second prompt",
            "traces": [
                {"from": "human", "value": "second prompt"},
                {"from": "gpt", "value": "second answer"},
            ],
            "tools": [],
            "metadata": {"id": "session-2", "source": "cli", "model_provider": "custom", "model": "Opus-Agent"},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = convert_traces_to_training_data(trace_file)

    assert [example["prompt"] for example in examples] == ["first prompt", "second prompt"]
    assert [example["metadata"]["session_id"] for example in examples] == ["session-1", "session-2"]
    assert examples[1]["messages"][1] == {"role": "assistant", "content": "second answer"}


def test_convert_hermes_raw_message_jsonl_without_meta(tmp_path: Path):
    trace_file = tmp_path / "legacy-hermes.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Build a CLI"}),
                json.dumps({"role": "assistant", "content": "Built it."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "hermes"
    assert example.metadata["session_id"] == "legacy-hermes"
    assert example.prompt == "Build a CLI"
    assert example.messages == [
        {"role": "user", "content": "Build a CLI"},
        {"role": "assistant", "content": "Built it."},
    ]


def test_convert_trace_uses_reasoning_text_when_summary_is_empty(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Write factorial"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "This is a simple factorial task.",
                    }
                ],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "def factorial(n): return 1"}],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "Write factorial"
    assert example.messages[-1]["role"] == "assistant"
    assert example.messages[-1]["reasoning_content"] == "This is a simple factorial task."


def test_convert_trace_accumulates_multiple_reasoning_events(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect repo"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"text": "First thought."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"text": "Second thought."}],
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

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[-1]["role"] == "assistant"
    assert example.messages[-1]["reasoning_content"] == "First thought.\n\nSecond thought."


def test_convert_codex_trace_attaches_late_reasoning_to_tool_call(tmp_path: Path):
    trace_file = tmp_path / "codex-late-reasoning.jsonl"
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
                "summary": [{"text": "I already issued the file creation command."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "created",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_2",
                "arguments": '{"cmd":"ls app.py"}',
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[1]["role"] == "assistant"
    assert example.messages[1]["tool_calls"][0]["id"] == "call_1"
    assert example.messages[1]["reasoning_content"] == "I already issued the file creation command."
    assert example.messages[2]["role"] == "tool"
    assert example.messages[3]["role"] == "assistant"
    assert example.messages[3]["tool_calls"][0]["id"] == "call_2"
    assert "reasoning_content" not in example.messages[3]


def test_convert_codex_trace_orders_reasoning_text_and_tool_call(tmp_path: Path):
    trace_file = tmp_path / "codex-text-reasoning-tool.jsonl"
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
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I'll create the file."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"text": "Need to write the file before running it."}],
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
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages == [
        {"role": "user", "content": "Create a file"},
        {
            "role": "assistant",
            "content": "I'll create the file.",
            "reasoning_content": "Need to write the file before running it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": {"cmd": "touch app.py"}},
                }
            ],
        },
    ]


def test_normalize_codex_trace_orders_reasoning_text_before_prior_tool_call(tmp_path: Path):
    trace_file = tmp_path / "codex-runtime-order.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect files"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": '{"cmd":"ls"}',
            },
        },
        {"type": "event_msg", "payload": {"type": "agent_reasoning"}},
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "I need to inspect the files."}],
            },
        },
        {"type": "event_msg", "payload": {"type": "agent_message"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I’ll inspect the files."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "README.md",
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    normalized = normalize_codex_trace_events(events)

    assert [event.get("payload", {}).get("type") for event in normalized if event.get("type") == "response_item"] == [
        "message",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
    ]

    example = convert_trace_to_training_example(trace_file)

    assert example.messages == [
        {"role": "user", "content": "Inspect files"},
        {
            "role": "assistant",
            "content": "I’ll inspect the files.",
            "reasoning_content": "I need to inspect the files.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": {"cmd": "ls"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "exec_command",
            "content": "README.md",
        },
    ]


def test_normalize_codex_trace_orders_text_before_prior_tool_call(tmp_path: Path):
    trace_file = tmp_path / "codex-text-tool.jsonl"
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
        {"type": "event_msg", "payload": {"type": "agent_message"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I’ll create the file."}],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    normalized = normalize_codex_trace_events(events)

    assert [event.get("payload", {}).get("type") for event in normalized if event.get("type") == "response_item"] == [
        "message",
        "message",
        "function_call",
    ]

    example = convert_trace_to_training_example(trace_file)

    assert example.messages == [
        {"role": "user", "content": "Create a file"},
        {
            "role": "assistant",
            "content": "I’ll create the file.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": {"cmd": "touch app.py"}},
                }
            ],
        },
    ]


def test_normalize_codex_trace_orders_empty_assistant_before_prior_tool_call(tmp_path: Path):
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need to write the stylesheet."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": '{"cmd":"cat > styles.css"}',
            },
        },
        {"type": "event_msg", "payload": {"type": "agent_message"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": " "}],
            },
        },
    ]

    normalized = normalize_codex_trace_events(events)

    assert [event.get("payload", {}).get("type") for event in normalized if event.get("type") == "response_item"] == [
        "reasoning",
        "message",
        "function_call",
    ]


def test_normalize_claude_code_trace_drops_empty_assistant_fragment():
    events = [
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:02.000Z",
            "message": {"id": "msg_1", "role": "assistant", "content": []},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:03.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Need to inspect first."}],
            },
        },
    ]

    normalized = normalize_claude_code_trace_events(events)

    assert len(normalized) == 1
    assert normalized[0]["message"]["content"][0]["type"] == "thinking"


def test_normalize_claude_code_trace_orders_text_before_prior_tool_use():
    events = [
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:02.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:03.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "text", "text": "I’ll inspect the file."}],
            },
        },
    ]

    normalized = normalize_claude_code_trace_events(events)

    assert normalized[0]["message"]["content"][0]["type"] == "text"
    assert normalized[1]["message"]["content"][0]["type"] == "tool_use"


def test_normalize_claude_code_trace_groups_message_fragments_and_drops_blank_text():
    events = [
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:02.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:03.000Z",
            "message": {"id": "msg_1", "role": "assistant", "content": [{"type": "text", "text": " "}]},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:04.000Z",
            "message": {"id": "msg_1", "role": "assistant", "content": [{"type": "text", "text": "."}]},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:05.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Need to inspect first."}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:06.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "redacted_thinking", "data": "opaque"}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2026-05-13T00:00:07.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {}}],
            },
        },
    ]

    normalized = normalize_claude_code_trace_events(events)

    assert [event["message"]["content"][0]["type"] for event in normalized] == [
        "thinking",
        "redacted_thinking",
        "tool_use",
        "tool_use",
    ]
    assert [event["timestamp"] for event in normalized] == [
        "2026-05-13T00:00:02.000Z",
        "2026-05-13T00:00:05.000Z",
        "2026-05-13T00:00:06.000Z",
        "2026-05-13T00:00:07.000Z",
    ]


def test_convert_claude_code_realistic_fragments_preserve_each_reasoning_turn(tmp_path: Path):
    trace_file = tmp_path / "claude-realistic-fragments.jsonl"
    events = [
        {
            "type": "user",
            "sessionId": "session-1",
            "uuid": "user-1",
            "message": {"role": "user", "content": "Explore the repo"},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "uuid": "assistant-1-thinking",
            "parentUuid": "user-1",
            "timestamp": "2026-05-13T00:00:01.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "thinking", "thinking": "First I need to inspect the project files."}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "uuid": "assistant-1-tool",
            "parentUuid": "assistant-1-thinking",
            "timestamp": "2026-05-13T00:00:02.000Z",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
            },
        },
        {
            "type": "user",
            "sessionId": "session-1",
            "uuid": "tool-result-1",
            "parentUuid": "assistant-1-tool",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md\nsrc"}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "uuid": "assistant-blank",
            "parentUuid": "tool-result-1",
            "timestamp": "2026-05-13T00:00:03.000Z",
            "message": {"id": "msg_2", "role": "assistant", "model": "claude-sonnet-4-6", "content": []},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "uuid": "assistant-period",
            "parentUuid": "assistant-blank",
            "timestamp": "2026-05-13T00:00:04.000Z",
            "message": {
                "id": "msg_2",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "."}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "uuid": "assistant-2-thinking",
            "parentUuid": "assistant-period",
            "timestamp": "2026-05-13T00:00:05.000Z",
            "message": {
                "id": "msg_2",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "thinking", "thinking": "Next I should read the README before editing."}],
            },
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "uuid": "assistant-2-tool",
            "parentUuid": "assistant-2-thinking",
            "timestamp": "2026-05-13T00:00:06.000Z",
            "message": {
                "id": "msg_2",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"file_path": "README.md"}}
                ],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assistant_messages = [message for message in example.messages if message["role"] == "assistant"]
    assert [message.get("reasoning_content") for message in assistant_messages] == [
        "First I need to inspect the project files.",
        "Next I should read the README before editing.",
    ]
    assert [message["tool_calls"][0]["function"]["name"] for message in assistant_messages] == ["Bash", "Read"]
    assert all(message.get("content", "") != "." for message in assistant_messages)


def test_convert_trace_normalizes_nested_json_encoded_tool_arguments(tmp_path: Path):
    trace_file = tmp_path / "codex-nested-tool-args.jsonl"
    events = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Apply these edits"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "str_replace_editor",
                "call_id": "call_1",
                "arguments": json.dumps(
                    {
                        "path": "/workspace/file.txt",
                        "edits": json.dumps(
                            [
                                {"oldText": "alpha", "newText": "beta"},
                                {"oldText": "gamma", "newText": "delta"},
                            ]
                        ),
                    }
                ),
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.messages[-1]["tool_calls"][0]["function"]["arguments"] == {
        "path": "/workspace/file.txt",
        "edits": [
            {"oldText": "alpha", "newText": "beta"},
            {"oldText": "gamma", "newText": "delta"},
        ],
    }


def test_convert_pi_trace_uses_thinking_blocks_and_tool_results(tmp_path: Path):
    trace_file = tmp_path / "pi-trace.jsonl"
    events = [
        {
            "type": "session",
            "id": "pi-session-1",
            "version": 3,
            "cwd": "/workspace/project",
        },
        {
            "type": "model_change",
            "id": "model-1",
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-20250514",
        },
        {
            "type": "thinking_level_change",
            "id": "thinking-1",
            "thinkingLevel": "high",
        },
        {
            "type": "session_info",
            "id": "session-info-1",
            "name": "Pi session",
        },
        {
            "type": "message",
            "id": "developer-1",
            "message": {
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a coding agent.\n\nAvailable tools:\n- bash: Execute shell commands\n- read: Read file contents\n",
                    }
                ],
            },
        },
        {
            "type": "message",
            "id": "user-1",
            "timestamp": "2026-05-17T00:00:00.000Z",
            "message": {
                "role": "user",
                "timestamp": "2026-05-17T00:00:01.000Z",
                "content": [{"type": "text", "text": "Review this repo"}],
            },
        },
        {
            "type": "message",
            "id": "assistant-1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should inspect the repository structure first."},
                    {"type": "text", "text": "I'll inspect the repository structure first."},
                    {
                        "type": "toolCall",
                        "id": "toolu_123",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "id": "tool-1",
            "message": {
                "role": "toolResult",
                "toolCallId": "toolu_123",
                "toolName": "bash",
                "content": [{"type": "text", "text": "README.md\nsrc\n"}],
            },
        },
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.prompt == "Review this repo"
    assert example.metadata["trace_type"] == "pi"
    assert example.metadata["model_provider"] == "anthropic"
    assert example.metadata["thinking_level"] == "high"
    assert example.metadata["first_message_timestamp"] == "2026-05-17T00:00:01.000Z"
    assert example.messages[0]["role"] == "system"
    assert "Available tools:" in example.messages[0]["content"]
    assert example.messages[2]["role"] == "assistant"
    assert example.messages[2]["content"] == "I'll inspect the repository structure first."
    assert example.messages[2]["reasoning_content"] == "I should inspect the repository structure first."
    assert example.messages[2]["tool_calls"] == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": {"command": "ls"},
            },
        }
    ]
    assert example.messages[3] == {
        "role": "tool",
        "tool_call_id": "toolu_123",
        "name": "bash",
        "content": "README.md\nsrc",
    }
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"bash", "read", "write", "edit"}.issubset(tool_names)
    bash_tool = next(tool for tool in example.tools if tool["function"]["name"] == "bash")
    assert bash_tool["function"]["parameters"]["properties"]["command"] == {"type": "string"}
