import json
from pathlib import Path

from teich.converter import convert_trace_to_training_example


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

    assert example.tools == [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        }
    ]
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

    assert example.tools == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": True,
                },
            },
        }
    ]


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
    assert example.tools == [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                    "additionalProperties": True,
                },
            },
        }
    ]


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
        {"type": "external_message", "role": "user", "content": "Inspect the project"},
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
    assert {tool["function"]["name"] for tool in example.tools} == {"Bash", "Edit"}


def test_convert_external_agent_trace(tmp_path: Path):
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
        {"type": "external_message", "role": "user", "content": "Build a CLI"},
        {"type": "external_message", "role": "assistant", "content": "Built it."},
    ]
    trace_file.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    example = convert_trace_to_training_example(trace_file)

    assert example.metadata["trace_type"] == "hermes-agent"
    assert example.metadata["session_id"] == "hermes-session"
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
            "message": {
                "role": "user",
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
    assert example.tools == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": True,
                },
            },
        }
    ]
