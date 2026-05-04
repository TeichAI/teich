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
