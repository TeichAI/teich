import json
from pathlib import Path

from teich.converter import (
    convert_trace_to_training_example,
    convert_traces_to_training_data,
    normalize_claude_code_trace_events,
    normalize_codex_trace_events,
)


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
    tool_names = {tool["function"]["name"] for tool in example.tools}
    assert {"Bash", "Edit", "TodoWrite"}.issubset(tool_names)
    bash_tool = next(tool for tool in example.tools if tool["function"]["name"] == "Bash")
    assert bash_tool["function"]["parameters"]["properties"]["command"]["type"] == "string"
    todo_tool = next(tool for tool in example.tools if tool["function"]["name"] == "TodoWrite")
    assert todo_tool["function"]["parameters"]["properties"]["todos"]["type"] == "array"


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


def test_convert_external_agent_trace_preserves_hermes_tool_calls_and_parent_metadata(tmp_path: Path):
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

    assert example.metadata["parent_session_id"] == "parent-session"
    assert example.metadata["tool_call_count"] == 1
    assert example.metadata["input_tokens"] == 12
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
    assert [tool["function"]["name"] for tool in example.tools] == ["delegate_task"]


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
        {"from": "human", "value": "Delegate a task"},
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
    assert example.tools[0]["function"]["name"] == "delegate_task"
    assert [tool["function"]["name"] for tool in example.tools] == ["delegate_task"]


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
