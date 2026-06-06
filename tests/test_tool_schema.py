from unittest.mock import patch

from teich.config import Config, MCPConfig
import pytest
from datasets import Dataset

from teich import prepare_data
from teich.tool_schema import snapshot_configured_tools, snapshot_mcp_tools, validate_tool_calls


class TinyChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}</{message['role']}>" for message in messages
        )
        if tokenize:
            return self(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
        output = {"input_ids": [ord(character) for character in text]}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(output["input_ids"])
        return output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)


def test_snapshot_configured_tools_includes_codex_builtins_and_mcp_tools():
    config = Config(
        agent={"provider": "codex"},
        mcp_servers=[MCPConfig(name="search", command="server", enabled_tools=["lookup"])],
    )
    mcp_tool = {
        "type": "function",
        "function": {
            "name": "search.lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("teich.tool_schema.snapshot_mcp_tools", return_value=[mcp_tool]):
        tools = snapshot_configured_tools(config)

    names = [tool["function"]["name"] for tool in tools]
    assert "bash" in names
    assert "exec_command" in names
    assert "apply_patch" in names
    assert "update_plan" in names
    assert "search.lookup" in names


def test_snapshot_configured_tools_uses_pi_builtins():
    config = Config(agent={"provider": "pi"})

    tools = snapshot_configured_tools(config)

    names = [tool["function"]["name"] for tool in tools]
    assert "bash" in names
    assert "read" in names
    assert "read_file" in names
    assert "write" in names
    assert "write_file" in names
    assert "edit" in names


def test_codex_builtin_tool_schema_matches_normalized_tool_calls():
    config = Config(agent={"provider": "codex"})
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "update_plan", "arguments": {"plan": []}},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": {"cmd": "pwd"}},
                    },
                ],
            }
        ],
        "tools": snapshot_configured_tools(config),
    }

    assert validate_tool_calls(row).ok is True


def test_pi_builtin_tool_schema_accepts_normalized_and_legacy_argument_names():
    config = Config(agent={"provider": "pi"})
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "pwd"}},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"cmd": "pwd"}},
                    },
                    {
                        "id": "call-3",
                        "type": "function",
                        "function": {"name": "write", "arguments": {"path": "demo.py", "content": "print(1)"}},
                    },
                    {
                        "id": "call-4",
                        "type": "function",
                        "function": {"name": "edit", "arguments": {"file_path": "demo.py", "edits": []}},
                    },
                ],
            }
        ],
        "tools": snapshot_configured_tools(config),
    }

    assert validate_tool_calls(row).ok is True


def test_snapshot_mcp_tools_normalizes_schema_and_applies_filters():
    mcp = MCPConfig(name="files", command="server", enabled_tools=["read"], disabled_tools=["write"])
    raw_tools = [
        {
            "type": "function",
            "function": {
                "name": "files.read",
                "description": "Read files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "files.write",
                "description": "Write files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
    ]

    with patch("teich.tool_schema._snapshot_stdio_mcp_tools", return_value=raw_tools):
        tools = snapshot_mcp_tools(mcp)

    assert [tool["function"]["name"] for tool in tools] == ["files.read"]


def test_validate_tool_calls_checks_declared_names_and_required_arguments():
    row = {
        "id": "row-1",
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"timeout_ms": 1000}},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "missing_tool", "arguments": {}},
                    },
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }

    report = validate_tool_calls(row)

    assert report.ok is False
    assert any("missing required argument 'command'" in error for error in report.errors)
    assert any("unexpected argument 'timeout_ms'" in error for error in report.errors)
    assert any("undeclared tool 'missing_tool'" in error for error in report.errors)


def test_validate_tool_calls_allows_null_for_optional_arguments():
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": {"command": "pwd", "timeout": None}},
                    },
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }

    assert validate_tool_calls(row).ok is True


def test_prepare_data_can_validate_tool_calls_before_rendering():
    dataset = Dataset.from_list(
        [
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
                                "function": {"name": "bash", "arguments": {}},
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "required": ["command"]},
                        },
                    }
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required argument 'command'"):
        prepare_data(dataset, TinyChatTokenizer(), validate_tools=True, verbose=False)
