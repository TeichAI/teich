from __future__ import annotations

import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .tool_schema import (
    CODEX_BUILTIN_TOOLS,
    CURSOR_BUILTIN_TOOLS,
    HERMES_BUILTIN_TOOLS,
    OPENCLAW_BUILTIN_TOOLS,
    PI_BUILTIN_TOOLS,
)

_INLINE_THINKING_BLOCK_PATTERN = re.compile(r"<(think|thinking)>(.*?)</\1>", re.DOTALL)
FIRST_MESSAGE_TIMESTAMP_METADATA_KEY = "first_message_timestamp"
PI_SYSTEM_PROMPT_CUSTOM_TYPE = "teich-system-prompt"
TEICH_AVAILABLE_TOOLS_CUSTOM_TYPE = "teich-available-tools"
TEICH_TRACE_CONTEXT_ITEM_TYPE = "teich_context"
TraceType = Literal["claude_code", "codex", "cursor", "droid", "external_agent", "hermes", "openclaw", "pi"]
_TIMESTAMP_KEYS = ("timestamp", "created_at", "createdAt")
_CURSOR_METADATA_KEYS = frozenset(
    {
        "cursor_headers",
        "cursor_key",
        "cursor_message_count",
        "cursor_row",
        "cursor_scope",
        "cursor_source_db",
        "cursor_storage_kind",
        "cursor_table",
        "cursor_workspace_id",
    }
)


def _object_parameters(
    properties: dict[str, Any] | None = None,
    required: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": True,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _function_schema(
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "description": description,
        "parameters": _object_parameters(properties, required),
    }


_CLAUDE_CODE_BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "Task": {
        "description": "Launch a subagent to complete a delegated task.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
                "subagent_type": {"type": "string"},
            },
            "required": ["prompt"],
            "additionalProperties": True,
        },
    },
    "Bash": {
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "timeout": {"type": "integer"},
                "run_in_background": {"type": "boolean"},
            },
            "required": ["command"],
            "additionalProperties": True,
        },
    },
    "BashOutput": {
        "description": "Read output from a running background shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "bash_id": {"type": "string"},
                "filter": {"type": "string"},
            },
            "required": ["bash_id"],
            "additionalProperties": True,
        },
    },
    "Edit": {
        "description": "Replace text in an existing file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": True,
        },
    },
    "Glob": {
        "description": "Find files by glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
            "additionalProperties": True,
        },
    },
    "Grep": {
        "description": "Search file contents by pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string"},
                "output_mode": {"type": "string"},
                "-A": {"type": "integer"},
                "-B": {"type": "integer"},
                "-C": {"type": "integer"},
                "head_limit": {"type": "integer"},
            },
            "required": ["pattern"],
            "additionalProperties": True,
        },
    },
    "KillBash": {
        "description": "Stop a running background shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "shell_id": {"type": "string"},
            },
            "required": ["shell_id"],
            "additionalProperties": True,
        },
    },
    "LS": {
        "description": "List files and directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "ignore": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path"],
            "additionalProperties": True,
        },
    },
    "MultiEdit": {
        "description": "Apply multiple text replacements to one file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["old_string", "new_string"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["file_path", "edits"],
            "additionalProperties": True,
        },
    },
    "NotebookEdit": {
        "description": "Edit a Jupyter notebook cell.",
        "parameters": {
            "type": "object",
            "properties": {
                "notebook_path": {"type": "string"},
                "cell_id": {"type": "string"},
                "new_source": {"type": "string"},
                "cell_type": {"type": "string"},
                "edit_mode": {"type": "string"},
            },
            "required": ["notebook_path", "new_source"],
            "additionalProperties": True,
        },
    },
    "NotebookRead": {
        "description": "Read a Jupyter notebook.",
        "parameters": {
            "type": "object",
            "properties": {
                "notebook_path": {"type": "string"},
            },
            "required": ["notebook_path"],
            "additionalProperties": True,
        },
    },
    "Read": {
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
            "additionalProperties": True,
        },
    },
    "TodoWrite": {
        "description": "Create or update the task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "priority": {"type": "string"},
                            "id": {"type": "string"},
                        },
                        "required": ["content", "status", "priority", "id"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["todos"],
            "additionalProperties": True,
        },
    },
    "WebFetch": {
        "description": "Fetch web content from a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["url", "prompt"],
            "additionalProperties": True,
        },
    },
    "WebSearch": {
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "allowed_domains": {"type": "array", "items": {"type": "string"}},
                "blocked_domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
    },
    "Write": {
        "description": "Write a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
            "additionalProperties": True,
        },
    },
    "PowerShell": _function_schema(
        "Run a PowerShell command.",
        {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "integer"},
            "run_in_background": {"type": "boolean"},
        },
        ("command",),
    ),
    "Skill": _function_schema(
        "Load and use a Claude Desktop skill.",
        {
            "skill": {"type": "string"},
            "args": {"type": "string"},
        },
        ("skill",),
    ),
    "ToolSearch": _function_schema(
        "Load deferred Claude Desktop or MCP tools by search query.",
        {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        ("query",),
    ),
    "TaskCreate": _function_schema(
        "Create a tracked task in the Claude Desktop task list.",
        {
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "activeForm": {"type": "string"},
        },
        ("subject", "description", "activeForm"),
    ),
    "TaskGet": _function_schema(
        "Fetch one tracked Claude Desktop task.",
        {"taskId": {"type": "string"}},
        ("taskId",),
    ),
    "TaskList": _function_schema(
        "List tracked Claude Desktop tasks.",
        {
            "status": {"type": "string"},
            "includeCompleted": {"type": "boolean"},
        },
    ),
    "TaskOutput": _function_schema(
        "Read output associated with a tracked Claude Desktop task.",
        {
            "taskId": {"type": "string"},
            "lines": {"type": "integer"},
        },
        ("taskId",),
    ),
    "TaskStop": _function_schema(
        "Stop a tracked Claude Desktop task.",
        {"taskId": {"type": "string"}},
        ("taskId",),
    ),
    "TaskUpdate": _function_schema(
        "Update the status of a tracked Claude Desktop task.",
        {
            "taskId": {"type": "string"},
            "status": {"type": "string"},
        },
        ("taskId", "status"),
    ),
    "CronCreate": _function_schema(
        "Create a scheduled Claude Desktop task.",
        {
            "name": {"type": "string"},
            "prompt": {"type": "string"},
            "schedule": {"type": "string"},
            "timezone": {"type": "string"},
        },
    ),
    "CronDelete": _function_schema(
        "Delete a scheduled Claude Desktop task.",
        {
            "cronId": {"type": "string"},
            "name": {"type": "string"},
        },
    ),
    "CronList": _function_schema("List scheduled Claude Desktop tasks."),
    "DesignSync": _function_schema(
        "Synchronize design context from Claude Desktop.",
        {"target": {"type": "string"}},
    ),
    "EnterPlanMode": _function_schema("Enter Claude Desktop plan mode.", {"prompt": {"type": "string"}}),
    "ExitPlanMode": _function_schema("Exit Claude Desktop plan mode.", {"plan": {"type": "string"}}),
    "EnterWorktree": _function_schema(
        "Enter a Claude Desktop worktree.",
        {
            "path": {"type": "string"},
            "branch": {"type": "string"},
        },
    ),
    "ExitWorktree": _function_schema("Exit the active Claude Desktop worktree."),
    "Monitor": _function_schema(
        "Monitor a long-running Claude Desktop task.",
        {
            "taskId": {"type": "string"},
            "interval": {"type": "string"},
        },
    ),
    "PushNotification": _function_schema(
        "Send a Claude Desktop push notification.",
        {
            "title": {"type": "string"},
            "message": {"type": "string"},
        },
    ),
    "RemoteTrigger": _function_schema(
        "Trigger a remote Claude Desktop action.",
        {
            "target": {"type": "string"},
            "prompt": {"type": "string"},
        },
    ),
    "mcp__Claude_Preview__preview_click": _function_schema(
        "Click an element in a Claude Preview browser session.",
        {
            "serverId": {"type": "string"},
            "selector": {"type": "string"},
        },
        ("serverId", "selector"),
    ),
    "mcp__Claude_Preview__preview_console_logs": _function_schema(
        "Read console logs from a Claude Preview browser session.",
        {
            "serverId": {"type": "string"},
            "level": {"type": "string"},
            "lines": {"type": "integer"},
        },
        ("serverId",),
    ),
    "mcp__Claude_Preview__preview_eval": _function_schema(
        "Evaluate JavaScript in a Claude Preview browser session.",
        {
            "serverId": {"type": "string"},
            "expression": {"type": "string"},
        },
        ("serverId", "expression"),
    ),
    "mcp__Claude_Preview__preview_resize": _function_schema(
        "Resize a Claude Preview browser session viewport.",
        {
            "serverId": {"type": "string"},
            "preset": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
        ("serverId",),
    ),
    "mcp__Claude_Preview__preview_screenshot": _function_schema(
        "Capture a screenshot from a Claude Preview browser session.",
        {"serverId": {"type": "string"}},
        ("serverId",),
    ),
    "mcp__Claude_Preview__preview_snapshot": _function_schema(
        "Capture a structured snapshot from a Claude Preview browser session.",
        {"serverId": {"type": "string"}},
        ("serverId",),
    ),
    "mcp__Claude_Preview__preview_start": _function_schema(
        "Start a Claude Preview browser session for an app.",
        {"name": {"type": "string"}},
        ("name",),
    ),
    "mcp__Claude_Preview__preview_stop": _function_schema(
        "Stop a Claude Preview browser session.",
        {"serverId": {"type": "string"}},
        ("serverId",),
    ),
    "mcp__huggingface__paper_search": _function_schema(
        "Search Hugging Face paper results.",
        {
            "query": {"type": "string"},
            "results_limit": {"type": "integer"},
            "concise_only": {"type": "boolean"},
        },
        ("query",),
    ),
}

_CLAUDE_CODE_DEFAULT_TOOL_NAMES = {
    "Task",
    "Bash",
    "BashOutput",
    "Edit",
    "Glob",
    "Grep",
    "KillBash",
    "LS",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
}

def _claude_deferred_tool_schema(name: str, *, include_parameters: bool = True) -> dict[str, Any]:
    schema = deepcopy(_CLAUDE_CODE_BUILTIN_TOOL_SCHEMAS.get(name) or {})
    if schema:
        return schema
    schema["description"] = _claude_deferred_tool_description(name)
    if include_parameters:
        schema["parameters"] = _object_parameters(_claude_deferred_tool_parameter_hints(name))
    return schema


def _claude_deferred_tool_description(name: str) -> str:
    if not name.startswith("mcp__"):
        return f"Call the Claude deferred tool {name}."
    parts = name.split("__", 2)
    if len(parts) != 3:
        return f"Call the Claude MCP tool {name}."
    _, server, tool = parts
    server_name = server.replace("_", " ").replace("-", " ").strip()
    tool_name = tool.replace("_", " ").replace("-", " ").strip()
    if server_name and tool_name:
        return f"Call the {server_name} MCP tool {tool_name}."
    return f"Call the Claude MCP tool {name}."


def _claude_deferred_tool_parameter_hints(name: str) -> dict[str, Any]:
    if not name.startswith("mcp__"):
        return {}
    parts = name.split("__", 2)
    if len(parts) != 3:
        return {}
    _, server, tool = parts
    server_key = server.lower()
    tool_key = tool.lower()

    if server_key == "claude_preview":
        return _claude_preview_parameter_hints(tool_key)
    if server_key == "claude_in_chrome":
        return _claude_chrome_parameter_hints(tool_key)
    if server_key == "computer-use":
        return _claude_computer_use_parameter_hints(tool_key)
    if server_key == "context7":
        return _claude_context7_parameter_hints(tool_key)
    if server_key == "fetch":
        return {"url": {"type": "string"}}
    if server_key == "huggingface":
        return _claude_huggingface_parameter_hints(tool_key)
    if server_key == "mcp-registry":
        return {
            "query": {"type": "string"},
            "server": {"type": "string"},
            "connector": {"type": "string"},
        }
    if server_key == "scheduled-tasks":
        return {
            "task_id": {"type": "string"},
            "title": {"type": "string"},
            "prompt": {"type": "string"},
            "schedule": {"type": "string"},
            "timezone": {"type": "string"},
        }
    if server_key == "searxng":
        return {"query": {"type": "string"}}
    if server_key == "ccd_directory":
        return {
            "path": {"type": "string"},
            "purpose": {"type": "string"},
        }
    if server_key == "ccd_session_mgmt":
        return {
            "session_id": {"type": "string"},
            "query": {"type": "string"},
            "message": {"type": "string"},
        }
    if server_key.startswith("claude_ai_"):
        return {
            "code": {"type": "string"},
            "redirect_uri": {"type": "string"},
        }
    return {}


def _claude_preview_parameter_hints(tool_key: str) -> dict[str, Any]:
    common = {"serverId": {"type": "string"}}
    if tool_key == "preview_start":
        return {"name": {"type": "string"}, "url": {"type": "string"}}
    if tool_key in {"preview_click", "preview_inspect"}:
        return {**common, "selector": {"type": "string"}}
    if tool_key == "preview_fill":
        return {
            **common,
            "selector": {"type": "string"},
            "value": {"type": "string"},
            "text": {"type": "string"},
        }
    if tool_key == "preview_eval":
        return {**common, "expression": {"type": "string"}}
    if tool_key == "preview_resize":
        return {
            **common,
            "preset": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        }
    if tool_key in {"preview_console_logs", "preview_logs"}:
        return {
            **common,
            "level": {"type": "string"},
            "lines": {"type": "integer"},
        }
    if tool_key == "preview_network":
        return {
            **common,
            "url": {"type": "string"},
            "method": {"type": "string"},
            "status": {"type": "integer"},
            "lines": {"type": "integer"},
        }
    if tool_key in {"preview_list", "preview_screenshot", "preview_snapshot", "preview_stop"}:
        return common
    return common


def _claude_chrome_parameter_hints(tool_key: str) -> dict[str, Any]:
    common = {
        "browserId": {"type": "string"},
        "tabId": {"type": "string"},
        "selector": {"type": "string"},
    }
    if tool_key in {"navigate", "tabs_create_mcp"}:
        return {**common, "url": {"type": "string"}}
    if tool_key in {"find", "get_page_text", "read_page"}:
        return {**common, "query": {"type": "string"}}
    if tool_key == "form_input":
        return {**common, "value": {"type": "string"}, "text": {"type": "string"}}
    if tool_key == "javascript_tool":
        return {**common, "code": {"type": "string"}, "expression": {"type": "string"}}
    if tool_key in {"read_console_messages", "read_network_requests"}:
        return {**common, "level": {"type": "string"}, "lines": {"type": "integer"}}
    if tool_key == "resize_window":
        return {**common, "width": {"type": "integer"}, "height": {"type": "integer"}}
    if tool_key in {"file_upload", "upload_image"}:
        return {**common, "path": {"type": "string"}, "file_path": {"type": "string"}}
    if tool_key == "shortcuts_execute":
        return {**common, "shortcut": {"type": "string"}, "name": {"type": "string"}}
    if tool_key == "browser_batch":
        return {"actions": {"type": "array", "items": {"type": "object", "additionalProperties": True}}}
    if tool_key in {"computer", "gif_creator"}:
        return {"action": {"type": "string"}, "input": {"type": "object", "additionalProperties": True}}
    return common


def _claude_computer_use_parameter_hints(tool_key: str) -> dict[str, Any]:
    pointer = {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "application": {"type": "string"},
        "display": {"type": "string"},
    }
    if tool_key in {
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "mouse_move",
        "left_mouse_down",
        "left_mouse_up",
    }:
        return pointer
    if tool_key == "left_click_drag":
        return {
            **pointer,
            "start_x": {"type": "integer"},
            "start_y": {"type": "integer"},
            "end_x": {"type": "integer"},
            "end_y": {"type": "integer"},
        }
    if tool_key in {"key", "hold_key"}:
        return {"key": {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}}}
    if tool_key == "type":
        return {"text": {"type": "string"}}
    if tool_key == "scroll":
        return {**pointer, "direction": {"type": "string"}, "amount": {"type": "integer"}}
    if tool_key == "open_application":
        return {"application": {"type": "string"}}
    if tool_key in {"request_access", "request_teach_access"}:
        return {
            "applications": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        }
    if tool_key in {"write_clipboard", "teach_step"}:
        return {"text": {"type": "string"}, "content": {"type": "string"}}
    if tool_key == "wait":
        return {"seconds": {"type": "number"}}
    if tool_key == "zoom":
        return {"level": {"type": "number"}, "direction": {"type": "string"}}
    if tool_key in {"computer_batch", "teach_batch"}:
        return {"actions": {"type": "array", "items": {"type": "object", "additionalProperties": True}}}
    return {
        "application": {"type": "string"},
        "display": {"type": "string"},
    }


def _claude_context7_parameter_hints(tool_key: str) -> dict[str, Any]:
    if tool_key == "resolve-library-id":
        return {"libraryName": {"type": "string"}}
    return {
        "context7CompatibleLibraryID": {"type": "string"},
        "topic": {"type": "string"},
        "tokens": {"type": "integer"},
    }


def _claude_huggingface_parameter_hints(tool_key: str) -> dict[str, Any]:
    if tool_key == "hf_whoami":
        return {}
    if "generate" in tool_key:
        return {
            "prompt": {"type": "string"},
            "model": {"type": "string"},
            "parameters": {"type": "object", "additionalProperties": True},
        }
    if "details" in tool_key:
        return {"repo_id": {"type": "string"}, "repo_type": {"type": "string"}}
    if "doc_fetch" in tool_key:
        return {"path": {"type": "string"}, "url": {"type": "string"}}
    return {
        "query": {"type": "string"},
        "results_limit": {"type": "integer"},
        "limit": {"type": "integer"},
        "concise_only": {"type": "boolean"},
    }


_DROID_BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "Read": {
        "description": "Read a file from the filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "number"},
                "limit": {"type": "number"},
                "image_quality": {"type": "string"},
            },
            "required": ["file_path"],
            "additionalProperties": True,
        },
    },
    "Edit": {
        "description": "Replace a string in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["file_path", "old_str", "new_str"],
            "additionalProperties": True,
        },
    },
    "Create": {
        "description": "Create a new file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
            "additionalProperties": True,
        },
    },
    "Execute": {
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "summary": {"type": "string"},
                "riskLevel": {"type": "string"},
                "riskLevelReason": {"type": "string"},
                "timeout": {"type": "number"},
                "fireAndForget": {"type": "boolean"},
            },
            "required": ["command"],
            "additionalProperties": True,
        },
    },
    "LS": {
        "description": "List the contents of a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {"type": "string"},
            },
            "required": ["directory_path"],
            "additionalProperties": True,
        },
    },
    "Glob": {
        "description": "Find files matching glob patterns.",
        "parameters": {
            "type": "object",
            "properties": {
                "patterns": {"type": "array", "items": {"type": "string"}},
                "folder": {"type": "string"},
            },
            "required": ["patterns"],
            "additionalProperties": True,
        },
    },
    "Grep": {
        "description": "Search file contents for a pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "output_mode": {"type": "string"},
                "case_insensitive": {"type": "boolean"},
                "fixed_string": {"type": "boolean"},
                "head_limit": {"type": "number"},
            },
            "required": ["pattern"],
            "additionalProperties": True,
        },
    },
    "TodoWrite": {
        "description": "Update the task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {"type": "string"},
            },
            "required": ["todos"],
            "additionalProperties": True,
        },
    },
    "FetchUrl": {
        "description": "Fetch the contents of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
            "additionalProperties": True,
        },
    },
    "Skill": {
        "description": "Invoke a skill.",
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
            },
            "required": ["skill"],
            "additionalProperties": True,
        },
    },
}


@dataclass(slots=True)
class TrainingExample:
    source_file: Path
    prompt: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "tools": self.tools,
            "metadata": self.metadata,
        }


def _normalize_timestamp_value(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = float(value)
    if timestamp > 10_000_000_000:
        timestamp /= 1000
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _timestamp_from_mapping(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in _TIMESTAMP_KEYS:
        timestamp = _normalize_timestamp_value(value.get(key))
        if timestamp:
            return timestamp
    return None


def _event_timestamp(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    for key in ("payload", "message"):
        timestamp = _timestamp_from_mapping(event.get(key))
        if timestamp:
            return timestamp
    return _timestamp_from_mapping(event)


def _first_message_timestamp_from_events(events: list[Any], predicate: Callable[[dict[str, Any]], bool]) -> str | None:
    for event in events:
        if isinstance(event, dict) and predicate(event):
            timestamp = _event_timestamp(event)
            if timestamp:
                return timestamp
    return None


def _add_first_message_timestamp(metadata: dict[str, Any], timestamp: str | None) -> None:
    if timestamp:
        metadata[FIRST_MESSAGE_TIMESTAMP_METADATA_KEY] = timestamp


def _is_user_role(role: Any) -> bool:
    return isinstance(role, str) and _normalize_role(role) == "user"


def _is_external_user_message(event: dict[str, Any]) -> bool:
    return event.get("type") == "external_message" and _is_user_role(event.get("role"))


def _is_nested_user_message(event: dict[str, Any]) -> bool:
    message = event.get("message")
    return isinstance(message, dict) and _is_user_role(message.get("role"))


def _is_claude_code_user_message(event: dict[str, Any]) -> bool:
    if _claude_queued_command_content(event) is not None:
        return True
    if event.get("type") != "user" or not _is_nested_user_message(event):
        return False
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    ):
        return False
    return _claude_training_user_content(event) is not None


def _is_droid_user_message(event: dict[str, Any]) -> bool:
    if event.get("type") != "message":
        return False
    message = event.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return not (
        isinstance(content, list)
        and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    )


def _first_text_block(content_blocks: Any) -> str:
    if isinstance(content_blocks, str):
        return _unwrap_teich_prompt_file(content_blocks).strip()
    if not isinstance(content_blocks, list):
        return ""
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return _unwrap_teich_prompt_file("\n".join(parts)).strip()


def _unwrap_teich_prompt_file(prompt: str) -> str:
    normalized = prompt.strip()
    match = re.fullmatch(
        r'<file\s+name=["\'][^"\']*\.teich-prompt\.txt["\']>\s*(?P<prompt>.*?)\s*</file>',
        normalized,
        flags=re.DOTALL,
    )
    if match:
        return match.group("prompt")
    return prompt


def _split_inline_thinking(text: str) -> tuple[str, str | None]:
    reasoning_parts: list[str] = []

    def _remove_match(match: re.Match[str]) -> str:
        reasoning = match.group(2).strip()
        if reasoning:
            reasoning_parts.append(reasoning)
        return ""

    content = _INLINE_THINKING_BLOCK_PATTERN.sub(_remove_match, text).strip()
    reasoning = "\n\n".join(reasoning_parts).strip()
    return content, reasoning or None


def _message_content_and_inline_reasoning(value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        return _split_inline_thinking(value)
    text = _first_text_block(value)
    reasoning = _pi_reasoning_content(value)
    return text, reasoning


def _has_same_system_message(messages: list[dict[str, Any]], content: str) -> bool:
    return any(
        message.get("role") == "system" and message.get("content") == content
        for message in messages
    )


def _pi_reasoning_content(content_blocks: Any) -> str | None:
    if not isinstance(content_blocks, list):
        return None
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "thinking":
            continue
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            parts.append(thinking.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _tool_result_content_text(payload: dict[str, Any]) -> str:
    return _first_text_block(payload.get("content"))


def _function_call_output_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, str):
        return output
    text = _first_text_block(output)
    if text:
        return text
    return str(output or "")


def _is_tool_not_found_result(tool_name: str | None, payload: dict[str, Any]) -> bool:
    content = _tool_result_content_text(payload).strip()
    if tool_name:
        return content == f"Tool {tool_name} not found"
    return content == "Tool  not found"


def _pi_teich_system_prompt_from_event(event: dict[str, Any]) -> str | None:
    if event.get("type") != "custom" or event.get("customType") != PI_SYSTEM_PROMPT_CUSTOM_TYPE:
        return None
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    system_prompt = data.get("systemPrompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return None
    return system_prompt.strip()


def _reasoning_summary(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary")
    parts: list[str] = []
    if isinstance(summary, list):
        for item in summary:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    result = "\n\n".join(parts).strip()
    if result:
        return result

    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _reasoning_summary_blocks_from_content(payload: dict[str, Any]) -> list[dict[str, str]]:
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    summary: list[dict[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            summary.append({"type": "summary_text", "text": text.strip()})
    return summary


def _custom_tool_arguments(name: str, value: Any) -> dict[str, Any]:
    if name == "apply_patch":
        return {"patch": value if isinstance(value, str) else _normalize_json_like_value(value)}
    return {"input": _normalize_json_like_value(value)}


def _custom_tool_output_value(value: Any) -> Any:
    if not isinstance(value, str):
        return _normalize_json_like_value(value)
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict) and isinstance(parsed.get("output"), str):
        return parsed["output"]
    return _normalize_json_like_value(parsed)


def normalize_codex_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict) or event.get("type") != "response_item":
        return event

    payload_type = payload.get("type")
    if payload_type == "reasoning":
        summary = payload.get("summary")
        if isinstance(summary, list) and any(
            isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
            for item in summary
        ):
            return event
        normalized_payload = dict(payload)
        normalized_payload["summary"] = _reasoning_summary_blocks_from_content(normalized_payload)
        normalized_event = dict(event)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    if payload_type == "custom_tool_call":
        name = payload.get("name")
        call_id = payload.get("call_id")
        if not isinstance(name, str) or not isinstance(call_id, str):
            return event
        normalized_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "input"}
        }
        normalized_payload.update(
            {
                "type": "function_call",
                "name": name,
                "call_id": call_id,
                "arguments": json.dumps(_custom_tool_arguments(name, payload.get("input")), ensure_ascii=False),
            }
        )
        normalized_event = dict(event)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    if payload_type == "custom_tool_call_output":
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            return event
        normalized_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "output"}
        }
        normalized_payload.update(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _custom_tool_output_value(payload.get("output")),
            }
        )
        normalized_event = dict(event)
        normalized_event["payload"] = normalized_payload
        return normalized_event

    return event


def _codex_payload_type(event: dict[str, Any]) -> str | None:
    if event.get("type") != "response_item":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_type = payload.get("type")
    return payload_type if isinstance(payload_type, str) else None


def _codex_reasoning_text(event: dict[str, Any]) -> str | None:
    if _codex_payload_type(event) != "reasoning":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    return _reasoning_summary(payload)


def _codex_assistant_message_has_text(event: dict[str, Any]) -> bool:
    if _codex_payload_type(event) != "message":
        return False
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("role") != "assistant":
        return False
    return bool(_first_text_block(payload.get("content")).strip())


def _codex_assistant_message_event(event: dict[str, Any]) -> bool:
    if _codex_payload_type(event) != "message":
        return False
    payload = event.get("payload")
    return isinstance(payload, dict) and payload.get("role") == "assistant"


def _events_with_rotated_timestamps(
    first_event: dict[str, Any],
    following_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _events_with_reassigned_timestamps(
        [first_event, *following_events],
        [*following_events, first_event],
    )


def _events_with_reassigned_timestamps(
    events: list[dict[str, Any]],
    timestamp_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_events = [dict(event) for event in events]
    for event_copy, timestamp_source in zip(ordered_events, timestamp_sources, strict=True):
        for key in ("timestamp", "created_at", "createdAt"):
            timestamp = timestamp_source.get(key)
            if timestamp is not None:
                event_copy[key] = timestamp
            elif key in event_copy:
                event_copy.pop(key, None)
    return ordered_events


def _codex_output_fragment(event: dict[str, Any]) -> bool:
    return _codex_payload_type(event) in {"function_call", "custom_tool_call"} or _codex_assistant_message_has_text(event)


def _codex_tool_call_fragment(event: dict[str, Any]) -> bool:
    return _codex_payload_type(event) in {"function_call", "custom_tool_call"}


def _codex_event_msg_payload_type(event: dict[str, Any]) -> str | None:
    if event.get("type") != "event_msg":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_type = payload.get("type")
    return payload_type if isinstance(payload_type, str) else None


def _codex_first_user_message_timestamp(events: list[Any]) -> str | None:
    timestamp = _first_message_timestamp_from_events(
        events,
        lambda event: _codex_event_msg_payload_type(event) == "user_message",
    )
    if timestamp:
        return timestamp
    return _first_message_timestamp_from_events(
        events,
        lambda event: _codex_payload_type(event) == "message"
        and isinstance(event.get("payload"), dict)
        and _is_user_role(event["payload"].get("role")),
    )


def _codex_assistant_prefix_marker(event: dict[str, Any]) -> bool:
    return _codex_event_msg_payload_type(event) in {"agent_reasoning", "agent_message"}


def _codex_reorderable_output_suffix_event(event: dict[str, Any], *, prefix_has_reasoning: bool) -> bool:
    if _codex_tool_call_fragment(event):
        return True
    if not prefix_has_reasoning:
        return False
    return _codex_assistant_message_event(event) or _codex_event_msg_payload_type(event) == "agent_message"


def _codex_assistant_prefix_block(
    events: list[dict[str, Any]],
    start: int,
) -> tuple[list[dict[str, Any]], int]:
    block: list[dict[str, Any]] = []
    saw_assistant_payload = False
    saw_message = False
    index = start
    while index < len(events):
        event = events[index]
        if _codex_assistant_prefix_marker(event):
            block.append(event)
            index += 1
            continue

        payload_type = _codex_payload_type(event)
        if payload_type == "reasoning":
            if saw_message:
                break
            block.append(event)
            saw_assistant_payload = True
            index += 1
            continue
        if _codex_assistant_message_event(event):
            block.append(event)
            saw_assistant_payload = True
            saw_message = True
            index += 1
            continue
        break

    if not saw_assistant_payload:
        return [], start
    return block, index


def normalize_codex_trace_events(
    events: list[dict[str, Any]],
    *,
    normalize_custom_tools: bool = True,
) -> list[dict[str, Any]]:
    if normalize_custom_tools:
        normalized_events = [normalize_codex_trace_event(event) for event in events]
    else:
        normalized_events = [
            normalize_codex_trace_event(event) if _codex_payload_type(event) == "reasoning" else event
            for event in events
        ]
    ordered_events: list[dict[str, Any]] = []
    index = 0
    while index < len(normalized_events):
        prefix_block, next_index = _codex_assistant_prefix_block(normalized_events, index)
        if prefix_block:
            prefix_has_reasoning = any(_codex_payload_type(event) == "reasoning" for event in prefix_block)
            output_suffix: list[dict[str, Any]] = []
            while ordered_events and _codex_reorderable_output_suffix_event(
                ordered_events[-1],
                prefix_has_reasoning=prefix_has_reasoning,
            ):
                output_suffix.insert(0, ordered_events.pop())
            if output_suffix:
                ordered_events.extend(
                    _events_with_reassigned_timestamps(
                        [*prefix_block, *output_suffix],
                        [*output_suffix, *prefix_block],
                    )
                )
            else:
                ordered_events.extend(prefix_block)
            index = next_index
            continue

        event = normalized_events[index]
        if _codex_reasoning_text(event):
            output_suffix = []
            while ordered_events and _codex_output_fragment(ordered_events[-1]):
                output_suffix.insert(0, ordered_events.pop())
            if output_suffix:
                ordered_events.extend(_events_with_rotated_timestamps(event, output_suffix))
                index += 1
                continue
        ordered_events.append(event)
        index += 1
    return ordered_events


def _normalize_json_like_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_like_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_like_value(item) for item in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return _normalize_json_like_value(parsed)


def _parse_function_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return _normalize_json_like_value(arguments) if arguments is not None else {}
    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        return _normalize_json_like_value(json.loads(stripped))
    except json.JSONDecodeError:
        return arguments


def _schema_identity(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)


def _infer_schema_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schemas = [_infer_schema_from_value(item) for item in value]
        schema: dict[str, Any] = {"type": "array"}
        if item_schemas:
            schema["items"] = _merge_schemas(item_schemas)
        return schema
    if isinstance(value, dict):
        return _infer_tool_parameters_schema([value])
    return {}


def _merge_object_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    properties_by_name: dict[str, list[dict[str, Any]]] = {}
    required_sets: list[set[str]] = []
    additional_properties = False
    for schema in schemas:
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, value in properties.items():
                if isinstance(value, dict):
                    properties_by_name.setdefault(name, []).append(value)
        required = schema.get("required")
        if isinstance(required, list):
            required_sets.append({item for item in required if isinstance(item, str)})
        else:
            required_sets.append(set())
        if schema.get("additionalProperties", True) is not False:
            additional_properties = True
    merged: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: _merge_schemas(property_schemas)
            for name, property_schemas in sorted(properties_by_name.items())
        },
        "additionalProperties": additional_properties,
    }
    if required_sets:
        required = sorted(set.intersection(*required_sets))
        if required:
            merged["required"] = required
    return merged


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema in schemas:
        if not schema:
            continue
        identity = _schema_identity(schema)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(schema)
    if not unique:
        return {}
    if len(unique) == 1:
        return unique[0]
    schema_types = {schema.get("type") for schema in unique if isinstance(schema.get("type"), str)}
    if schema_types == {"object"}:
        return _merge_object_schemas(unique)
    if schema_types == {"array"}:
        item_schemas = [schema.get("items") for schema in unique if isinstance(schema.get("items"), dict)]
        merged: dict[str, Any] = {"type": "array"}
        if item_schemas:
            merged["items"] = _merge_schemas(item_schemas)
        return merged
    return {"anyOf": unique}


def _infer_tool_parameters_schema(argument_samples: list[Any]) -> dict[str, Any]:
    dict_samples = [sample for sample in argument_samples if isinstance(sample, dict)]
    if not dict_samples:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    properties: dict[str, dict[str, Any]] = {}
    all_keys = sorted({key for sample in dict_samples for key in sample})
    for key in all_keys:
        observed = [_infer_schema_from_value(sample[key]) for sample in dict_samples if key in sample]
        properties[key] = _merge_schemas(observed)
    required = sorted(set.intersection(*(set(sample.keys()) for sample in dict_samples))) if dict_samples else []
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    return schema


def _parse_tool_descriptions_from_text(text: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_section:
            if line == "Available tools:":
                in_section = True
            continue
        if not line:
            if descriptions:
                break
            continue
        if not line.startswith("- "):
            if descriptions:
                break
            continue
        name, separator, description = line[2:].partition(":")
        tool_name = name.strip()
        tool_description = description.strip()
        if separator and tool_name and tool_description:
            descriptions[tool_name] = tool_description
    return descriptions


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    if role == "model":
        return "assistant"
    return role


def _build_tool_entry(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name}
    if isinstance(schema, dict) and schema:
        function.update(schema)
    return {"type": "function", "function": function}


def _tool_entries_from_schema_map(
    schemas: dict[str, dict[str, Any]],
    names: set[str] | None = None,
) -> list[dict[str, Any]]:
    return [_build_tool_entry(name, deepcopy(schemas.get(name) or {})) for name in sorted(names or set(schemas))]


def _tool_entry_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function") if isinstance(tool, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _tool_entry_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return {}
    return {key: deepcopy(value) for key, value in function.items() if key != "name"}


def _normalize_explicit_tool_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("type") == "function":
        function = value.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                normalized = deepcopy(value)
                normalized["type"] = "function"
                normalized["function"] = dict(normalized["function"])
                normalized["function"]["name"] = name.strip()
                return normalized
    name = value.get("name")
    if isinstance(name, str) and name.strip():
        schema = {key: deepcopy(val) for key, val in value.items() if key != "name"}
        return _build_tool_entry(name.strip(), schema)
    return None


def _add_explicit_tools(
    raw_tools: Any,
    explicit_tools: dict[str, dict[str, Any]],
    tool_names: set[str],
    tool_schemas: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(raw_tools, list):
        return
    for raw_tool in raw_tools:
        tool = _normalize_explicit_tool_entry(raw_tool)
        if tool is None:
            continue
        name = _tool_entry_name(tool)
        if name is None:
            continue
        explicit_tools[name] = tool
        tool_names.add(name)
        schema = _tool_entry_schema(tool)
        if schema:
            tool_schemas[name] = {**tool_schemas.get(name, {}), **schema}


def _normalize_trace_type_name(trace_type: Any) -> str:
    return trace_type.strip().lower().replace("-", "_") if isinstance(trace_type, str) else ""


def _builtin_tools_for_trace_type(trace_type: Any) -> list[dict[str, Any]]:
    normalized = _normalize_trace_type_name(trace_type)
    if normalized == "codex":
        return CODEX_BUILTIN_TOOLS
    if normalized == "cursor":
        return CURSOR_BUILTIN_TOOLS
    if normalized == "droid":
        return _tool_entries_from_schema_map(_DROID_BUILTIN_TOOL_SCHEMAS)
    if normalized == "claude_code":
        return _tool_entries_from_schema_map(_CLAUDE_CODE_BUILTIN_TOOL_SCHEMAS, _CLAUDE_CODE_DEFAULT_TOOL_NAMES)
    if normalized in {"hermes", "hermes_agent"}:
        return HERMES_BUILTIN_TOOLS
    if normalized == "pi":
        return PI_BUILTIN_TOOLS
    if normalized == "openclaw":
        return OPENCLAW_BUILTIN_TOOLS
    return []


def _add_builtin_tools_for_trace_type(
    trace_type: Any,
    _explicit_tools: dict[str, dict[str, Any]],
    tool_names: set[str],
    tool_schemas: dict[str, dict[str, Any]],
) -> None:
    for tool in _builtin_tools_for_trace_type(trace_type):
        name = _tool_entry_name(tool)
        if name is None:
            continue
        tool_names.add(name)
        schema = _tool_entry_schema(tool)
        if schema:
            tool_schemas[name] = {**schema, **tool_schemas.get(name, {})}


def _merge_tools_with_builtin_trace_tools(raw_tools: Any, trace_type: Any) -> list[dict[str, Any]]:
    row_tools = raw_tools if isinstance(raw_tools, list) else []
    if not _builtin_tools_for_trace_type(trace_type):
        return row_tools
    explicit_tools: dict[str, dict[str, Any]] = {}
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    _add_builtin_tools_for_trace_type(trace_type, explicit_tools, tool_names, tool_schemas)
    _add_explicit_tools(row_tools, explicit_tools, tool_names, tool_schemas)
    return _build_tools_from_snapshots_and_calls(tool_names, tool_schemas, {}, explicit_tools)


def _build_tools_from_snapshots_and_calls(
    tool_names: set[str],
    tool_schemas: dict[str, dict[str, Any]],
    tool_argument_samples: dict[str, list[Any]],
    explicit_tools: dict[str, dict[str, Any]] | None = None,
    tool_descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    explicit_tools = explicit_tools or {}
    tool_descriptions = tool_descriptions or {}
    for name in sorted(tool_names):
        fallback_schema = deepcopy(tool_schemas.get(name) or {})
        explicit_schema = _tool_entry_schema(explicit_tools[name]) if name in explicit_tools else {}
        schema = {**fallback_schema, **explicit_schema}
        if name in tool_descriptions and "description" not in explicit_schema:
            schema["description"] = tool_descriptions[name]
        if "parameters" not in schema:
            schema["parameters"] = _infer_tool_parameters_schema(tool_argument_samples.get(name, []))
        tools.append(_build_tool_entry(name, schema))
    return tools


def _claude_code_tool_schema_from_definition(tool: dict[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    description = tool.get("description")
    if isinstance(description, str) and description.strip():
        schema["description"] = description.strip()

    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        parameters = tool.get("input_schema")
    if not isinstance(parameters, dict):
        parameters = tool.get("inputSchema")
    if not isinstance(parameters, dict):
        raw_schema = tool.get("schema")
        if isinstance(raw_schema, dict):
            parameters = raw_schema.get("parameters") if isinstance(raw_schema.get("parameters"), dict) else raw_schema
    if not isinstance(parameters, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            parameters = function.get("parameters")
    if isinstance(parameters, dict):
        schema["parameters"] = parameters
    return schema


def _is_openclaw_session_header(event: Any) -> bool:
    if not isinstance(event, dict) or event.get("type") != "session":
        return False
    cwd = event.get("cwd")
    return isinstance(cwd, str) and ".openclaw" in cwd


def _external_session_source(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    source = payload.get("source") if isinstance(payload, dict) else None
    return source.strip().lower() if isinstance(source, str) else ""


def _is_cursor_trace_row(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    if event.get("type") in {"cursor_session_meta", "cursor_available_tools"}:
        return True
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        return False
    trace_type = metadata.get("trace_type")
    if isinstance(trace_type, str) and trace_type.strip().lower().replace("-", "_") == "cursor":
        return True
    if any(key in metadata for key in _CURSOR_METADATA_KEYS):
        return True
    return isinstance(event.get("raw_cursor"), dict)


def _detect_trace_type(events: list[Any], default: TraceType | None = "codex") -> TraceType | None:
    first_event = next((event for event in events if isinstance(event, dict)), None)
    if _is_openclaw_session_header(first_event):
        return "openclaw"
    if any(_is_openclaw_session_header(event) for event in events):
        return "openclaw"

    for event in events:
        if _is_hermes_conversation(event):
            return "hermes"
        if not isinstance(event, dict):
            continue
        if _is_openclaw_session_header(event):
            return "openclaw"
        if _is_hermes_trace_row(event):
            return "hermes"
        if _is_cursor_trace_row(event):
            return "cursor"
        event_type = event.get("type")
        if event_type == "external_session_meta":
            source = _external_session_source(event)
            if source in {"claude", "claude-code", "claude_code"}:
                return "claude_code"
            if source in {"hermes", "hermes-agent", "hermes_agent"}:
                return "hermes"
            return "external_agent"
        if event_type == "hermes_session_meta":
            return "hermes"
        if _is_hermes_export_session(event):
            return "hermes"
        if isinstance(event.get("role"), str) and event_type is None:
            return "hermes"
        if event_type in {"assistant", "user", "system", "result"} and (
            isinstance(event.get("session_id"), str) or isinstance(event.get("sessionId"), str)
        ):
            return "claude_code"
        if event_type in {"session_meta", "turn_context", "response_item", "event_msg"}:
            return "codex"
        if event_type == "session_start" and isinstance(event.get("cwd"), str) and event.get("version") is not None:
            return "droid"
        if event_type in {
            "session",
            "message",
            "session_info",
            "model_change",
            "thinking_level_change",
            "compaction",
            "branch_summary",
            "custom",
            "custom_message",
            "label",
        }:
            return "pi"
    return default


def detect_trace_type(events: list[Any]) -> TraceType | None:
    """Return the supported raw trace type for parsed JSON events, or None."""
    return _detect_trace_type(events, default=None)


def _claude_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return _unwrap_teich_prompt_file(content).strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif block_type == "tool_result":
            text = block.get("content")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return _unwrap_teich_prompt_file("\n".join(parts)).strip()


def _claude_reasoning_from_content(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"thinking", "reasoning", "reasoning_text"}:
            text = block.get("thinking")
            if not isinstance(text, str) or not text.strip():
                text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    result = "\n\n".join(parts).strip()
    return result or None


def _claude_assistant_content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event.get("type") != "assistant":
        return []
    payload = event.get("message")
    if not isinstance(payload, dict):
        return []
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _claude_message_id(event: dict[str, Any]) -> str | None:
    if event.get("type") != "assistant":
        return None
    payload = event.get("message")
    if not isinstance(payload, dict):
        return None
    message_id = payload.get("id")
    return message_id if isinstance(message_id, str) and message_id else None


def _claude_assistant_event_has_reasoning(event: dict[str, Any]) -> bool:
    blocks = _claude_assistant_content_blocks(event)
    if _claude_reasoning_from_content(blocks):
        return True
    return any(block.get("type") == "redacted_thinking" for block in blocks)


def _claude_assistant_event_has_tool_use(event: dict[str, Any]) -> bool:
    for block in _claude_assistant_content_blocks(event):
        if block.get("type") == "tool_use":
            return True
    return False


def _claude_assistant_event_has_text_output(event: dict[str, Any]) -> bool:
    for block in _claude_assistant_content_blocks(event):
        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return True
    return False


def _same_claude_session(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_session = left.get("session_id") or left.get("sessionId")
    right_session = right.get("session_id") or right.get("sessionId")
    if isinstance(left_session, str) and isinstance(right_session, str):
        return left_session == right_session
    return True


def _claude_emptyish_text(text: Any) -> bool:
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    return not stripped or stripped in {".", "..", "...", "…"}


def _drop_claude_assistant_fragment(event: dict[str, Any]) -> bool:
    if event.get("type") != "assistant":
        return False
    if _claude_assistant_event_has_reasoning(event) or _claude_assistant_event_has_tool_use(event):
        return False

    blocks = _claude_assistant_content_blocks(event)
    if not blocks:
        return True
    text_types = {"text", "input_text", "output_text"}
    return all(block.get("type") in text_types and _claude_emptyish_text(block.get("text")) for block in blocks)


def _claude_assistant_fragment_rank(event: dict[str, Any]) -> int:
    if _claude_assistant_event_has_reasoning(event):
        return 0
    if _claude_assistant_event_has_text_output(event):
        return 1
    if _claude_assistant_event_has_tool_use(event):
        return 2
    return 3


def _normalize_claude_assistant_group(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept_events = [event for event in events if not _drop_claude_assistant_fragment(event)]
    if len(kept_events) <= 1:
        return kept_events

    indexed_events = list(enumerate(kept_events))
    indexed_events.sort(key=lambda item: (_claude_assistant_fragment_rank(item[1]), item[0]))
    ordered_events = [event for _, event in indexed_events]
    if ordered_events == kept_events:
        return kept_events
    return _events_with_reassigned_timestamps(ordered_events, kept_events)


def normalize_claude_code_trace_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_events: list[dict[str, Any]] = []
    index = 0
    while index < len(events):
        event = events[index]
        if not isinstance(event, dict):
            ordered_events.append(event)
            index += 1
            continue
        if event.get("type") == "assistant":
            message_id = _claude_message_id(event)
            group = [event]
            index += 1
            while index < len(events):
                next_event = events[index]
                if not isinstance(next_event, dict) or next_event.get("type") != "assistant":
                    break
                if not _same_claude_session(event, next_event):
                    break
                next_message_id = _claude_message_id(next_event)
                if (message_id is not None or next_message_id is not None) and next_message_id != message_id:
                    break
                group.append(next_event)
                index += 1
            ordered_events.extend(_normalize_claude_assistant_group(group))
            continue
        ordered_events.append(event)
        index += 1
    return ordered_events


def _append_or_merge_assistant_message(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    if not messages or messages[-1].get("role") != "assistant":
        messages.append(message)
        return

    existing = messages[-1]
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        existing_reasoning = existing.get("reasoning_content")
        if isinstance(existing_reasoning, str) and existing_reasoning.strip():
            existing["reasoning_content"] = f"{existing_reasoning.strip()}\n\n{reasoning.strip()}"
        else:
            existing["reasoning_content"] = reasoning.strip()

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        existing_content = existing.get("content")
        if isinstance(existing_content, str) and existing_content.strip():
            existing["content"] = f"{existing_content.strip()}\n\n{content.strip()}"
        else:
            existing["content"] = content.strip()

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        existing_tool_calls = existing.setdefault("tool_calls", [])
        if isinstance(existing_tool_calls, list):
            existing_tool_calls.extend(tool_calls)
        else:
            existing["tool_calls"] = tool_calls


def _claude_tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    return _first_text_block(content)


def _claude_local_command_parts(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    name_match = re.search(r"<command-name>\s*(?P<name>.*?)\s*</command-name>", stripped, flags=re.DOTALL)
    args_match = re.search(r"<command-args>\s*(?P<args>.*?)\s*</command-args>", stripped, flags=re.DOTALL)
    if not name_match or not args_match:
        return None
    name = re.sub(r"\s+", " ", name_match.group("name")).strip()
    args = args_match.group("args").strip()
    return name, args


def _is_claude_local_command_artifact(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(
        (
            "<local-command-caveat>",
            "<local-command-stdout>",
            "<local-command-stderr>",
        )
    )


def _claude_training_user_content(event: dict[str, Any]) -> str | None:
    payload = event.get("message")
    if not isinstance(payload, dict):
        return None
    content_blocks = payload.get("content")
    content = _claude_text_from_content(content_blocks)
    if not content:
        return None

    command = _claude_local_command_parts(content)
    if command is not None:
        command_name, command_args = command
        if command_name == "/goal" and command_args:
            return command_args
        return None

    if _is_claude_local_command_artifact(content):
        return None
    if event.get("isMeta") is True:
        return None
    return content


def _is_claude_synthetic_session_limit(event: dict[str, Any], content: str) -> bool:
    error_kind = event.get("error")
    return (
        event.get("isApiErrorMessage") is True
        and isinstance(error_kind, str)
        and error_kind.strip().lower() in {"rate_limit", "session_limit"}
        and "session limit" in content.lower()
    )


def _is_claude_synthetic_no_response_requested(event: dict[str, Any], content: str) -> bool:
    payload = event.get("message")
    if not isinstance(payload, dict):
        return False
    return payload.get("model") == "<synthetic>" and content.strip() == "No response requested."


def _is_claude_synthetic_assistant_artifact(event: dict[str, Any], content: str) -> bool:
    return _is_claude_synthetic_session_limit(event, content) or _is_claude_synthetic_no_response_requested(event, content)


def _claude_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _claude_attachment(event: dict[str, Any]) -> dict[str, Any] | None:
    attachment = event.get("attachment")
    return attachment if isinstance(attachment, dict) else None


def _claude_queued_command_content(event: dict[str, Any]) -> str | None:
    if event.get("type") != "attachment":
        return None
    attachment = _claude_attachment(event)
    if not attachment or attachment.get("type") != "queued_command":
        return None
    if attachment.get("commandMode") not in {None, "prompt"}:
        return None
    prompt = attachment.get("prompt")
    return prompt.strip() if isinstance(prompt, str) and prompt.strip() else None


def _claude_context_from_attachment(attachment: dict[str, Any]) -> str | None:
    attachment_type = attachment.get("type")
    if attachment_type == "deferred_tools_delta":
        added_names = _claude_string_list(attachment.get("addedNames"))
        if added_names:
            return "Claude Code deferred tools available through ToolSearch:\n" + "\n".join(
                f"- {name}" for name in added_names
            )
        return None
    if attachment_type == "mcp_instructions_delta":
        added_names = _claude_string_list(attachment.get("addedNames"))
        added_blocks = _claude_string_list(attachment.get("addedBlocks"))
        parts: list[str] = []
        if added_names:
            parts.append("Claude MCP instructions added for: " + ", ".join(added_names))
        parts.extend(added_blocks)
        return "\n\n".join(parts).strip() or None
    if attachment_type == "skill_listing":
        content = attachment.get("content")
        if isinstance(content, str) and content.strip():
            return "Claude Code available skills:\n" + content.strip()
        return None
    if attachment_type == "command_permissions":
        allowed_tools = _claude_string_list(attachment.get("allowedTools"))
        if allowed_tools:
            return "Claude Code command permissions allow: " + ", ".join(allowed_tools)
        return "Claude Code command permissions allow no additional tools."
    if attachment_type == "date_change":
        new_date = attachment.get("newDate")
        if isinstance(new_date, str) and new_date.strip():
            return "Current date: " + new_date.strip()
        return None
    if attachment_type == "hook_additional_context":
        content = _claude_string_list(attachment.get("content"))
        if not content:
            return None
        hook_name = attachment.get("hookName")
        header = "Claude Code hook context"
        if isinstance(hook_name, str) and hook_name.strip():
            header += f" ({hook_name.strip()})"
        return header + ":\n" + "\n".join(content)
    if attachment_type == "edited_text_file":
        filename = attachment.get("filename")
        snippet = attachment.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            return None
        if isinstance(filename, str) and filename.strip():
            return f"Claude Code edited file context for {filename.strip()}:\n{snippet.strip()}"
        return "Claude Code edited file context:\n" + snippet.strip()
    if attachment_type == "task_reminder":
        content = _claude_string_list(attachment.get("content"))
        if content:
            return "Claude Code task reminder:\n" + "\n".join(content)
        return None
    if attachment_type == "plan_mode_exit":
        plan_path = attachment.get("planFilePath")
        if isinstance(plan_path, str) and plan_path.strip():
            return "Claude Code exited plan mode. Plan file: " + plan_path.strip()
        return "Claude Code exited plan mode."
    return None


def _claude_context_from_system_event(event: dict[str, Any]) -> str | None:
    subtype = event.get("subtype")
    if subtype in {None, "local_command", "turn_duration"}:
        return None
    if subtype == "stop_hook_summary":
        content = _claude_string_list(event.get("hookAdditionalContext"))
        if content:
            return "Claude Code stop hook context:\n" + "\n".join(content)
        return None
    content = event.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    if subtype == "away_summary":
        return "Claude Code away summary:\n" + content.strip()
    if subtype == "informational":
        return "Claude Code notice:\n" + content.strip()
    return f"Claude Code system event ({subtype}):\n{content.strip()}"


def _append_claude_system_context(
    messages: list[dict[str, Any]],
    system_contexts: list[str],
    seen_contexts: set[str],
    content: str | None,
) -> None:
    if not content:
        return
    content = content.strip()
    if not content or content in seen_contexts:
        return
    seen_contexts.add(content)
    system_contexts.append(content)
    if messages and messages[-1].get("role") == "system":
        existing = messages[-1].get("content")
        if isinstance(existing, str) and existing.strip():
            messages[-1]["content"] = f"{existing.strip()}\n\n{content}"
            messages[-1]["masked"] = True
            return
    messages.append({"role": "system", "content": content, "masked": True})


def _convert_claude_code_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    events = normalize_claude_code_trace_events(events)
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set(_CLAUDE_CODE_DEFAULT_TOOL_NAMES)
    tool_schemas: dict[str, dict[str, Any]] = deepcopy(_CLAUDE_CODE_BUILTIN_TOOL_SCHEMAS)
    tool_argument_samples: dict[str, list[Any]] = {}
    explicit_tools: dict[str, dict[str, Any]] = {}
    session_meta: dict[str, Any] = {}
    system_contexts: list[str] = []
    seen_system_contexts: set[str] = set()
    deferred_tool_names: set[str] = set()
    mcp_instruction_names: set[str] = set()
    attachment_types: set[str] = set()
    session_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    total_cost_usd: Any = None
    entrypoint: str | None = None
    user_type: str | None = None
    git_branch: str | None = None
    cwd: str | None = None
    cli_version: str | None = None
    mode: str | None = None
    permission_mode: str | None = None
    prompt = ""
    first_message_timestamp = _first_message_timestamp_from_events(
        events,
        lambda event: _is_external_user_message(event) or _is_claude_code_user_message(event),
    )

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if entrypoint is None and isinstance(event.get("entrypoint"), str):
            entrypoint = event["entrypoint"]
        if user_type is None and isinstance(event.get("userType"), str):
            user_type = event["userType"]
        if git_branch is None and isinstance(event.get("gitBranch"), str):
            git_branch = event["gitBranch"]
        if cwd is None and isinstance(event.get("cwd"), str):
            cwd = event["cwd"]
        if cli_version is None and isinstance(event.get("version"), str):
            cli_version = event["version"]
        if event_type == "external_session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                session_meta = payload
                value = payload.get("id")
                if isinstance(value, str) and value.strip():
                    session_id = value.strip()
                value = payload.get("model")
                if isinstance(value, str) and value.strip():
                    model = value.strip()
                _add_explicit_tools(payload.get("tools"), explicit_tools, tool_names, tool_schemas)
            continue
        if event_type == "external_message":
            role = event.get("role")
            content = event.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            normalized_role = _normalize_role(role)
            if normalized_role == "user" and content.strip() and not prompt:
                prompt = content.strip()
            messages.append({"role": normalized_role, "content": content})
            continue
        if event_type == "mode":
            value = event.get("mode")
            if isinstance(value, str) and value.strip():
                mode = value.strip()
            continue
        if event_type == "permission-mode":
            value = event.get("permissionMode")
            if isinstance(value, str) and value.strip():
                permission_mode = value.strip()
            continue
        if event_type == "attachment":
            if isinstance(event.get("session_id"), str):
                session_id = event["session_id"]
            elif isinstance(event.get("sessionId"), str):
                session_id = event["sessionId"]
            attachment = _claude_attachment(event)
            if not attachment:
                continue
            attachment_type = attachment.get("type")
            if isinstance(attachment_type, str) and attachment_type.strip():
                attachment_types.add(attachment_type.strip())
            if attachment_type == "deferred_tools_delta":
                added_names = _claude_string_list(attachment.get("addedNames"))
                deferred_tool_names.update(added_names)
                for name in added_names:
                    tool_names.add(name)
                    tool_schemas.setdefault(name, _claude_deferred_tool_schema(name))
            elif attachment_type == "mcp_instructions_delta":
                mcp_instruction_names.update(_claude_string_list(attachment.get("addedNames")))
            queued_command = _claude_queued_command_content(event)
            if queued_command:
                if not prompt:
                    prompt = queued_command
                messages.append({"role": "user", "content": queued_command})
                continue
            _append_claude_system_context(
                messages,
                system_contexts,
                seen_system_contexts,
                _claude_context_from_attachment(attachment),
            )
            continue
        if event_type == "system":
            if isinstance(event.get("session_id"), str):
                session_id = event["session_id"]
            elif isinstance(event.get("sessionId"), str):
                session_id = event["sessionId"]
            if event.get("subtype") == "init":
                value = event.get("model")
                if isinstance(value, str) and value.strip():
                    model = value.strip()
                tools = event.get("tools")
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, str) and tool.strip():
                            tool_name = tool.strip()
                            tool_names.add(tool_name)
                            tool_schemas.setdefault(tool_name, _claude_deferred_tool_schema(tool_name))
                        elif isinstance(tool, dict):
                            name = tool.get("name")
                            if isinstance(name, str) and name.strip():
                                tool_name = name.strip()
                                tool_names.add(tool_name)
                                schema = _claude_code_tool_schema_from_definition(tool)
                                if schema:
                                    tool_schemas[tool_name] = {**tool_schemas.get(tool_name, {}), **schema}
            else:
                _append_claude_system_context(
                    messages,
                    system_contexts,
                    seen_system_contexts,
                    _claude_context_from_system_event(event),
                )
            continue
        if event_type in {"user", "assistant"}:
            if isinstance(event.get("session_id"), str):
                session_id = event["session_id"]
            elif isinstance(event.get("sessionId"), str):
                session_id = event["sessionId"]

        if event_type == "user":
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            content_blocks = payload.get("content")
            if isinstance(content_blocks, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content_blocks
            ):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_call_id = block.get("tool_use_id") or block.get("tool_call_id")
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        continue
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": "unknown_tool",
                            "content": _claude_tool_result_text(block),
                        }
                    )
                continue
            content = _claude_training_user_content(event)
            if content and not prompt:
                prompt = content
            if content:
                messages.append({"role": "user", "content": content})
            continue

        if event_type == "assistant":
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            value = payload.get("model")
            if isinstance(value, str) and value.strip():
                model = value.strip()
            payload_usage = payload.get("usage")
            if isinstance(payload_usage, dict):
                usage = payload_usage
            content_blocks = payload.get("content")
            content = _claude_text_from_content(content_blocks)
            if _is_claude_synthetic_assistant_artifact(event, content):
                continue
            message: dict[str, Any] = {"role": "assistant", "content": content}
            reasoning_content = _claude_reasoning_from_content(content_blocks)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if event.get("isApiErrorMessage") is True:
                message["teich_provider_error"] = True
            error_kind = event.get("error")
            if isinstance(error_kind, str) and error_kind.strip():
                message["teich_provider_error"] = error_kind.strip()
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    if not tool_call_id or not tool_name:
                        continue
                    arguments = _normalize_json_like_value(block.get("input") or {})
                    tool_names.add(tool_name)
                    tool_schemas.setdefault(
                        tool_name,
                        _claude_deferred_tool_schema(tool_name, include_parameters=False),
                    )
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if content or reasoning_content or tool_calls:
                _append_or_merge_assistant_message(messages, message)
            continue

        if event_type == "result":
            value = event.get("session_id")
            if not isinstance(value, str):
                value = event.get("sessionId")
            if isinstance(value, str) and value.strip():
                session_id = value.strip()
            payload_usage = event.get("usage")
            if isinstance(payload_usage, dict):
                usage = payload_usage
            if event.get("total_cost_usd") is not None:
                total_cost_usd = event.get("total_cost_usd")
            result = event.get("result")
            if isinstance(result, str) and result.strip():
                last_content = messages[-1].get("content") if messages else None
                if last_content != result.strip():
                    messages.append({"role": "assistant", "content": result.strip()})

    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        explicit_tools,
    )
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )
    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_id or session_meta.get("id") or trace_file.stem,
        "trace_type": "claude-code",
        "model_provider": session_meta.get("model_provider") or "anthropic",
        "model": model or session_meta.get("model"),
        "cwd": session_meta.get("cwd") or cwd,
        "cli_version": session_meta.get("cli_version") or cli_version,
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    if entrypoint:
        metadata["entrypoint"] = entrypoint
    if user_type:
        metadata["user_type"] = user_type
    if git_branch:
        metadata["git_branch"] = git_branch
    if mode:
        metadata["mode"] = mode
    if permission_mode:
        metadata["permission_mode"] = permission_mode
    if system_contexts:
        metadata["system_prompt"] = "\n\n".join(system_contexts)
        metadata["claude_context_count"] = len(system_contexts)
    if attachment_types:
        metadata["claude_attachment_types"] = sorted(attachment_types)
    if deferred_tool_names:
        metadata["claude_deferred_tools"] = sorted(deferred_tool_names)
    if mcp_instruction_names:
        metadata["claude_mcp_instruction_names"] = sorted(mcp_instruction_names)
    if usage:
        metadata["usage"] = usage
    if total_cost_usd is not None:
        metadata["total_cost_usd"] = total_cost_usd
    _add_first_message_timestamp(metadata, first_message_timestamp)
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _droid_settings_path(trace_file: Path) -> Path:
    return trace_file.with_suffix(".settings.json")


def _load_droid_settings_sidecar(trace_file: Path) -> dict[str, Any]:
    settings_path = _droid_settings_path(trace_file)
    if not settings_path.exists():
        return {}
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def _convert_droid_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set(_DROID_BUILTIN_TOOL_SCHEMAS)
    tool_schemas: dict[str, dict[str, Any]] = deepcopy(_DROID_BUILTIN_TOOL_SCHEMAS)
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_names_by_call_id: dict[str, str] = {}
    session_id: str | None = None
    cwd: str | None = None
    title: str | None = None
    prompt = ""
    first_message_timestamp = _first_message_timestamp_from_events(events, _is_droid_user_message)

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session_start":
            value = event.get("id")
            if isinstance(value, str) and value.strip():
                session_id = value.strip()
            value = event.get("cwd")
            if isinstance(value, str) and value.strip():
                cwd = value.strip()
            value = event.get("sessionTitle") or event.get("title")
            if isinstance(value, str) and value.strip():
                title = value.strip()
            continue
        if event_type != "message":
            continue
        payload = event.get("message")
        if not isinstance(payload, dict):
            continue
        if payload.get("visibility") == "user_only":
            continue
        role = payload.get("role")
        content_blocks = payload.get("content")

        if role == "user":
            if isinstance(content_blocks, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content_blocks
            ):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_call_id = block.get("tool_use_id") or block.get("tool_call_id")
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        continue
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_names_by_call_id.get(tool_call_id, "unknown_tool"),
                            "content": _claude_tool_result_text(block),
                        }
                    )
                continue
            content = _claude_text_from_content(content_blocks)
            if content and not prompt and payload.get("visibility") != "llm_only":
                prompt = content
            if content:
                messages.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            content = _claude_text_from_content(content_blocks)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            reasoning_content = _claude_reasoning_from_content(content_blocks)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    if not tool_call_id or not tool_name:
                        continue
                    arguments = _normalize_json_like_value(block.get("input") or {})
                    tool_names.add(tool_name)
                    tool_names_by_call_id[tool_call_id] = tool_name
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if content or reasoning_content or tool_calls:
                _append_or_merge_assistant_message(messages, message)

    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        {},
    )
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )
    settings = _load_droid_settings_sidecar(trace_file)
    model = settings.get("model")
    provider_lock = settings.get("providerLock")
    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_id or trace_file.stem,
        "trace_type": "droid",
        "model_provider": provider_lock if isinstance(provider_lock, str) and provider_lock.strip() else "factory",
        "model": model if isinstance(model, str) and model.strip() else None,
        "cwd": cwd,
        "title": title,
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    _add_first_message_timestamp(metadata, first_message_timestamp)
    usage = settings.get("tokenUsage")
    if isinstance(usage, dict) and usage:
        metadata["usage"] = usage
    for settings_key, metadata_key in (("reasoningEffort", "reasoning_effort"), ("autonomyLevel", "autonomy_level")):
        value = settings.get(settings_key)
        if isinstance(value, str) and value.strip():
            metadata[metadata_key] = value.strip()
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _convert_hermes_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    explicit_tools: dict[str, dict[str, Any]] = {}
    session_meta: dict[str, Any] = {}
    prompt = ""
    message_events = events
    _add_builtin_tools_for_trace_type("hermes", explicit_tools, tool_names, tool_schemas)
    sidecar_meta = _load_hermes_metadata_sidecar(trace_file)
    if sidecar_meta:
        session_meta = sidecar_meta
        _add_explicit_tools(sidecar_meta.get("tools"), explicit_tools, tool_names, tool_schemas)
    if len(events) == 1 and _is_hermes_trace_row(events[0]):
        row = events[0]
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            session_meta = metadata
            _add_explicit_tools(metadata.get("tools"), explicit_tools, tool_names, tool_schemas)
        row_prompt = row.get("task")
        if isinstance(row_prompt, str) and row_prompt.strip():
            prompt = row_prompt.strip()
        tools = row.get("tools")
        if isinstance(tools, list):
            _add_explicit_tools(tools, explicit_tools, tool_names, tool_schemas)
        message_events = _hermes_conversation_to_events(row.get("traces"))
    elif len(events) == 1 and _is_hermes_conversation(events[0]):
        message_events = _hermes_conversation_to_events(events[0])
    elif len(events) == 1 and isinstance(events[0], dict) and isinstance(events[0].get("messages"), list):
        session_meta = {key: value for key, value in events[0].items() if key != "messages"}
        _add_explicit_tools(events[0].get("tools"), explicit_tools, tool_names, tool_schemas)
        message_events = [event for event in events[0]["messages"] if isinstance(event, dict)]
    first_message_timestamp = _first_message_timestamp_from_events(
        message_events,
        lambda event: _is_user_role(event.get("role")),
    )

    for event in message_events:
        if not isinstance(event, dict):
            continue
        if event.get("type") in {"hermes_session_meta", "external_session_meta"}:
            payload = event.get("payload")
            if isinstance(payload, dict):
                session_meta = payload
                _add_explicit_tools(payload.get("tools"), explicit_tools, tool_names, tool_schemas)
                _add_explicit_tools(payload.get("available_tools"), explicit_tools, tool_names, tool_schemas)
            continue

        role = event.get("role")
        if not isinstance(role, str):
            continue
        normalized_role = _normalize_role(role)
        content, inline_reasoning = _message_content_and_inline_reasoning(event.get("content"))
        if normalized_role == "user" and content.strip() and not prompt:
            prompt = content.strip()

        if normalized_role == "tool":
            message: dict[str, Any] = {"role": "tool", "content": content}
            tool_call_id = event.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                message["tool_call_id"] = tool_call_id
            tool_name = event.get("name") or event.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                message["name"] = tool_name
                tool_names.add(tool_name)
            messages.append(message)
            continue

        message = {"role": normalized_role, "content": content}
        if normalized_role == "assistant":
            reasoning = event.get("reasoning_content") or event.get("reasoning") or inline_reasoning
            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning_content"] = reasoning.strip()
            for key in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
                value = event.get(key)
                if value:
                    message[key] = _normalize_json_like_value(value)

        raw_tool_calls = _normalize_json_like_value(event.get("tool_calls"))
        tool_calls: list[dict[str, Any]] = []
        if isinstance(raw_tool_calls, list):
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    continue
                function = raw_tool_call.get("function")
                function_name = None
                arguments: Any = {}
                if isinstance(function, dict):
                    value = function.get("name")
                    if isinstance(value, str) and value:
                        function_name = value
                    arguments = _parse_function_arguments(function.get("arguments"))
                value = raw_tool_call.get("name")
                if function_name is None and isinstance(value, str) and value:
                    function_name = value
                if not function_name:
                    continue
                tool_call_id = raw_tool_call.get("id") or event.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    tool_call_id = f"{function_name}_{len(tool_calls) + 1}"
                tool_names.add(function_name)
                tool_argument_samples.setdefault(function_name, []).append(arguments)
                tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments,
                        },
                    }
                )
        if tool_calls:
            message["tool_calls"] = tool_calls
        if content.strip() or normalized_role == "assistant" or tool_calls or "reasoning_content" in message:
            messages.append(message)

    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        explicit_tools,
    )
    metadata = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("id") or trace_file.stem,
        "trace_type": "hermes",
        "source": session_meta.get("source"),
        "teich_export_status": session_meta.get("teich_export_status"),
        "teich_partial": session_meta.get("teich_partial"),
        "model_provider": session_meta.get("model_provider"),
        "configured_model_provider": session_meta.get("configured_model_provider"),
        "model": session_meta.get("model"),
        "configured_context_length": session_meta.get("configured_context_length"),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "parent_session_id": session_meta.get("parent_session_id"),
        "hermes_source": session_meta.get("hermes_source"),
        "message_count": session_meta.get("message_count"),
        "tool_call_count": session_meta.get("tool_call_count"),
        "input_tokens": session_meta.get("input_tokens"),
        "output_tokens": session_meta.get("output_tokens"),
        "cache_read_tokens": session_meta.get("cache_read_tokens"),
        "cache_write_tokens": session_meta.get("cache_write_tokens"),
        "reasoning_tokens": session_meta.get("reasoning_tokens"),
        "total_tokens": session_meta.get("total_tokens"),
        "estimated_cost_usd": session_meta.get("estimated_cost_usd"),
        "actual_cost_usd": session_meta.get("actual_cost_usd"),
        "total_cost": session_meta.get("total_cost"),
        "cost_status": session_meta.get("cost_status"),
        "cost_source": session_meta.get("cost_source"),
        "billing_provider": session_meta.get("billing_provider"),
        "billing_base_url": session_meta.get("billing_base_url"),
        "billing_mode": session_meta.get("billing_mode"),
        "system_prompt": session_meta.get("system_prompt"),
        "started_at": session_meta.get("started_at"),
        "ended_at": session_meta.get("ended_at"),
        "end_reason": session_meta.get("end_reason"),
        "api_call_count": session_meta.get("api_call_count"),
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    _add_first_message_timestamp(metadata, first_message_timestamp)
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _hermes_metadata_path(trace_file: Path) -> Path:
    return trace_file.with_suffix(".metadata.json")


def _load_hermes_metadata_sidecar(trace_file: Path) -> dict[str, Any]:
    metadata_path = _hermes_metadata_path(trace_file)
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _is_hermes_conversation(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and isinstance(item.get("from"), str) and isinstance(item.get("value"), str) for item in value)
    )


def _is_hermes_trace_row(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and isinstance(value.get("traces"), list)
        and _is_hermes_conversation(value.get("traces"))
    )


def _extract_xml_blocks(value: str, tag: str) -> tuple[list[str], str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    blocks = [match.group(1).strip() for match in pattern.finditer(value)]
    stripped = pattern.sub("", value).strip()
    return blocks, stripped


def _hermes_conversation_to_events(conversation: Any) -> list[dict[str, Any]]:
    if not _is_hermes_conversation(conversation):
        return []
    events: list[dict[str, Any]] = []
    for item in conversation:
        source_role = item.get("from")
        value = item.get("value") or ""
        timestamp = _timestamp_from_mapping(item)
        if source_role == "system":
            event = {"role": "system", "content": value}
            if timestamp:
                event["timestamp"] = timestamp
            events.append(event)
        elif source_role == "human":
            event = {"role": "user", "content": value}
            if timestamp:
                event["timestamp"] = timestamp
            events.append(event)
        elif source_role == "tool":
            tool_blocks, fallback_content = _extract_xml_blocks(value, "tool_response")
            parsed = _normalize_json_like_value(tool_blocks[0]) if tool_blocks else fallback_content
            if not isinstance(parsed, dict):
                parsed = {"content": parsed}
            event = {
                "role": "tool",
                "content": json.dumps(parsed.get("content"), ensure_ascii=False)
                if isinstance(parsed.get("content"), (dict, list))
                else str(parsed.get("content") or ""),
                "tool_call_id": parsed.get("tool_call_id"),
                "tool_name": parsed.get("name"),
            }
            if timestamp:
                event["timestamp"] = timestamp
            events.append(event)
        elif source_role == "gpt":
            thinking_blocks, without_thinking = _extract_xml_blocks(value, "think")
            tool_blocks, content = _extract_xml_blocks(without_thinking, "tool_call")
            tool_calls: list[dict[str, Any]] = []
            for index, block in enumerate(tool_blocks, start=1):
                parsed = _normalize_json_like_value(block)
                if not isinstance(parsed, dict):
                    continue
                name = parsed.get("name")
                if not isinstance(name, str) or not name:
                    continue
                tool_calls.append(
                    {
                        "id": f"{name}_{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {},
                        },
                    }
                )
            event: dict[str, Any] = {"role": "assistant", "content": content}
            if thinking_blocks:
                event["reasoning_content"] = "\n\n".join(block for block in thinking_blocks if block)
            if tool_calls:
                event["tool_calls"] = tool_calls
            if timestamp:
                event["timestamp"] = timestamp
            events.append(event)
    return events


def _convert_external_agent_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    explicit_tools: dict[str, dict[str, Any]] = {}
    session_meta: dict[str, Any] = {}
    prompt = ""
    first_message_timestamp = _first_message_timestamp_from_events(
        events,
        _is_external_user_message,
    )
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "external_session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                session_meta = payload
                _add_explicit_tools(payload.get("tools"), explicit_tools, tool_names, tool_schemas)
                _add_explicit_tools(payload.get("available_tools"), explicit_tools, tool_names, tool_schemas)
            continue
        if event.get("type") != "external_message":
            continue
        role = event.get("role")
        content = event.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        normalized_role = _normalize_role(role)
        if normalized_role == "user" and content.strip() and not prompt:
            prompt = content.strip()
        if normalized_role == "tool":
            message: dict[str, Any] = {"role": "tool", "content": content}
            tool_call_id = event.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                message["tool_call_id"] = tool_call_id
            tool_name = event.get("name") or event.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                message["name"] = tool_name
                tool_names.add(tool_name)
            messages.append(message)
            continue

        message = {"role": normalized_role, "content": content}
        reasoning = event.get("reasoning_content") or event.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            message["reasoning_content"] = reasoning.strip()
        raw_tool_calls = _normalize_json_like_value(event.get("tool_calls"))
        tool_calls: list[dict[str, Any]] = []
        if isinstance(raw_tool_calls, list):
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    continue
                function = raw_tool_call.get("function")
                function_name = None
                arguments: Any = {}
                if isinstance(function, dict):
                    value = function.get("name")
                    if isinstance(value, str) and value:
                        function_name = value
                    arguments = _parse_function_arguments(function.get("arguments"))
                value = raw_tool_call.get("name")
                if function_name is None and isinstance(value, str) and value:
                    function_name = value
                if not function_name:
                    continue
                tool_call_id = raw_tool_call.get("id") or event.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    tool_call_id = f"{function_name}_{len(tool_calls) + 1}"
                tool_names.add(function_name)
                tool_argument_samples.setdefault(function_name, []).append(arguments)
                tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments,
                        },
                    }
                )
        if tool_calls:
            message["tool_calls"] = tool_calls
        if content.strip() or normalized_role == "assistant" or tool_calls or "reasoning_content" in message:
            messages.append(message)
    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        explicit_tools,
    )
    metadata = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("id") or trace_file.stem,
        "trace_type": session_meta.get("source") or "external-agent",
        "teich_export_status": session_meta.get("teich_export_status"),
        "teich_partial": session_meta.get("teich_partial"),
        "model_provider": session_meta.get("model_provider"),
        "configured_model_provider": session_meta.get("configured_model_provider"),
        "model": session_meta.get("model"),
        "configured_context_length": session_meta.get("configured_context_length"),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "parent_session_id": session_meta.get("parent_session_id"),
        "hermes_source": session_meta.get("hermes_source"),
        "message_count": session_meta.get("message_count"),
        "tool_call_count": session_meta.get("tool_call_count"),
        "input_tokens": session_meta.get("input_tokens"),
        "output_tokens": session_meta.get("output_tokens"),
        "cache_read_tokens": session_meta.get("cache_read_tokens"),
        "cache_write_tokens": session_meta.get("cache_write_tokens"),
        "reasoning_tokens": session_meta.get("reasoning_tokens"),
        "total_tokens": session_meta.get("total_tokens"),
        "estimated_cost_usd": session_meta.get("estimated_cost_usd"),
        "actual_cost_usd": session_meta.get("actual_cost_usd"),
        "total_cost": session_meta.get("total_cost"),
        "cost_status": session_meta.get("cost_status"),
        "cost_source": session_meta.get("cost_source"),
        "billing_provider": session_meta.get("billing_provider"),
        "billing_base_url": session_meta.get("billing_base_url"),
        "billing_mode": session_meta.get("billing_mode"),
        "system_prompt": session_meta.get("system_prompt"),
        "started_at": session_meta.get("started_at"),
        "ended_at": session_meta.get("ended_at"),
        "end_reason": session_meta.get("end_reason"),
        "api_call_count": session_meta.get("api_call_count"),
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    _add_first_message_timestamp(metadata, first_message_timestamp)
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _convert_cursor_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    explicit_tools: dict[str, dict[str, Any]] = {}
    session_meta: dict[str, Any] = {}
    tool_names_by_call_id: dict[str, str] = {}
    prompt = ""
    first_message_timestamp = _first_message_timestamp_from_events(
        events,
        lambda event: event.get("role") == "user"
        and isinstance(event.get("message"), dict)
        and not _cursor_event_contains_tool_result(event),
    )
    _add_builtin_tools_for_trace_type("cursor", explicit_tools, tool_names, tool_schemas)

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "cursor_session_meta":
            session_meta = event
            continue
        if event_type == "cursor_available_tools":
            _add_explicit_tools(event.get("tools"), explicit_tools, tool_names, tool_schemas)
            continue

        role = event.get("role")
        payload = event.get("message")
        if not isinstance(role, str) or not isinstance(payload, dict):
            continue
        content_blocks = payload.get("content")

        if role == "user":
            if isinstance(content_blocks, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content_blocks
            ):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_call_id = block.get("tool_use_id") or block.get("tool_call_id")
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        continue
                    tool_message: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": block.get("name") or tool_names_by_call_id.get(tool_call_id, "unknown_tool"),
                        "content": _claude_tool_result_text(block),
                    }
                    if block.get("is_error") is True:
                        tool_message["is_error"] = True
                    messages.append(tool_message)
                continue
            content = _claude_text_from_content(content_blocks)
            if content and not prompt:
                prompt = content
            if content:
                messages.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            content = _claude_text_from_content(content_blocks)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            reasoning_content = _claude_reasoning_from_content(content_blocks)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_name, str) or not tool_name:
                        continue
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        tool_call_id = f"{tool_name}_{len(tool_calls) + 1}"
                    arguments = _normalize_json_like_value(block.get("input") or {})
                    tool_names.add(tool_name)
                    tool_names_by_call_id[tool_call_id] = tool_name
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if content or reasoning_content or tool_calls:
                _append_or_merge_assistant_message(messages, message)
            continue

        if role == "system":
            content = _claude_text_from_content(content_blocks)
            if content:
                messages.append({"role": "system", "content": content})

    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        explicit_tools,
    )
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )
    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("session_id") or trace_file.stem,
        "trace_type": "cursor",
        "model": session_meta.get("model"),
        "source": session_meta.get("source") or "cursor",
        "source_db": session_meta.get("source_db"),
        "cursor_scope": session_meta.get("cursor_scope"),
        "cursor_workspace_id": session_meta.get("cursor_workspace_id"),
        "cursor_table": session_meta.get("cursor_table"),
        "cursor_key": session_meta.get("cursor_key"),
        "cursor_storage_kind": session_meta.get("cursor_storage_kind"),
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    _add_first_message_timestamp(metadata, first_message_timestamp)
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _cursor_event_contains_tool_result(event: dict[str, Any]) -> bool:
    payload = event.get("message")
    if not isinstance(payload, dict):
        return False
    content = payload.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def load_trace_file(trace_file: Path) -> list[Any]:
    return _load_trace_file(trace_file)


def _load_trace_file(trace_file: Path, *, skip_invalid_lines: bool = False) -> list[Any]:
    events: list[Any] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if skip_invalid_lines:
                    continue
                raise
            events.append(value)
    return events


def _convert_codex_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
) -> TrainingExample:
    events = normalize_codex_trace_events(events)
    first_message_timestamp = _codex_first_user_message_timestamp(events)
    messages: list[dict[str, Any]] = []
    pending_reasoning_parts: list[str] = []
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_descriptions: dict[str, str] = {}
    tool_call_names: dict[str, str] = {}
    explicit_tools: dict[str, dict[str, Any]] = {}
    session_meta: dict[str, Any] = {}
    turn_contexts: list[dict[str, Any]] = []
    system_prompt: str | None = None
    prompt = ""
    _add_builtin_tools_for_trace_type("codex", explicit_tools, tool_names, tool_schemas)

    for event in events:
        if not isinstance(event, dict):
            continue
        event = normalize_codex_trace_event(event)
        event_type = event.get("type")
        payload = event.get("payload")
        if event_type == "session_meta" and isinstance(payload, dict):
            session_meta = payload
            _add_explicit_tools(payload.get("tools"), explicit_tools, tool_names, tool_schemas)
            base_instructions = payload.get("base_instructions")
            if isinstance(base_instructions, dict):
                text = base_instructions.get("text")
                if isinstance(text, str) and text.strip():
                    system_prompt = system_prompt or text.strip()
                    if not _has_same_system_message(messages, text):
                        messages.append({"role": "system", "content": text})
                        tool_descriptions.update(_parse_tool_descriptions_from_text(text))
            continue
        if event_type == "turn_context" and isinstance(payload, dict):
            turn_contexts.append(payload)
            continue
        if event_type != "response_item" or not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        if payload_type == TEICH_TRACE_CONTEXT_ITEM_TYPE:
            value = payload.get("system_prompt")
            if isinstance(value, str) and value.strip():
                system_prompt = system_prompt or value.strip()
            _add_explicit_tools(payload.get("tools"), explicit_tools, tool_names, tool_schemas)
            continue

        if payload_type == "reasoning":
            reasoning = _reasoning_summary(payload)
            if reasoning:
                previous_message = messages[-1] if messages else None
                previous_has_output = (
                    isinstance(previous_message, dict)
                    and previous_message.get("role") == "assistant"
                    and (
                        bool(str(previous_message.get("content") or "").strip())
                        or bool(previous_message.get("tool_calls"))
                    )
                )
                if previous_has_output:
                    _append_or_merge_assistant_message(
                        messages,
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": reasoning,
                        },
                    )
                else:
                    pending_reasoning_parts.append(reasoning)
            continue

        if payload_type == "message":
            role = payload.get("role")
            if not isinstance(role, str):
                continue
            normalized_role = _normalize_role(role)
            content = _first_text_block(payload.get("content"))
            if normalized_role == "user" and content and not prompt:
                prompt = content
            message: dict[str, Any] = {
                "role": normalized_role,
                "content": content,
            }
            if normalized_role == "assistant" and pending_reasoning_parts:
                message["reasoning_content"] = "\n\n".join(pending_reasoning_parts)
                pending_reasoning_parts = []
            messages.append(message)
            continue

        if payload_type == "function_call":
            name = payload.get("name")
            call_id = payload.get("call_id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                continue
            tool_names.add(name)
            tool_call_names[call_id] = name
            arguments = _parse_function_arguments(payload.get("arguments"))
            tool_argument_samples.setdefault(name, []).append(arguments)
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
            if messages and messages[-1].get("role") == "assistant":
                assistant_message = messages[-1]
                if pending_reasoning_parts:
                    existing_reasoning = assistant_message.get("reasoning_content")
                    pending_reasoning = "\n\n".join(pending_reasoning_parts)
                    if isinstance(existing_reasoning, str) and existing_reasoning.strip():
                        assistant_message["reasoning_content"] = (
                            f"{existing_reasoning.strip()}\n\n{pending_reasoning}"
                        )
                    else:
                        assistant_message["reasoning_content"] = pending_reasoning
                    pending_reasoning_parts = []
                existing_tool_calls = assistant_message.setdefault("tool_calls", [])
                if isinstance(existing_tool_calls, list):
                    existing_tool_calls.append(tool_call)
                else:
                    assistant_message["tool_calls"] = [tool_call]
            else:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call],
                }
                if pending_reasoning_parts:
                    assistant_message["reasoning_content"] = "\n\n".join(pending_reasoning_parts)
                    pending_reasoning_parts = []
                messages.append(assistant_message)
            continue

        if payload_type == "function_call_output":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            tool_name = tool_call_names.get(call_id)
            if tool_name:
                tool_names.add(tool_name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name or "unknown_tool",
                    "content": _function_call_output_text(payload),
                }
            )
            continue

        if payload_type == "tool_schema":
            name = payload.get("name")
            schema = payload.get("schema")
            if isinstance(name, str) and isinstance(schema, dict):
                tool_names.add(name)
                tool_schemas[name] = schema

    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        explicit_tools,
        tool_descriptions,
    )
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )

    metadata = {
        "source_file": trace_file.name,
        "session_id": session_meta.get("id") or trace_file.stem,
        "trace_type": "codex",
        "source": session_meta.get("source"),
        "model_provider": session_meta.get("model_provider"),
        "model": (
            session_meta.get("model")
            or next(
                (
                    context.get("model")
                    for context in reversed(turn_contexts)
                    if isinstance(context.get("model"), str) and context.get("model")
                ),
                None,
            )
        ),
        "cwd": session_meta.get("cwd"),
        "cli_version": session_meta.get("cli_version"),
        "system_prompt": system_prompt or session_meta.get("system_prompt"),
        "turn_count": len(turn_contexts),
    }
    _add_first_message_timestamp(metadata, first_message_timestamp)
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _convert_pi_trace_to_training_example(
    trace_file: Path,
    events: list[dict[str, Any]],
    *,
    trace_type: Literal["openclaw", "pi"] = "pi",
) -> TrainingExample:
    messages: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_argument_samples: dict[str, list[Any]] = {}
    tool_descriptions: dict[str, str] = {}
    explicit_tools: dict[str, dict[str, Any]] = {}
    session_header: dict[str, Any] = {}
    model_change: dict[str, Any] = {}
    session_names: list[str] = []
    thinking_level: str | None = None
    teich_system_prompt: str | None = None
    prompt = ""
    invalid_tool_call_ids: set[str] = set()
    include_teich_pi_metadata = trace_type == "pi"
    _add_builtin_tools_for_trace_type(trace_type, explicit_tools, tool_names, tool_schemas)
    first_message_timestamp = _first_message_timestamp_from_events(
        events,
        lambda event: event.get("type") == "message" and _is_nested_user_message(event),
    )

    for event in events:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        payload = event.get("message")
        if not isinstance(payload, dict) or payload.get("role") != "toolResult":
            continue
        tool_call_id = payload.get("toolCallId")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        tool_name = payload.get("toolName") if isinstance(payload.get("toolName"), str) else None
        if _is_tool_not_found_result(tool_name, payload):
            invalid_tool_call_ids.add(tool_call_id)

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session":
            session_header = event
            continue
        if event_type == "model_change":
            model_change = event
            continue
        if event_type == "thinking_level_change":
            level = event.get("thinkingLevel")
            if isinstance(level, str) and level.strip():
                thinking_level = level.strip()
            continue
        if event_type == "session_info":
            name = event.get("name")
            if isinstance(name, str) and name.strip():
                session_names.append(name.strip())
            continue
        if event_type == "custom":
            if include_teich_pi_metadata:
                teich_system_prompt = teich_system_prompt or _pi_teich_system_prompt_from_event(event)
            if event.get("customType") == TEICH_AVAILABLE_TOOLS_CUSTOM_TYPE:
                data = event.get("data")
                if isinstance(data, dict):
                    _add_explicit_tools(data.get("tools"), explicit_tools, tool_names, tool_schemas)
            continue
        if event_type != "message":
            continue

        payload = event.get("message")
        if not isinstance(payload, dict):
            continue
        role = payload.get("role")
        if not isinstance(role, str):
            continue

        if role == "toolResult":
            tool_call_id = payload.get("toolCallId")
            if not isinstance(tool_call_id, str):
                continue
            if tool_call_id in invalid_tool_call_ids:
                continue
            tool_name = payload.get("toolName")
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name or "unknown_tool",
                "content": _first_text_block(payload.get("content")),
            }
            if payload.get("isError") is True:
                tool_message["is_error"] = True
            messages.append(tool_message)
            continue

        normalized_role = _normalize_role(role)
        content_blocks = payload.get("content")
        content = _first_text_block(content_blocks)

        if role == "developer" and content:
            tool_descriptions.update(_parse_tool_descriptions_from_text(content))

        if normalized_role == "user":
            if content and not prompt:
                prompt = content
            messages.append({"role": normalized_role, "content": content})
            continue

        message: dict[str, Any] = {
            "role": normalized_role,
            "content": content,
        }
        if normalized_role == "assistant":
            reasoning_content = _pi_reasoning_content(content_blocks)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "toolCall":
                        continue
                    tool_call_id = block.get("id")
                    tool_name = block.get("name")
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    if not tool_call_id or not tool_name or tool_call_id in invalid_tool_call_ids:
                        continue
                    tool_names.add(tool_name)
                    arguments = _parse_function_arguments(block.get("arguments"))
                    tool_argument_samples.setdefault(tool_name, []).append(arguments)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                message["tool_calls"] = tool_calls
            if not message["content"] and "reasoning_content" not in message and "tool_calls" not in message:
                continue
        elif not content:
            continue
        messages.append(message)

    if teich_system_prompt and not _has_same_system_message(messages, teich_system_prompt):
        messages.insert(0, {"role": "system", "content": teich_system_prompt})

    tools = _build_tools_from_snapshots_and_calls(
        tool_names,
        tool_schemas,
        tool_argument_samples,
        explicit_tools,
        tool_descriptions,
    )
    if not prompt:
        prompt = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )

    metadata: dict[str, Any] = {
        "source_file": trace_file.name,
        "session_id": session_header.get("id") or trace_file.stem,
        "trace_type": trace_type,
        "model_provider": model_change.get("provider"),
        "model": model_change.get("modelId"),
        "cwd": session_header.get("cwd"),
        "cli_version": None,
        "turn_count": sum(1 for message in messages if message.get("role") == "user"),
    }
    if thinking_level:
        metadata["thinking_level"] = thinking_level
    if teich_system_prompt:
        metadata["system_prompt"] = teich_system_prompt
    if session_names:
        metadata["session_names"] = session_names
        metadata["session_name"] = session_names[-1]
    _add_first_message_timestamp(metadata, first_message_timestamp)
    return TrainingExample(
        source_file=trace_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def _is_hermes_export_session(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("messages"), list) and (
        value.get("source") == "cli"
        or isinstance(value.get("started_at"), (int, float))
        or isinstance(value.get("parent_session_id"), str)
    )


def _is_structured_training_row(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("messages"), list) and not _is_hermes_export_session(value)


def _normalize_training_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        return None
    normalized_role = _normalize_role(role.strip())
    content, inline_reasoning = _message_content_and_inline_reasoning(message.get("content"))
    normalized: dict[str, Any] = {
        "role": normalized_role,
        "content": content,
    }
    if normalized_role == "assistant":
        reasoning_content = message.get("reasoning_content")
        if not isinstance(reasoning_content, str) or not reasoning_content.strip():
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                reasoning_content = thinking.strip()
        if not isinstance(reasoning_content, str) or not reasoning_content.strip():
            reasoning_content = inline_reasoning
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            normalized["reasoning_content"] = reasoning_content.strip()
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized["tool_calls"] = tool_calls
    if normalized_role == "tool":
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            normalized["tool_call_id"] = tool_call_id
        tool_name = message.get("name")
        if isinstance(tool_name, str) and tool_name:
            normalized["name"] = tool_name
        if message.get("is_error") is True:
            normalized["is_error"] = True
    return normalized


def normalize_training_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    return [
        normalized_message
        for normalized_message in (_normalize_training_message(message) for message in messages)
        if normalized_message is not None
    ]


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    return next(
        (
            message.get("content", "")
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )


def _structured_training_example_from_row(
    source_file: Path,
    row: dict[str, Any],
    row_index: int,
) -> TrainingExample:
    raw_messages = row.get("messages")
    first_message_timestamp = _first_message_timestamp_from_events(
        raw_messages if isinstance(raw_messages, list) else [],
        lambda message: _is_user_role(message.get("role")),
    )
    messages = normalize_training_messages(raw_messages)
    if not messages:
        system = row.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            messages.append({"role": "user", "content": prompt.strip()})
        response = row.get("response") if isinstance(row.get("response"), str) else ""
        assistant_message: dict[str, Any] = {"role": "assistant", "content": response}
        thinking = row.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            assistant_message["reasoning_content"] = thinking.strip()
        if assistant_message["content"] or "reasoning_content" in assistant_message:
            messages.append(assistant_message)
    raw_tools = row.get("tools") if isinstance(row.get("tools"), list) else []
    prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else _prompt_from_messages(messages)
    metadata = dict(row.get("metadata")) if isinstance(row.get("metadata"), dict) else {}
    if isinstance(row.get("model"), str) and row.get("model"):
        metadata.setdefault("model", row["model"])
    if isinstance(row.get("usage"), dict) and row.get("usage"):
        metadata.setdefault("usage", row["usage"])
    metadata.setdefault("source_file", source_file.name)
    metadata.setdefault("source_line", row_index)
    metadata.setdefault(
        "trace_type",
        "chat" if any(key in row for key in ("system", "thinking", "response", "model")) and not raw_tools else "structured",
    )
    tools = _merge_tools_with_builtin_trace_tools(raw_tools, metadata.get("trace_type"))
    if first_message_timestamp:
        metadata.setdefault(FIRST_MESSAGE_TIMESTAMP_METADATA_KEY, first_message_timestamp)
    return TrainingExample(
        source_file=source_file,
        prompt=prompt,
        messages=messages,
        tools=tools,
        metadata=metadata,
    )


def convert_trace_to_training_example(trace_file: Path) -> TrainingExample:
    events = load_trace_file(trace_file)
    if events and all(_is_structured_training_row(event) for event in events):
        if len(events) != 1:
            raise ValueError(
                f"Structured training data file {trace_file} contains {len(events)} rows; use convert_traces_to_training_data or load_traces instead."
            )
        return _structured_training_example_from_row(trace_file, events[0], 1)
    trace_type = _detect_trace_type(events)
    if trace_type == "claude_code":
        return _convert_claude_code_trace_to_training_example(trace_file, events)
    if trace_type == "droid":
        return _convert_droid_trace_to_training_example(trace_file, events)
    if trace_type == "hermes":
        return _convert_hermes_trace_to_training_example(trace_file, events)
    if trace_type == "cursor":
        return _convert_cursor_trace_to_training_example(trace_file, events)
    if trace_type == "external_agent":
        return _convert_external_agent_trace_to_training_example(trace_file, events)
    if trace_type == "openclaw":
        return _convert_pi_trace_to_training_example(trace_file, events, trace_type="openclaw")
    if trace_type == "pi":
        return _convert_pi_trace_to_training_example(trace_file, events)
    return _convert_codex_trace_to_training_example(trace_file, events)


def _jsonl_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*.jsonl")
        if path.is_file() and not {"partials", "failures"}.intersection(path.relative_to(source).parts)
    )


def _convert_jsonl_file_to_training_rows(jsonl_file: Path, *, skip_invalid_lines: bool = False) -> list[dict[str, Any]]:
    rows = _load_trace_file(jsonl_file, skip_invalid_lines=skip_invalid_lines)
    if not rows:
        return []
    if all(_is_structured_training_row(row) for row in rows):
        return [
            _structured_training_example_from_row(jsonl_file, row, row_index).to_dict()
            for row_index, row in enumerate(rows, start=1)
        ]
    trace_type = _detect_trace_type(rows)
    if trace_type == "claude_code":
        return [_convert_claude_code_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "droid":
        return [_convert_droid_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "hermes":
        if all(_is_hermes_trace_row(row) for row in rows):
            return [
                _convert_hermes_trace_to_training_example(jsonl_file, [row]).to_dict()
                for row in rows
            ]
        if all(isinstance(row, dict) and isinstance(row.get("messages"), list) for row in rows):
            return [
                _convert_hermes_trace_to_training_example(jsonl_file, [row]).to_dict()
                for row in rows
            ]
        return [_convert_hermes_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "cursor":
        return [_convert_cursor_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "external_agent":
        return [_convert_external_agent_trace_to_training_example(jsonl_file, rows).to_dict()]
    if trace_type == "openclaw":
        return [_convert_pi_trace_to_training_example(jsonl_file, rows, trace_type="openclaw").to_dict()]
    if trace_type == "pi":
        return [_convert_pi_trace_to_training_example(jsonl_file, rows).to_dict()]
    return [_convert_codex_trace_to_training_example(jsonl_file, rows).to_dict()]


def convert_traces_to_training_data(traces_dir: Path | str, *, skip_invalid_lines: bool = False) -> list[dict[str, Any]]:
    source = Path(traces_dir) if not isinstance(traces_dir, Path) else traces_dir
    trace_files = _jsonl_files(source)
    rows: list[dict[str, Any]] = []
    for path in trace_files:
        rows.extend(_convert_jsonl_file_to_training_rows(path, skip_invalid_lines=skip_invalid_lines))
    return rows
