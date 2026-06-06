from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Config, MCPConfig

MCP_PROTOCOL_VERSION = "2025-06-18"


CODEX_BUILTIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run shell commands in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                    "workdir": {"type": "string"},
                },
                "required": ["command"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                    "yield_time_ms": {"type": "integer"},
                    "max_output_tokens": {"type": "integer"},
                    "shell": {"type": "string"},
                    "login": {"type": "boolean"},
                    "tty": {"type": "boolean"},
                    "justification": {"type": "string"},
                    "prefix_rule": {"type": "array"},
                    "sandbox_permissions": {"type": "string"},
                },
                "required": ["cmd"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified patch to files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string"},
                },
                "required": ["patch"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Update the current task plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "plan": {"type": "array"},
                },
                "required": ["plan"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Inspect a local image file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": True,
            },
        },
    },
]


PI_BUILTIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run shell commands in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cmd": {"type": "string"},
                    "cwd": {"type": "string"},
                    "description": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "anyOf": [
                    {"required": ["command"]},
                    {"required": ["cmd"]},
                ],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "file_path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "anyOf": [
                    {"required": ["path"]},
                    {"required": ["file_path"]},
                ],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write file contents in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["content"],
                "anyOf": [
                    {"required": ["path"]},
                    {"required": ["file_path"]},
                ],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write file contents in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit file contents in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "file_path": {"type": "string"},
                    "edits": {"type": "array"},
                },
                "required": ["edits"],
                "anyOf": [
                    {"required": ["path"]},
                    {"required": ["file_path"]},
                ],
                "additionalProperties": True,
            },
        },
    },
]


@dataclass
class ToolCallValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("Tool-call validation failed:\n" + "\n".join(f"- {error}" for error in self.errors))


def _tool_identity(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    return name if isinstance(name, str) else ""


def _dedupe_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = _tool_identity(tool)
        if name:
            by_name[name] = tool
    return [by_name[name] for name in sorted(by_name)]


def _tool_parameter_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    parameters = function.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _parse_tool_call_arguments(arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _json_schema_type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_argument_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
) -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _json_schema_type_matches(value, schema_type):
        errors.append(f"{path}: expected {schema_type}, got {type(value).__name__}")
        return errors
    if isinstance(schema_type, list) and not any(
        isinstance(item, str) and _json_schema_type_matches(value, item)
        for item in schema_type
    ):
        expected = "|".join(item for item in schema_type if isinstance(item, str))
        errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
        return errors
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value {value!r} is not in enum {enum!r}")
    if not isinstance(value, dict):
        return errors
    for combinator in ("anyOf", "oneOf"):
        options = schema.get(combinator)
        if not isinstance(options, list):
            continue
        option_errors = [
            _validate_argument_schema(value, option, path=path)
            for option in options
            if isinstance(option, dict)
        ]
        if option_errors and not any(not branch_errors for branch_errors in option_errors):
            errors.extend(min(option_errors, key=len))
    required = schema.get("required")
    required_names = {name for name in required if isinstance(name, str)} if isinstance(required, list) else set()
    if isinstance(required, list):
        for required_name in required:
            if isinstance(required_name, str) and required_name not in value:
                errors.append(f"{path}: missing required argument {required_name!r}")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, item in value.items():
            property_schema = properties.get(name)
            if item is None and name not in required_names:
                continue
            if isinstance(property_schema, dict):
                errors.extend(_validate_argument_schema(item, property_schema, path=f"{path}.{name}"))
        if schema.get("additionalProperties", True) is False:
            extra_keys = sorted(key for key in value if key not in properties)
            for key in extra_keys:
                errors.append(f"{path}: unexpected argument {key!r}")
    return errors


def _row_id(row: dict[str, Any], explicit_row_id: Any) -> Any:
    if explicit_row_id is not None:
        return explicit_row_id
    for key in ("row_id", "id", "source_key", "raw_index"):
        value = row.get(key)
        if value is not None:
            return value
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("session_id", "id", "source_file"):
            value = metadata.get(key)
            if value is not None:
                return value
    return None


def validate_tool_calls(
    row: dict[str, Any],
    *,
    row_id: Any = None,
) -> ToolCallValidationReport:
    if not isinstance(row, dict):
        return ToolCallValidationReport(ok=False, errors=["row must be a mapping"])
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ToolCallValidationReport(ok=False, errors=["row is missing list-valued 'messages'"])
    tools = row.get("tools") or []
    if not isinstance(tools, list):
        return ToolCallValidationReport(ok=False, errors=["row has non-list 'tools'"])

    tool_schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = _tool_identity(tool)
        if name:
            tool_schemas[name] = _tool_parameter_schema(tool)

    errors: list[str] = []
    label = _row_id(row, row_id)
    row_prefix = f"row {label}: " if label is not None else ""
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"assistant", "model"}:
            continue
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            errors.append(f"{row_prefix}message {message_index}: tool_calls must be a list")
            continue
        for tool_call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                errors.append(f"{row_prefix}message {message_index} tool_call {tool_call_index}: tool call must be an object")
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                errors.append(f"{row_prefix}message {message_index} tool_call {tool_call_index}: missing function object")
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"{row_prefix}message {message_index} tool_call {tool_call_index}: missing function name")
                continue
            if name not in tool_schemas:
                errors.append(f"{row_prefix}message {message_index} tool_call {tool_call_index}: undeclared tool {name!r}")
                continue
            arguments = _parse_tool_call_arguments(function.get("arguments"))
            if not isinstance(arguments, dict):
                errors.append(f"{row_prefix}message {message_index} tool_call {tool_call_index}: arguments must be a JSON object")
                continue
            errors.extend(
                f"{row_prefix}message {message_index} tool_call {tool_call_index} {error}"
                for error in _validate_argument_schema(arguments, tool_schemas[name], path=name)
            )
    return ToolCallValidationReport(ok=not errors, errors=errors)


def _mcp_tool_to_openai_tool(server_name: str, tool: dict[str, Any]) -> dict[str, Any] | None:
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    schema = tool.get("inputSchema") or tool.get("input_schema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
    function: dict[str, Any] = {
        "name": f"{server_name}.{name.strip()}",
        "parameters": schema,
    }
    description = tool.get("description") or tool.get("title")
    if isinstance(description, str) and description.strip():
        function["description"] = description.strip()
    return {"type": "function", "function": function}


def _json_rpc_request(request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        request["params"] = params
    return request


def _raise_json_rpc_error(response: dict[str, Any], server_name: str) -> None:
    error = response.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            raise RuntimeError(f"MCP server '{server_name}' returned JSON-RPC error: {message.strip()}")
        raise RuntimeError(f"MCP server '{server_name}' returned JSON-RPC error: {json.dumps(error, ensure_ascii=False)}")


def _stdio_mcp_request(process: subprocess.Popen[str], request: dict[str, Any], server_name: str) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError(f"MCP server '{server_name}' did not expose stdio streams.")
    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"MCP server '{server_name}' closed stdout before responding.")
        response = json.loads(line)
        if isinstance(response, dict) and response.get("id") == request.get("id"):
            _raise_json_rpc_error(response, server_name)
            return response


def _stdio_mcp_notify(process: subprocess.Popen[str], method: str, params: dict[str, Any] | None = None) -> None:
    if process.stdin is None:
        return
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _mcp_environment(mcp: MCPConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(mcp.env)
    for name in mcp.env_vars:
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _snapshot_stdio_mcp_tools(mcp: MCPConfig) -> list[dict[str, Any]]:
    if not mcp.command:
        return []
    process = subprocess.Popen(
        [mcp.command, *mcp.args],
        cwd=mcp.cwd or None,
        env=_mcp_environment(mcp),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        _stdio_mcp_request(
            process,
            _json_rpc_request(
                1,
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "teich", "version": "0.1.0"},
                },
            ),
            mcp.name,
        )
        _stdio_mcp_notify(process, "notifications/initialized")
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        request_id = 2
        while True:
            params = {"cursor": cursor} if cursor else None
            response = _stdio_mcp_request(process, _json_rpc_request(request_id, "tools/list", params), mcp.name)
            request_id += 1
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"MCP server '{mcp.name}' returned an invalid tools/list response.")
            page_tools = result.get("tools")
            if isinstance(page_tools, list):
                for tool in page_tools:
                    if isinstance(tool, dict):
                        normalized = _mcp_tool_to_openai_tool(mcp.name, tool)
                        if normalized is not None:
                            tools.append(normalized)
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return tools
            cursor = next_cursor
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _http_mcp_headers(mcp: MCPConfig) -> dict[str, str]:
    headers = {"content-type": "application/json", "accept": "application/json"}
    headers.update(mcp.http_headers)
    for header_name, env_name in mcp.env_http_headers.items():
        value = os.getenv(env_name)
        if value:
            headers[header_name] = value
    if mcp.bearer_token_env_var:
        token = os.getenv(mcp.bearer_token_env_var)
        if token:
            headers["authorization"] = f"Bearer {token}"
    return headers


def _http_mcp_request(mcp: MCPConfig, request: dict[str, Any]) -> dict[str, Any]:
    if not mcp.url:
        return {}
    http_request = Request(
        mcp.url,
        data=json.dumps(request).encode("utf-8"),
        headers=_http_mcp_headers(mcp),
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=mcp.startup_timeout_sec or 30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MCP server '{mcp.name}' returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"MCP server '{mcp.name}' request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"MCP server '{mcp.name}' returned a non-object JSON-RPC response.")
    _raise_json_rpc_error(payload, mcp.name)
    return payload


def _snapshot_http_mcp_tools(mcp: MCPConfig) -> list[dict[str, Any]]:
    _http_mcp_request(
        mcp,
        _json_rpc_request(
            1,
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "teich", "version": "0.1.0"},
            },
        ),
    )
    tools: list[dict[str, Any]] = []
    cursor: str | None = None
    request_id = 2
    while True:
        params = {"cursor": cursor} if cursor else None
        response = _http_mcp_request(mcp, _json_rpc_request(request_id, "tools/list", params))
        request_id += 1
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"MCP server '{mcp.name}' returned an invalid tools/list response.")
        page_tools = result.get("tools")
        if isinstance(page_tools, list):
            for tool in page_tools:
                if isinstance(tool, dict):
                    normalized = _mcp_tool_to_openai_tool(mcp.name, tool)
                    if normalized is not None:
                        tools.append(normalized)
        next_cursor = result.get("nextCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return tools
        cursor = next_cursor


def snapshot_mcp_tools(mcp: MCPConfig) -> list[dict[str, Any]]:
    if not mcp.enabled:
        return []
    tools = _snapshot_http_mcp_tools(mcp) if mcp.url else _snapshot_stdio_mcp_tools(mcp)
    enabled = set(mcp.enabled_tools)
    disabled = set(mcp.disabled_tools)
    filtered: list[dict[str, Any]] = []
    for tool in tools:
        name = _tool_identity(tool)
        short_name = name.rsplit(".", 1)[-1]
        if enabled and short_name not in enabled and name not in enabled:
            continue
        if short_name in disabled or name in disabled:
            continue
        filtered.append(tool)
    return filtered


def snapshot_configured_tools(config: Config) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    provider = config.get_agent_provider()
    if provider == "codex":
        tools.extend(CODEX_BUILTIN_TOOLS)
    elif provider == "pi":
        tools.extend(PI_BUILTIN_TOOLS)
    for mcp in config.mcp_servers:
        if not mcp.enabled:
            continue
        try:
            tools.extend(snapshot_mcp_tools(mcp))
        except Exception:
            if mcp.required:
                raise
    return _dedupe_tools(tools)


def write_tools_snapshot(destination: Path, tools: list[dict[str, Any]]) -> None:
    if tools:
        destination.write_text(json.dumps(tools, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif destination.exists():
        destination.unlink()
