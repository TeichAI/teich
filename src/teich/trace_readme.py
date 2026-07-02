from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .converter import NON_DATA_TRACE_DIR_NAMES, convert_traces_to_training_data

README_SAMPLE_MAX_CHARS = 4_000
README_SAMPLE_STRING_MAX_CHARS = 600
README_SAMPLE_MAX_ITEMS = 6
README_SAMPLE_MAX_DEPTH = 5
README_INLINE_TOOLS_MAX_CHARS = 80_000
TEICH_TRAINING_DOCS_URL = "https://github.com/TeichAI/teich/blob/main/docs/training.md"
TEICH_PREPARE_DOCS_URL = "https://github.com/TeichAI/teich/blob/main/docs/prepare-data.md"
EXTRACTION_PROVIDERS = {"claude", "codex", "cursor", "hermes", "pi"}

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DEFAULT_CARD_TEMPLATE = "dataset_card.md.j2"
# Frontmatter keys the card owns; a user's ``card_extra`` may not shadow them.
_RESERVED_CARD_KEYS = frozenset(
    {"pretty_name", "task_categories", "tags", "configs", "size_categories", "license"}
)
# Hugging Face dataset-card ``size_categories`` buckets, ascending by upper bound.
_SIZE_BUCKETS: tuple[tuple[int, str], ...] = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
    (10_000_000_000, "1B<n<10B"),
    (100_000_000_000, "10B<n<100B"),
    (1_000_000_000_000, "100B<n<1T"),
)


def size_category(row_count: int) -> str | None:
    """Map a dataset row count to its Hugging Face ``size_categories`` bucket.

    Returns ``None`` for an empty/unknown dataset (``row_count <= 0``) so the card
    omits the key rather than advertising a bogus size.
    """
    if row_count <= 0:
        return None
    for upper, label in _SIZE_BUCKETS:
        if row_count < upper:
            return label
    return "n>1T"


def normalize_extraction_provider(provider: str | None) -> str | None:
    if not isinstance(provider, str):
        return None
    normalized = provider.strip().lower().replace("_", "-")
    if normalized == "claude-code":
        normalized = "claude"
    return normalized if normalized in EXTRACTION_PROVIDERS else None


def extraction_readme_tags(provider: str) -> list[str]:
    normalized = normalize_extraction_provider(provider) or provider.strip().lower().replace("_", "-")
    ordered_tags = ["agent-traces", "format:agent-traces", normalized, "distillation", "teich"]
    tags: list[str] = []
    seen: set[str] = set()
    for tag in ordered_tags:
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def extraction_provider_from_existing_readme(readme_path: Path) -> str | None:
    try:
        text = readme_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"\bteich\s+extract\s+([A-Za-z0-9_-]+)\b", text)
    if match:
        return normalize_extraction_provider(match.group(1))
    return None


def extraction_provider_from_dataset_rows(traces_dir: Path) -> str | None:
    if not traces_dir.exists():
        return None
    for trace_file in sorted(traces_dir.rglob("*.jsonl")):
        if not trace_file.is_file():
            continue
        try:
            relative_parts = trace_file.relative_to(traces_dir).parts
        except ValueError:
            relative_parts = trace_file.parts
        if any(part in NON_DATA_TRACE_DIR_NAMES for part in relative_parts):
            continue
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    if line_index >= 100:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metadata = row.get("metadata") if isinstance(row, dict) else None
                    if not isinstance(metadata, dict):
                        metadata = {}
                    if metadata.get("source") == "cursor" or any(
                        key in metadata for key in ("cursor_scope", "cursor_table", "cursor_workspace_id")
                    ):
                        return "cursor"
        except OSError:
            continue
    return None


def extraction_provider_for_existing_dataset(traces_dir: Path) -> str | None:
    return extraction_provider_from_existing_readme(traces_dir / "README.md") or extraction_provider_from_dataset_rows(
        traces_dir
    )


def _path_is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _merge_tool_parameters(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    object_schemas = [schema for schema in schemas if isinstance(schema, dict) and schema]
    if not object_schemas:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    if len(object_schemas) == 1:
        return object_schemas[0]
    properties: dict[str, list[dict[str, Any]]] = {}
    required_sets: list[set[str]] = []
    additional_properties = False
    for schema in object_schemas:
        schema_properties = schema.get("properties")
        if isinstance(schema_properties, dict):
            for key, value in schema_properties.items():
                if isinstance(value, dict):
                    properties.setdefault(key, []).append(value)
        required = schema.get("required")
        if isinstance(required, list):
            required_sets.append({item for item in required if isinstance(item, str)})
        else:
            required_sets.append(set())
        if schema.get("additionalProperties", True) is not False:
            additional_properties = True
    merged_properties: dict[str, dict[str, Any]] = {}
    for key, values in sorted(properties.items()):
        unique_values: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            identity = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if identity in seen:
                continue
            seen.add(identity)
            unique_values.append(value)
        if len(unique_values) == 1:
            merged_properties[key] = unique_values[0]
        else:
            merged_properties[key] = {"anyOf": unique_values}
    merged: dict[str, Any] = {
        "type": "object",
        "properties": merged_properties,
        "additionalProperties": additional_properties,
    }
    if required_sets:
        required = sorted(set.intersection(*required_sets))
        if required:
            merged["required"] = required
    return merged


def _dataset_tools(trace_files: Iterable[Path]) -> list[dict[str, Any]]:
    merged_by_name: dict[str, dict[str, Any]] = {}
    for trace_file in trace_files:
        try:
            examples = convert_traces_to_training_data(trace_file, skip_invalid_lines=True)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for example in examples:
            tools = example.get("tools") if isinstance(example, dict) else None
            if not isinstance(tools, list):
                continue
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                function = tool.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                entry = merged_by_name.setdefault(name, {"type": "function", "function": {"name": name}})
                merged_function = entry["function"]
                if not isinstance(merged_function, dict):
                    continue
                description = function.get("description")
                if isinstance(description, str) and description and "description" not in merged_function:
                    merged_function["description"] = description
                schema = function.get("parameters")
                if isinstance(schema, dict):
                    existing_schema = merged_function.get("parameters")
                    schema_list = [existing_schema] if isinstance(existing_schema, dict) else []
                    schema_list.append(schema)
                    merged_function["parameters"] = _merge_tool_parameters(schema_list)
    return [merged_by_name[name] for name in sorted(merged_by_name)]


def _split_data_files(traces_dir: Path, excluded_dirs: Iterable[Path] = ()) -> list[tuple[str, str]]:
    """Dataset-card split -> file glob, reflecting the actual routing folders.

    When routed split folders (passed/failed/borderline) hold data, expose them as
    HF splits. Otherwise a single ``train`` split: the top-level ``*.jsonl`` files
    plus a recursive ``<dir>/**/*.jsonl`` for each data-bearing subdir that isn't a
    known non-data dir (partials/failures/bench/verification/...). Nested extractions
    (e.g. Cursor's project-relative transcripts) live only in subdirs, so a bare
    top-level ``*.jsonl`` would advertise an empty dataset while the card still counts
    those rows — HF's ``data_files`` config must reach them.

    ``excluded_dirs`` (e.g. a custom ``output.failures_dir`` under ``traces_dir``) are
    skipped too — they hold ignored/failed runs and their basename may not be in the
    reserved non-data set, so they must not be advertised as a train split.
    """
    routing = ("passed", "failed", "borderline")
    splits = [
        (name, f"{name}/*.jsonl")
        for name in routing
        if (traces_dir / name).is_dir() and any((traces_dir / name).glob("*.jsonl"))
    ]
    if splits:
        return splits
    traces_resolved = traces_dir.resolve()
    excluded_names = {
        excluded.resolve().relative_to(traces_resolved).parts[0]
        for excluded in excluded_dirs
        if excluded.resolve() != traces_resolved and _path_is_relative_to(excluded, traces_dir)
    }
    patterns = ["*.jsonl"]
    if traces_dir.is_dir():
        for child in sorted(traces_dir.iterdir()):
            if not child.is_dir() or child.name in NON_DATA_TRACE_DIR_NAMES or child.name in routing:
                continue
            if child.name in excluded_names:
                continue
            if any(child.rglob("*.jsonl")):
                patterns.append(f"{child.name}/**/*.jsonl")
    return [("train", pattern) for pattern in patterns]


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}... [truncated {omitted} chars]"


def _readme_sample_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= README_SAMPLE_MAX_DEPTH:
        return "[truncated: nested value]"
    if isinstance(value, str):
        return _truncate_text(value, README_SAMPLE_STRING_MAX_CHARS)
    if isinstance(value, list):
        items = [_readme_sample_value(item, depth=depth + 1) for item in value[:README_SAMPLE_MAX_ITEMS]]
        if len(value) > README_SAMPLE_MAX_ITEMS:
            items.append(f"[truncated: {len(value) - README_SAMPLE_MAX_ITEMS} items omitted]")
        return items
    if isinstance(value, dict):
        items = list(value.items())
        sampled = {
            str(key): _readme_sample_value(item, depth=depth + 1)
            for key, item in items[:README_SAMPLE_MAX_ITEMS]
        }
        if len(items) > README_SAMPLE_MAX_ITEMS:
            sampled["__truncated__"] = f"{len(items) - README_SAMPLE_MAX_ITEMS} keys omitted"
        return sampled
    return value


def _readme_sample_line(raw_line: str) -> str:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError:
        return json.dumps({"raw": _truncate_text(raw_line, README_SAMPLE_STRING_MAX_CHARS)}, ensure_ascii=False)
    return json.dumps(_readme_sample_value(value), ensure_ascii=False)


def _sample_lines(trace_files: Iterable[Path], sample_size: int = 1) -> list[str]:
    total_chars = 0
    samples: list[str] = []
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    if not line.strip():
                        continue
                    sample = _readme_sample_line(line)
                    projected_total = total_chars + len(sample) + (1 if samples else 0)
                    if projected_total > README_SAMPLE_MAX_CHARS:
                        return samples
                    samples.append(sample)
                    total_chars = projected_total
                    if len(samples) >= sample_size:
                        return samples
        except OSError:
            continue
    return samples


def _row_count(trace_files: Iterable[Path]) -> int:
    total = 0
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return total


def _sample_entry(trace_files: Iterable[Path]) -> dict[str, Any] | None:
    for trace_file in trace_files:
        try:
            with trace_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        return entry
                    break
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _is_structured_dataset(trace_files: Iterable[Path]) -> bool:
    sample_entry = _sample_entry(trace_files)
    return (
        isinstance(sample_entry, dict)
        and isinstance(sample_entry.get("messages"), list)
        and sample_entry.get("source") != "cli"
    )


def _is_agent_trace_row_dataset(trace_files: Iterable[Path]) -> bool:
    sample_entry = _sample_entry(trace_files)
    return isinstance(sample_entry, dict) and isinstance(sample_entry.get("traces"), list)


def _tools_json(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, indent=2, ensure_ascii=False)


def _should_externalize_tools(tools: list[dict[str, Any]]) -> bool:
    return bool(tools) and len(_tools_json(tools)) > README_INLINE_TOOLS_MAX_CHARS


def _tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _tools_details_block(tools: list[dict[str, Any]]) -> list[str]:
    tools_json = _tools_json(tools)
    if len(tools_json) > README_INLINE_TOOLS_MAX_CHARS:
        tool_names = sorted(name for tool in tools if (name := _tool_name(tool)))
        return [
            "## Tool schema snapshot",
            "",
            (
                "The complete dataset-level tool schema snapshot was written to `tools.json` because it is "
                "too large to embed safely in the Hugging Face dataset card."
            ),
            "",
            "<details>",
            "<summary>Tool names in snapshot</summary>",
            "",
            "```json",
            json.dumps(
                {
                    "tool_count": len(tools),
                    "tools": tool_names,
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "</details>",
            "",
        ]
    return [
        "## Tool schema snapshot",
        "",
        "<details>",
        "<summary>Training-ready tool schema snapshot</summary>",
        "",
        "```json",
        tools_json,
        "```",
        "",
        "</details>",
        "",
    ]


def _reward_stats(traces_dir: Path, trace_files: Iterable[Path]) -> dict[str, int] | None:
    """Summarize verifier outcomes for the dataset card.

    Two reward-labeled paths write sidecars in different shapes: the prompt-mode seed
    verifier writes ``verification/<stem>.json`` with a ``passed`` bool, while bench
    harvest writes ``metadata/<stem>.json`` with a ``split`` (passed/failed/borderline)
    and a numeric ``reward``. Scan both so the counts describe an all-prompts, all-bench,
    or mixed dataset. Only sidecars whose stem matches a live dataset row are counted (so
    a stale/orphaned sidecar can't inflate the count), and each stem is counted once.
    Returns None when nothing is verified.
    """
    live_stems = {path.stem for path in trace_files}
    counted: set[str] = set()
    passed = failed = borderline = numeric = 0

    def _numeric(reward: Any) -> bool:
        return isinstance(reward, (int, float)) and not isinstance(reward, bool)

    verification_dir = traces_dir / "verification"
    if verification_dir.is_dir():
        for sidecar in sorted(verification_dir.glob("*.json")):
            if sidecar.stem not in live_stems or sidecar.stem in counted:
                continue
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("passed"), bool):
                continue
            counted.add(sidecar.stem)
            passed += 1 if data["passed"] else 0
            failed += 0 if data["passed"] else 1
            numeric += 1 if _numeric(data.get("reward")) else 0

    metadata_dir = traces_dir / "metadata"
    if metadata_dir.is_dir():
        for sidecar in sorted(metadata_dir.glob("*.json")):
            if sidecar.stem not in live_stems or sidecar.stem in counted:
                continue
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            split = data.get("split")
            if split == "passed":
                passed += 1
            elif split == "failed":
                failed += 1
            elif split == "borderline":
                borderline += 1  # a partial score (0<r<1): verified, but neither pass nor fail
            else:
                continue  # unknown split: not a verified outcome
            counted.add(sidecar.stem)
            numeric += 1 if _numeric(data.get("reward")) else 0

    total = passed + failed + borderline
    if total == 0:
        return None
    return {"total": total, "passed": passed, "failed": failed, "borderline": borderline, "numeric": numeric}


def _sanitize_card_extra(card_extra: dict[str, Any] | None) -> dict[str, Any]:
    """Drop reserved frontmatter keys the card owns; keep user keys in order."""
    if not card_extra:
        return {}
    return {key: value for key, value in card_extra.items() if key not in _RESERVED_CARD_KEYS}


def _card_extra_yaml(card_extra: dict[str, Any] | None) -> str:
    """Serialize ``card_extra`` to YAML frontmatter lines (empty dict -> "")."""
    sanitized = _sanitize_card_extra(card_extra)
    if not sanitized:
        return ""
    import yaml

    return yaml.safe_dump(sanitized, sort_keys=False, default_flow_style=False)


def _render_card(context: dict[str, Any], template_path: Path | None = None) -> str:
    """Render the dataset card from ``context`` via Jinja2 (default or user template)."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    if template_path is not None:
        loader_dir = str(template_path.parent)
        template_name = template_path.name
    else:
        loader_dir = str(_TEMPLATE_DIR)
        template_name = _DEFAULT_CARD_TEMPLATE
    env = Environment(
        loader=FileSystemLoader(loader_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template(template_name)
    return template.render(**context)


def build_traces_readme(
    *,
    pretty_name: str,
    trace_files: list[Path],
    tags: list[str],
    model_id: str | None = None,
    repo_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    extraction_provider: str | None = None,
    reward_stats: dict[str, int] | None = None,
    data_files: list[tuple[str, str]] | None = None,
    license: str | None = None,
    card_extra: dict[str, Any] | None = None,
    readme_template: Path | None = None,
) -> str:
    structured_dataset = _is_structured_dataset(trace_files)
    agent_trace_rows = _is_agent_trace_row_dataset(trace_files)
    dataset_tools = tools if tools is not None else _dataset_tools(trace_files)
    dataset_reference = repo_id or "username/repo"
    row_count = _row_count(trace_files)
    sample_lines = _sample_lines(trace_files)
    sample_block = "\n".join(sample_lines)
    effective_tags = list(tags)
    routed = any(split != "train" for split, _ in data_files or [])
    if (reward_stats or routed) and "reward-labeled" not in effective_tags:
        effective_tags.append("reward-labeled")
    normalized_provider = (
        normalize_extraction_provider(extraction_provider)
        or extraction_provider.strip().lower().replace("_", "-")
        or "claude"
        if extraction_provider
        else None
    )
    size = size_category(row_count)
    is_row_dataset = structured_dataset or agent_trace_rows
    context: dict[str, Any] = {
        "pretty_name": pretty_name,
        "tags": effective_tags,
        "data_files": list(data_files or [("train", "**/*.jsonl")]),
        "license": license,
        "size_categories": [size] if size else None,
        "card_extra_yaml": _card_extra_yaml(card_extra),
        "model_id": model_id,
        "reward_stats": reward_stats,
        "dataset_tools": dataset_tools,
        "externalize_tools": _should_externalize_tools(dataset_tools),
        "tools_details_block": "\n".join(_tools_details_block(dataset_tools)).rstrip("\n")
        if dataset_tools
        else "",
        "extraction_provider": normalized_provider,
        "structured_dataset": structured_dataset,
        "agent_trace_rows": agent_trace_rows,
        "sample_block": sample_block,
        "dataset_reference": dataset_reference,
        "training_docs_url": TEICH_TRAINING_DOCS_URL,
        "prepare_docs_url": TEICH_PREPARE_DOCS_URL,
        "intro_line": (
            "This directory contains newline-delimited JSON training examples generated by teich."
            if structured_dataset
            else "This directory contains raw agent trace files generated by teich."
        ),
        "rows_line": f"Rows: {row_count}" if is_row_dataset else f"JSONL files: {len(trace_files)}",
    }
    return _render_card(context, readme_template)


def write_traces_readme(
    traces_dir: Path,
    *,
    pretty_name: str,
    tags: list[str],
    model_id: str | None = None,
    repo_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    excluded_dirs: list[Path] | None = None,
    extraction_provider: str | None = None,
    license: str | None = None,
    card_extra: dict[str, Any] | None = None,
    readme_template: Path | None = None,
) -> Path:
    trace_files = sorted(
        path
        for path in traces_dir.rglob("*.jsonl")
        if path.is_file()
        and not NON_DATA_TRACE_DIR_NAMES.intersection(path.relative_to(traces_dir).parts)
        and not any(_path_is_relative_to(path, excluded_dir) for excluded_dir in excluded_dirs or [])
    )
    dataset_tools = tools if tools is not None else _dataset_tools(trace_files)
    readme_path = traces_dir / "README.md"
    readme_path.write_text(
        build_traces_readme(
            pretty_name=pretty_name,
            trace_files=trace_files,
            tags=tags,
            model_id=model_id,
            repo_id=repo_id,
            tools=dataset_tools,
            extraction_provider=extraction_provider,
            reward_stats=_reward_stats(traces_dir, trace_files),
            data_files=_split_data_files(traces_dir, excluded_dirs or []),
            license=license,
            card_extra=card_extra,
            readme_template=readme_template,
        ),
        encoding="utf-8",
    )
    tools_path = traces_dir / "tools.json"
    if _should_externalize_tools(dataset_tools):
        tools_path.write_text(_tools_json(dataset_tools) + "\n", encoding="utf-8")
    elif tools_path.exists():
        tools_path.unlink()
    return readme_path
