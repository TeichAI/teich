#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from jinja2 import Environment
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_local_symbols() -> tuple[Any, ...]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    for module_name in list(sys.modules):
        if module_name == "teich" or module_name.startswith("teich."):
            sys.modules.pop(module_name, None)
    teich_module = importlib.import_module("teich")
    formatter_module = importlib.import_module("teich.formatter")
    return (
        teich_module.load_traces,
        formatter_module._build_preview,
        formatter_module._chat_template_looks_like_gemma,
        formatter_module._chat_template_supports_assistant_mask,
        formatter_module._fast_mask_row,
        formatter_module._gemma_mask_row,
        formatter_module._mask_row,
        formatter_module._offset_mask_row,
        formatter_module._render_chat,
        formatter_module._render_chat_with_generation_prompt,
        formatter_module._resolve_chat_template_renderer,
        formatter_module._resolve_assistant_prompt_prefixes,
        formatter_module._resolve_effective_max_length,
        formatter_module._resolve_text_tokenizer,
        formatter_module._supports_offsets,
        formatter_module._validate_chat_template_kwargs,
    )


(
    load_traces,
    _build_preview,
    _chat_template_looks_like_gemma,
    _chat_template_supports_assistant_mask,
    _fast_mask_row,
    _gemma_mask_row,
    _mask_row,
    _offset_mask_row,
    _render_chat,
    _render_chat_with_generation_prompt,
    _resolve_chat_template_renderer,
    _resolve_assistant_prompt_prefixes,
    _resolve_effective_max_length,
    _resolve_text_tokenizer,
    _supports_offsets,
    _validate_chat_template_kwargs,
) = _load_local_symbols()

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "ibm-granite/granite-4.1-3b",
    "ibm-granite/granite-4.1-8b",
    "ibm-granite/granite-4.1-30b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=_default_source())
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--preview-chars", type=int, default=1200)
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--artifact-dir", default="output/model_mask_audit_artifacts")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--no-builtins", action="store_true")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    return parser.parse_args()


def _default_source() -> str | None:
    trace_example = ROOT / "trace_example.jsonl"
    if trace_example.exists():
        return str(trace_example)
    output_dir = ROOT / "output"
    if output_dir.exists():
        return str(output_dir)
    return None


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


def _builtin_rows() -> list[dict[str, Any]]:
    tools = _tool_schema()
    return [
        {
            "case": "builtin_simple_chat",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Summarize the repo."},
                {"role": "assistant", "content": "This repo turns traces into training data.", "reasoning_content": "Inspect the package purpose."},
            ],
            "tools": [],
            "metadata": {"trace_type": "builtin"},
        },
        {
            "case": "builtin_tool_call",
            "messages": [
                {"role": "user", "content": "List the files."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need repo contents first.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {"command": "dir"}},
                        }
                    ],
                },
            ],
            "tools": tools,
            "metadata": {"trace_type": "builtin"},
        },
        {
            "case": "builtin_tool_roundtrip",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Inspect and summarize."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need one shell command.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {"command": "dir src"}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "teich"},
                {"role": "user", "content": "Now summarize."},
                {"role": "assistant", "content": "The repo exposes conversion and formatting code under src.", "reasoning_content": "Answer after tool output."},
            ],
            "tools": tools,
            "metadata": {"trace_type": "builtin"},
        },
    ]


def _load_source_rows(source: str | None, split: str, max_examples: int | None) -> list[dict[str, Any]]:
    if not source:
        return []
    dataset = load_traces(source, split=split, max_examples=max_examples)
    rows: list[dict[str, Any]] = []
    for index in range(dataset.num_rows):
        item = dataset[index]
        metadata = item.get("metadata") or {}
        label = metadata.get("trace_file") or metadata.get("session_id") or metadata.get("trace_type") or f"row_{index}"
        rows.append(
            {
                "case": f"source_{index}_{label}",
                "messages": item.get("messages") or [],
                "tools": item.get("tools") or [],
                "metadata": metadata,
            }
        )
    return rows


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _load_source_rows(args.source, args.split, args.max_examples)
    if not args.no_builtins:
        rows.extend(_builtin_rows())
    if not rows:
        raise ValueError("No audit rows available.")
    return rows


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return slug or "artifact"


def _decode_token(text_tokenizer: Any, token_id: int) -> str:
    try:
        return text_tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return text_tokenizer.decode([token_id])


def _build_delimited_preview(text_tokenizer: Any, input_ids: list[int], labels: list[int]) -> str:
    parts: list[str] = []
    current_is_supervised: bool | None = None
    for token_id, label in zip(input_ids, labels):
        is_supervised = label != -100
        if current_is_supervised is None:
            current_is_supervised = is_supervised
            parts.append("<<<SUPERVISED>>>" if is_supervised else "<<<MASKED>>>")
        elif current_is_supervised != is_supervised:
            parts.append("<<<END_SUPERVISED>>>" if current_is_supervised else "<<<END_MASKED>>>")
            current_is_supervised = is_supervised
            parts.append("<<<SUPERVISED>>>" if is_supervised else "<<<MASKED>>>")
        parts.append(_decode_token(text_tokenizer, token_id))
    if current_is_supervised is not None:
        parts.append("<<<END_SUPERVISED>>>" if current_is_supervised else "<<<END_MASKED>>>")
    return "".join(parts)


def _write_case_artifact(
    artifact_dir: Path,
    model_id: str,
    case: dict[str, Any],
    preview_chars: int,
) -> str:
    model_dir = artifact_dir / _safe_slug(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    case_file = model_dir / f"{_safe_slug(str(case['case']))}.txt"
    lines = [
        f"model: {model_id}",
        f"case: {case['case']}",
        f"status: {case['status']}",
        f"path: {case['path']}",
    ]
    if case["status"] == "ok":
        lines.extend(
            [
                f"tokens: {case['tokens']}",
                f"supervised_tokens: {case['supervised_tokens']}",
                f"masked_tokens: {case['masked_tokens']}",
                f"supervised_ratio: {case['supervised_ratio']}",
                "",
                "=== DELIMITED PREVIEW ===",
                _truncate(case["delimited_preview"], preview_chars),
                "",
                "=== FULL DELIMITED PREVIEW ===",
                case["delimited_preview"],
                "",
                "=== GENERATION PROMPT DETAILS ===",
                json.dumps(case.get("generation_prompt", {}), indent=2, ensure_ascii=False),
                "",
                "=== ASSISTANT PROMPT PREFIXES ===",
                json.dumps(case.get("assistant_prompt_prefixes", []), indent=2, ensure_ascii=False),
                "",
                "=== RAW FORMATTED TEXT ===",
                case.get("text", ""),
                "",
                "=== ANSI PREVIEW ===",
                case["preview"],
            ]
        )
    else:
        lines.extend(["", "=== ERROR ===", str(case["error"])])
    case_file.write_text("\n".join(lines), encoding="utf-8")
    return str(case_file)


@dataclass
class _ManualTemplateRenderer:
    chat_template: str
    template_config: dict[str, Any]

    def __post_init__(self) -> None:
        self._environment = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
        self._template = self._environment.from_string(self.chat_template)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        if tokenize:
            raise TypeError("manual renderer only supports tokenize=False")
        render_context = {
            **self.template_config,
            **kwargs,
            "messages": messages,
            "tools": tools or [],
            "add_generation_prompt": add_generation_prompt,
        }
        rendered = self._template.render(**render_context)
        if not isinstance(rendered, str):
            raise TypeError("manual chat template render must return a string")
        return rendered


class _ManualTextTokenizer:
    def __init__(self, tokenizer_path: str, tokenizer_config: dict[str, Any]) -> None:
        self._tokenizer = RawTokenizer.from_file(tokenizer_path)
        self.chat_template = tokenizer_config.get("chat_template")
        self.model_max_length = tokenizer_config.get("model_max_length")

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        result: dict[str, Any] = {"input_ids": list(encoding.ids)}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(encoding.ids)
        if return_offsets_mapping:
            result["offset_mapping"] = [tuple(offset) for offset in encoding.offsets]
        return result

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _load_manual_gemma_renderer(model_id: str, args: argparse.Namespace) -> tuple[Any, Any, Any, str] | None:
    if not model_id.startswith("google/gemma-4-"):
        return None
    download_kwargs = {
        "repo_id": model_id,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
    }
    tokenizer_config_path = hf_hub_download(filename="tokenizer_config.json", **download_kwargs)
    tokenizer_path = hf_hub_download(filename="tokenizer.json", **download_kwargs)
    try:
        chat_template_path = hf_hub_download(filename="chat_template.jinja", **download_kwargs)
        chat_template = Path(chat_template_path).read_text(encoding="utf-8")
    except Exception:
        tokenizer_config_for_template = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))
        chat_template = tokenizer_config_for_template.get("chat_template")
        if not isinstance(chat_template, str) or not chat_template:
            raise
    tokenizer_config = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))
    manual_tokenizer = _ManualTextTokenizer(tokenizer_path, tokenizer_config)
    renderer = _ManualTemplateRenderer(chat_template=chat_template, template_config=tokenizer_config)
    return manual_tokenizer, manual_tokenizer, renderer, "Manual Gemma template/tokenizer"


def _load_renderer(model_id: str, args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    load_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    try:
        renderer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        text_tokenizer = _resolve_text_tokenizer(renderer)
        return renderer, text_tokenizer, renderer, "AutoTokenizer"
    except Exception as tokenizer_exc:
        try:
            manual = _load_manual_gemma_renderer(model_id, args)
            if manual is not None:
                return manual
        except Exception:
            pass
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
            text_tokenizer = _resolve_text_tokenizer(processor)
            renderer = _resolve_chat_template_renderer(processor, text_tokenizer)
            return processor, text_tokenizer, renderer, f"AutoProcessor (tokenizer fallback: {tokenizer_exc})"
        except Exception:
            raise tokenizer_exc


def _generation_prompt_details(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        base_render = _render_chat(renderer, messages, tools, template_kwargs)
        prompt_render = _render_chat_with_generation_prompt(renderer, messages, tools, template_kwargs)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    starts_with_base = prompt_render.startswith(base_render)
    prompt_suffix = prompt_render[len(base_render) :] if starts_with_base else ""
    return {
        "status": "ok",
        "starts_with_base": starts_with_base,
        "base_length": len(base_render),
        "prompt_length": len(prompt_render),
        "suffix": prompt_suffix,
        "base_render": base_render,
        "prompt_render": prompt_render,
    }


def _mask_with_path(
    row: dict[str, Any],
    renderer: Any,
    text_tokenizer: Any,
    template_kwargs: dict[str, Any],
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]],
    effective_max_length: int | None,
) -> tuple[str, dict[str, Any]]:
    messages = row.get("messages") or []
    tools = row.get("tools") or []
    attempts = [
        (
            "fast",
            lambda: _fast_mask_row(
                renderer,
                text_tokenizer,
                messages,
                tools,
                template_kwargs,
                effective_max_length,
                True,
            ),
        ),
        (
            "gemma",
            lambda: _gemma_mask_row(
                renderer,
                text_tokenizer,
                messages,
                tools,
                template_kwargs,
                True,
                effective_max_length,
                True,
            ),
        ),
        (
            "offset",
            lambda: _offset_mask_row(
                renderer,
                text_tokenizer,
                messages,
                tools,
                template_kwargs,
                assistant_prompt_prefix_cache,
                True,
                effective_max_length,
                True,
                False,
            ),
        ),
    ]
    for path_name, attempt in attempts:
        masked = attempt()
        if masked is not None:
            return path_name, masked
    return (
        "fallback",
        _mask_row(
            {"messages": messages, "tools": tools},
            renderer,
            text_tokenizer,
            "messages",
            "tools",
            template_kwargs,
            assistant_prompt_prefix_cache,
            True,
            effective_max_length,
            True,
            False,
        ),
    )


def _audit_model(model_id: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, text_tokenizer, renderer, loader = _load_renderer(model_id, args)
    template_kwargs = _validate_chat_template_kwargs(json.loads(args.chat_template_kwargs))
    effective_max_length = _resolve_effective_max_length(args.max_length, text_tokenizer)
    assistant_prompt_prefix_cache: dict[str, tuple[str, ...]] = {}
    cases: list[dict[str, Any]] = []
    path_counts: Counter[str] = Counter()
    artifact_dir = Path(args.artifact_dir)

    for row in rows:
        case_name = row.get("case") or "unnamed"
        messages = row.get("messages") or []
        tools = row.get("tools") or []
        generation_prompt = _generation_prompt_details(renderer, messages, tools, template_kwargs)
        assistant_prompt_prefixes = list(
            _resolve_assistant_prompt_prefixes(
                renderer,
                messages,
                tools,
                template_kwargs,
                assistant_prompt_prefix_cache,
            )
        )
        try:
            path_name, masked = _mask_with_path(
                row,
                renderer,
                text_tokenizer,
                template_kwargs,
                assistant_prompt_prefix_cache,
                effective_max_length,
            )
            if args.skip_previews:
                preview = ""
                delimited_preview = ""
            else:
                preview = _build_preview(text_tokenizer, masked["input_ids"], masked["labels"])
                delimited_preview = _build_delimited_preview(text_tokenizer, masked["input_ids"], masked["labels"])
            labels = masked["labels"]
            supervised_tokens = sum(1 for label in labels if label != -100)
            masked_tokens = len(labels) - supervised_tokens
            result = {
                "case": case_name,
                "status": "ok",
                "path": path_name,
                "tokens": len(labels),
                "supervised_tokens": supervised_tokens,
                "masked_tokens": masked_tokens,
                "supervised_ratio": round(supervised_tokens / len(labels), 4) if labels else 0.0,
                "text": masked.get("text", ""),
                "preview": preview,
                "delimited_preview": delimited_preview,
                "generation_prompt": generation_prompt,
                "assistant_prompt_prefixes": assistant_prompt_prefixes,
            }
            path_counts[path_name] += 1
        except Exception as exc:
            result = {
                "case": case_name,
                "status": "error",
                "path": "error",
                "error": str(exc),
                "generation_prompt": generation_prompt,
                "assistant_prompt_prefixes": assistant_prompt_prefixes,
            }
            path_counts["error"] += 1
        result["artifact_file"] = _write_case_artifact(artifact_dir, model_id, result, args.preview_chars)
        cases.append(result)

    return {
        "model": model_id,
        "tokenizer_class": type(tokenizer).__name__,
        "renderer_class": type(renderer).__name__,
        "loader": loader,
        "supports_offsets": _supports_offsets(text_tokenizer),
        "supports_assistant_mask": _chat_template_supports_assistant_mask(renderer),
        "looks_like_gemma": _chat_template_looks_like_gemma(renderer),
        "effective_max_length": effective_max_length,
        "path_counts": dict(path_counts),
        "cases": cases,
    }


def _render_model_report(console: Console, report: dict[str, Any], preview_chars: int) -> None:
    header = Table(show_header=False)
    header.add_column(style="cyan")
    header.add_column()
    header.add_row("model", str(report["model"]))
    header.add_row("tokenizer", str(report["tokenizer_class"]))
    header.add_row("renderer", str(report["renderer_class"]))
    header.add_row("loader", str(report["loader"]))
    header.add_row("offsets", str(report["supports_offsets"]))
    header.add_row("assistant_mask", str(report["supports_assistant_mask"]))
    header.add_row("gemma_like", str(report["looks_like_gemma"]))
    header.add_row("max_length", str(report["effective_max_length"]))
    header.add_row("paths", json.dumps(report["path_counts"], sort_keys=True))
    console.print(Panel(header, title=str(report["model"])))

    cases_table = Table()
    cases_table.add_column("case", style="cyan")
    cases_table.add_column("status")
    cases_table.add_column("path")
    cases_table.add_column("tokens", justify="right")
    cases_table.add_column("supervised", justify="right")
    cases_table.add_column("masked", justify="right")
    cases_table.add_column("ratio", justify="right")
    cases_table.add_column("artifact")
    for case in report["cases"]:
        if case["status"] == "ok":
            cases_table.add_row(
                str(case["case"]),
                str(case["status"]),
                str(case["path"]),
                str(case["tokens"]),
                str(case["supervised_tokens"]),
                str(case["masked_tokens"]),
                str(case["supervised_ratio"]),
                str(case["artifact_file"]),
            )
        else:
            cases_table.add_row(
                str(case["case"]),
                str(case["status"]),
                str(case["path"]),
                "-",
                "-",
                "-",
                "-",
                str(case["artifact_file"]),
            )
    console.print(cases_table)

    for case in report["cases"]:
        if case["status"] == "ok":
            console.print(
                Panel(
                    _truncate(case["delimited_preview"], preview_chars),
                    title=f"{report['model']} :: {case['case']} :: {case['path']}",
                )
            )
        else:
            console.print(Panel(str(case["error"]), title=f"{report['model']} :: {case['case']} :: error", border_style="red"))


def main() -> int:
    args = parse_args()
    console = Console()
    rows = _load_rows(args)
    reports: list[dict[str, Any]] = []
    summary = Table(title="model_mask_audit summary")
    summary.add_column("model", style="cyan")
    summary.add_column("ok", justify="right")
    summary.add_column("error", justify="right")
    summary.add_column("fast", justify="right")
    summary.add_column("gemma", justify="right")
    summary.add_column("offset", justify="right")
    summary.add_column("fallback", justify="right")

    for model_id in args.models:
        try:
            report = _audit_model(model_id, rows, args)
            reports.append(report)
            counts = Counter(case["path"] for case in report["cases"])
            ok_count = sum(1 for case in report["cases"] if case["status"] == "ok")
            error_count = sum(1 for case in report["cases"] if case["status"] == "error")
            summary.add_row(
                model_id,
                str(ok_count),
                str(error_count),
                str(counts.get("fast", 0)),
                str(counts.get("gemma", 0)),
                str(counts.get("offset", 0)),
                str(counts.get("fallback", 0)),
            )
            _render_model_report(console, report, args.preview_chars)
        except Exception as exc:
            reports.append({"model": model_id, "status": "error", "error": str(exc)})
            summary.add_row(model_id, "0", "1", "0", "0", "0", "0")
            console.print(Panel(str(exc), title=f"{model_id} :: load error", border_style="red"))

    console.print(summary)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
