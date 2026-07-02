"""CLI for teich."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from threading import Event, RLock, Thread
from typing import Any

import typer
from huggingface_hub import HfApi
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperCommand, TyperGroup

from .anonymize import anonymize_path
from .config import Config
from .converter import NON_DATA_TRACE_DIR_NAMES, convert_traces_to_training_data
from .extract import CURSOR_EXTRACTION_NOTICE, ExtractProvider, extract_local_sessions
from .runner import (
    ChatRunner,
    ClaudeCodeRunner,
    CodexRunner,
    HermesRunner,
    PiRunner,
    SessionProgressUpdate,
    TraceMetrics,
    completed_prompt_keys_from_outputs,
    prompt_inputs_for_run,
    unique_prompt_inputs_by_completion_key,
)
from .trace_readme import extraction_readme_tags, write_traces_readme
from .tool_schema import snapshot_configured_tools

console = Console()
ROOT_EXTRA_HELP = """\
Common workflows:
  teich init my-project
  teich generate -c config.yaml
  teich extract claude --model fable-5 --out data
  teich convert data --out teich-training.jsonl
  teich studio

For command-specific options and examples:
  teich generate --help
  teich extract --help
  teich convert --help
  teich studio --help
"""
EXTRACT_HELP = "Extract existing local agent sessions into a staged Teich dataset folder."
EXTRACT_EXTRA_HELP = """\
Common examples:
  teich extract claude --model fable-5 --out data
  teich extract claude --sessions-dir /path/to/.claude --out data
  teich extract claude --sessions-dir /path/to/.claude/projects --out data
  teich extract codex --sessions-dir /path/to/.codex --out data
  teich extract codex --sessions-dir /path/to/.codex/sessions --out data
  teich extract pi --sessions-dir /path/to/.pi --out data
  teich extract pi --sessions-dir /path/to/.pi/agent/sessions --out data
  teich extract pi --sessions-dir /path/to/.pi/sessions --out data
  teich extract hermes --sessions-dir /path/to/.hermes --out data
  teich extract hermes --sessions-dir /path/to/.hermes/state.db --out data
  teich extract cursor --sessions-dir /path/to/Cursor/User/workspaceStorage --out data
  teich extract cursor --sessions-dir /path/to/Cursor/User/globalStorage/state.vscdb --out data

Default stores:
  claude  CLAUDE_CONFIG_DIR/projects, CLAUDE_HOME/projects, ~/.claude/projects
  codex   CODEX_HOME/sessions, ~/.codex/sessions
  pi      PI_SESSION_DIR, PI_CODING_AGENT_DIR/sessions, ~/.pi/agent/sessions, ~/.pi/sessions
  hermes  HERMES_STATE_DB, HERMES_HOME/state.db, ~/.hermes/state.db
  cursor  CURSOR_WORKSPACE_STORAGE, CURSOR_GLOBAL_STORAGE_DB, Cursor/User/workspaceStorage,
          Cursor/User/globalStorage/state.vscdb, ~/.cursor-server/data/User/workspaceStorage

After extraction:
  teich convert data --out teich-training.jsonl
  This writes standalone OpenAI-style JSONL rows with prompt, messages, tools, and metadata for trainers that do not import Teich.
"""
GENERATE_EXTRA_HELP = """\
Typical generated project flow:
  teich init my-project
  cd my-project
  edit prompts.jsonl and config.yaml
  teich generate -c config.yaml
  teich generate -c config.yaml --resume

Outputs:
  output/    raw traces, converted JSONL rows, README.md, and sometimes tools.json
  sandbox/   workspace snapshots for agent providers
  failures/  failed or interrupted traces excluded from resume/upload

Use teich extract instead when you already have local Claude, Codex, Pi, Hermes, or Cursor sessions.
"""
CONVERT_EXTRA_HELP = """\
Examples:
  teich convert data --out teich-training.jsonl
  teich convert output/session.jsonl --out session.training.jsonl

Output format:
  Newline-delimited JSON rows with prompt, messages, tools, and metadata.
  This is useful when your trainer can consume standalone OpenAI-style message rows without importing Teich.

Use prepare_data() and mask_data() when you want tokenizer-specific rendering and exact response-only labels.
"""
ANONYMIZE_EXTRA_HELP = """\
Examples:
  teich anonymize output --output output_anonymized
  teich anonymize data --in-place

What is scrubbed:
  API keys, email addresses, and home-directory usernames.
  Embedded base64 media payloads are preserved for conversation context.

This is a best-effort safety pass. Review data before publishing or uploading.
"""
INIT_EXTRA_HELP = """\
Creates:
  config.yaml    generation settings
  prompts.jsonl  starter prompt rows

Next:
  edit config.yaml and prompts.jsonl
  set TEICH_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY
  teich generate -c config.yaml
"""
STUDIO_EXTRA_HELP = """\
Examples:
  teich studio
  teich studio ./my-project
  teich studio --host 127.0.0.1 --port 8420 --no-open

Studio edits the same config.yaml and prompts.jsonl files used by teich generate.
"""
POOL_EXTRA_HELP = """\
This namespace is reserved for a future Teich community pool upload backend.
For Hugging Face dataset uploads today, use teich generate or teich extract and confirm the upload prompt.
"""


class ExtraHelpCommand(TyperCommand):
    extra_help = ""

    def format_help(self, ctx: typer.Context, formatter: typer.HelpFormatter) -> None:
        super().format_help(ctx, formatter)
        if self.extra_help:
            formatter.write("\n" + self.extra_help)


class ExtraHelpGroup(TyperGroup):
    extra_help = ""

    def format_help(self, ctx: typer.Context, formatter: typer.HelpFormatter) -> None:
        super().format_help(ctx, formatter)
        if self.extra_help:
            formatter.write("\n" + self.extra_help)


def _help_command(name: str, extra_help: str) -> type[ExtraHelpCommand]:
    return type(name, (ExtraHelpCommand,), {"extra_help": extra_help})


def _help_group(name: str, extra_help: str) -> type[ExtraHelpGroup]:
    return type(name, (ExtraHelpGroup,), {"extra_help": extra_help})


app = typer.Typer(
    name="teich",
    help="Generate, extract, convert, prepare, and inspect agent training data.",
    no_args_is_help=True,
    cls=_help_group("TeichHelpGroup", ROOT_EXTRA_HELP),
)
pool_app = typer.Typer(
    help="Reserved Teich community pool commands.",
    cls=_help_group("PoolHelpGroup", POOL_EXTRA_HELP),
)
app.add_typer(pool_app, name="pool")
UPLOAD_IGNORE_PATTERNS = ["partials/**", "failures/**", "bench/**"]
UPLOAD_METADATA_PATTERNS = ["README.md", "tools.json"]


def _upload_ignore_patterns(cfg: Config) -> list[str]:
    patterns = list(UPLOAD_IGNORE_PATTERNS)
    try:
        relative_failures_dir = cfg.output.failures_dir.resolve().relative_to(cfg.output.traces_dir.resolve())
    except ValueError:
        return patterns
    if relative_failures_dir.parts:
        pattern = f"{relative_failures_dir.as_posix()}/**"
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def _has_non_empty_trace_outputs(traces_dir: Path) -> bool:
    if not traces_dir.exists():
        return False
    for path in traces_dir.rglob("*.jsonl"):
        if not path.is_file():
            continue
        try:
            if any(part in NON_DATA_TRACE_DIR_NAMES for part in path.relative_to(traces_dir).parts):
                continue
        except ValueError:
            pass
        if path.stat().st_size > 0:
            return True
    return False


def _existing_dataset_modes(traces_dir: Path) -> set[str]:
    """Which generation modes already wrote dataset rows into ``traces_dir``.

    bench rows are ``bench-<task>.jsonl``; anything else (incl. routed
    ``passed/``|``failed/`` traces) is prompts. Used to keep bench and prompts as
    separate, homogeneous datasets — a mixed dir is unpleasant to consume.
    """
    modes: set[str] = set()
    if not traces_dir.exists():
        return modes
    for path in traces_dir.rglob("*.jsonl"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if NON_DATA_TRACE_DIR_NAMES.intersection(path.relative_to(traces_dir).parts):
            continue
        modes.add("bench" if path.name.startswith("bench-") else "prompts")
    return modes


def _write_output_metadata(cfg: Config) -> Path:
    return write_traces_readme(
        cfg.output.traces_dir,
        pretty_name=cfg.output.pretty_name,
        tags=cfg.get_dataset_tags(),
        model_id=cfg.model.model,
        repo_id=cfg.get_publish_repo_id(),
        tools=snapshot_configured_tools(cfg),
        excluded_dirs=[cfg.output.failures_dir],
        license=cfg.output.license,
        card_extra=cfg.output.card_extra,
        readme_template=cfg.output.readme_template,
    )


def _new_jsonl_outputs(traces_dir: Path, started_at: datetime, results: list[Path]) -> list[Path]:
    output_paths = {path.resolve() for path in results if path.exists()}
    threshold = started_at.timestamp() - 1
    for path in traces_dir.glob("*.jsonl"):
        if path.is_file() and path.stat().st_mtime >= threshold:
            output_paths.add(path.resolve())
    return sorted(output_paths)


def _try_write_completed_output_metadata(cfg: Config) -> Path | None:
    if not _has_non_empty_trace_outputs(cfg.output.traces_dir):
        return None
    try:
        return _write_output_metadata(cfg)
    except Exception as exc:
        console.print(f"[yellow]Warning: failed to write README for completed outputs: {exc}[/yellow]")
        return None


def _upload_dataset_folder(
    *,
    folder_path: Path,
    repo_id: str,
    hf_token: str | None,
    private: bool,
    ignore_patterns: list[str],
) -> str:
    api = HfApi(token=hf_token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    delete_file = getattr(api, "delete_file", None)
    if callable(delete_file):
        try:
            delete_file(
                path_in_repo="tools.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Remove legacy teich tools snapshot",
            )
        except Exception:
            pass
    upload_large_folder = getattr(api, "upload_large_folder", None)
    if callable(upload_large_folder):
        metadata_patterns = [pattern for pattern in UPLOAD_METADATA_PATTERNS if (folder_path / pattern).is_file()]
        if metadata_patterns:
            api.upload_folder(
                folder_path=str(folder_path),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Upload teich dataset metadata",
                allow_patterns=metadata_patterns,
                ignore_patterns=ignore_patterns,
            )
        upload_large_folder(
            repo_id=repo_id,
            folder_path=str(folder_path),
            repo_type="dataset",
            private=private,
            ignore_patterns=list(dict.fromkeys([*ignore_patterns, *UPLOAD_METADATA_PATTERNS])),
        )
    else:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload teich dataset output",
            ignore_patterns=ignore_patterns,
        )
    return str(repo_url)


def _publish_dataset_to_hub(cfg: Config) -> str:
    repo_id = cfg.get_publish_repo_id()
    if not repo_id:
        raise ValueError("No publish.repo_id configured")
    return _upload_dataset_folder(
        folder_path=cfg.output.traces_dir,
        repo_id=repo_id,
        hf_token=cfg.get_hf_token(),
        private=cfg.publish.private,
        ignore_patterns=_upload_ignore_patterns(cfg),
    )


def _prompt_publish_completed_outputs(cfg: Config) -> str | None:
    repo_id = cfg.get_publish_repo_id()
    if not repo_id or not _has_non_empty_trace_outputs(cfg.output.traces_dir):
        return None
    should_publish = typer.confirm(
        f"Upload successful traces to Hugging Face dataset {repo_id}?",
        default=False,
    )
    if not should_publish:
        console.print("[yellow]Skipping Hugging Face upload for completed outputs.[/yellow]")
        return None
    repo_url = _publish_dataset_to_hub(cfg)
    console.print(f"[green]Published completed outputs: {repo_url}[/green]")
    return repo_url


def _print_interrupted_upload_hint(cfg: Config) -> None:
    repo_id = cfg.get_publish_repo_id()
    if not repo_id:
        return
    private_flag = " --private" if cfg.publish.private else ""
    exclude_flags = " ".join(f'--exclude "{pattern}"' for pattern in _upload_ignore_patterns(cfg))
    upload_command = f"hf upload {repo_id} . . --repo-type dataset {exclude_flags}{private_flag}"
    console.print("[yellow]Skipping Hugging Face upload for interrupted run.[/yellow]")
    console.print("[cyan]To upload the completed outputs later, run this from the output folder:[/cyan]")
    console.print(f"  {upload_command}", soft_wrap=True)


def _extract_dataset_config(
    provider: ExtractProvider,
    output: Path,
    *,
    model_filter: str | None = None,
    repo_id: str | None = None,
    hf_token: str | None = None,
    private: bool = False,
) -> Config:
    model_id = model_filter or f"{provider}-local-sessions"
    pretty_model = model_filter or provider
    return Config.model_validate(
        {
            "agent": {"provider": provider},
            "model": {"model": model_id},
            "output": {
                "traces_dir": output,
                "pretty_name": f"{pretty_model} Agent Traces",
            },
            "publish": {
                "repo_id": repo_id,
                "hf_token": hf_token,
                "private": private,
            },
        }
    )


def _write_extract_readme(
    provider: ExtractProvider,
    output: Path,
    *,
    model_filter: str | None = None,
    repo_id: str | None = None,
) -> Path:
    cfg = _extract_dataset_config(provider, output, model_filter=model_filter, repo_id=repo_id)
    return write_traces_readme(
        cfg.output.traces_dir,
        pretty_name=cfg.output.pretty_name,
        tags=extraction_readme_tags(provider),
        model_id=None,
        repo_id=cfg.get_publish_repo_id(),
        extraction_provider=provider,
        license=cfg.output.license,
        card_extra=cfg.output.card_extra,
        readme_template=cfg.output.readme_template,
    )


def _prompt_huggingface_upload(
    provider: ExtractProvider,
    output: Path,
    *,
    model_filter: str | None = None,
    private: bool = False,
) -> str | None:
    should_upload = typer.confirm("Would you like to upload to Hugging Face?", default=False)
    if not should_upload:
        console.print("[yellow]Skipping Hugging Face upload.[/yellow]")
        return None
    repo_id = typer.prompt("Hugging Face dataset repo id (owner/name)").strip()
    try:
        cfg = _extract_dataset_config(provider, output, model_filter=model_filter, repo_id=repo_id, private=private)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not cfg.get_publish_repo_id():
        console.print("[red]Hugging Face dataset repo id is required.[/red]")
        raise typer.Exit(1)
    hf_token = cfg.get_hf_token()
    if not hf_token:
        hf_token = typer.prompt("HF_TOKEN", hide_input=True).strip()
        cfg = _extract_dataset_config(
            provider,
            output,
            model_filter=model_filter,
            repo_id=repo_id,
            hf_token=hf_token,
            private=private,
        )
        if not cfg.get_hf_token():
            console.print("[red]HF_TOKEN is required to upload to Hugging Face.[/red]")
            raise typer.Exit(1)
    readme_path = _write_extract_readme(provider, output, model_filter=model_filter, repo_id=repo_id)
    console.print(f"[green]Updated README:[/green] {readme_path}")
    repo_url = _upload_dataset_folder(
        folder_path=cfg.output.traces_dir,
        repo_id=repo_id,
        hf_token=cfg.get_hf_token(),
        private=cfg.publish.private,
        ignore_patterns=UPLOAD_IGNORE_PATTERNS,
    )
    console.print(f"[green]Published dataset:[/green] {repo_url}")
    return repo_url


def _run_extract_command(
    provider: ExtractProvider,
    output: Path,
    sessions_dir: list[Path] | None,
    *,
    model_filter: str | None = None,
    skip_anonymize: bool = False,
    private: bool = False,
) -> None:
    console.print(Panel.fit("Teich Extract", style="bold blue"))
    if provider == "cursor":
        console.print(f"[yellow]{CURSOR_EXTRACTION_NOTICE}[/yellow]", soft_wrap=True)
    result = extract_local_sessions(
        provider,
        output_dir=output,
        sources=sessions_dir,
        model_filter=model_filter,
        clear_destination=True,
    )
    if not result.source_paths:
        console.print(f"[red]No local {provider} session folders found.[/red]")
        console.print("[yellow]Pass one or more explicit folders with --sessions-dir.[/yellow]")
        raise typer.Exit(1)
    if not result.copied_files:
        model_clause = f" matching model '{model_filter}'" if model_filter else ""
        console.print(f"[yellow]Found {len(result.source_paths)} source path(s), but no {provider} sessions{model_clause} were extracted.[/yellow]")
        for source in result.source_paths:
            console.print(f"  - {source}")
        raise typer.Exit(1)
    stale_readme_path = output / "README.md"
    if stale_readme_path.exists() and stale_readme_path.is_file():
        stale_readme_path.unlink()
    anonymize_report = None if skip_anonymize else anonymize_path(output, output, in_place=True)
    readme_path = _write_extract_readme(provider, output, model_filter=model_filter)
    extracted_message = f"Extracted {result.count} {provider} trace{'s' if result.count != 1 else ''}"
    if model_filter:
        extracted_message += f" with {model_filter}"
    console.print(f"[green]{extracted_message}[/green]", soft_wrap=True)
    if anonymize_report is None:
        console.print(
            "[yellow]Skipped anonymization because --no-anon was passed. Review the data before sharing or uploading.[/yellow]",
            soft_wrap=True,
        )
    else:
        totals = anonymize_report.totals
        console.print(
            "[cyan]Automatically scrambled[/cyan] "
            f"{totals.get('api_key', 0)} API keys, "
            f"{totals.get('email', 0)} email addresses, and "
            f"{totals.get('username', 0)} username references",
            soft_wrap=True,
        )
    console.print(f"[green]Wrote README:[/green] {readme_path}", soft_wrap=True)
    console.print(f"[cyan]Data was written to[/cyan] {output}", soft_wrap=True)
    for source in result.source_paths:
        console.print(f"[dim]source: {source}[/dim]")
    _prompt_huggingface_upload(provider, output, model_filter=model_filter, private=private)


def _extract_sources_option() -> list[Path] | None:
    return typer.Option(
        None,
        "--sessions-dir",
        "-s",
        help=(
            "Explicit agent root, native session folder, state.db file, or JSONL file. "
            "Examples: .claude, .claude/projects, .codex, .codex/sessions, .pi, "
            ".pi/agent/sessions, .pi/sessions, .hermes, .hermes/state.db, "
            "Cursor/User/workspaceStorage, Cursor/User/globalStorage/state.vscdb. Can be passed more than once."
        ),
    )


def _extract_output_option() -> Path:
    return typer.Option(Path("data"), "--output", "--out", "-o", help="Output directory root")


def _extract_model_option() -> str | None:
    return typer.Option(None, "--model", "-m", help="Only extract traces whose model metadata contains this value.")


def _extract_no_anonymize_option() -> bool:
    return typer.Option(False, "--no-anon", "--no-anonymize", help="Skip automatic anonymization of extracted traces.")


@app.command(
    help=EXTRACT_HELP,
    short_help="Extract existing local agent sessions.",
    no_args_is_help=True,
    cls=_help_command("ExtractHelpCommand", EXTRACT_EXTRA_HELP),
)
def extract(
    provider: ExtractProvider = typer.Argument(
        ...,
        metavar="PROVIDER",
        help="Provider to extract: claude, codex, cursor, pi, or hermes.",
    ),
    output: Path = _extract_output_option(),
    sessions_dir: list[Path] | None = _extract_sources_option(),
    model: str | None = _extract_model_option(),
    no_anon: bool = _extract_no_anonymize_option(),
    private: bool = typer.Option(False, "--private", help="Create the Hugging Face dataset as private if upload is confirmed."),
) -> None:
    """Extract local sessions for one provider."""
    _run_extract_command(provider, output, sessions_dir, model_filter=model, skip_anonymize=no_anon, private=private)


@app.command(
    help="Copy trace files and scrub common secrets/user identifiers.",
    short_help="Anonymize trace files.",
    cls=_help_command("AnonymizeHelpCommand", ANONYMIZE_EXTRA_HELP),
)
def anonymize(
    input_path: Path = typer.Argument(Path("output"), help="Trace file or directory to anonymize"),
    output: Path = typer.Option(
        Path("output_anonymized"),
        "--output",
        "-o",
        help="Anonymized output path. Ignored when --in-place is set.",
    ),
    in_place: bool = typer.Option(False, "--in-place", help="Overwrite files in the input path"),
) -> None:
    """Replace emails, home-directory usernames, and API keys with deterministic dummy values."""
    try:
        report = anonymize_path(input_path, output, in_place=in_place)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    totals = report.totals
    destination = report.input_path if in_place else report.output_path
    console.print(f"[green]Anonymized {report.files_changed}/{len(report.files)} file(s) into {destination}[/green]")
    if totals:
        console.print(
            "[cyan]Replacements:[/cyan] "
            + " ".join(f"{key}={value}" for key, value in sorted(totals.items()))
        )
    else:
        console.print("[yellow]No emails, usernames, or API keys were detected.[/yellow]")


@app.command(
    help="Convert raw or extracted traces into standalone OpenAI-style training JSONL rows.",
    short_help="Convert traces to training JSONL.",
    cls=_help_command("ConvertHelpCommand", CONVERT_EXTRA_HELP),
)
def convert(
    input_path: Path = typer.Argument(..., help="Raw trace JSONL file or folder to convert"),
    output: Path = typer.Option(
        Path("teich.jsonl"),
        "--output",
        "--out",
        "-o",
        help="Output JSONL file containing standalone OpenAI-style training rows",
    ),
) -> None:
    """Convert raw or extracted traces into standalone OpenAI-style training JSONL rows."""
    if not input_path.exists():
        console.print(f"[red]Input path not found: {input_path}[/red]")
        raise typer.Exit(1)
    if input_path.resolve() == output.resolve():
        console.print("[red]Output path must be different from the input path.[/red]")
        raise typer.Exit(1)

    rows = convert_traces_to_training_data(input_path)
    if not rows:
        console.print(f"[red]No supported trace rows found in {input_path}.[/red]")
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    console.print(f"[green]Converted {len(rows)} trace row{'s' if len(rows) != 1 else ''} to {output}[/green]")
    console.print("[cyan]Output format:[/cyan] Teich JSONL with prompt, messages, tools, and metadata.")


@pool_app.command(
    "upload",
    help="Reserved placeholder for future Teich pool uploads.",
    short_help="Reserved pool upload placeholder.",
)
def pool_upload(
    path: Path = typer.Argument(Path("output_anonymized"), help="Anonymized trace directory to upload later"),
) -> None:
    """Placeholder for future Teich pool upload support."""
    console.print("[yellow]teich pool upload is not wired to a deployed pool backend yet.[/yellow]")
    console.print(f"[cyan]Ready path:[/cyan] {path}")
    console.print("[dim]The command is reserved until the site has its database and upload API deployed.[/dim]")
    raise typer.Exit(1)


@app.command(
    help="Run a configured prompt batch and write trace/data artifacts.",
    short_help="Generate traces from config.",
    cls=_help_command("GenerateHelpCommand", GENERATE_EXTRA_HELP),
)
def generate(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config", "-c",
        help="Path to configuration file",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Override output directory",
    ),
    concurrency: int = typer.Option(
        None,
        "--concurrency", "-j",
        min=1,
        help="Number of prompts to run in parallel",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Skip prompts that already have completed answers in the output directory",
    ),
    mode: str = typer.Option(
        "prompts",
        "--mode",
        help="Generation mode: 'prompts' (default) or 'bench' (run benchmark tasks from bench.sources).",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Bench mode (harbor sources only): re-download a remote harbor bench source even if it's already cached. No effect on swe-bench, which uses Hugging Face's own dataset cache.",
    ),
) -> None:
    """Generate agent traces: plain traces from prompts (--mode prompts), or reward-labeled benchmark traces from bench.sources (--mode bench)."""
    console.print(Panel.fit("Teich", style="bold blue"))

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    cfg = Config.from_yaml(config)
    if output:
        cfg.output.traces_dir = output
    if concurrency is not None:
        cfg.max_concurrency = concurrency

    if mode not in {"prompts", "bench"}:
        console.print(f"[red]Unknown --mode '{mode}'. Use 'prompts' or 'bench'.[/red]")
        raise typer.Exit(1)
    conflicting_modes = _existing_dataset_modes(cfg.output.traces_dir) - {mode}
    if conflicting_modes:
        other = ", ".join(sorted(conflicting_modes))
        console.print(
            f"[red]{cfg.output.traces_dir} already contains {other}-mode data; "
            f"refusing to mix it with {mode}-mode output.[/red]"
        )
        console.print(
            "[yellow]Keep bench and prompts as separate datasets (a mixed dataset is unpleasant "
            "to consume). Point --output at a fresh directory, or use a separate config/project "
            "with its own output.traces_dir and publish.repo_id.[/yellow]"
        )
        raise typer.Exit(1)
    if mode == "bench":
        from .bench import run_bench  # lazy: harbor is an optional extra

        try:
            written = run_bench(cfg, console=console, resume=resume, refresh=refresh)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            # An earlier source may have harvested rows before a later one failed; write the card
            # for that partial dataset (like prompt mode does on error). Don't auto-publish.
            readme_path = _try_write_completed_output_metadata(cfg)
            if readme_path:
                console.print(f"[green]Wrote README: {readme_path}[/green]")
            raise typer.Exit(1) from exc
        if written:
            readme_path = _write_output_metadata(cfg)
            console.print(f"[green]Wrote README: {readme_path}[/green]")
            if cfg.get_publish_repo_id():
                repo_url = _publish_dataset_to_hub(cfg)
                console.print(f"[green]Published dataset: {repo_url}[/green]")
        else:
            console.print("[yellow]bench: no rows written; skipping dataset card.[/yellow]")
        return

    prompt_inputs = unique_prompt_inputs_by_completion_key(cfg.get_prompt_inputs())
    if not prompt_inputs:
        console.print("[red]No prompts configured. Add prompts in config.yaml or prompts.jsonl.[/red]")
        raise typer.Exit(1)
    agent_provider = cfg.get_agent_provider()

    # Ensure output dir exists
    cfg.output.traces_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        completed_keys = completed_prompt_keys_from_outputs(
            cfg.output.traces_dir,
            excluded_dirs=[cfg.output.failures_dir],
        )
        pending_prompt_inputs = prompt_inputs_for_run(
            prompt_inputs,
            cfg.output.traces_dir,
            resume=True,
            excluded_dirs=[cfg.output.failures_dir],
        )
        skipped_count = len(prompt_inputs) - len(pending_prompt_inputs)
    else:
        completed_keys = set()
        pending_prompt_inputs = prompt_inputs
        skipped_count = 0
    effective_concurrency = (
        max(1, min(cfg.max_concurrency, len(pending_prompt_inputs)))
        if pending_prompt_inputs
        else cfg.max_concurrency
    )

    console.print(f"[green]Loaded config: {config}[/green]")
    if resume:
        console.print(
            f"[yellow]Resume enabled: found {len(completed_keys)} completed output prompts; "
            f"skipping {skipped_count} configured prompts.[/yellow]"
        )
    if not pending_prompt_inputs:
        console.print("[green]All configured prompts already have completed outputs. Nothing to run.[/green]")
        readme_path = _try_write_completed_output_metadata(cfg)
        if readme_path:
            console.print(f"[green]Wrote README: {readme_path}[/green]")
        return
    console.print(
        f"[blue]Processing {len(pending_prompt_inputs)} prompts with concurrency {effective_concurrency}...[/blue]\n"
    )

    # Run generation
    try:
        if agent_provider == "codex":
            runner = CodexRunner(cfg)
            if cfg.agent.codex.use_host_auth:
                console.print(
                    "[yellow]Codex host-auth enabled: using your ChatGPT subscription via "
                    f"{cfg.get_codex_host_auth_source()}.[/yellow]"
                )
                console.print(
                    "[yellow]Heads up: once the rotating token is refreshed, your host Codex login "
                    "will be invalidated — run `codex login` on the host afterward to restore it.[/yellow]"
                )
            if cfg.agent.langfuse.enabled:
                console.print(
                    "[cyan]Codex Langfuse tracing enabled: uploading each session to "
                    f"{cfg.agent.langfuse.base_url}.[/cyan]"
                )
        elif agent_provider == "pi":
            runner = PiRunner(cfg)
        elif agent_provider in {"claude", "claude-code", "claude_code"}:
            runner = ClaudeCodeRunner(cfg)
            if cfg.agent.langfuse.enabled:
                console.print(
                    "[cyan]Claude Code Langfuse tracing enabled: uploading each session to "
                    f"{cfg.agent.langfuse.base_url}.[/cyan]"
                )
        elif agent_provider in {"hermes", "hermes-agent", "hermes_agent"}:
            runner = HermesRunner(cfg)
        elif agent_provider == "chat":
            runner = ChatRunner(cfg)
        else:
            console.print(
                "[red]Unsupported agent provider: "
                f"{agent_provider}. Supported providers: codex, pi, claude-code, hermes, chat.[/red]"
            )
            raise typer.Exit(1)

        generation_started_at = datetime.now(timezone.utc)
        with BatchProgressReporter(console) as reporter:
            results = runner.run_all(
                max_concurrency=cfg.max_concurrency,
                progress_callback=reporter.update,
                prompt_inputs=pending_prompt_inputs,
                resume=resume,
            )
        readme_path = _write_output_metadata(cfg)
        totals = reporter.snapshot_totals()
        generated_files = _new_jsonl_outputs(cfg.output.traces_dir, generation_started_at, results)

        console.print(f"\n[bold green]Success! Generated {len(generated_files)} JSONL files:[/bold green]")
        for r in generated_files:
            console.print(f"  - {r}")
        console.print(
            "\n[cyan]Usage:[/cyan] "
            f"tokens={_format_total_tokens(totals)} input={_format_total_token_field(totals, 'input_tokens')} "
            f"output={_format_total_token_field(totals, 'output_tokens')} "
            f"reasoning={_format_total_token_field(totals, 'reasoning_tokens')} "
            f"cache_read={_format_total_token_field(totals, 'cache_read_tokens')}"
        )
        console.print(f"[cyan]API cost:[/cyan] {_format_total_cost(totals)}")
        if agent_provider != "chat":
            console.print(f"[green]Saved sandboxes: {cfg.output.sandbox_dir}[/green]")
        console.print(f"\n[green]Wrote README: {readme_path}[/green]")
        if cfg.get_publish_repo_id():
            repo_url = _publish_dataset_to_hub(cfg)
            console.print(f"[green]Published dataset: {repo_url}[/green]")

    except KeyboardInterrupt:
        readme_path = _try_write_completed_output_metadata(cfg)
        if readme_path:
            console.print(f"\n[green]Wrote README for completed outputs: {readme_path}[/green]")
            _print_interrupted_upload_hint(cfg)
        console.print("\n[yellow]Interrupted. Completed outputs remain on disk.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        readme_path = _try_write_completed_output_metadata(cfg)
        if readme_path:
            console.print(f"\n[green]Wrote README for completed outputs: {readme_path}[/green]")
            _prompt_publish_completed_outputs(cfg)
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _format_elapsed(update: SessionProgressUpdate) -> str:
    if update.started_at is None:
        return "--"
    finished_at = update.finished_at or datetime.now(timezone.utc)
    elapsed = max(0, int((finished_at - update.started_at).total_seconds()))
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_api_tokens(metrics: TraceMetrics | None) -> str:
    if not metrics:
        return "--"
    if not metrics.has_token_usage:
        return "N/A"
    return str(metrics.total_tokens)


def _format_cost(metrics: TraceMetrics | None) -> str:
    if not metrics:
        return "--"
    if not metrics.has_cost:
        return "N/A"
    return f"${metrics.total_cost:.6f}"


def _format_total_tokens(totals: dict[str, float | int | bool]) -> str:
    if not totals.get("has_token_usage"):
        return "N/A"
    return str(totals["total_tokens"])


def _format_total_token_field(totals: dict[str, float | int | bool], key: str) -> str:
    if not totals.get("has_token_usage"):
        return "N/A"
    return str(totals[key])


def _format_total_cost(totals: dict[str, float | int | bool]) -> str:
    if not totals.get("has_cost"):
        return "N/A"
    return f"${float(totals['total_cost']):.6f}"


class BatchProgressReporter:
    def __init__(self, progress_console: Console):
        self.console = progress_console
        self.enabled = progress_console.is_terminal
        self._lock = RLock()
        self._updates: dict[str, SessionProgressUpdate] = {}
        self._live: Live | None = None
        self._refresh_stop: Event | None = None
        self._refresh_thread: Thread | None = None

    def __enter__(self) -> BatchProgressReporter:
        if self.enabled:
            self._live = Live(self._render(), console=self.console, refresh_per_second=4)
            self._live.__enter__()
            self._refresh_stop = Event()
            self._refresh_thread = Thread(
                target=self._refresh_live_loop,
                name="teich-progress-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._refresh_stop is not None:
            self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=2)
            self._refresh_thread = None
        self._refresh_stop = None
        if self._live is not None:
            self._live.__exit__(exc_type, exc, exc_tb)
            self._live = None

    def _refresh_live_loop(self) -> None:
        while self._refresh_stop is not None and not self._refresh_stop.wait(1):
            self._refresh_live_once()

    def _refresh_live_once(self) -> None:
        with self._lock:
            live = self._live
        if live is not None:
            live.update(self._render(), refresh=True)

    def update(self, update: SessionProgressUpdate) -> None:
        with self._lock:
            previous = self._updates.get(update.prompt_id)
            if previous and update.started_at is None:
                update.started_at = previous.started_at
            if previous and update.metrics_delta and update.metrics is not None:
                if previous.metrics is not None:
                    metrics = previous.metrics.copy()
                    metrics.add_metrics(update.metrics)
                    update.metrics = metrics
                update.metrics_delta = False
            elif previous and update.metrics is None and update.status == "running":
                update.metrics = previous.metrics
            self._updates[update.prompt_id] = update
            if self._live is not None:
                self._live.update(self._render(), refresh=True)

    def snapshot_totals(self) -> dict[str, float | int]:
        with self._lock:
            totals = {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "est_total_tokens": 0,
                "total_cost": 0.0,
                "has_token_usage": False,
                "has_cost": False,
            }
            for update in self._updates.values():
                metrics = update.metrics
                if not metrics:
                    continue
                if metrics.has_token_usage:
                    totals["has_token_usage"] = True
                totals["input_tokens"] += metrics.input_tokens
                totals["output_tokens"] += metrics.output_tokens
                totals["reasoning_tokens"] += metrics.reasoning_tokens
                totals["cache_read_tokens"] += metrics.cache_read_tokens
                totals["cache_write_tokens"] += metrics.cache_write_tokens
                totals["total_tokens"] += metrics.total_tokens
                totals["est_total_tokens"] += metrics.est_total_tokens
                if metrics.has_cost:
                    totals["has_cost"] = True
                totals["total_cost"] += metrics.total_cost
            return totals

    def _render(self):
        with self._lock:
            summary = Table.grid(expand=True)
            summary.add_column(justify="left")
            summary.add_column(justify="left")
            queued = sum(1 for update in self._updates.values() if update.status == "queued")
            running = sum(1 for update in self._updates.values() if update.status == "running")
            completed = sum(1 for update in self._updates.values() if update.status == "completed")
            failed = sum(1 for update in self._updates.values() if update.status == "failed")
            totals = self.snapshot_totals()
            summary.add_row(
                f"queued={queued} running={running} completed={completed} failed={failed}",
                f"tokens={_format_total_tokens(totals)} cost={_format_total_cost(totals)}",
            )

            table = Table(title="Generation Progress")
            table.add_column("#", justify="right", no_wrap=True)
            table.add_column("status", no_wrap=True)
            table.add_column("elapsed", no_wrap=True)
            table.add_column("tokens", justify="right", no_wrap=True)
            table.add_column("cost", justify="right", no_wrap=True)
            table.add_column("model", no_wrap=True)
            table.add_column("prompt")
            table.add_column("details")
            active_updates = [
                update
                for update in self._updates.values()
                if update.status in {"queued", "running"}
            ]
            for update in sorted(active_updates, key=lambda item: item.prompt_index):
                metrics = update.metrics
                model = metrics.model if metrics and metrics.model else "--"
                details = "--"
                if update.details:
                    details = update.details
                elif update.trace_path is not None:
                    details = str(update.trace_path.name)
                elif update.error:
                    details = update.error
                table.add_row(
                    str(update.prompt_index),
                    update.status,
                    _format_elapsed(update),
                    _format_api_tokens(metrics),
                    _format_cost(metrics),
                    model,
                    update.prompt_preview,
                    details,
                )
            return Group(Panel.fit(summary, title="Batch Summary"), table)


@app.command(
    help="Create a starter config.yaml and prompts.jsonl project.",
    short_help="Initialize a Teich project.",
    cls=_help_command("InitHelpCommand", INIT_EXTRA_HELP),
)
def init(
    path: Path = typer.Argument(Path("."), help="Directory to initialize"),
) -> None:
    """Initialize a new teich project."""
    console.print(Panel.fit("Initialize Project", style="bold blue"))

    path.mkdir(parents=True, exist_ok=True)

    # Create config.yaml
    config_path = path / "config.yaml"
    if not config_path.exists():
        config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        console.print(f"[green]Created: {config_path}[/green]")
    else:
        console.print(f"[yellow]Already exists: {config_path}[/yellow]")

    # Create prompts.jsonl
    prompts_path = path / "prompts.jsonl"
    if not prompts_path.exists():
        prompts_path.write_text(PROMPTS_TEMPLATE, encoding="utf-8")
        console.print(f"[green]Created: {prompts_path}[/green]")
    else:
        console.print(f"[yellow]Already exists: {prompts_path}[/yellow]")

    console.print(f"\n[bold green]Initialized in {path.absolute()}[/bold green]")
    console.print("\n[yellow]Next:[/yellow]")
    console.print("1. Set TEICH_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY in env or config.yaml")
    console.print("2. Add prompt rows to prompts.jsonl")
    console.print("3. Run: [cyan]uvx teich generate -c config.yaml[/cyan]")


CONFIG_TEMPLATE = '''# Teich configuration
#
# Quick start:
# 1. Choose agent.provider: codex, pi, claude-code, hermes, or chat
# 2. Set model.model to the model you want to run
# 3. Keep prompts in prompts.jsonl, or add inline prompts below
# 4. Prefer TEICH_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY in your environment for secrets
# 5. Run: uvx teich generate -c config.yaml

agent:
  # codex = Codex CLI in Docker
  # pi = Pi coding agent in Docker
  # claude-code = Claude Code CLI in Docker
  # hermes = Hermes Agent CLI in Docker
  # chat = direct text-only dataset generation via an OpenAI-compatible API
  provider: pi

  # Codex-only: use your ChatGPT subscription instead of an API key. Teich seeds
  # one auth.json snapshot from your host login ($CODEX_HOME/auth.json or
  # ~/.codex/auth.json) into auth_dir, then runs a host-side broker that owns the
  # rotating refresh token. Each container gets its own seeded auth.json (refresh
  # token swapped for a per-run secret) and refreshes through the broker, which
  # single-flights rotation, so any max_concurrency is safe.
  # NOTE: the first rotation invalidates your host `codex` login; run `codex
  # login` again afterward to restore it. auth_dir holds live credentials, so
  # Teich keeps it out of output/sandbox/failures and gitignores it.
  # codex:
  #   use_host_auth: true
  #   host_auth_file: null            # default: $CODEX_HOME/auth.json or ~/.codex/auth.json
  #   auth_dir: ./.teich/codex-auth

  # Trace each agent session to Langfuse (https://langfuse.com). Works for Codex
  # and Claude Code -- each uses its own native integration (Codex plugin, Claude
  # Code Stop hook) and Teich passes the credentials into the container. Side-
  # channel only: it reads transcripts, fails open, and doesn't change agent
  # behavior or Teich's output files. All three credentials are required when
  # enabled. base_url may be Langfuse Cloud or a self-hosted URL; for a host-local
  # instance use http://host.docker.internal:<port>.
  # langfuse:
  #   enabled: true
  #   public_key: pk-lf-...
  #   secret_key: sk-lf-...
  #   base_url: https://cloud.langfuse.com

model:
  # Model id passed to the selected agent/provider.
  model: deepseek/deepseek-v4-flash

  # Codex approval behavior. Common values: never, on-request.
  approval_policy: never

  # Codex sandbox mode.
  sandbox: danger-full-access

  # Optional reasoning / thinking level.
  # - Codex: forwarded as model_reasoning_effort
  # - Pi: normalized to low / medium / high when supported
  # - Claude Code / Hermes: model/provider specific
  # Hermes also enables built-in toolsets:
  # safe,terminal,file,skills,memory,session_search,delegation
  reasoning_effort: medium

  # Optional Codex reasoning-summary detail, forwarded as model_reasoning_summary.
  # Values: auto | concise | detailed | none. Set to "detailed" to capture rich
  # reasoning summaries in traces (Codex's default can emit empty summaries).
  # Leave null to use Codex's default.
  reasoning_summary: null

  # Optional Codex service tier. Set to "fast" to enable Codex fast mode: it
  # runs ~1.5x faster at a higher credit rate and requires ChatGPT subscription
  # auth (agent.codex.use_host_auth) plus a supported model such as gpt-5.5 or
  # gpt-5.4. Leave null for the standard tier.
  service_tier: null

  # Optional context length override for providers that support it.
  # Useful for Hermes custom endpoints when /v1/models reports a served
  # context below Hermes' minimum but the endpoint supports a larger window.
  context_length: null

  # Optional Pi-specific provider overrides.
  # Teich already defaults maxTokens to 131072 even if this block is omitted.
  # Uncomment to customize the Pi/OpenRouter model entry further.
  # pi_model_overrides:
  #   maxTokens: 131072
  #   compat:
  #     maxTokensField: max_tokens

# Optional API/provider configuration.
# Leave this section commented out to use the runtime defaults.
#
# Common setups:
# - OpenAI: provider=openai
# - OpenRouter: provider=openrouter, base_url=https://openrouter.ai/api/v1
# - Local OpenAI-compatible server: provider=openai, base_url=http://localhost:1234/v1
#
# You can also put secrets in env vars instead of committing them here:
#   TEICH_API_KEY=...
#   OPENROUTER_API_KEY=...
#   OPENAI_API_KEY=...
#   TEICH_BASE_URL=...
#   TEICH_PROVIDER=...
#   TEICH_MODEL=...
api:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key: null
  # Pi uses OpenRouter through chat/completions even if this is set to responses;
  # the Responses adapter can stall before the first native session event.
  wire_api: responses

# Optional MCP servers exposed inside the agent runtime.
# mcp_servers:
#   - name: filesystem
#     command: npx
#     args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

# Prompts can come from a file, inline below, or both.
# Relative paths are resolved from the location of this config file.
# JSONL is recommended because it safely supports multiline prompts, prompt-level system messages, and follow_up_prompts.
# CSV is still supported, but prompt files with commas/newlines are easier to maintain as JSONL.
prompts_file: prompts.jsonl

# Optional inline prompts. Use objects when you need metadata or chat follow-up turns.
# prompts:
#   - prompt: "Draft a compact project plan"
#     system: "Answer as a concise project manager."
#     follow_up_prompts:
#       - "Revise it for a solo developer"
#       - "Add a risk checklist"
prompts: []

output:
  # Where generated .jsonl files are written.
  # - codex / pi: normalized copies of native agent session traces
  # - claude-code: native Claude Code transcript JSONL from .claude/projects/...
  # - hermes: native Hermes session export JSONL from generated runs, including delegated subagents
  # - chat: text-only training rows with messages/prompt/response/thinking fields
  traces_dir: ./output

  # Where Teich stores workspace snapshots for each run.
  sandbox_dir: ./sandbox

  # Where Teich stores failed or interrupted traces for debugging.
  # These files are excluded from resume detection, conversion, README generation, and uploads.
  failures_dir: ./failures

  # Harbor bench-mode intermediates (downloaded sources, raw trials, normalized sessions).
  # Defaults to a `bench` dir beside traces_dir (parallel to sandbox/failures), never inside
  # the dataset. Set a path to override.
  bench_dir: null

  # Used in the generated trace README.
  pretty_name: "My Agent Traces"

# Optional direct upload to a Hugging Face dataset repo.
# Tags are auto-generated from the provider and model.
publish:
  repo_id: null
  hf_token: null
  private: false

# Benchmark mode (`teich generate --mode bench`). Each source is run by a backend selected
# by `type`: `harbor` (the harbor package; needs teich[harbor], Python 3.12+) or `swe-bench`
# (the swebench package; needs teich[swe]). teich runs the configured agent on each task/
# instance, harvests the plain native trace into output/{passed,failed,borderline}/, and
# writes scores + provenance to output/metadata/<stem>.json. Intermediates live under
# output.bench_dir (a sibling `bench` dir by default), outside the dataset/uploads.
# `--resume` skips already-harvested tasks; `--refresh` re-downloads remote harbor sources
# (no effect on swe-bench, which uses Hugging Face's own dataset cache).
# Keep bench its own project: dedicated output.traces_dir + publish.repo_id.
bench:
  sources: []
    # - { type: harbor,    source: terminal-bench@2.0 }          # registry spec
    # - { type: harbor,    source: ./local-tasks }               # local task dir / dir of task dirs
    # - { type: swe-bench, source: SWE-bench/SWE-bench_Verified, split: test }
    # - { type: swe-bench, source: SWE-bench/SWE-bench_Lite, instances: [django__django-12345] }
    #   # optional per-source: repo (git/HF registry), version, instances [..], backend (harbor)

# Number of prompts to run in parallel.
max_concurrency: 1

# Per-session timeout in seconds.
timeout_seconds: 600

# Legacy global API key field.
# Prefer env vars or api.api_key above for new configs.
openai_api_key: null

# Optional developer/system instructions injected into every agent run.
# Applied across codex (developer_instructions), claude-code and pi
# (--append-system-prompt), and hermes (AGENTS.md) — additive, so the agent's
# built-in base prompt is preserved. Handy for nudging chain-of-thought into the
# trace, e.g.:
#   developer_instructions: |
#     Think out loud so your problem-solving process is visible. Before each tool
#     call or edit, briefly explain what you're doing and why; after a command or
#     test runs, state what you concluded before the next step.
developer_instructions: null
'''


def _configure_studio_event_loop_policy(
    *,
    platform: str = sys.platform,
    asyncio_module: Any | None = None,
) -> bool:
    if platform != "win32":
        return False
    if asyncio_module is None:
        import asyncio as asyncio_module

    policy_cls = getattr(asyncio_module, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is None:
        return False
    asyncio_module.set_event_loop_policy(policy_cls())
    return True


@app.command(
    help="Launch the local Teich Studio browser UI.",
    short_help="Launch Teich Studio.",
    cls=_help_command("StudioHelpCommand", STUDIO_EXTRA_HELP),
)
def studio(
    path: Path = typer.Argument(Path("."), help="Project directory (created/initialized if needed)"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the studio server to"),
    port: int = typer.Option(8420, "--port", help="Port to bind the studio server to"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the studio in your browser"),
) -> None:
    """Launch the Teich Studio browser UI."""
    import threading
    import webbrowser

    _configure_studio_event_loop_policy()

    try:
        import uvicorn

        from .studio.server import create_app
    except ImportError as exc:
        console.print(f"[red]Teich Studio requires fastapi and uvicorn: {exc}[/red]")
        console.print("Install them with: [cyan]pip install 'teich[studio]'[/cyan]")
        raise typer.Exit(1)

    project_dir = path.resolve()
    studio_app = create_app(project_dir)
    url = f"http://{host}:{port}"

    console.print(Panel.fit("Teich Studio", style="bold blue"))
    console.print(f"[green]Project:[/green] {project_dir}")
    console.print(f"[green]Studio running at:[/green] [bold cyan]{url}[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(studio_app, host=host, port=port, log_level="warning")


PROMPTS_TEMPLATE = '''{"prompt":"Build a simple todo list app in React"}
{"prompt":"Create a Python script that fetches weather data from an API"}
{"github_repo":"armand0e/perplexica-mcp","prompt":"Add a small usability improvement and update the tests"}
{"system":"Answer as a concise project manager.","prompt":"Draft a compact project plan"}
{"prompt":"Draft a compact project plan","follow_up_prompts":["Revise it for a solo developer","Add a risk checklist"]}
'''


def main():
    app()


if __name__ == "__main__":
    main()
