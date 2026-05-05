"""CLI for teich."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import typer
from huggingface_hub import HfApi
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .config import Config
from .runner import ChatRunner, CodexRunner, PiRunner, SessionProgressUpdate, TraceMetrics
from .trace_readme import write_traces_readme

console = Console()
app = typer.Typer(
    name="teich",
    help="Generate agent training data using Codex, Pi, or chat",
    no_args_is_help=True,
)


def _publish_dataset_to_hub(cfg: Config) -> str:
    repo_id = cfg.get_publish_repo_id()
    if not repo_id:
        raise ValueError("No publish.repo_id configured")
    api = HfApi(token=cfg.get_hf_token())
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=cfg.publish.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(cfg.output.traces_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload teich dataset output",
    )
    return str(repo_url)


@app.command()
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
) -> None:
    """Generate training traces from prompts."""
    console.print(Panel.fit("Teich", style="bold blue"))

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    cfg = Config.from_yaml(config)
    if output:
        cfg.output.traces_dir = output
    if concurrency is not None:
        cfg.max_concurrency = concurrency

    prompt_inputs = cfg.get_prompt_inputs()
    if not prompt_inputs:
        console.print("[red]No prompts configured. Add prompts in config.yaml or prompts.csv.[/red]")
        raise typer.Exit(1)

    # Ensure output dir exists
    cfg.output.traces_dir.mkdir(parents=True, exist_ok=True)
    effective_concurrency = (
        max(1, min(cfg.max_concurrency, len(prompt_inputs))) if prompt_inputs else cfg.max_concurrency
    )

    console.print(f"[green]Loaded config: {config}[/green]")
    console.print(
        f"[blue]Processing {len(prompt_inputs)} prompts with concurrency {effective_concurrency}...[/blue]\n"
    )

    # Run generation
    try:
        agent_provider = cfg.get_agent_provider()
        if agent_provider == "codex":
            runner = CodexRunner(cfg)
        elif agent_provider == "pi":
            runner = PiRunner(cfg)
        elif agent_provider == "chat":
            runner = ChatRunner(cfg)
        else:
            console.print(
                f"[red]Unsupported agent provider: {agent_provider}. Supported providers: codex, pi, chat.[/red]"
            )
            raise typer.Exit(1)

        with BatchProgressReporter(console) as reporter:
            results = runner.run_all(
                max_concurrency=cfg.max_concurrency,
                progress_callback=reporter.update,
            )
        dataset_tags = cfg.get_dataset_tags()
        readme_path = write_traces_readme(
            cfg.output.traces_dir,
            pretty_name=cfg.output.pretty_name,
            tags=dataset_tags,
            model_id=cfg.model.model,
        )
        totals = reporter.snapshot_totals()

        console.print(f"\n[bold green]Success! Generated {len(results)} JSONL files:[/bold green]")
        for r in results:
            console.print(f"  - {r}")
        console.print(
            "\n[cyan]Usage:[/cyan] "
            f"tokens={totals['total_tokens']} input={totals['input_tokens']} "
            f"output={totals['output_tokens']} reasoning={totals['reasoning_tokens']} "
            f"cache_read={totals['cache_read_tokens']}"
        )
        console.print(f"[cyan]API cost:[/cyan] ${totals['total_cost']:.6f}")
        if agent_provider != "chat":
            console.print(f"[green]Saved sandboxes: {cfg.output.sandbox_dir}[/green]")
        console.print(f"\n[green]Wrote README: {readme_path}[/green]")
        if cfg.get_publish_repo_id():
            repo_url = _publish_dataset_to_hub(cfg)
            console.print(f"[green]Published dataset: {repo_url}[/green]")

    except Exception as e:
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
    if not metrics or not metrics.total_tokens:
        return "--"
    return str(metrics.total_tokens)


def _format_cost(metrics: TraceMetrics | None) -> str:
    if not metrics:
        return "--"
    return f"${metrics.total_cost:.6f}"


class BatchProgressReporter:
    def __init__(self, progress_console: Console):
        self.console = progress_console
        self.enabled = progress_console.is_terminal
        self._lock = RLock()
        self._updates: dict[str, SessionProgressUpdate] = {}
        self._live: Live | None = None

    def __enter__(self) -> BatchProgressReporter:
        if self.enabled:
            self._live = Live(self._render(), console=self.console, refresh_per_second=4)
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, exc_tb)
            self._live = None

    def update(self, update: SessionProgressUpdate) -> None:
        with self._lock:
            previous = self._updates.get(update.prompt_id)
            if previous and update.started_at is None:
                update.started_at = previous.started_at
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
            }
            for update in self._updates.values():
                metrics = update.metrics
                if not metrics:
                    continue
                totals["input_tokens"] += metrics.input_tokens
                totals["output_tokens"] += metrics.output_tokens
                totals["reasoning_tokens"] += metrics.reasoning_tokens
                totals["cache_read_tokens"] += metrics.cache_read_tokens
                totals["cache_write_tokens"] += metrics.cache_write_tokens
                totals["total_tokens"] += metrics.total_tokens
                totals["est_total_tokens"] += metrics.est_total_tokens
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
                f"tokens={totals['total_tokens']} cost=${totals['total_cost']:.6f}",
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
                if update.trace_path is not None:
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


@app.command()
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

    # Create prompts.csv
    prompts_path = path / "prompts.csv"
    if not prompts_path.exists():
        prompts_path.write_text(PROMPTS_TEMPLATE, encoding="utf-8")
        console.print(f"[green]Created: {prompts_path}[/green]")
    else:
        console.print(f"[yellow]Already exists: {prompts_path}[/yellow]")

    console.print(f"\n[bold green]Initialized in {path.absolute()}[/bold green]")
    console.print("\n[yellow]Next:[/yellow]")
    console.print("1. Set TEICH_API_KEY / OPENAI_API_KEY in env or config.yaml")
    console.print("2. Add prompt rows to prompts.csv")
    console.print("3. Run: [cyan]uvx teich generate -c config.yaml[/cyan]")


CONFIG_TEMPLATE = '''# Teich configuration
#
# Quick start:
# 1. Choose agent.provider: codex, pi, or chat
# 2. Set model.model to the model you want to run
# 3. Keep prompts in prompts.csv, or add inline prompts below
# 4. Prefer TEICH_API_KEY / OPENAI_API_KEY in your environment for secrets
# 5. Run: uvx teich generate -c config.yaml

agent:
  # codex = Codex CLI in Docker
  # pi = Pi coding agent in Docker
  # chat = direct text-only dataset generation via an OpenAI-compatible API
  provider: codex

model:
  # Model id passed to the selected agent/provider.
  model: codex-mini-latest

  # Codex approval behavior. Common values: never, on-request.
  approval_policy: never

  # Codex sandbox mode.
  sandbox: danger-full-access

  # Optional reasoning / thinking level.
  # - Codex: forwarded as model_reasoning_effort
  # - Pi: normalized to low / medium / high when supported
  reasoning_effort: null

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
# Prefer putting secrets in env vars instead of committing them here:
#   TEICH_API_KEY=...
#   TEICH_BASE_URL=...
#   TEICH_PROVIDER=...
#   TEICH_MODEL=...
# api:
#   provider: openai
#   base_url: null
#   api_key: null
#   wire_api: responses

# Optional MCP servers exposed inside the agent runtime.
# mcp_servers:
#   - name: filesystem
#     command: npx
#     args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

# Prompts can come from a file, inline below, or both.
# Relative paths are resolved from the location of this config file.
# CSV prompt files must include a `prompt` column.
prompts_file: prompts.csv

# Optional inline prompts.
prompts: []

output:
  # Where generated .jsonl files are written.
  # - codex / pi: raw normalized agent traces
  # - chat: text-only training rows with messages/prompt/response/thinking fields
  traces_dir: ./output

  # Where Teich stores workspace snapshots for each run.
  sandbox_dir: ./sandbox

  # Used in the generated trace README.
  pretty_name: "My Agent Traces"

# Optional direct upload to a Hugging Face dataset repo.
# Tags are auto-generated from the provider and model.
publish:
  repo_id: null
  hf_token: null
  private: false

# Number of prompts to run in parallel.
max_concurrency: 1

# Per-session timeout in seconds.
timeout_seconds: 600

# Legacy global API key field.
# Prefer env vars or api.api_key above for new configs.
openai_api_key: null

# Optional developer instructions injected into the runtime.
developer_instructions: null
'''

PROMPTS_TEMPLATE = '''image,github_repo,prompt
None,None,"Build a simple todo list app in React"
None,None,"Create a Python script that fetches weather data from an API"
None,armand0e/perplexica-mcp,"Add a small usability improvement and update the tests"
'''


def main():
    app()


if __name__ == "__main__":
    main()
