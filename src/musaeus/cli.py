"""cli.py — `musaeus`, the command-line front door.

Same Agent as the server, driven from a terminal instead of HTTP. Two commands:
  - `musaeus chat`     -> an interactive REPL: read a line, print the Agent's answer.
  - `musaeus version`  -> print the installed version.

typer gives us the argument parsing; rich gives us readable output (a titled panel,
colour, a spinner while the model thinks). Everything routes through the same
load_settings -> build_llm -> Agent wiring the server uses, so the two entry points
can never drift apart.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .agent import Agent
from .config import load_settings
from .llm import build_llm

app = typer.Typer(
    add_completion=False,
    help="Musaeus — an open, local-first LLM agent you actually own.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    provider: str = typer.Option("local", help="local | anthropic | openai | google"),
    model: str | None = typer.Option(None, help="Override the provider's default model."),
    system: str | None = typer.Option(None, help="Optional system prompt for the agent."),
) -> None:
    """Interactive REPL: type a message, get the agent's answer. Ctrl-C or Ctrl-D to quit."""
    settings = load_settings(provider=provider, model=model)
    agent = Agent(build_llm(settings), system=system)

    console.print(
        Panel.fit(
            f"[bold]Musaeus[/bold] · provider=[cyan]{settings.provider}[/cyan] "
            f"· model=[cyan]{settings.model_for}[/cyan]\n"
            "[dim]Type a message. Ctrl-C or Ctrl-D to exit.[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            message = console.input("[bold green]you[/bold green] › ").strip()
        except (KeyboardInterrupt, EOFError):
            # Clean exit on Ctrl-C / Ctrl-D — a REPL should never dump a traceback.
            console.print("\n[dim]bye[/dim]")
            raise typer.Exit()

        if not message:
            continue
        if message in {"/exit", "/quit"}:
            console.print("[dim]bye[/dim]")
            raise typer.Exit()

        try:
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                answer = agent.run(message)
        except KeyboardInterrupt:
            # Interrupt mid-request: abandon this turn, keep the REPL alive.
            console.print("[yellow]…cancelled[/yellow]")
            continue
        except Exception as exc:  # a bad request shouldn't kill the whole session
            console.print(f"[red]error[/red] {type(exc).__name__}: {exc}")
            continue

        console.print(Panel(answer, title="musaeus", border_style="magenta"))


@app.command()
def version() -> None:
    """Print the installed Musaeus version."""
    try:
        from importlib.metadata import version as _v

        console.print(_v("musaeus"))
    except Exception:
        # Not installed as a distribution (e.g. running from a source checkout).
        from musaeus import __version__

        console.print(__version__)


if __name__ == "__main__":
    app()
