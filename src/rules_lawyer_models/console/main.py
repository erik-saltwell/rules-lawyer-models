# src/reddit_rpg_miner/cli/main.py
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import click
import typer
from dotenv import load_dotenv
from rich.console import Console

from rules_lawyer_models.commands import AnalyzeSequenceLengths, CommmandProtocol
from rules_lawyer_models.console.rich_logging_protocol import RichConsoleLogger
from rules_lawyer_models.core import RunContext
from rules_lawyer_models.utils import CommonPaths
from rules_lawyer_models.utils.logging_config import configure_logging

load_dotenv()
configure_logging()


def current_command_name() -> str:
    ctx = click.get_current_context()
    assert ctx.command.name is not None
    return ctx.command.name


def check_all_erors(errors: list[str], console: Console) -> None:
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(code=1)


app = typer.Typer(
    name="rules-lawyer-models",
    add_completion=True,
    help="CLI for rules-lawyer-models",
)


@app.command("test")
def test() -> None:
    """Simple smoke test command."""
    console = Console()
    logger = RichConsoleLogger(console)
    paths = CommonPaths("reddit_rpg_post_classifier")
    ctxt: RunContext = RunContext(paths, logger)
    command: CommmandProtocol = AnalyzeSequenceLengths()
    command.execute(ctxt)


@app.command("test-two")
def reddit_rpg_post_classifier() -> None: ...


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if not value:
        return

    # IMPORTANT: distribution name (pyproject.toml [project].name), often hyphenated.
    # Example: "my-tool" even if your import package is "my_tool".
    DIST_NAME = "rules-lawyer-models"

    console = Console()

    try:
        pkg_version = dist_version(DIST_NAME)
        md = metadata(DIST_NAME)
        try:
            pkg_name = md["Name"]
        except KeyError:
            pkg_name = DIST_NAME

        console.print(f"{pkg_name} {pkg_version}")
    except PackageNotFoundError:
        # Running from source without an installed distribution
        console.print(f"{DIST_NAME} 0.0.0+unknown")

    raise typer.Exit()


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Root command group for reddit_rpg_miner."""
    # Intentionally empty: this forces Typer to keep subcommands like `test`.
    pass


if __name__ == "__main__":
    app()
