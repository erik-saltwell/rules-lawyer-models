# src/reddit_rpg_miner/cli/main.py
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import click
import typer
from dotenv import load_dotenv
from rich.console import Console

import rules_lawyer_models.console.console_validation as checks
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
    from transformers import PreTrainedTokenizerBase

    from rules_lawyer_models.core.loaders import load_tokenizer
    from rules_lawyer_models.exploration.token_length import compute_tokens
    from rules_lawyer_models.utils.base_model_name import BaseModelName

    tokenizer: PreTrainedTokenizerBase = load_tokenizer(BaseModelName.QWEN_25_14B_4BIT_BASE)
    other_count = compute_tokens("Other", tokenizer)
    reules_question_count = compute_tokens("Question", tokenizer)
    print(f"Other token count: {other_count}")
    print(f"Rules Question token count: {reules_question_count}")


@app.command("reddit-rpg-post-classifier")
def reddit_rpg_post_classifier() -> None:
    console = Console()
    console.print("[green]Hello from test[/green]")
    errors: list[str] = []

    model_name: str = current_command_name().replace("-", "_")
    errors.extend(checks._validate_directory_name(model_name))

    check_all_erors(errors, console)


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
