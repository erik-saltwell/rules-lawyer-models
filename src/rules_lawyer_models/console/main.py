# src/reddit_rpg_miner/cli/main.py
from __future__ import annotations

import unsloth  # isort: skip  # Must precede all transformers imports

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import click
import typer
from dotenv import load_dotenv
from rich.console import Console

from rules_lawyer_models.commands import CommmandProtocol, VerifyTemplateData
from rules_lawyer_models.commands.compute_batch_size import ComputeBatchSizeCommand
from rules_lawyer_models.console.rich_logging_protocol import RichConsoleLogger
from rules_lawyer_models.core import RunContext
from rules_lawyer_models.training.training_options import FragmentID, TrainingLength, TrainingRunConfiguration
from rules_lawyer_models.utils import CommonPaths
from rules_lawyer_models.utils.dataset_name import DatasetName
from rules_lawyer_models.utils.logging_config import configure_logging
from rules_lawyer_models.utils.model_name import BaseModelName, TargetModelName

load_dotenv()
configure_logging()
print(unsloth.__version__)


def current_command_name() -> str:
    ctx = click.get_current_context()
    assert ctx.command.name is not None
    return ctx.command.name


def check_all_errors(errors: list[str], console: Console) -> None:
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
    paths = CommonPaths(DatasetName.REDDIT_RPG_POST_CLASSIFICATION)
    ctxt: RunContext = RunContext(paths, logger)

    dataset_name = DatasetName.REDDIT_RPG_POST_CLASSIFICATION
    base_model_name = BaseModelName.QWEN_25_3B_05BIT_INSTRUCT
    system_prompt_id = FragmentID.RPG_POST_CLASSIFICATION_PROMPT
    target_model_name: TargetModelName = TargetModelName.REDDIT_RPG_POST_CLASSIFICATION

    run_configuration: TrainingRunConfiguration = TrainingRunConfiguration.construct_base(
        dataset_name, "train", "text", base_model_name, target_model_name, TrainingLength(True, 1), system_prompt_id
    )

    # command: CommmandProtocol = VerifyTemplateData(
    #     1, dataset_name, base_model_name, system_prompt_id, "content", "label", "text", "eval"
    # )

    # command.execute(ctxt)

    # ctxt.logger.report_message(" ")

    # analyze_seq_command: CommmandProtocol = AnalyzeSequenceLengths(
    #     dataset_name, base_model_name, system_prompt_id, "content", "label", "text"
    # )

    # analyze_seq_command.execute(ctxt)

    # ctxt.logger.report_message(" ")

    compute_batch_command: CommmandProtocol = ComputeBatchSizeCommand(run_configuration, 200, "content", "label", 1024)
    compute_batch_command.execute(ctxt)

    # run_configuration: TrainingRunConfiguration = TrainingRunConfiguration.construct_base(
    #     DatasetName.REDDIT_RPG_POST_CLASSIFICATION,
    #     "train",
    #     BaseModelName.QWEN_25_3B_05BIT_INSTRUCT,
    #     TargetModelName.REDDIT_RPG_POST_CLASSIFICATION,
    #     TrainingLength(True, 50),
    #     FragmentID.RPG_POST_CLASSIFICATION_PROMPT,
    # )

    # command: ComputeBatchSizeCommand = ComputeBatchSizeCommand(run_configuration, 100, "content", "label")
    # command.execute(ctxt)


@app.command("verify-template")
def verify_template() -> None:
    """Verify template data by printing the top rows."""
    console = Console()
    logger = RichConsoleLogger(console)
    paths = CommonPaths("reddit_rpg_post_classifier")
    ctxt: RunContext = RunContext(paths, logger)
    dataset_name = DatasetName.REDDIT_RPG_POST_CLASSIFICATION
    base_model_name = BaseModelName.QWEN_25_3B_05BIT_INSTRUCT
    system_prompt_id = FragmentID.RPG_POST_CLASSIFICATION_PROMPT
    command: CommmandProtocol = VerifyTemplateData(
        1, dataset_name, base_model_name, system_prompt_id, "content", "label", "text", "eval"
    )
    command.execute(ctxt)


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
