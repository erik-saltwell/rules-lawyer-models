"""Reusable sweep orchestration — parameterised wrapper around W&B sweep lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from rules_lawyer_models.training.training_options_factory import TrainingMetaOptions
from rules_lawyer_models.training.training_run_configuration import TrainingRunConfiguration
from rules_lawyer_models.utils.dataset_name import DatasetName
from rules_lawyer_models.utils.model_data import BaseModelName, TargetModelName
from rules_lawyer_models.utils.text_fragments import FragmentID

if TYPE_CHECKING:
    from rules_lawyer_models.core import RunContext


def run_sweep(
    # Model identity
    dataset_name: DatasetName,
    base_model_name: BaseModelName,
    target_model_name: TargetModelName,
    system_prompt_id: FragmentID,
    # Dataset config
    dataset_split_name: str,
    content_column_name: str,
    label_column_name: str,
    training_column_name: str,
    positive_class: str,
    clean_prediction_fn: Callable[[str], str],
    # Run config
    run_configuration: TrainingRunConfiguration,
    # Sweep config
    sweep_count: int,
    sweep_method: str,
    sweep_metric_name: str,
    sweep_metric_goal: str,
    # Context
    ctxt: RunContext,
    # Optional overrides
    sweep_config: dict | None = None,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 3412,
) -> None:
    """Run a full W&B hyperparameter sweep.

    Handles the complete lifecycle: data preparation, sweep registration,
    trial definition, and agent execution.
    """
    # Lazy imports to avoid circular dependency chains through training/__init__.py
    from datasets import Dataset, DatasetDict

    import wandb
    from rules_lawyer_models.commands.trial_runner import run_single_trial
    from rules_lawyer_models.data.dataset_helper import (
        prep_classification_dataset_for_trainng,
        split_dataset,
    )
    from rules_lawyer_models.serialization import load_dataset_from_hf, load_tokenizer_from_hf

    str_label_column_name = "str_" + label_column_name
    predictions_column_name = "predictions"

    # --- Phase 1: Prepare data ONCE (expensive, reused across trials) ---
    tokenizer = load_tokenizer_from_hf(base_model_name)
    dataset_all: Dataset = load_dataset_from_hf(dataset_name)[dataset_split_name]
    dataset_all = prep_classification_dataset_for_trainng(
        dataset_all,
        base_model_name,
        content_column_name,
        label_column_name,
        str_label_column_name,
        training_column_name,
        system_prompt_id,
        tokenizer,
    )
    dataset_dict: DatasetDict = split_dataset(dataset_all, validation_split, test_split, seed, label_column_name)

    # --- Phase 2: Create and register the sweep ---
    if sweep_config is None:
        sweep_config = TrainingMetaOptions.get_default_sweep_config(
            metric_name=sweep_metric_name,
            metric_goal=sweep_metric_goal,
        )
    sweep_config["method"] = sweep_method
    sweep_config["metric"] = {"goal": sweep_metric_goal, "name": sweep_metric_name}

    wandb.login()
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=target_model_name,
    )

    # --- Phase 3: Define the trial callback ---
    def sweep_trial() -> None:
        """Called by wandb.agent for each trial."""
        with wandb.init() as _run:
            config = dict(wandb.config)
            meta_options = TrainingMetaOptions.from_dict(config)

            run_single_trial(
                meta_options=meta_options,
                run_configuration=run_configuration,
                dataset_dict=dataset_dict,
                ctxt=ctxt,
                content_column_name=content_column_name,
                str_label_column_name=str_label_column_name,
                predictions_column_name=predictions_column_name,
                positive_class=positive_class,
                clean_prediction_fn=clean_prediction_fn,
                report_to_wandb=True,
            )

    # --- Phase 4: Launch the sweep agent ---
    wandb.agent(sweep_id, function=sweep_trial, count=sweep_count)
