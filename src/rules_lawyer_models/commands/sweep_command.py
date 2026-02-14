"""W&B Hyperparameter Sweep command for IMDB test dataset."""

from __future__ import annotations

from dataclasses import dataclass

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.training import (
    EvalationSettings,
    StepSize,
    TrainingLength,
    TrainingRunConfiguration,
    run_sweep,
)
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID, TargetModelName


@dataclass
class SweepCommand:
    sweep_count: int = 5
    sweep_method: str = "random"

    def execute(self, ctxt: RunContext) -> None:
        run_configuration = TrainingRunConfiguration(
            base_model_name=BaseModelName.QWEN_25_1_5B_INSTRUCT,
            target_model_name=TargetModelName.IMDB_TEST,
            training_column_name="train",
            system_prompt_id=FragmentID.IMDB_TEST_PROMPT,
            step_size=StepSize(1536, 4, 4),
            training_length=TrainingLength(use_steps=False, max_epochs=0.05),
            evaluation_settings=EvalationSettings.from_enabled(50),
            train_on_outputs_only=True,
        )

        run_sweep(
            dataset_name=DatasetName.IMDB_TEST,
            base_model_name=BaseModelName.QWEN_25_1_5B_INSTRUCT,
            target_model_name=TargetModelName.IMDB_TEST,
            system_prompt_id=FragmentID.IMDB_TEST_PROMPT,
            dataset_split_name="train",
            content_column_name="text",
            label_column_name="label",
            training_column_name="train",
            positive_class="pos",
            clean_prediction_fn=lambda text: "pos" if "pos" in text.lower() else "neg",
            run_configuration=run_configuration,
            sweep_count=self.sweep_count,
            sweep_method=self.sweep_method,
            sweep_metric_name="f1",
            sweep_metric_goal="maximize",
            ctxt=ctxt,
        )
