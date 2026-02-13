from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from trl.trainer.sft_trainer import SFTTrainer

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data.dataset_helper import prep_classification_dataset_for_trainng, split_dataset
from rules_lawyer_models.evaluation import (
    BinaryClassificationMetric,
    LoggerMetricsReporter,
    MetricsManager,
    MetricsReportingManager,
    WandbMetricsReporter,
    accuracy,
    f1_score,
    precision,
)
from rules_lawyer_models.exploration import (
    TokenLengthData,
    analyze_token_lengths,
    dataset_exploration,
    get_percent_samples_within_sequence_length,
)
from rules_lawyer_models.serialization import load_dataset_from_hf, load_tokenizer_from_hf
from rules_lawyer_models.training.training_options_factory import TrainingMetaOptions
from rules_lawyer_models.training.training_pipeline import create_trainer, load_base_model, run_training
from rules_lawyer_models.training.training_run_configuration import (
    EvalationSettings,
    StepSize,
    TrainingLength,
    TrainingRunConfiguration,
)
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID
from rules_lawyer_models.utils.model_data import TargetModelName


@dataclass
class IntegrationTestCommand:
    def execute(self, ctxt: RunContext) -> None:
        dataset_name = DatasetName.IMDB_TEST
        base_model_name = BaseModelName.QWEN_25_1_5B_INSTRUCT
        system_prompt_id = FragmentID.IMDB_TEST_PROMPT
        target_model_name: TargetModelName = TargetModelName.IMDB_TEST

        split_name = "train"
        content_column_name = "text"
        label_column_name = "label"
        train_column_name = "train"
        str_label_column_name = "str_" + label_column_name
        predictions_column_name = "predictions"
        # eval_column_name = "eval"

        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(base_model_name)

        dataset_all: Dataset = load_dataset_from_hf(dataset_name)[split_name]
        dataset_all = prep_classification_dataset_for_trainng(
            dataset_all,
            base_model_name,
            content_column_name,
            label_column_name,
            str_label_column_name,
            train_column_name,
            system_prompt_id,
            tokenizer,
        )

        dataset_exploration.dump_first_row_to_logger(dataset_all, ctxt.logger)
        ctxt.logger.add_break(2)

        # self.analyze_sequence_length(dataset_all, tokenizer, train_column_name, ctxt)
        # 1536 tokens only leaves out 21 samples.
        # stress_dataset = make_stress_split(dataset_all, 500, train_column_name, tokenizer)

        # )
        # max_batch_size: int = find_max_batch_size()
        max_sequence_len: int = 1536
        max_batch_size: int = 32
        gradient_accumulation_steps = 2
        run_configuration: TrainingRunConfiguration = self.create_run_config(
            max_sequence_len,
            max_batch_size,
            gradient_accumulation_steps,
            base_model_name,
            target_model_name,
            train_column_name,
            system_prompt_id,
        )
        training_meta_options = self.create_training_meta_options()

        # max_batch_size = find_max_batch_size(
        #     run_configuration,
        #     max_sequence_len,
        #     ctxt,
        #     dataset_all,
        #     BatchSizeStrategy.DOUBLE,
        #     starting_batch_size=1,
        #     max_batch_size=128,
        # )

        dataset_dict: DatasetDict = split_dataset(dataset_all, 0.1, 0.1, run_configuration.seed, label_column_name)

        model, tokenizer = load_base_model(
            run_configuration, training_meta_options.to_training_options(), base_model_name
        )

        trainer: SFTTrainer = create_trainer(
            model,
            tokenizer,
            run_configuration,
            training_meta_options.to_training_options(),
            True,
            dataset_dict["train"],
            dataset_dict["eval"],
        )
        run_training(model, tokenizer, trainer, run_configuration, training_meta_options.to_training_options())

        def clean_prediction(text: str) -> str:
            return "pos" if "pos" in text.lower() else "neg"

        positive_class = "pos"
        metrics_manager = MetricsManager(
            [
                BinaryClassificationMetric(
                    inputs_column_name=content_column_name,
                    ground_truth_column_name=str_label_column_name,
                    predictions_column_name=predictions_column_name,
                    positive_class=positive_class,
                    aggregate_predictions=accuracy,
                ),
                BinaryClassificationMetric(
                    inputs_column_name=content_column_name,
                    ground_truth_column_name=str_label_column_name,
                    predictions_column_name=predictions_column_name,
                    positive_class=positive_class,
                    aggregate_predictions=precision,
                ),
                BinaryClassificationMetric(
                    inputs_column_name=content_column_name,
                    ground_truth_column_name=str_label_column_name,
                    predictions_column_name=predictions_column_name,
                    positive_class=positive_class,
                    aggregate_predictions=f1_score,
                ),
            ]
        )

        results = metrics_manager.calculate_metrics(
            dataset_dict["test"],
            model,
            tokenizer,
            system_prompt_id,
            content_column_name,
            predictions_column_name,
            clean_prediction,
            max_new_tokens=10,
        )

        reporting_manager = MetricsReportingManager(
            [
                LoggerMetricsReporter(ctxt.logger),
                WandbMetricsReporter(),
            ]
        )
        reporting_manager.report(results)

    def create_training_meta_options(self) -> TrainingMetaOptions:
        meta_options: TrainingMetaOptions = TrainingMetaOptions(
            rank=16,
            alpha_multiplier=2,
            use_projection_modules=True,
            lora_dropout=0.0,
            warmup_ratio=0.025,
            learning_rate=2e-4,
            optim="paged_adamw_8bit",
            weight_decay=0.1,
            lr_schedular_type="linear",
        )
        return meta_options

    def create_run_config(
        self,
        max_sequence_len: int,
        max_batch_size: int,
        gradient_accumulation_setps: int,
        base_model_name: BaseModelName,
        target_model_name: TargetModelName,
        training_column_name: str,
        system_prompt_id: FragmentID,
    ) -> TrainingRunConfiguration:
        step_size: StepSize = StepSize(
            max_sequence_length=max_sequence_len,
            per_device_batch_size=max_batch_size,
            gradient_accumulation_steps=gradient_accumulation_setps,
        )
        training_length: TrainingLength = TrainingLength(use_steps=False, max_epochs=1.0)
        evaluation_settings: EvalationSettings = EvalationSettings.from_enabled(50)

        return TrainingRunConfiguration(
            base_model_name=base_model_name,
            target_model_name=target_model_name,
            training_column_name=training_column_name,
            system_prompt_id=system_prompt_id,
            step_size=step_size,
            training_length=training_length,
            evaluation_settings=evaluation_settings,
            train_on_outputs_only=True,
        )

    def analyze_sequence_length(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase, train_column_name: str, ctxt: RunContext
    ) -> None:
        result: TokenLengthData = analyze_token_lengths(dataset, train_column_name, tokenizer)

        ctxt.logger.report_table_message(result._asdict())
        self.produce_coverage_report_from_target(dataset, train_column_name, 1024, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, train_column_name, 1536, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, train_column_name, 2048, tokenizer, ctxt)

    def produce_coverage_report_from_target(
        self,
        dataset: Dataset,
        column_name: str,
        target_sequence_len: int,
        tokenizer: PreTrainedTokenizerBase,
        ctxt: RunContext,
    ) -> None:
        coverage_percent: float = get_percent_samples_within_sequence_length(
            dataset, column_name, tokenizer, target_sequence_len
        )
        coverage_loss_count = (1.0 - coverage_percent) * len(dataset)
        ctxt.logger.report_message(
            f"Coverage for {target_sequence_len}: {coverage_percent}%. Loss: {coverage_loss_count} samples."
        )
