from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data.dataset_helper import prep_classification_dataset_for_trainng
from rules_lawyer_models.exploration import (
    TokenLengthData,
    analyze_token_lengths,
    dataset_exploration,
    get_percent_samples_within_sequence_length,
)
from rules_lawyer_models.serialization import load_dataset_from_hf, load_tokenizer_from_hf
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID


@dataclass
class IntegrationTestCommand:
    def execute(self, ctxt: RunContext) -> None:
        dataset_name = DatasetName.IMDB_TEST
        base_model_name = BaseModelName.QWEN_25_1_5B_INSTRUCT
        system_prompt_id = FragmentID.IMDB_TEST_PROMPT
        # target_model_name: TargetModelName = TargetModelName.IMDB_TEST

        split_name = "train"
        content_column_name = "text"
        label_column_name = "label"
        train_column_name = "train"
        str_label_column_name = "str_" + label_column_name
        # eval_column_name = "eval"

        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(base_model_name)

        dataset_dict: DatasetDict = load_dataset_from_hf(dataset_name)
        dataset_all: Dataset = dataset_dict[split_name]
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
        # 1536 tokens only leave out 21 samples.

        return

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
