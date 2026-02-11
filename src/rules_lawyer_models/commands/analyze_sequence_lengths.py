from __future__ import annotations

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data import add_string_label_column
from rules_lawyer_models.data.template_helper import add_training_column
from rules_lawyer_models.exploration import (
    TokenLengthData,
    analyze_token_lengths,
    get_percent_samples_within_sequence_length,
)
from rules_lawyer_models.serialization import load_dataset_from_hf, load_tokenizer_from_hf
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID, get_fragment

from .command_protocol import CommmandProtocol


class AnalyzeSequenceLengths(CommmandProtocol):
    def __init__(
        self,
        dataset_name: DatasetName,
        base_model_name: BaseModelName,
        system_prompt_id: FragmentID,
        content_column_name: str,
        labels_column_name: str,
        training_column_name: str,
    ):
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.system_prompt_id = system_prompt_id
        self.content_column_name = content_column_name
        self.labels_column_name = labels_column_name
        self.training_column_name = training_column_name

    def execute(self, ctxt: RunContext) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_name)
        dataset: Dataset = datasets["train"]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_name)
        string_labels_column_name = "str_" + self.labels_column_name
        dataset = add_string_label_column(dataset, self.labels_column_name, string_labels_column_name)
        dataset = add_training_column(
            self.base_model_name,
            dataset,
            self.content_column_name,
            string_labels_column_name,
            self.training_column_name,
            get_fragment(self.system_prompt_id),
            tokenizer,
        )

        result: TokenLengthData = analyze_token_lengths(dataset, self.training_column_name, tokenizer)

        ctxt.logger.report_table_message(result._asdict())
        self.produce_coverage_report_from_target(dataset, self.training_column_name, 1024, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, self.training_column_name, 1536, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, self.training_column_name, 2048, tokenizer, ctxt)

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
