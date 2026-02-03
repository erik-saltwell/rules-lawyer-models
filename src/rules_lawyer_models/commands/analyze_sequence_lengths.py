from __future__ import annotations

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data import add_string_label_column, add_templated_column
from rules_lawyer_models.exploration import (
    TokenLengthData,
    analyze_token_lengths,
    get_percent_samples_within_sequence_length,
)
from rules_lawyer_models.serialization import load_dataset_from_hf, load_tokenizer_from_hf
from rules_lawyer_models.utils import get_fragment

from .command_protocol import CommmandProtocol


class AnalyzeSequenceLengths(CommmandProtocol):
    def execute(self, ctxt: RunContext) -> None:
        output_column_name: str = "text"
        datasets: DatasetDict = load_dataset_from_hf(ctxt.dataset_name)
        dataset: Dataset = datasets["train"]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(ctxt.base_model_name)
        dataset = add_string_label_column(dataset, "label", "str_label")
        dataset = add_templated_column(
            dataset, "content", "str_label", get_fragment(ctxt.system_prompt_name), tokenizer
        )
        result: TokenLengthData = analyze_token_lengths(dataset, output_column_name, tokenizer)

        ctxt.logger.report_table_message(result._asdict())
        self.produce_coverage_report_from_target(dataset, output_column_name, 1024, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, output_column_name, 1536, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, output_column_name, 2048, tokenizer, ctxt)

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
