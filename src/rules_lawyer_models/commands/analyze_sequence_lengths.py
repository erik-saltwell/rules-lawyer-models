from __future__ import annotations

from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext, load_tokenizer
from rules_lawyer_models.data import add_string_label_column, add_templated_column
from rules_lawyer_models.exploration import (
    TokenLengthData,
    analyze_token_lengths,
    get_percent_samples_within_sequence_length,
)
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID, get_fragment

from .command_protocol import CommmandProtocol


class AnalyzeSequenceLengths(CommmandProtocol):
    def execute(self, ctxt: RunContext) -> None:
        datasets: DatasetDict = cast(DatasetDict, load_dataset(DatasetName.REDDIT_RPG_POST_CLASSIFICATION.value))
        dataset: Dataset = datasets["train"]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer(BaseModelName.QWEN_25_14B_4BIT_INSTRUCT)
        dataset = add_string_label_column(dataset, "label", "str_label")
        dataset = add_templated_column(
            dataset, "content", "str_label", get_fragment(FragmentID.RPG_POST_CLASSIFICATION_PROMPT), tokenizer
        )
        result: TokenLengthData = analyze_token_lengths(dataset, "text", tokenizer)

        ctxt.logger.report_table_message(result._asdict())
        self.produce_coverage_report_from_target(dataset, "text", 1024, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, "text", 1536, tokenizer, ctxt)
        self.produce_coverage_report_from_target(dataset, "text", 2048, tokenizer, ctxt)

    def produce_coverage_report_from_target(
        self,
        dataset: Dataset,
        column_name: str,
        target_sequence_len: int,
        tokenizer: PreTrainedTokenizerBase,
        ctxt: RunContext,
    ) -> None:
        coverage_percent: float = get_percent_samples_within_sequence_length(
            dataset, "text", tokenizer, target_sequence_len
        )
        coverage_loss_count = (1.0 - coverage_percent) * len(dataset)
        ctxt.logger.report_message(
            f"Coverage for {target_sequence_len}: {coverage_percent}%. Loss: {coverage_loss_count} samples."
        )
