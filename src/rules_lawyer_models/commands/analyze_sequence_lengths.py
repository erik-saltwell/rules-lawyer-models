from __future__ import annotations

from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.commands.command_protocol import CommmandProtocol
from rules_lawyer_models.core.loaders import load_tokenizer
from rules_lawyer_models.core.run_context import RunContext
from rules_lawyer_models.exploration.token_length import TokenLengthData, analyze_token_lengths


class AnalyzeSequenceLengths(CommmandProtocol):
    def execute(self, ctxt: RunContext) -> None:
        datasets: DatasetDict = cast(DatasetDict, load_dataset(ctxt.dataset_name.value))
        dataset: Dataset = datasets["train"]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer(ctxt.base_model_name)
        result: TokenLengthData = analyze_token_lengths(dataset, "text", tokenizer)
        ctxt.logger.report_message(f"{result.p90}")
