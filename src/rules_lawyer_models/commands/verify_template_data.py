from __future__ import annotations

from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data import add_string_label_column, add_templated_column
from rules_lawyer_models.serialization import load_tokenizer_from_hf
from rules_lawyer_models.utils import get_fragment

from .command_protocol import CommmandProtocol


class VerifyTemplateData(CommmandProtocol):
    def __init__(self, num_rows: int) -> None:
        self.num_rows = num_rows

    def execute(self, ctxt: RunContext) -> None:
        datasets: DatasetDict = cast(DatasetDict, load_dataset(ctxt.dataset_name.value))
        dataset: Dataset = datasets["train"]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(ctxt.base_model_name)
        dataset = add_string_label_column(dataset, "label", "str_label")
        dataset = add_templated_column(
            dataset, "content", "str_label", get_fragment(ctxt.system_prompt_name), tokenizer
        )

        for i in range(min(self.num_rows, len(dataset))):
            ctxt.logger.report_message(f"--- Row {i} ---")
            ctxt.logger.report_message(dataset[i]["text"])
