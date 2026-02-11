from __future__ import annotations

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data import add_string_label_column, add_training_column
from rules_lawyer_models.serialization import load_dataset_from_hf, load_tokenizer_from_hf
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID, get_fragment

from .command_protocol import CommmandProtocol


class VerifyTemplateData(CommmandProtocol):
    def __init__(
        self,
        num_rows: int,
        dataset_name: DatasetName,
        base_model_name: BaseModelName,
        system_prompt_id: FragmentID,
        content_column_name: str,
        labels_column_name: str,
        training_column_name: str,
        eval_column_name: str,
    ) -> None:
        self.num_rows = num_rows
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.system_prompt_id = system_prompt_id
        self.content_column_name = content_column_name
        self.labels_column_name = labels_column_name
        self.training_column_name = training_column_name
        self.eval_column_name = eval_column_name

    def execute(self, ctxt: RunContext) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_name)
        dataset: Dataset = datasets["train"]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_name)
        string_labels_column_name = "str_" + self.labels_column_name
        dataset = add_string_label_column(dataset, self.labels_column_name, string_labels_column_name)
        # dataset = dataset.take(2)
        dataset = add_training_column(
            self.base_model_name,
            dataset,
            self.content_column_name,
            string_labels_column_name,
            self.training_column_name,
            get_fragment(self.system_prompt_id),
            tokenizer,
        )
        # dataset = add_eval_column(
        #     self.base_model_name,
        #     dataset,
        #     self.content_column_name,
        #     string_labels_column_name,
        #     self.eval_column_name,
        #     get_fragment(self.system_prompt_id),
        #     tokenizer,
        # )

        for i in range(min(self.num_rows, len(dataset))):
            ctxt.logger.report_message(f"--- Row {i} TRAINING ---")
            ctxt.logger.report_message(dataset[i][self.training_column_name])
            # txt.logger.report_message(f"--- Row {i} eval ---")
            # ctxt.logger.report_message(dataset[i][self.eval_column_name])
