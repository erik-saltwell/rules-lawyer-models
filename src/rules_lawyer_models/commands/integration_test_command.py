from dataclasses import dataclass

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.data.dataset_helper import add_string_label_column
from rules_lawyer_models.data.template_helper import add_training_column
from rules_lawyer_models.exploration import dataset_exploration
from rules_lawyer_models.serialization.dataset_serializer import load_dataset_from_hf
from rules_lawyer_models.serialization.token_serializer import load_tokenizer_from_hf
from rules_lawyer_models.utils import BaseModelName, DatasetName, FragmentID
from rules_lawyer_models.utils.text_fragments import get_fragment


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
        dataset_all = dataset_dict[split_name]
        dataset_all = add_string_label_column(dataset_all, label_column_name, str_label_column_name)
        dataset_all = add_training_column(
            base_model_name,
            dataset_all,
            content_column_name,
            str_label_column_name,
            train_column_name,
            get_fragment(system_prompt_id),
            tokenizer,
        )

        dataset_exploration.dump_first_row_to_logger(dataset_all, ctxt.logger)

        return
