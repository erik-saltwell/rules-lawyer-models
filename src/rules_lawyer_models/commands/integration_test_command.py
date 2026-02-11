from dataclasses import dataclass

from rules_lawyer_models.core import RunContext


@dataclass
class IntegrationTestCommand:
    def execute(self, ctxt: RunContext) -> None:
        ...
        # dataset_name = DatasetName.IMDB_TEST
        # base_model_name = BaseModelName.QWEN_25_1_5B_INSTRUCT
        # system_prompt_id = FragmentID.IMDB_TEST_PROMPT
        # target_model_name: TargetModelName = TargetModelName.IMDB_TEST

        # split_name = "train"
        # content_column_name = "text"
        # label_column_name = "label"
        # train_column_name = "train"
        # str_label_column_name = "str_" + label_column_name
        # eval_column_name = "eval"
