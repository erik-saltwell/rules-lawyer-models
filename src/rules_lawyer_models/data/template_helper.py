from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.utils.model_name import BaseModelName
from rules_lawyer_models.utils.text_fragments import FragmentID, get_fragment

model_fragment_map: dict[BaseModelName, FragmentID] = {
    BaseModelName.QWEN_25_14B_4BIT_INSTRUCT: FragmentID.RPG_POST_CLASSIFICATION_PROMPT,
}


def get_template_skeleton(model_name: BaseModelName) -> str:
    fragment_id = model_fragment_map.get(model_name)
    if fragment_id is None:
        raise ValueError(f"No template fragment found for model {model_name}")
    return get_fragment(fragment_id)


def add_templated_column_for_non_instruct_models(
    dataset: Dataset,
    skeleton_prompt: str,
    input_column_name: str,
    output_column_name: str,
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def format_prompts(examples):
        instructions = [system_prompt for _ in examples["input"]]
        inputs = examples[input_column_name]
        outputs = examples[output_column_name]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs, strict=True):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text: str = skeleton_prompt.format(instruction, input, output)
            text = text + EOS_TOKEN  # pyright: ignore
            texts.append(text)
        return {
            "text": texts,
        }

    dataset = dataset.map(
        format_prompts,
        batched=True,
    )
    return dataset
