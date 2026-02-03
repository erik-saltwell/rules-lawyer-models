from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.utils.model_name import BaseModelName
from rules_lawyer_models.utils.text_fragments import FragmentID, get_fragment

model_fragment_map: dict[BaseModelName, FragmentID] = {
    BaseModelName.QWEN_25_14B_4BIT_BASE: FragmentID.ALPACA_PROMPT_TEMPLATE,
}


def get_template_skeleton(model_name: BaseModelName) -> str:
    fragment_id = model_fragment_map.get(model_name)
    if fragment_id is None:
        raise ValueError(f"No template fragment found for model {model_name}")
    return get_fragment(fragment_id)


def base_model_is_instruct(model_name: BaseModelName) -> bool:
    return model_name not in model_fragment_map


def add_templated_column_for_instruct_models(
    dataset: Dataset,
    input_column_name: str,
    output_column_name: str,
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    if EOS_TOKEN is None:
        raise ValueError("The tokenizer does not have an EOS token defined.")

    def formatting_prompts_func(examples):
        inputs = examples[input_column_name]
        outputs = examples[output_column_name]
        texts = []
        for input, output in zip(inputs, outputs, strict=True):
            # Create a message list suitable for ChatML/Instruct templates
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input},
                {"role": "assistant", "content": output},
            ]

            # Apply the Qwen 2.5 specific chat template [7, 10, 15, 18]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Append EOS token to ensure the model learns boundary termination
            texts.append(text + EOS_TOKEN)  # pyright: ignore
        return {"text": texts}

    # Map the formatted prompts into the dataset for the SFTTrainer
    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset


def add_templated_column_for_non_instruct_models(
    dataset: Dataset,
    skeleton_prompt: str,
    input_column_name: str,
    output_column_name: str,
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    if EOS_TOKEN is None:
        raise ValueError("The tokenizer does not have an EOS token defined.")

    def format_prompts(examples):
        instructions = [system_prompt for _ in examples[input_column_name]]
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


def add_templated_column(
    dataset: Dataset,
    input_column_name: str,
    output_column_name: str,
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    if base_model_is_instruct(tokenizer.name_or_path):
        return add_templated_column_for_instruct_models(
            dataset=dataset,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
    else:
        skeleton_prompt = get_template_skeleton(BaseModelName(tokenizer.name_or_path))
        return add_templated_column_for_non_instruct_models(
            dataset=dataset,
            skeleton_prompt=skeleton_prompt,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
