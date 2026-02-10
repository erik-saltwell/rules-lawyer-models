from collections.abc import Callable, Mapping

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.utils.model_data import ModelData, get_model_data
from rules_lawyer_models.utils.model_name import BaseModelName
from rules_lawyer_models.utils.text_fragments import get_fragment


def _generate_eos(tokenizer: PreTrainedTokenizerBase) -> str:
    eos = tokenizer.eos_token
    if eos is None:
        raise ValueError("The tokenizer does not have an EOS token defined.")
    if isinstance(eos, list):
        eos = eos[0]
    eos_token: str = eos
    return eos_token


def _apply_simple_template(
    instruction: str, input: str, output: str, template: str, tokenizer: PreTrainedTokenizerBase, eos: str | None
) -> str:
    return_value: str = template.format(instruction, input, output)
    if eos and not return_value.strip().endswith(eos):
        return_value = return_value.rstrip() + eos
    return return_value


def _apply_chat_template(
    instruction: str, input: str, output: str, tokenizer: PreTrainedTokenizerBase, eos: str | None
) -> str:
    return_value: str
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input},
    ]

    if output:
        messages.append({"role": "assistant", "content": output})
        return_value = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        if eos and not return_value.rstrip().endswith(eos):
            return_value = return_value.rstrip() + eos
    else:
        return_value = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return return_value


RowFormatter = Callable[[str, str], str]


def _apply_template(
    data: Mapping[str, list[str]],
    format_row: RowFormatter,
    content_column_name: str,
    labels_column_name: str,
    new_column_name: str,
) -> dict[str, list[str]]:
    inputs = data[content_column_name]
    outputs = data[labels_column_name]
    return {new_column_name: [format_row(content, label) for content, label in zip(inputs, outputs, strict=True)]}


def add_training_column(
    model_name: BaseModelName,
    dataset: Dataset,
    content_column_name: str,
    labels_column_name: str,
    training_column_name: str,
    instruction_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    eos = _generate_eos(tokenizer)
    model_data: ModelData = get_model_data(model_name)
    if model_data.is_instruct:

        def format_row(inp: str, out: str) -> str:
            return _apply_chat_template(instruction_prompt, inp, out, tokenizer, eos)
    else:
        if model_data.training_fragment_id is None:
            raise ValueError(f"No training template fragment for base model {model_name}")
        template = get_fragment(model_data.training_fragment_id)

        def format_row(inp: str, out: str) -> str:
            return _apply_simple_template(instruction_prompt, inp, out, template, tokenizer, eos)

    return dataset.map(
        lambda data: _apply_template(data, format_row, content_column_name, labels_column_name, training_column_name),
        batched=True,
    )


def add_eval_column(
    model_name: BaseModelName,
    dataset: Dataset,
    content_column_name: str,
    labels_column_name: str,
    eval_column_name: str,
    instruction_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    model_data: ModelData = get_model_data(model_name)
    if model_data.is_instruct:

        def format_row(inp: str, _out: str) -> str:
            return _apply_chat_template(instruction_prompt, inp, "", tokenizer, None)
    else:
        if model_data.eval_fragment_id is None:
            raise ValueError(f"No eval template fragment for base model {model_name}")
        template = get_fragment(model_data.eval_fragment_id)

        def format_row(inp: str, _out: str) -> str:
            return _apply_simple_template(instruction_prompt, inp, "", template, tokenizer, None)

    return dataset.map(
        lambda data: _apply_template(data, format_row, content_column_name, labels_column_name, eval_column_name),
        batched=True,
    )
