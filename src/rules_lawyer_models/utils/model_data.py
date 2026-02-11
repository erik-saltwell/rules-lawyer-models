from __future__ import annotations

import enum
from dataclasses import dataclass

from rules_lawyer_models.utils.text_fragments import FragmentID

# Separator constants by model family
chat_ml_instruction_seperator = "<|im_start|>user\n"
chat_ml_response_seperator = "<|im_start|>assistant\n"
llama_instruction_seperator = "<|start_header_id|>user<|end_header_id|>\n\n"
llama_response_seperator = "<|start_header_id|>assistant<|end_header_id|>\n\n"
mistral_instruction_seperator = "[INST]"
mistral_response_seperator = "[/INST]"
gemma_instruction_seperator = "<start_of_turn>user\n"
gemma_response_seperator = "<start_of_turn>model\n"
phi_instruction_seperator = "<|user|>\n"
phi_response_seperator = "<|assistant|>\n"


class BaseModelName(enum.StrEnum):
    QWEN_25_14B_4BIT_BASE = "unsloth/Qwen2.5-14B-bnb-4bit"
    QWEN_25_14B_4BIT_INSTRUCT = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    QWEN_25_3B_4BIT_INSTRUCT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    QWEN_25_3B_05BIT_INSTRUCT = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    QWEN_25_1_5B_INSTRUCT = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    NONE = "none"


class TargetModelName(enum.StrEnum):
    REDDIT_RPG_POST_CLASSIFICATION = "reddit_rpg_post_classification"
    IMDB_TEST = "imdb_sentiment_test"
    NONE = "none"


@dataclass(frozen=True)
class ModelData:
    is_instruct: bool
    instruction_seperator: str | None = None
    response_seperator: str | None = None
    training_fragment_id: FragmentID | None = None
    eval_fragment_id: FragmentID | None = None


_model_registry: dict[BaseModelName, ModelData] = {
    BaseModelName.QWEN_25_1_5B_INSTRUCT: ModelData(
        is_instruct=True,
        instruction_seperator=chat_ml_instruction_seperator,
        response_seperator=chat_ml_response_seperator,
    ),
    BaseModelName.QWEN_25_14B_4BIT_BASE: ModelData(
        is_instruct=False,
        training_fragment_id=FragmentID.ALPACA_PROMPT_TEMPLATE,
        eval_fragment_id=FragmentID.ALPACA_PROMPT_TEMPLATE,
    ),
    BaseModelName.QWEN_25_14B_4BIT_INSTRUCT: ModelData(
        is_instruct=True,
        instruction_seperator=chat_ml_instruction_seperator,
        response_seperator=chat_ml_response_seperator,
    ),
    BaseModelName.QWEN_25_3B_4BIT_INSTRUCT: ModelData(
        is_instruct=True,
        instruction_seperator=chat_ml_instruction_seperator,
        response_seperator=chat_ml_response_seperator,
    ),
    BaseModelName.QWEN_25_3B_05BIT_INSTRUCT: ModelData(
        is_instruct=True,
        instruction_seperator=chat_ml_instruction_seperator,
        response_seperator=chat_ml_response_seperator,
    ),
}


def get_model_data(model_name: BaseModelName) -> ModelData:
    if model_name not in _model_registry:
        raise KeyError(f"No model data registered for {model_name}")
    return _model_registry[model_name]
