from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(slots=True)
class ModelData:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
