from unsloth.tokenizer_utils import load_correct_tokenizer  # isort: skip
from typing import cast

from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.utils.model_name import BaseModelName


def load_tokenizer_from_hf(model_id: BaseModelName) -> PreTrainedTokenizerBase:
    """Load and return the tokenizer for the given base model name."""

    tokenizer = cast(PreTrainedTokenizerBase, load_correct_tokenizer(model_id))
    return tokenizer
