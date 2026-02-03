from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rules_lawyer_models.utils.model_name import BaseModelName


def load_tokenizer_from_hf(model_id: BaseModelName) -> PreTrainedTokenizerBase:
    """Load and return the tokenizer for the given base model name."""

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
    return tokenizer
