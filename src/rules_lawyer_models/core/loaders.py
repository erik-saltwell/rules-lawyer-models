from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

import rules_lawyer_models.utils as utils


def load_tokenizer(model_id: utils.BaseModelName) -> PreTrainedTokenizerBase:
    """Load and return the tokenizer for the given base model name."""

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
    return tokenizer
