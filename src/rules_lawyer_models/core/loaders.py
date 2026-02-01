from typing import Any

from unsloth import FastLanguageModel

import rules_lawyer_models.utils as utils


def load_tokenizer(model_id: utils.BaseModelName) -> Any:
    """Load and return the tokenizer for the given base model name."""

    tokenizer = FastLanguageModel.from_pretrained(model_id).get_tokenizer()
    return tokenizer
