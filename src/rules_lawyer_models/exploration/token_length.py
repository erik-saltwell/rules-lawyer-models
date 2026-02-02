from transformers import PreTrainedTokenizerBase


def compute_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Compute the number of tokens in the given text."""
    # Simple whitespace-based tokenization for demonstration purposes.
    return len(tokenizer.encode(text))
