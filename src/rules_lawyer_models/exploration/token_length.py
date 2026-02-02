from typing import NamedTuple

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


class TokenLengthData(NamedTuple):
    p80: int
    p90: int
    p95: int
    p98: int
    p99: int
    p99_5: int
    p100: int


def compute_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Compute the number of tokens in the given text."""
    # Simple whitespace-based tokenization for demonstration purposes.
    return len(tokenizer.encode(text))


def analyze_token_lengths(dataset: Dataset, column_name: str, tokenizer: PreTrainedTokenizerBase) -> TokenLengthData:
    lengths = np.array([compute_tokens(row[column_name], tokenizer) for row in dataset])  # pyright: ignore
    p80, p90, p95, p98, p99, p99_5, p100 = np.percentile(lengths, [80, 90, 95, 98, 99, 99.5, 100]).astype(int)
    return TokenLengthData(
        p80=int(p80),
        p90=int(p90),
        p95=int(p95),
        p98=int(p98),
        p99=int(p99),
        p99_5=int(p99_5),
        p100=int(p100),
    )
