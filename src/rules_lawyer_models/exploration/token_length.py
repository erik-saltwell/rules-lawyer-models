from rules_lawyer_models.commands import RunContext


def compute_tokens(text: str, ctxt: RunContext) -> int:
    """Compute the number of tokens in the given text."""
    # Simple whitespace-based tokenization for demonstration purposes.
    tokens = text.split()
    return len(tokens)
