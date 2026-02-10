from __future__ import annotations

from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Value
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.core.run_context import RunContext
from rules_lawyer_models.exploration.token_length import compute_tokens


def split_dataset(
    datasets: DatasetDict,
    dataset_name: str,
    validation_percent_of_total: float,
    test_percent_of_total: float,
    ctxt: RunContext,
    stratify_by_column_name: str | None = None,
) -> DatasetDict:
    """Splits a dataset into train, validation, and test sets.

    Args:
        datasets (DatasetDict): The original dataset dictionary.
        dataset_name (str): The name of the dataset to split.
        test_percent_of_total (float): The percentage of the total dataset to allocate to the test set.
        validation_percent_of_total (float): The percentage of the total dataset to allocate to the validation set.
        ctxt (RunContext): The run context containing configuration and logging.
        stratify_by_column_name (str, optional): The name of the label column. Defaults to None.

    Returns:
        DatasetDict: A new DatasetDict with 'train', 'validation', and 'test' splits.
    """

    if test_percent_of_total + validation_percent_of_total >= 1.0:
        raise ValueError("The sum of test_percent_of_total and val_percent_of_total must be less than 1.0")

    non_train_percent_of_total = test_percent_of_total + validation_percent_of_total
    if non_train_percent_of_total == 0.0:
        raise ValueError("At least one of test_percent_of_total or val_percent_of_total must be greater than 0.0")

    validation_percent_of_non_train = validation_percent_of_total / non_train_percent_of_total

    train_nontrain_sets = datasets[dataset_name].train_test_split(
        test_size=non_train_percent_of_total,
        stratify_by_column=stratify_by_column_name,
        seed=ctxt.seed,
    )
    test_val_sets = train_nontrain_sets["test"].train_test_split(
        test_size=validation_percent_of_non_train,
        stratify_by_column=stratify_by_column_name,
        seed=ctxt.seed,
    )
    splits = DatasetDict(
        {
            "train": train_nontrain_sets["train"],
            "validation": test_val_sets["test"],
            "test": test_val_sets["train"],
        }
    )
    return splits


def make_stress_split(
    dataset: Dataset,
    number_of_rows: int,
    text_column_name: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Return a subset containing the rows with the longest token counts.

    Args:
        dataset: The source dataset.
        number_of_rows: How many rows to keep.
        text_column_name: The column containing text to measure.
        tokenizer: Tokenizer used to count tokens.

    Returns:
        A Dataset with the top *number_of_rows* rows sorted by descending token count.
    """
    token_counts = [compute_tokens(text, tokenizer) for text in dataset[text_column_name]]
    dataset = dataset.add_column("_token_count", token_counts)  # pyright: ignore
    dataset = dataset.sort("_token_count", reverse=True)
    dataset = dataset.select(range(min(number_of_rows, len(dataset))))
    dataset = dataset.remove_columns("_token_count")
    return dataset


def take(dataset: Dataset, count: int) -> Dataset:
    """Return a new Dataset with at most *count* records from the start."""
    return dataset.select(range(min(count, len(dataset))))


def add_string_label_column(
    dataset: Dataset,
    classlabel_column_name: str,
    new_column_name: str,
) -> Dataset:
    """
    Return a new Dataset with an extra column `new_column_name` containing the
    string form of a ClassLabel column.

    - If `classlabel_column_name` is a ClassLabel feature, values are converted via int2str().
    - If it is already a string column, values are copied (as strings).
    """
    if classlabel_column_name not in dataset.column_names:
        raise KeyError(f"Column '{classlabel_column_name}' not found. Available columns: {dataset.column_names}")

    if new_column_name in dataset.column_names:
        raise ValueError(f"Column '{new_column_name}' already exists. Choose a different name.")

    feature: Any = dataset.features.get(classlabel_column_name)
    if feature is None:
        raise KeyError(
            f"No feature metadata found for column '{classlabel_column_name}'. "
            "Was this Dataset constructed without features?"
        )

    def _to_str(x: Any) -> str | None:
        if x is None:
            return None
        # Many datasets store ClassLabel values as ints (sometimes numpy ints).
        try:
            return feature.int2str(int(x))  # type: ignore
        except Exception:
            return str(x)

    # Preferred path: ClassLabel mapping.
    if isinstance(feature, ClassLabel):

        def _map_a(batch: dict[str, list[Any]]) -> dict[str, list[str | None]]:
            labels = batch[classlabel_column_name]
            return {new_column_name: [_to_str(v) for v in labels]}

        return dataset.map(_map_a, batched=True)

    # If it’s already a string column, just copy/normalize to str.
    if isinstance(feature, Value) and feature.dtype == "string":

        def _map_b(batch: dict[str, list[Any]]) -> dict[str, list[str | None]]:
            labels = batch[classlabel_column_name]
            return {new_column_name: [None if v is None else str(v) for v in labels]}

        return dataset.map(_map_b, batched=True)

    # Fallback: try to convert anyway (useful if the dataset lost ClassLabel metadata).
    def _map_c(batch: dict[str, list[Any]]) -> dict[str, list[str | None]]:
        labels = batch[classlabel_column_name]
        return {new_column_name: [None if v is None else str(v) for v in labels]}

    return dataset.map(_map_c, batched=True)
