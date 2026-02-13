from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING, Any

from datasets import ClassLabel, Dataset, DatasetDict, Value, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from rules_lawyer_models.exploration.token_length import compute_tokens
from rules_lawyer_models.utils import BaseModelName, FragmentID, get_fragment
from rules_lawyer_models.utils.logging_protocol import LoggingProtocol

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.model_generator import ModelGenerator

from .template_helper import add_training_column


def split_dataset(
    dataset: Dataset,
    validation_percent_of_total: float,
    test_percent_of_total: float,
    seed: int,
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

    train_nontrain_sets = dataset.train_test_split(
        test_size=non_train_percent_of_total,
        stratify_by_column=stratify_by_column_name,
        seed=seed,
    )
    test_val_sets = train_nontrain_sets["test"].train_test_split(
        test_size=validation_percent_of_non_train,
        stratify_by_column=stratify_by_column_name,
        seed=seed,
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


def union_datasets(a: Dataset, b: Dataset) -> Dataset:
    """Concatenate two Datasets after validating they share the same columns.

    Raises:
        ValueError: If the column names differ between the two datasets.
    """
    cols_a = set(a.column_names)
    cols_b = set(b.column_names)
    if cols_a != cols_b:
        only_a = cols_a - cols_b
        only_b = cols_b - cols_a
        parts: list[str] = []
        if only_a:
            parts.append(f"only in first: {sorted(only_a)}")
        if only_b:
            parts.append(f"only in second: {sorted(only_b)}")
        raise ValueError(f"Column mismatch — {'; '.join(parts)}")
    return concatenate_datasets([a, b])


def add_predictions_column(
    dataset: Dataset,
    system_prompt_id: FragmentID,
    content_column_name: str,
    predictions_column_name: str,
    generator: ModelGenerator,
    logger: LoggingProtocol | None = None,
) -> Dataset:
    """Generate a prediction for every row and add it as a new column.

    Args:
        dataset: Source dataset.
        system_prompt_id: Fragment ID for the system prompt.
        content_column_name: Column containing the user input text.
        predictions_column_name: Name of the new column to create.
        generator: A ModelGenerator configured for inference.
        logger: Optional logger for progress reporting.

    Returns:
        A new Dataset with the predictions column appended.
    """
    system_prompt = get_fragment(system_prompt_id)
    texts = dataset[content_column_name]
    if logger is not None:
        predictions: list[str] = []
        with logger.progress("Generating predictions", total=len(texts)) as progress:
            for text in texts:
                predictions.append(generator.generate(system_prompt, text))
                progress.advance()
    else:
        predictions = [generator.generate(system_prompt, text) for text in texts]
    return dataset.add_column(predictions_column_name, predictions)  # pyright: ignore


def rebalance_minority_class(
    dataset: Dataset,
    label_column_name: str,
    target_minority_percent: float,
    tolerance: float,
    seed: int = 42,
) -> Dataset:
    """Undersample the majority class so the minority class represents approximately the target percentage.

    Args:
        dataset: The source dataset.
        label_column_name: Column containing class labels.
        target_minority_percent: Desired fraction (0–1) for the minority class.
        tolerance: Acceptable absolute deviation from the target (e.g. 0.05 for ±5%).
        seed: Random seed for reproducible sampling.

    Returns:
        A rebalanced Dataset where the minority class is within tolerance of the target.

    Raises:
        ValueError: If inputs are invalid or the target cannot be achieved.
    """
    if not 0.0 < target_minority_percent < 1.0:
        raise ValueError(f"target_minority_percent must be between 0 and 1 exclusive, got {target_minority_percent}")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if label_column_name not in dataset.column_names:
        raise KeyError(f"Column '{label_column_name}' not found. Available columns: {dataset.column_names}")

    labels: list[Any] = dataset[label_column_name]
    counts: dict[Any, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    minority_class = min(counts, key=counts.__getitem__)
    minority_count = counts[minority_class]
    current_percent = minority_count / len(dataset)

    if abs(current_percent - target_minority_percent) <= tolerance:
        return dataset

    target_non_minority = round(minority_count * (1.0 - target_minority_percent) / target_minority_percent)
    non_minority_indices = [i for i, label in enumerate(labels) if label != minority_class]

    if target_non_minority >= len(non_minority_indices):
        raise ValueError(
            f"Cannot achieve target {target_minority_percent:.1%} ± {tolerance:.1%}: "
            f"minority class '{minority_class}' has {minority_count} rows but would need "
            f"at most {target_non_minority} non-minority rows (have {len(non_minority_indices)})"
        )

    rng = _random.Random(seed)
    sampled_non_minority = rng.sample(non_minority_indices, target_non_minority)

    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    rebalanced = dataset.select(sorted(minority_indices + sampled_non_minority))

    result_percent = minority_count / len(rebalanced)
    if abs(result_percent - target_minority_percent) > tolerance:
        raise ValueError(
            f"Rebalancing landed at {result_percent:.2%} which is outside "
            f"target {target_minority_percent:.1%} ± {tolerance:.1%}"
        )

    return rebalanced


def prep_classification_dataset_for_trainng(
    dataset: Dataset,
    base_model_name: BaseModelName,
    content_column_name: str,
    label_column_name: str,
    string_label_column_name: str,
    training_column_name: str,
    system_prompt_id: FragmentID,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    return_set: Dataset = add_string_label_column(dataset, label_column_name, string_label_column_name)
    return_set = add_training_column(
        base_model_name,
        return_set,
        content_column_name,
        string_label_column_name,
        training_column_name,
        get_fragment(system_prompt_id),
        tokenizer,
    )

    return return_set
