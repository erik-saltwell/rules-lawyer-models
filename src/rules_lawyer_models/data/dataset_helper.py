from __future__ import annotations

from datasets import DatasetDict

from rules_lawyer_models.core.run_context import RunContext


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
        stratify_by_column_name (str, optional): The name of the label column. Defaults to "".

    Returns:
        DatasetDict: A new DatasetDict with 'train', 'validation', and 'test' splits.
    """
    if test_percent_of_total + validation_percent_of_total >= 1.0:
        raise ValueError("The sum of test_percent_of_total and val_percent_of_total must be less than 1.0")

    non_train_percent_of_total = test_percent_of_total + validation_percent_of_total
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
