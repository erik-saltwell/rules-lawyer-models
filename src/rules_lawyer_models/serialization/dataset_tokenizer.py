from typing import cast

from datasets import DatasetDict, load_dataset, load_from_disk

from rules_lawyer_models.utils.common_paths import CommonPaths
from rules_lawyer_models.utils.dataset_name import DatasetName


def load_dataset_from_hf(dataset_name: DatasetName) -> DatasetDict:
    return cast(DatasetDict, load_dataset(dataset_name))


def load_dataset_from_disk(paths: CommonPaths) -> DatasetDict:
    path = paths.computed_datasets
    return cast(DatasetDict, load_from_disk(str(path)))


def save_dataset_to_disk(dataset: DatasetDict, paths: CommonPaths) -> None:
    path = paths.computed_datasets
    dataset.save_to_disk(str(path))
