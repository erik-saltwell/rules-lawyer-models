from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from rules_lawyer_models.data.dataset_helper import make_stress_split, split_dataset


@pytest.fixture()
def sample_dataset() -> DatasetDict:
    """A 1000-row dataset with a 'data_type' column (3 classes).

    Class distribution: 500 × alpha, 300 × beta, 200 × gamma.
    """
    labels = [0] * 500 + [1] * 300 + [2] * 200
    features = Features({"text": Value("string"), "data_type": ClassLabel(names=["alpha", "beta", "gamma"])})
    ds = Dataset.from_dict(
        {"text": [f"row_{i}" for i in range(1000)], "data_type": labels},
        features=features,
    )
    return DatasetDict({"train": ds})


@pytest.fixture()
def mock_ctxt() -> MagicMock:
    ctxt = MagicMock()
    ctxt.seed = 42
    return ctxt


class TestSplitDataset:
    def test_70_20_10_split(self, sample_dataset: DatasetDict, mock_ctxt: MagicMock) -> None:
        splits = split_dataset(
            datasets=sample_dataset,
            dataset_name="train",
            validation_percent_of_total=0.2,
            test_percent_of_total=0.1,
            ctxt=mock_ctxt,
        )

        total = len(sample_dataset["train"])
        assert set(splits.keys()) == {"train", "validation", "test"}

        assert len(splits["train"]) == pytest.approx(total * 0.7, abs=total * 0.05)
        assert len(splits["validation"]) == pytest.approx(total * 0.2, abs=total * 0.05)
        assert len(splits["test"]) == pytest.approx(total * 0.1, abs=total * 0.05)

        # All rows accounted for, no duplicates
        assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == total

    def test_stratify_preserves_class_distribution(self, sample_dataset: DatasetDict, mock_ctxt: MagicMock) -> None:
        splits = split_dataset(
            datasets=sample_dataset,
            dataset_name="train",
            validation_percent_of_total=0.2,
            test_percent_of_total=0.1,
            ctxt=mock_ctxt,
            stratify_by_column_name="data_type",
        )

        original_dist = {0: 0.5, 1: 0.3, 2: 0.2}

        for split_name in ("train", "validation", "test"):
            split = splits[split_name]
            n = len(split)
            for label, expected_ratio in original_dist.items():
                count = split["data_type"].count(label)
                actual_ratio = count / n
                assert actual_ratio == pytest.approx(expected_ratio, abs=0.05), (
                    f"Split '{split_name}': data_type {label} ratio {actual_ratio:.3f} "
                    f"deviates from expected {expected_ratio:.3f}"
                )


def _word_count(text: str, _tokenizer: object) -> int:
    return len(text.split())


class TestMakeStressSplit:
    @patch("rules_lawyer_models.data.dataset_helper.compute_tokens", side_effect=_word_count)
    def test_returns_largest_rows(self, _mock_compute) -> None:
        texts = [
            "one",  # 1 word
            "one two three four",  # 4 words
            "one two",  # 2 words
            "one two three",  # 3 words
            "one two three four five",  # 5 words
        ]
        ds = Dataset.from_dict({"text": texts, "id": list(range(len(texts)))})

        result = make_stress_split(ds, number_of_rows=3, text_column_name="text", tokenizer=None)

        assert len(result) == 3
        assert list(result["id"]) == [4, 1, 3]

    @patch("rules_lawyer_models.data.dataset_helper.compute_tokens", side_effect=_word_count)
    def test_request_more_than_available(self, _mock_compute) -> None:
        ds = Dataset.from_dict({"text": ["a b", "a"], "id": [0, 1]})

        result = make_stress_split(ds, number_of_rows=10, text_column_name="text", tokenizer=None)

        assert len(result) == 2
        assert list(result["id"]) == [0, 1]
