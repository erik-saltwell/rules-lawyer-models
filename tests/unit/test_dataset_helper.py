from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from rules_lawyer_models.data.dataset_helper import (
    add_string_label_column,
    make_stress_split,
    rebalance_minority_class,
    split_dataset,
)


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
    def test_returns_largest_rows(self, _mock_compute: MagicMock) -> None:
        texts = [
            "one",  # 1 word
            "one two three four",  # 4 words
            "one two",  # 2 words
            "one two three",  # 3 words
            "one two three four five",  # 5 words
        ]
        ds = Dataset.from_dict({"text": texts, "id": list(range(len(texts)))})

        result = make_stress_split(ds, number_of_rows=3, text_column_name="text", tokenizer=MagicMock())

        assert len(result) == 3
        assert list(result["id"]) == [4, 1, 3]

    @patch("rules_lawyer_models.data.dataset_helper.compute_tokens", side_effect=_word_count)
    def test_request_more_than_available(self, _mock_compute: MagicMock) -> None:
        ds = Dataset.from_dict({"text": ["a b", "a"], "id": [0, 1]})

        result = make_stress_split(ds, number_of_rows=10, text_column_name="text", tokenizer=MagicMock())

        assert len(result) == 2
        assert list(result["id"]) == [0, 1]


class TestAddStringLabelColumn:
    def test_classlabel_to_string(self) -> None:
        features = Features(
            {
                "label": ClassLabel(names=["cat", "dog", "bird"]),
            }
        )
        ds = Dataset.from_dict({"label": [0, 1, 2, 0]}, features=features)

        result = add_string_label_column(ds, "label", "label_str")

        assert "label_str" in result.column_names
        assert list(result["label_str"]) == ["cat", "dog", "bird", "cat"]

    def test_string_column_copied(self) -> None:
        features = Features({"name": Value("string")})
        ds = Dataset.from_dict({"name": ["alice", "bob"]}, features=features)

        result = add_string_label_column(ds, "name", "name_copy")

        assert list(result["name_copy"]) == ["alice", "bob"]

    def test_missing_column_raises_key_error(self) -> None:
        ds = Dataset.from_dict({"x": [1, 2]})

        with pytest.raises(KeyError, match="not_here"):
            add_string_label_column(ds, "not_here", "out")

    def test_duplicate_column_raises_value_error(self) -> None:
        features = Features({"label": ClassLabel(names=["a", "b"])})
        ds = Dataset.from_dict({"label": [0, 1]}, features=features)

        with pytest.raises(ValueError, match="already exists"):
            add_string_label_column(ds, "label", "label")

    def test_preserves_original_columns(self) -> None:
        features = Features(
            {
                "label": ClassLabel(names=["x", "y"]),
                "data": Value("string"),
            }
        )
        ds = Dataset.from_dict({"label": [0, 1], "data": ["a", "b"]}, features=features)

        result = add_string_label_column(ds, "label", "label_str")

        assert list(result["label"]) == [0, 1]
        assert list(result["data"]) == ["a", "b"]


class TestRebalanceMinorityClass:
    @pytest.fixture()
    def imbalanced_dataset(self) -> Dataset:
        """100 minority + 900 majority rows."""
        labels = ["pos"] * 100 + ["neg"] * 900
        return Dataset.from_dict({"text": [f"row_{i}" for i in range(1000)], "label": labels})

    def test_rebalances_to_target(self, imbalanced_dataset: Dataset) -> None:
        result = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05)

        minority_count = list(result["label"]).count("pos")
        actual_percent = minority_count / len(result)
        assert actual_percent == pytest.approx(0.3, abs=0.05)

    def test_all_minority_rows_preserved(self, imbalanced_dataset: Dataset) -> None:
        result = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05)

        assert list(result["label"]).count("pos") == 100

    def test_total_rows_reduced(self, imbalanced_dataset: Dataset) -> None:
        result = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05)

        assert len(result) < len(imbalanced_dataset)

    def test_already_within_tolerance_returns_unchanged(self) -> None:
        labels = ["pos"] * 30 + ["neg"] * 70
        ds = Dataset.from_dict({"label": labels})

        result = rebalance_minority_class(ds, "label", target_minority_percent=0.3, tolerance=0.05)

        assert len(result) == len(ds)

    def test_seed_is_deterministic(self, imbalanced_dataset: Dataset) -> None:
        r1 = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05, seed=99)
        r2 = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05, seed=99)

        assert list(r1["text"]) == list(r2["text"])

    def test_different_seeds_give_different_samples(self, imbalanced_dataset: Dataset) -> None:
        r1 = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05, seed=1)
        r2 = rebalance_minority_class(imbalanced_dataset, "label", target_minority_percent=0.3, tolerance=0.05, seed=2)

        assert list(r1["text"]) != list(r2["text"])

    def test_multiclass_uses_smallest_as_minority(self) -> None:
        labels = ["a"] * 50 + ["b"] * 300 + ["c"] * 650
        ds = Dataset.from_dict({"label": labels})

        result = rebalance_minority_class(ds, "label", target_minority_percent=0.2, tolerance=0.05)

        minority_count = list(result["label"]).count("a")
        assert minority_count == 50
        assert minority_count / len(result) == pytest.approx(0.2, abs=0.05)

    def test_missing_column_raises_key_error(self) -> None:
        ds = Dataset.from_dict({"x": [1, 2]})

        with pytest.raises(KeyError, match="no_col"):
            rebalance_minority_class(ds, "no_col", target_minority_percent=0.3, tolerance=0.05)

    def test_target_zero_raises_value_error(self) -> None:
        ds = Dataset.from_dict({"label": ["a", "b"]})

        with pytest.raises(ValueError, match="between 0 and 1"):
            rebalance_minority_class(ds, "label", target_minority_percent=0.0, tolerance=0.05)

    def test_target_one_raises_value_error(self) -> None:
        ds = Dataset.from_dict({"label": ["a", "b"]})

        with pytest.raises(ValueError, match="between 0 and 1"):
            rebalance_minority_class(ds, "label", target_minority_percent=1.0, tolerance=0.05)

    def test_negative_tolerance_raises_value_error(self) -> None:
        ds = Dataset.from_dict({"label": ["a", "b"]})

        with pytest.raises(ValueError, match="tolerance must be positive"):
            rebalance_minority_class(ds, "label", target_minority_percent=0.3, tolerance=-0.01)

    def test_cannot_achieve_target_raises_value_error(self) -> None:
        # 5 minority, 10 majority — targeting 2% minority would need 245 majority rows
        labels = ["rare"] * 5 + ["common"] * 10
        ds = Dataset.from_dict({"label": labels})

        with pytest.raises(ValueError, match="Cannot achieve target"):
            rebalance_minority_class(ds, "label", target_minority_percent=0.02, tolerance=0.005)
