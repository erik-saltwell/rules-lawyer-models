from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from rules_lawyer_models.data.template_helper import add_templated_column_for_non_instruct_models


def _make_tokenizer(eos_token: str = "</s>") -> MagicMock:
    tok = MagicMock()
    tok.eos_token = eos_token
    return tok


class TestAddTemplatedColumnForNonInstructModels:
    def test_adds_text_column_with_eos(self) -> None:
        ds = Dataset.from_dict(
            {
                "input": ["Hello", "World"],
                "output": ["Hi", "Earth"],
            }
        )
        skeleton = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
        tokenizer = _make_tokenizer("</s>")

        result = add_templated_column_for_non_instruct_models(
            dataset=ds,
            skeleton_prompt=skeleton,
            input_column_name="input",
            output_column_name="output",
            system_prompt="You are helpful.",
            tokenizer=tokenizer,
        )

        assert "text" in result.column_names
        for text in result["text"]:
            assert text.endswith("</s>")
            assert "### Instruction:" in text
            assert "You are helpful." in text

    def test_preserves_original_columns(self) -> None:
        ds = Dataset.from_dict(
            {
                "input": ["A"],
                "output": ["B"],
            }
        )
        skeleton = "{} {} {}"
        tokenizer = _make_tokenizer("<eos>")

        result = add_templated_column_for_non_instruct_models(
            dataset=ds,
            skeleton_prompt=skeleton,
            input_column_name="input",
            output_column_name="output",
            system_prompt="sys",
            tokenizer=tokenizer,
        )

        assert "input" in result.column_names
        assert "output" in result.column_names
        assert list(result["input"]) == ["A"]
        assert list(result["output"]) == ["B"]

    def test_formats_each_row_independently(self) -> None:
        ds = Dataset.from_dict(
            {
                "input": ["x1", "x2", "x3"],
                "output": ["y1", "y2", "y3"],
            }
        )
        skeleton = "[{};{};{}]"
        tokenizer = _make_tokenizer("|END|")

        result = add_templated_column_for_non_instruct_models(
            dataset=ds,
            skeleton_prompt=skeleton,
            input_column_name="input",
            output_column_name="output",
            system_prompt="SYS",
            tokenizer=tokenizer,
        )

        assert len(result) == 3
        assert result["text"][0] == "[SYS;x1;y1]|END|"
        assert result["text"][1] == "[SYS;x2;y2]|END|"
        assert result["text"][2] == "[SYS;x3;y3]|END|"
