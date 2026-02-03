from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from rules_lawyer_models.data.template_helper import (
    add_templated_column_for_instruct_models,
    add_templated_column_for_non_instruct_models,
)


def _make_tokenizer(eos_token: str = "</s>") -> MagicMock:
    tok = MagicMock()
    tok.eos_token = eos_token
    return tok


def _fake_apply_chat_template(messages: list[dict[str, str]], **_kwargs: object) -> str:
    """Simplified ChatML-style formatter for testing."""
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>{msg['content']}")
    return "\n".join(parts)


class TestAddTemplatedColumnForInstructModels:
    def test_adds_text_column_with_eos(self) -> None:
        ds = Dataset.from_dict({"input": ["Hello"], "output": ["Hi"]})
        tokenizer = _make_tokenizer("</s>")
        tokenizer.apply_chat_template = MagicMock(side_effect=_fake_apply_chat_template)

        result = add_templated_column_for_instruct_models(
            dataset=ds,
            input_column_name="input",
            output_column_name="output",
            system_prompt="Be helpful.",
            tokenizer=tokenizer,
        )

        assert "text" in result.column_names
        assert result["text"][0].endswith("</s>")

    def test_passes_correct_messages_to_chat_template(self) -> None:
        ds = Dataset.from_dict({"q": ["What?"], "a": ["Answer."]})
        tokenizer = _make_tokenizer("<eos>")
        tokenizer.apply_chat_template = MagicMock(side_effect=_fake_apply_chat_template)

        add_templated_column_for_instruct_models(
            dataset=ds,
            input_column_name="q",
            output_column_name="a",
            system_prompt="SYS",
            tokenizer=tokenizer,
        )

        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "What?"},
                {"role": "assistant", "content": "Answer."},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def test_formats_each_row_independently(self) -> None:
        ds = Dataset.from_dict({"input": ["a", "b"], "output": ["x", "y"]})
        tokenizer = _make_tokenizer("|E|")
        tokenizer.apply_chat_template = MagicMock(side_effect=_fake_apply_chat_template)

        result = add_templated_column_for_instruct_models(
            dataset=ds,
            input_column_name="input",
            output_column_name="output",
            system_prompt="S",
            tokenizer=tokenizer,
        )

        assert len(result) == 2
        assert result["text"][0] == "<|system|>S\n<|user|>a\n<|assistant|>x|E|"
        assert result["text"][1] == "<|system|>S\n<|user|>b\n<|assistant|>y|E|"

    def test_preserves_original_columns(self) -> None:
        ds = Dataset.from_dict({"input": ["A"], "output": ["B"]})
        tokenizer = _make_tokenizer("</s>")
        tokenizer.apply_chat_template = MagicMock(side_effect=_fake_apply_chat_template)

        result = add_templated_column_for_instruct_models(
            dataset=ds,
            input_column_name="input",
            output_column_name="output",
            system_prompt="sys",
            tokenizer=tokenizer,
        )

        assert list(result["input"]) == ["A"]
        assert list(result["output"]) == ["B"]


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
