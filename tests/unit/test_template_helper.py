from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from rules_lawyer_models.data.template_helper import (
    _apply_chat_template,
    _apply_simple_template,
    _apply_template,
    _base_model_is_instruct,
    _generate_eos,
    add_eval_column,
    add_training_column,
)
from rules_lawyer_models.utils.model_name import BaseModelName

# ── Constants ────────────────────────────────────────────────────

INSTRUCT_MODEL = BaseModelName.QWEN_25_3B_05BIT_INSTRUCT
NON_INSTRUCT_MODEL = BaseModelName.QWEN_25_14B_4BIT_BASE


# ── Helpers ──────────────────────────────────────────────────────


def _make_tokenizer(eos_token: str | None = "</s>") -> MagicMock:
    tok = MagicMock()
    tok.eos_token = eos_token
    return tok


def _fake_apply_chat_template(messages: list[dict[str, str]], **_kwargs: object) -> str:
    """Simplified ChatML-style formatter for testing."""
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>{msg['content']}")
    return "\n".join(parts)


def _make_instruct_tokenizer(eos: str = "</s>") -> MagicMock:
    tok = _make_tokenizer(eos)
    tok.apply_chat_template = MagicMock(side_effect=_fake_apply_chat_template)
    return tok


# ── 1. _generate_eos ────────────────────────────────────────────


class TestGenerateEos:
    def test_returns_eos_string(self) -> None:
        tok = _make_tokenizer("</s>")
        assert _generate_eos(tok) == "</s>"

    def test_raises_on_none_eos(self) -> None:
        tok = _make_tokenizer(None)
        with pytest.raises(ValueError, match="EOS token"):
            _generate_eos(tok)

    def test_returns_first_element_when_list(self) -> None:
        tok = MagicMock()
        tok.eos_token = ["</s>", "<end>"]
        assert _generate_eos(tok) == "</s>"


# ── 2. _apply_chat_template ─────────────────────────────────────


class TestApplyChatTemplate:
    def test_training_mode_includes_assistant_and_eos(self) -> None:
        tok = _make_instruct_tokenizer("|E|")
        result = _apply_chat_template("SYS", "hello", "world", tok, "|E|")

        assert "<|assistant|>world" in result
        assert result.endswith("|E|")
        tok.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def test_eval_mode_no_assistant_no_eos(self) -> None:
        tok = _make_instruct_tokenizer()
        result = _apply_chat_template("SYS", "hello", "", tok, None)

        assert "<|assistant|>" not in result
        assert not result.endswith("</s>")
        tok.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hello"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_training_mode_with_none_eos(self) -> None:
        tok = _make_instruct_tokenizer()
        result = _apply_chat_template("SYS", "in", "out", tok, None)

        assert "<|assistant|>out" in result
        assert result == "<|system|>SYS\n<|user|>in\n<|assistant|>out"

    def test_system_and_user_always_present(self) -> None:
        tok = _make_instruct_tokenizer()

        result_train = _apply_chat_template("inst", "inp", "out", tok, "</s>")
        assert "<|system|>inst" in result_train
        assert "<|user|>inp" in result_train

        result_eval = _apply_chat_template("inst", "inp", "", tok, None)
        assert "<|system|>inst" in result_eval
        assert "<|user|>inp" in result_eval

    def test_no_duplicate_eos_when_template_already_ends_with_eos(self) -> None:
        tok = _make_tokenizer("|E|")
        tok.apply_chat_template = MagicMock(return_value="content|E|")
        result = _apply_chat_template("S", "in", "out", tok, "|E|")
        assert result.count("|E|") == 1

    def test_no_duplicate_eos_when_template_has_trailing_whitespace(self) -> None:
        tok = _make_tokenizer("|E|")
        tok.apply_chat_template = MagicMock(return_value="content|E|\n")
        result = _apply_chat_template("S", "in", "out", tok, "|E|")
        assert result.count("|E|") == 1


# ── 3. _apply_simple_template ───────────────────────────────────


class TestApplySimpleTemplate:
    def test_formats_template_with_eos(self) -> None:
        tok = _make_tokenizer("</s>")
        result = _apply_simple_template("inst", "inp", "out", "{} {} {}", tok, "</s>")
        assert result == "inst inp out</s>"

    def test_formats_template_without_eos(self) -> None:
        tok = _make_tokenizer()
        result = _apply_simple_template("inst", "inp", "out", "{} {} {}", tok, None)
        assert result == "inst inp out"

    def test_structured_skeleton(self) -> None:
        tok = _make_tokenizer("</s>")
        skeleton = "### Instruction:\n{}\n### Input:\n{}\n### Response:\n{}"
        result = _apply_simple_template("Be helpful", "question", "answer", skeleton, tok, "</s>")
        assert "### Instruction:\nBe helpful" in result
        assert "### Input:\nquestion" in result
        assert "### Response:\nanswer" in result
        assert result.endswith("</s>")

    def test_no_duplicate_eos_when_template_already_ends_with_eos(self) -> None:
        tok = _make_tokenizer("</s>")
        result = _apply_simple_template("inst", "inp", "out</s>", "{} {} {}", tok, "</s>")
        assert result.count("</s>") == 1

    def test_no_duplicate_eos_when_template_has_trailing_whitespace(self) -> None:
        tok = _make_tokenizer("</s>")
        result = _apply_simple_template("inst", "inp", "out</s>\n", "{} {} {}", tok, "</s>")
        assert result.count("</s>") == 1


# ── 4. _apply_template ──────────────────────────────────────────


class TestApplyTemplate:
    def test_applies_formatter_to_each_row(self) -> None:
        data: dict[str, list[str]] = {"content": ["a", "b", "c"], "label": ["1", "2", "3"]}
        calls: list[tuple[str, str]] = []

        def track_formatter(inp: str, out: str) -> str:
            calls.append((inp, out))
            return f"{inp}:{out}"

        result = _apply_template(data, track_formatter, "content", "label", "text")

        assert calls == [("a", "1"), ("b", "2"), ("c", "3")]
        assert result == {"text": ["a:1", "b:2", "c:3"]}

    def test_returns_correct_column_name(self) -> None:
        data: dict[str, list[str]] = {"c": ["x"], "l": ["y"]}
        result = _apply_template(data, lambda i, o: f"{i}{o}", "c", "l", "my_col")
        assert "my_col" in result
        assert result["my_col"] == ["xy"]

    def test_raises_on_mismatched_column_lengths(self) -> None:
        data: dict[str, list[str]] = {"content": ["a", "b"], "label": ["1", "2", "3"]}
        with pytest.raises(ValueError):
            _apply_template(data, lambda i, o: "", "content", "label", "text")

    def test_handles_empty_batch(self) -> None:
        data: dict[str, list[str]] = {"content": [], "label": []}
        result = _apply_template(data, lambda i, o: "", "content", "label", "text")
        assert result == {"text": []}


# ── 5. _base_model_is_instruct ──────────────────────────────────


class TestBaseModelIsInstruct:
    def test_instruct_model_returns_true(self) -> None:
        assert _base_model_is_instruct(INSTRUCT_MODEL) is True

    def test_base_model_returns_false(self) -> None:
        assert _base_model_is_instruct(NON_INSTRUCT_MODEL) is False


# ── 6. add_training_column — instruct path ──────────────────────


class TestAddTrainingColumnInstruct:
    def test_adds_named_column(self) -> None:
        ds = Dataset.from_dict({"content": ["hello"], "label": ["Question"]})
        tok = _make_instruct_tokenizer()

        result = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert "text" in result.column_names

    def test_preserves_original_columns(self) -> None:
        ds = Dataset.from_dict({"content": ["A"], "label": ["B"]})
        tok = _make_instruct_tokenizer()

        result = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert list(result["content"]) == ["A"]
        assert list(result["label"]) == ["B"]

    def test_each_row_formatted_with_chat_template(self) -> None:
        ds = Dataset.from_dict({"content": ["a", "b"], "label": ["x", "y"]})
        tok = _make_instruct_tokenizer("|E|")

        result = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "text", "S", tok)

        assert result["text"][0] == "<|system|>S\n<|user|>a\n<|assistant|>x|E|"
        assert result["text"][1] == "<|system|>S\n<|user|>b\n<|assistant|>y|E|"

    def test_eos_appended(self) -> None:
        ds = Dataset.from_dict({"content": ["hello"], "label": ["world"]})
        tok = _make_instruct_tokenizer("</s>")

        result = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert result["text"][0].endswith("</s>")

    def test_formats_multiple_rows_independently(self) -> None:
        ds = Dataset.from_dict({"content": ["a", "b", "c"], "label": ["1", "2", "3"]})
        tok = _make_instruct_tokenizer("|E|")

        result = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "out", "S", tok)

        assert len(result) == 3
        for i, (c, lbl) in enumerate(zip(["a", "b", "c"], ["1", "2", "3"], strict=False)):
            assert f"<|user|>{c}" in result["out"][i]
            assert f"<|assistant|>{lbl}" in result["out"][i]


# ── 7. add_training_column — non-instruct path ──────────────────


class TestAddTrainingColumnNonInstruct:
    def test_adds_named_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rules_lawyer_models.data.template_helper._get_training_template",
            lambda _: "{} {} {}",
        )
        ds = Dataset.from_dict({"content": ["hello"], "label": ["world"]})
        tok = _make_tokenizer("</s>")

        result = add_training_column(NON_INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert "text" in result.column_names

    def test_uses_skeleton_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rules_lawyer_models.data.template_helper._get_training_template",
            lambda _: "[{};{};{}]",
        )
        ds = Dataset.from_dict({"content": ["inp"], "label": ["out"]})
        tok = _make_tokenizer("|E|")

        result = add_training_column(NON_INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert result["text"][0] == "[SYS;inp;out]|E|"

    def test_eos_appended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rules_lawyer_models.data.template_helper._get_training_template",
            lambda _: "{} {} {}",
        )
        ds = Dataset.from_dict({"content": ["a"], "label": ["b"]})
        tok = _make_tokenizer("</s>")

        result = add_training_column(NON_INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert result["text"][0].endswith("</s>")

    def test_preserves_original_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rules_lawyer_models.data.template_helper._get_training_template",
            lambda _: "{} {} {}",
        )
        ds = Dataset.from_dict({"content": ["A"], "label": ["B"]})
        tok = _make_tokenizer("</s>")

        result = add_training_column(NON_INSTRUCT_MODEL, ds, "content", "label", "text", "SYS", tok)

        assert list(result["content"]) == ["A"]
        assert list(result["label"]) == ["B"]


# ── 8. add_eval_column — instruct path ──────────────────────────


class TestAddEvalColumnInstruct:
    def test_adds_named_column(self) -> None:
        ds = Dataset.from_dict({"content": ["hello"], "label": ["Question"]})
        tok = _make_instruct_tokenizer()

        result = add_eval_column(INSTRUCT_MODEL, ds, "content", "label", "eval", "SYS", tok)

        assert "eval" in result.column_names

    def test_no_eos_appended(self) -> None:
        ds = Dataset.from_dict({"content": ["hello"], "label": ["world"]})
        tok = _make_instruct_tokenizer("</s>")

        result = add_eval_column(INSTRUCT_MODEL, ds, "content", "label", "eval", "SYS", tok)

        assert not result["eval"][0].endswith("</s>")

    def test_no_assistant_message(self) -> None:
        ds = Dataset.from_dict({"content": ["hello"], "label": ["world"]})
        tok = _make_instruct_tokenizer()

        add_eval_column(INSTRUCT_MODEL, ds, "content", "label", "eval", "SYS", tok)

        tok.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hello"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_preserves_original_columns(self) -> None:
        ds = Dataset.from_dict({"content": ["A"], "label": ["B"]})
        tok = _make_instruct_tokenizer()

        result = add_eval_column(INSTRUCT_MODEL, ds, "content", "label", "eval", "SYS", tok)

        assert list(result["content"]) == ["A"]
        assert list(result["label"]) == ["B"]


# ── 9. add_eval_column — non-instruct path ──────────────────────


class TestAddEvalColumnNonInstruct:
    def test_adds_named_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rules_lawyer_models.data.template_helper._get_eval_template",
            lambda _: "{} {} {}",
        )
        ds = Dataset.from_dict({"content": ["hello"], "label": ["world"]})
        tok = _make_tokenizer("</s>")

        result = add_eval_column(NON_INSTRUCT_MODEL, ds, "content", "label", "eval", "SYS", tok)

        assert "eval" in result.column_names

    def test_no_eos_appended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rules_lawyer_models.data.template_helper._get_eval_template",
            lambda _: "{} {} {}",
        )
        ds = Dataset.from_dict({"content": ["a"], "label": ["b"]})
        tok = _make_tokenizer("</s>")

        result = add_eval_column(NON_INSTRUCT_MODEL, ds, "content", "label", "eval", "SYS", tok)

        assert not result["eval"][0].endswith("</s>")


# ── 10. Edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    def test_single_row_dataset(self) -> None:
        ds = Dataset.from_dict({"content": ["only"], "label": ["one"]})
        tok = _make_instruct_tokenizer()

        train = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "text", "S", tok)
        eval_ = add_eval_column(INSTRUCT_MODEL, ds, "content", "label", "eval", "S", tok)

        assert len(train) == 1
        assert len(eval_) == 1

    def test_custom_column_names(self) -> None:
        ds = Dataset.from_dict({"post": ["hello"], "answer": ["world"]})
        tok = _make_instruct_tokenizer()

        result = add_training_column(INSTRUCT_MODEL, ds, "post", "answer", "formatted", "SYS", tok)

        assert "formatted" in result.column_names
        assert "<|user|>hello" in result["formatted"][0]

    def test_empty_instruction_prompt(self) -> None:
        ds = Dataset.from_dict({"content": ["hi"], "label": ["there"]})
        tok = _make_instruct_tokenizer()

        result = add_training_column(INSTRUCT_MODEL, ds, "content", "label", "text", "", tok)

        assert "<|system|>" in result["text"][0]
