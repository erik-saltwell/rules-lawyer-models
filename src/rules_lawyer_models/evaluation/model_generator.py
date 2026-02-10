from unsloth import FastLanguageModel  # isort: skip  # Must precede transformers

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelGenerator:
    """Wraps a trained model for inference/generation.

    Accepts the model and tokenizer available after training (e.g. from
    the SFTTrainer), switches the model to evaluation mode via
    FastLanguageModel.for_inference(), and exposes a generate() method
    that formats a system-prompt + user-input via the tokenizer's chat
    template and returns the cleaned model output.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        clean_prediction: Callable[[str], str],
        max_new_tokens: int = 128,
    ) -> None:
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer
        self._clean_prediction = clean_prediction
        self._max_new_tokens = max_new_tokens

    def _build_prompt(self, system_prompt: str, user_input: str) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        return str(
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate(self, system_prompt: str, user_input: str) -> str:
        prompt = self._build_prompt(system_prompt, user_input)
        input_ids = torch.tensor(
            [self._tokenizer.encode(prompt)],
            device=self._model.device,
        )

        outputs = self._model.generate(  # type: ignore[operator]
            input_ids=input_ids,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        generated_tokens = outputs[0][input_ids.shape[1] :]
        raw_output: str = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._clean_prediction(raw_output)
