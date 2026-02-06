from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rules_lawyer_models.evaluation.metric_result import MetricResult


@runtime_checkable
class MetricProtocol(Protocol):
    """Protocol for all metric observers."""

    @property
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        ...

    def compute(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_data: list[dict],
        step: int,
        epoch: float | None = None,
        predictions: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MetricResult:
        """Compute the metric value."""
        ...

    def reset(self) -> None:
        """Reset any accumulated state."""
        ...
