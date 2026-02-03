from __future__ import annotations

from dataclasses import dataclass

from rules_lawyer_models.utils import (
    BaseModelName,
    CommonPaths,
    DatasetName,
    LoggingProtocol,
    TargetModelName,
)


@dataclass(slots=True)
class RunContext:
    common_paths: CommonPaths
    logger: LoggingProtocol
    seed: int = 3817
    model_name: TargetModelName = TargetModelName.NONE
    base_model_name: BaseModelName = BaseModelName.NONE
    dataset_name: DatasetName = DatasetName.NONE
