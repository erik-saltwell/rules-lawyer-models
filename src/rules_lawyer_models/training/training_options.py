from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from datasets import Dataset

STEPS: str = "steps"
NO: str = "no"


@dataclass
class TrainingLength:
    use_steps: bool = False
    max_steps: int = -1
    max_epochs: float = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalationSettings:
    evaluation_enabled: bool
    evaluation_steps: int
    evaluation_dataset: Dataset | None

    @property
    def evaluation_strategy(self) -> str:
        return STEPS if self.evaluation_enabled else NO

    @property
    def save_strategy(self) -> str:
        return STEPS if self.evaluation_enabled else NO

    @property
    def load_best_model_at_end(self) -> bool:
        return self.evaluation_enabled

    @property
    def metric_for_best_model(self) -> str | None:
        return "eval_loss" if self.evaluation_enabled else None

    @property
    def greater_is_better(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "evaluation_dataset"}

    @classmethod
    def from_disabled(cls) -> EvalationSettings:
        return EvalationSettings(False, -1, None)

    @classmethod
    def from_enabled(cls, evaluation_dataset: Dataset, evaluation_steps: int) -> EvalationSettings:
        return EvalationSettings(True, evaluation_steps, evaluation_dataset)


@dataclass
class StepSize:
    max_sequence_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int

    @property
    def total_step_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingOptions:
    r: int
    target_modules: list[str]
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    use_loftq: bool
    loftq_bits: int
    loftq_iter: int
    warmup_ratio: float
    learning_rate: float
    optim: str
    weight_decay: float
    lr_schedular_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrainingOptions:
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def simple_defaults(cls) -> TrainingOptions:
        return TrainingOptions(
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.1,
            use_rslora=False,
            use_loftq=False,
            loftq_bits=4,
            loftq_iter=1,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_schedular_type="linear",
        )
