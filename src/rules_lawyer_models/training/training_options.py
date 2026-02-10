from __future__ import annotations

from dataclasses import asdict, dataclass, fields


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
