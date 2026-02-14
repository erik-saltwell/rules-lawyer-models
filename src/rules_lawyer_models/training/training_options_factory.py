from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from rules_lawyer_models.training.training_options import TrainingOptions


@dataclass
class TrainingMetaOptions:
    rank: int
    alpha_multiplier: int
    use_projection_modules: bool
    lora_dropout: float
    warmup_ratio: float
    learning_rate: float
    optim: str
    weight_decay: float
    lr_schedular_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_training_options(self) -> TrainingOptions:
        return TrainingOptions(
            r=self.rank,
            target_modules=TrainingMetaOptions._get_training_modules(self.use_projection_modules),
            lora_alpha=self.rank * self.alpha_multiplier,
            lora_dropout=self.lora_dropout,
            use_rslora=(self.rank > 16),
            use_loftq=False,
            loftq_bits=4,
            loftq_iter=1,
            warmup_ratio=self.warmup_ratio,
            learning_rate=self.learning_rate,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_schedular_type=self.lr_schedular_type,
        )

    @classmethod
    def _get_training_modules(cls, include_projection_modules: bool = True) -> list[str]:
        training_modules: list[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        if include_projection_modules:
            training_modules.extend(
                [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
        return training_modules

    @classmethod
    def from_dict(cls, data: dict) -> TrainingMetaOptions:
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def get_simple_default(cls) -> TrainingMetaOptions:
        return TrainingMetaOptions(
            rank=16,
            alpha_multiplier=1,
            use_projection_modules=True,
            lora_dropout=0.1,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_schedular_type="linear",
        )

    @classmethod
    def get_default_sweep_config(cls, metric_name: str = "f1", metric_goal: str = "maximize") -> dict[str, Any]:
        return {
            "method": "random",
            "metric": {"goal": metric_goal, "name": metric_name},
            "parameters": {
                "rank": {"values": [8, 16, 32]},
                "alpha_multiplier": {"values": [1, 2]},
                "use_projection_modules": {"values": [True, False]},
                "warmup_ratio": {"values": [0.05, 0.1]},
                "lr_schedular_type": {"values": ["linear", "cosine"]},
                "optim": {"values": ["adamw_8bit", "sgd"]},
                "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 2e-3},
                "lora_dropout": {"distribution": "uniform", "min": 0, "max": 0.1},
                "weight_decay": {"distribution": "uniform", "min": 0.01, "max": 0.1},
            },
        }


def create_training_options(factory_settings: TrainingMetaOptions) -> TrainingOptions:
    return factory_settings.to_training_options()


# loftq example
# loftq_config = {"bits": 4},   # Configure 4-bit quantization

# Learning rate
# Typical Range: 2e-4 (0.0002) to 5e-6 (0.000005).
# 🟩 For normal LoRA/QLoRA Fine-tuning, we recommend 2e-4 as a starting point.

# r
# 8, 16, 32, 64, 128
# Choose 16 or 32

# lora_alpha:
# r or r *2

# lora_dropout:
# 0 defualt
# range: 0 - 0.1

# weight_decay
# default: 0.01
# range: 0.01-0.1

# warmup steps
# default: 5-10% of steps

# schedular_type
# default: "linear"
# range: ["linear", "cosine"]

# target modules:
# Attention: q_proj, k_proj, v_proj, o_proj
# MLP: gate_proj, up_proj, down_proj

# loftq
# loftq_bits (int, defaults to 4): Specifies the bit precision for the quantized backbone weights (e.g., 4 or 8).
# loftq_iter (int, defaults to 1): The number of alternating optimization iterations
#          between the LoRA adapters and the quantized weights.
# loftq_backend (str): Specifies the quantization backend. Common choices include bitsandbytes (for 4-bit).
# loftq_dtype (torch.dtype, optional): The data type for the quantized weights, often torch.qint4 or torch.qint8.
