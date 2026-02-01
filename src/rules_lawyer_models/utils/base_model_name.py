import enum


class BaseModelName(enum.StrEnum):
    QWEN_25_14B_4BIT_BASE = "unsloth/Qwen2.5-14B-bnb-4bit"
    QWEN_25_14B_4BIT_INSTRUCT = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
