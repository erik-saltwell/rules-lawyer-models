import enum


class BaseModelName(enum.StrEnum):
    QWEN_25_14B_4BIT_BASE = "unsloth/Qwen2.5-14B-bnb-4bit"
    QWEN_25_14B_4BIT_INSTRUCT = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    QWEN_25_3B_4BIT_INSTRUCT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    QWEN_25_3B_05BIT_INSTRUCT = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    NONE = "none"


class TargetModelName(enum.StrEnum):
    REDDIT_RPG_POST_CLASSIFICATION = "reddit_rpg_post_classification"
    NONE = "none"
