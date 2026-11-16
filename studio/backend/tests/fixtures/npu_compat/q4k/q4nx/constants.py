# Trimmed to the shape of ROCm/FLM_Q4NX_Converter's q4nx/constants.py at the pinned q4k commit.
from enum import IntEnum, auto


class ModelArch(IntEnum):
    QWEN3 = auto()
    QWEN35_08B = auto()
    QWEN35_9B = auto()
    QWEN35_OMNI = auto()
    QWEN35_A3B = auto()


ModelArchNames: dict[ModelArch, list[str]] = {
    ModelArch.QWEN35_08B: ["qwen35-0.8B", "qwen3.5-0.8B"],
    ModelArch.QWEN35_9B: ["qwen35-9B", "qwen3.5-9B"],
    ModelArch.QWEN35_OMNI: ["qwen35-omni", "qwen3.5-omni"],
    ModelArch.QWEN35_A3B: ["qwen35moe"],
    ModelArch.QWEN3: ["qwen3"],
}
