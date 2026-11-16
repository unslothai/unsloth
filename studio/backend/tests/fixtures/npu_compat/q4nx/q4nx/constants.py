# Trimmed to the shape of ROCm/FLM_Q4NX_Converter's q4nx/constants.py at the pinned q4nx commit.
from enum import IntEnum, auto


class ModelArch(IntEnum):
    QWEN3VL = auto()
    QWEN3 = auto()
    QWEN35_08B = auto()
    QWEN2VL = auto()
    LLAMA = auto()
    LFM2 = auto()
    GPT_OSS = auto()
    GEMMA4 = auto()


ModelArchNames: dict[ModelArch, list[str]] = {
    ModelArch.QWEN35_08B: ["qwen35-0.8B", "qwen3.5-0.8B"],
    ModelArch.QWEN3VL: ["qwen3vl", "qwen3-vl"],
    ModelArch.QWEN2VL: ["qwen2.5-Vl"],
    ModelArch.QWEN3: ["qwen3"],
    ModelArch.LLAMA: ["llama"],
    ModelArch.LFM2: ["lfm2"],
    ModelArch.GPT_OSS: ["gpt-oss"],
    ModelArch.GEMMA4: ["gemma4", "Gemma-4"],
}
