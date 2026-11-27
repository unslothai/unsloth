# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ModelScope's public API, answered in Hugging Face Hub shapes."""

from __future__ import annotations

import asyncio
import re
import threading
import time
import weakref
from typing import Any, Awaitable, Callable, Optional

import httpx

from hub.utils.hf_errors import not_on_modelscope

MODELSCOPE = "https://www.modelscope.cn"
PAGE_SIZE = 50

_SHA = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_REVISIONS = {"", "main", "master", "HEAD", "refs/heads/main", "refs/heads/master"}
_HIDDEN_DATASET_FILES = {"dataset_infos.json"}
_TREE_PAGE = 500
_TREE_MAX_PAGES = 200

# HF ids ModelScope hosts under another org; only copies whose loaded files all match (LFS sha256, else size).
_ALIASES = {
    "model": {
        "unslothai/Qwen3-ASR-0.6B-GGUF": "ggml-org/Qwen3-ASR-0.6B-GGUF",
        "unslothai/Qwen3-ASR-1.7B-GGUF": "ggml-org/Qwen3-ASR-1.7B-GGUF",
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro": "AI-ModelScope/FLUX.1-dev-ControlNet-Union-Pro",
        "akjindal53244/Llama-3.1-Storm-8B": "AI-ModelScope/Llama-3.1-Storm-8B",
        "Falconsai/nsfw_image_detection": "AI-ModelScope/nsfw_image_detection",
        "google-t5/t5-base": "AI-ModelScope/t5-base",
        "google-t5/t5-small": "AI-ModelScope/t5-small",
        "MahmoudAshraf/mms-300m-1130-forced-aligner": "AI-ModelScope/mms-300m-1130-forced-aligner",
        "NeoQuasar/Kronos-base": "AI-ModelScope/Kronos-base",
        "NeoQuasar/Kronos-mini": "AI-ModelScope/Kronos-mini",
        "NeoQuasar/Kronos-small": "AI-ModelScope/Kronos-small",
        "NeoQuasar/Kronos-Tokenizer-2k": "AI-ModelScope/Kronos-Tokenizer-2k",
        "NeoQuasar/Kronos-Tokenizer-base": "AI-ModelScope/Kronos-Tokenizer-base",
        "patrickjohncyh/fashion-clip": "AI-ModelScope/fashion-clip",
        "rizvandwiki/gender-classification": "AI-ModelScope/gender-classification",
        "SWivid/F5-TTS": "AI-ModelScope/F5-TTS",
        "tencent/AuK": "Tencent-Hunyuan/AuK",
        "tencent/AuK-Flash": "Tencent-Hunyuan/AuK-Flash",
        "tencent/Hunyuan-0.5B-Instruct": "Tencent-Hunyuan/Hunyuan-0.5B-Instruct",
        "tencent/Hunyuan-0.5B-Instruct-AWQ-Int4": "Tencent-Hunyuan/Hunyuan-0.5B-Instruct-AWQ-Int4",
        "tencent/Hunyuan-0.5B-Instruct-FP8": "Tencent-Hunyuan/Hunyuan-0.5B-Instruct-FP8",
        "tencent/Hunyuan-0.5B-Instruct-GPTQ-Int4": "Tencent-Hunyuan/Hunyuan-0.5B-Instruct-GPTQ-Int4",
        "tencent/Hunyuan-0.5B-Pretrain": "Tencent-Hunyuan/Hunyuan-0.5B-Pretrain",
        "tencent/Hunyuan-1.8B-Instruct": "Tencent-Hunyuan/Hunyuan-1.8B-Instruct",
        "tencent/Hunyuan-1.8B-Instruct-AWQ-Int4": "Tencent-Hunyuan/Hunyuan-1.8B-Instruct-AWQ-Int4",
        "tencent/Hunyuan-1.8B-Instruct-FP8": "Tencent-Hunyuan/Hunyuan-1.8B-Instruct-FP8",
        "tencent/Hunyuan-1.8B-Instruct-GPTQ-Int4": "Tencent-Hunyuan/Hunyuan-1.8B-Instruct-GPTQ-Int4",
        "tencent/Hunyuan-1.8B-Pretrain": "Tencent-Hunyuan/Hunyuan-1.8B-Pretrain",
        "tencent/Hunyuan-4B-Instruct": "Tencent-Hunyuan/Hunyuan-4B-Instruct",
        "tencent/Hunyuan-4B-Instruct-AWQ-Int4": "Tencent-Hunyuan/Hunyuan-4B-Instruct-AWQ-Int4",
        "tencent/Hunyuan-4B-Instruct-FP8": "Tencent-Hunyuan/Hunyuan-4B-Instruct-FP8",
        "tencent/Hunyuan-4B-Instruct-GPTQ-Int4": "Tencent-Hunyuan/Hunyuan-4B-Instruct-GPTQ-Int4",
        "tencent/Hunyuan-4B-Pretrain": "Tencent-Hunyuan/Hunyuan-4B-Pretrain",
        "tencent/Hunyuan-7B-Instruct": "Tencent-Hunyuan/Hunyuan-7B-Instruct",
        "tencent/Hunyuan-7B-Pretrain": "Tencent-Hunyuan/Hunyuan-7B-Pretrain",
        "tencent/Hunyuan-7B-Pretrain-0124": "Tencent-Hunyuan/Hunyuan-7B-Pretrain-0124",
        "tencent/Hunyuan-A13B-Instruct-FP8": "Tencent-Hunyuan/Hunyuan-A13B-Instruct-FP8",
        "tencent/Hunyuan-GameCraft-1.0": "Tencent-Hunyuan/Hunyuan-GameCraft-1.0",
        "tencent/Hunyuan-MT-7B": "Tencent-Hunyuan/Hunyuan-MT-7B",
        "tencent/Hunyuan-MT-7B-fp8": "Tencent-Hunyuan/Hunyuan-MT-7B-fp8",
        "tencent/Hunyuan-MT-Chimera-7B": "Tencent-Hunyuan/Hunyuan-MT-Chimera-7B",
        "tencent/Hunyuan-MT-Chimera-7B-fp8": "Tencent-Hunyuan/Hunyuan-MT-Chimera-7B-fp8",
        "tencent/Hunyuan3D-1": "Tencent-Hunyuan/Hunyuan3D-1",
        "tencent/Hunyuan3D-2": "Tencent-Hunyuan/Hunyuan3D-2",
        "tencent/Hunyuan3D-2.1": "Tencent-Hunyuan/Hunyuan3D-2.1",
        "tencent/Hunyuan3D-2mini": "Tencent-Hunyuan/Hunyuan3D-2mini",
        "tencent/Hunyuan3D-2mv": "Tencent-Hunyuan/Hunyuan3D-2mv",
        "tencent/Hunyuan3D-Omni": "Tencent-Hunyuan/Hunyuan3D-Omni",
        "tencent/Hunyuan3D-Part": "Tencent-Hunyuan/Hunyuan3D-Part",
        "tencent/HunyuanCustom": "Tencent-Hunyuan/HunyuanCustom",
        "tencent/HunyuanImage-2.1": "Tencent-Hunyuan/HunyuanImage-2.1",
        "tencent/HunyuanImage-3.0": "Tencent-Hunyuan/HunyuanImage-3.0",
        "tencent/HunyuanImage-3.0-Instruct": "Tencent-Hunyuan/HunyuanImage-3.0-Instruct",
        "tencent/HunyuanImage-3.0-Instruct-Distil": "Tencent-Hunyuan/HunyuanImage-3.0-Instruct-Distil",
        "tencent/HunyuanPortrait": "Tencent-Hunyuan/HunyuanPortrait",
        "tencent/HunyuanVideo": "Tencent-Hunyuan/HunyuanVideo",
        "tencent/HunyuanVideo-1.5": "Tencent-Hunyuan/HunyuanVideo-1.5",
        "tencent/HunyuanVideo-Avatar": "Tencent-Hunyuan/HunyuanVideo-Avatar",
        "tencent/HunyuanVideo-I2V": "Tencent-Hunyuan/HunyuanVideo-I2V",
        "tencent/HunyuanVideo-PromptRewrite": "Tencent-Hunyuan/HunyuanVideo-PromptRewrite",
        "tencent/HunyuanWorld-1": "Tencent-Hunyuan/HunyuanWorld-1",
        "tencent/HunyuanWorld-Mirror": "Tencent-Hunyuan/HunyuanWorld-Mirror",
        "tencent/HunyuanWorld-Voyager": "Tencent-Hunyuan/HunyuanWorld-Voyager",
        "tencent/HY-Embodied-0.5": "Tencent-Hunyuan/HY-Embodied-0.5",
        "tencent/Hy-Embodied-0.5-VLA-RoboTwin": "Tencent-Hunyuan/Hy-Embodied-0.5-VLA-RoboTwin",
        "tencent/Hy-Embodied-0.5-VLA-UMI": "Tencent-Hunyuan/Hy-Embodied-0.5-VLA-UMI",
        "tencent/Hy-Embodied-RxBrain-1.0": "Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0",
        "tencent/Hy-Memory-30B-A3B": "Tencent-Hunyuan/Hy-Memory-30B-A3B",
        "tencent/HY-Motion-1.0": "Tencent-Hunyuan/HY-Motion-1.0",
        "tencent/HY-MT1.5-1.8B": "Tencent-Hunyuan/HY-MT1.5-1.8B",
        "tencent/HY-MT1.5-1.8B-FP8": "Tencent-Hunyuan/HY-MT1.5-1.8B-FP8",
        "tencent/HY-MT1.5-1.8B-GGUF": "Tencent-Hunyuan/HY-MT1.5-1.8B-GGUF",
        "tencent/HY-MT1.5-1.8B-GPTQ-Int4": "Tencent-Hunyuan/HY-MT1.5-1.8B-GPTQ-Int4",
        "tencent/HY-MT1.5-7B": "Tencent-Hunyuan/HY-MT1.5-7B",
        "tencent/HY-MT1.5-7B-FP8": "Tencent-Hunyuan/HY-MT1.5-7B-FP8",
        "tencent/HY-MT1.5-7B-GGUF": "Tencent-Hunyuan/HY-MT1.5-7B-GGUF",
        "tencent/HY-MT1.5-7B-GPTQ-Int4": "Tencent-Hunyuan/HY-MT1.5-7B-GPTQ-Int4",
        "tencent/Hy-MT2-1.8B": "Tencent-Hunyuan/Hy-MT2-1.8B",
        "tencent/Hy-MT2-1.8B-1.25Bit-GGUF": "Tencent-Hunyuan/Hy-MT2-1.8B-1.25Bit-GGUF",
        "tencent/Hy-MT2-1.8B-2Bit-GGUF": "Tencent-Hunyuan/Hy-MT2-1.8B-2Bit-GGUF",
        "tencent/Hy-MT2-1.8B-FP8": "Tencent-Hunyuan/Hy-MT2-1.8B-FP8",
        "tencent/Hy-MT2-1.8B-GGUF": "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
        "tencent/Hy-MT2-30B-A3B-FP8": "Tencent-Hunyuan/Hy-MT2-30B-A3B-FP8",
        "tencent/Hy-MT2-7B": "Tencent-Hunyuan/Hy-MT2-7B",
        "tencent/Hy-MT2-7B-FP8": "Tencent-Hunyuan/Hy-MT2-7B-FP8",
        "tencent/Hy-MT2-7B-GGUF": "Tencent-Hunyuan/Hy-MT2-7B-GGUF",
        "tencent/Hy-MT3-1.8B": "Tencent-Hunyuan/Hy-MT3-1.8B",
        "tencent/Hy-MT3-1.8B-FP8": "Tencent-Hunyuan/Hy-MT3-1.8B-FP8",
        "tencent/Hy-MT3-1.8B-GGUF": "Tencent-Hunyuan/Hy-MT3-1.8B-GGUF",
        "tencent/Hy-MT3-30B-A3B": "Tencent-Hunyuan/Hy-MT3-30B-A3B",
        "tencent/Hy-MT3-30B-A3B-FP8": "Tencent-Hunyuan/Hy-MT3-30B-A3B-FP8",
        "tencent/Hy-MT3-30B-A3B-GGUF": "Tencent-Hunyuan/Hy-MT3-30B-A3B-GGUF",
        "tencent/Hy-MT3-7B": "Tencent-Hunyuan/Hy-MT3-7B",
        "tencent/Hy-MT3-7B-FP8": "Tencent-Hunyuan/Hy-MT3-7B-FP8",
        "tencent/Hy-MT3-7B-GGUF": "Tencent-Hunyuan/Hy-MT3-7B-GGUF",
        "tencent/Hy-Nomas": "Tencent-Hunyuan/Hy-Nomas",
        "tencent/HY-OmniWeaving": "Tencent-Hunyuan/HY-OmniWeaving",
        "tencent/Hy-SimulMT-30B-A3B": "Tencent-Hunyuan/Hy-SimulMT-30B-A3B",
        "tencent/Hy-SimulMT-30B-A3B-FP8": "Tencent-Hunyuan/Hy-SimulMT-30B-A3B-FP8",
        "tencent/Hy-SimulMT-30B-A3B-GGUF": "Tencent-Hunyuan/Hy-SimulMT-30B-A3B-GGUF",
        "tencent/Hy-SimulMT-30B-A3B-Interleave": "Tencent-Hunyuan/Hy-SimulMT-30B-A3B-Interleave",
        "tencent/Hy-SimulMT-30B-A3B-Interleave-FP8": "Tencent-Hunyuan/Hy-SimulMT-30B-A3B-Interleave-FP8",
        "tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF": "Tencent-Hunyuan/Hy-SimulMT-30B-A3B-Interleave-GGUF",
        "tencent/HY-Video-PRFL": "Tencent-Hunyuan/HY-Video-PRFL",
        "tencent/HY-World-2.0": "Tencent-Hunyuan/HY-World-2.0",
        "tencent/HY-WorldPlay": "Tencent-Hunyuan/HY-WorldPlay",
        "tencent/Hy3": "Tencent-Hunyuan/Hy3",
        "tencent/Hy3-FP8": "Tencent-Hunyuan/Hy3-FP8",
        "tencent/Hy3-preview": "Tencent-Hunyuan/Hy3-preview",
        "tencent/Hy3-preview-Base": "Tencent-Hunyuan/Hy3-preview-Base",
        "tencent/HY3D-Bench": "Tencent-Hunyuan/HY3D-Bench",
        "tencent/Hy4": "Tencent-Hunyuan/Hy4",
        "tencent/Hy4-base": "Tencent-Hunyuan/Hy4-base",
        "tencent/Hy4-flash-base": "Tencent-Hunyuan/Hy4-flash-base",
        "tencent/Hy4-FP8": "Tencent-Hunyuan/Hy4-FP8",
        "tencent/Hy4-preview": "Tencent-Hunyuan/Hy4-preview",
        "tencent/Hy4-preview-FP8": "Tencent-Hunyuan/Hy4-preview-FP8",
        "tencent/R3-embedding-0.6b": "Tencent-Hunyuan/R3-embedding-0.6b",
        "tencent/R3-rerank-0.6b": "Tencent-Hunyuan/R3-rerank-0.6b",
        "tencent/Tencent-Hunyuan-Large": "Tencent-Hunyuan/Tencent-Hunyuan-Large",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "AI-ModelScope/TinyLlama-1.1B-Chat-v1.0",
        "ZhengPeng7/BiRefNet": "modelscope/BiRefNet",
    },
    "dataset": {
        "openai/gsm8k": "AI-ModelScope/gsm8k",
        "unsloth/alpaca-cleaned": "AI-ModelScope/alpaca-cleaned",
        "garage-bAInd/Open-Platypus": "swift/Open-Platypus",
        "AI4Math/MathVista": "evalscope/MathVista",
        "AI4Math/MathVerse": "evalscope/MathVerse",
        "ylecun/mnist": "modelscope/mnist",
        "MrDragonFox/Elise": "AI-ModelScope/Elise",
        "tatsu-lab/alpaca": "OmniData/alpaca",
        "philschmid/guanaco-sharegpt-style": "OmniData/guanaco-sharegpt-style",
        "agibot-world/AgiBotWorldChallenge-2026": "agibot_world/AgiBotWorldChallenge-2026",
        "cornell-movie-review-data/rotten_tomatoes": "AI-ModelScope/rotten_tomatoes",
        "databricks/databricks-dolly-15k": "AI-ModelScope/databricks-dolly-15k",
        "GokuScraper/seedance-2-prompts-datasets": "AI-ModelScope/seedance-2-prompts-datasets",
        "Idavidrein/gpqa": "AI-ModelScope/gpqa",
        "KAKA22/SpreadsheetBench": "AI-ModelScope/SpreadsheetBench",
        "KAKA22/SpreadsheetBench-v2": "AI-ModelScope/SpreadsheetBench-v2",
        "Lichess/standard-chess-games": "AI-ModelScope/standard-chess-games",
        "lmms-eval/Video-MME": "AI-ModelScope/Video-MME",
        "mandarjoshi/trivia_qa": "evalscope/trivia_qa",
        "Maxwell-Jia/AIME_2024": "AI-ModelScope/aime_2024",
        "meta-math/MetaMathQA": "swift/MetaMathQA",
        "nkp37/OpenVid-1M": "AI-ModelScope/OpenVid-1M",
        "nvidia/OpenMathInstruct-2": "nv-community/OpenMathInstruct-2",
        "nvidia/OpenMathReasoning": "nv-community/OpenMathReasoning",
        "nvidia/SAGE-10k": "nv-community/SAGE-10k",
        "nvidia/simready-central-status": "nv-community/simready-central-status",
        "openai/openai_humaneval": "openai-mirror/openai_humaneval",
        "project-numina/NuminaMath-CoT": "AI-MO/NuminaMath-CoT",
        "project-numina/olympiads": "AI-MO/olympiads",
        "Rowan/hellaswag": "evalscope/hellaswag",
        "semianalysisai/cc-traces-weka-062126": "evalscope/cc-traces-weka-062126",
        "tasksource/procedural-typed-decisions": "AI-ModelScope/procedural-typed-decisions",
        "tencent/ArtifactsBenchmark": "Tencent-Hunyuan/ArtifactsBenchmark",
        "tencent/C3-BenchMark": "Tencent-Hunyuan/C3-BenchMark",
        "tencent/IFMTBench": "Tencent-Hunyuan/IFMTBench",
        "truthfulqa/truthful_qa": "evalscope/truthful_qa",
    },
}
# Model orgs renamed on ModelScope; _ALIASES wins, _ORG_KEPT keep their id (UNSLOTH_MODELSCOPE_ORG_ALIAS_NETWORK=1 tests/test_modelscope_org_aliases.py).
_ORG_ALIASES = {
    "MiniMaxAI": "MiniMax",
    "OpenMOSS-Team": "openmoss",
    "zai-org": "ZhipuAI",
    "01-ai": "01ai",
    "tencent": "tencent-community",
    "openai": "openai-mirror",
    "nvidia": "nv-community",
    "Alibaba-NLP": "iic",
}
_ORG_KEPT = (
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-200K",
    "01-ai/Yi-6B",
    "01-ai/Yi-6B-200K",
    "01-ai/Yi-6B-Chat-4bits",
    "01-ai/Yi-9B-200K",
    "01-ai/Yi-Coder-9B-Chat",
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    "Alibaba-NLP/Simulation_LLM_wiki_14B_V2",
    "Alibaba-NLP/Simulation_LLM_wiki_7B_V2",
    "MiniMaxAI/MiniMax-Text-01-hf",
    "nvidia/ArtiFixer",
    "nvidia/Audio2Emotion-v2.2",
    "nvidia/Audio2Emotion-v3.0",
    "nvidia/Audio2Face-3D-v2.3-Mark",
    "nvidia/Audio2Face-3D-v2.3.1-Claire",
    "nvidia/Audio2Face-3D-v2.3.1-James",
    "nvidia/cari4d_commercial",
    "nvidia/ChronoEdit-14B-Diffusers",
    "nvidia/COMPASS",
    "nvidia/corrdiff-cmip6-era5",
    "nvidia/Cosmos-1.0-Autoregressive-12B",
    "nvidia/Cosmos-1.0-Autoregressive-13B-Video2World",
    "nvidia/Cosmos-1.0-Autoregressive-4B",
    "nvidia/Cosmos-1.0-Autoregressive-5B-Video2World",
    "nvidia/Cosmos-1.0-Diffusion-14B-Text2World",
    "nvidia/Cosmos-1.0-Diffusion-14B-Video2World",
    "nvidia/Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8",
    "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
    "nvidia/Cosmos-1.0-Diffusion-7B-Video2World",
    "nvidia/Cosmos-1.0-Guardrail",
    "nvidia/Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
    "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
    "nvidia/Cosmos-Guardrail1",
    "nvidia/Cosmos-Predict1-12B",
    "nvidia/Cosmos-Predict1-13B-Video2World",
    "nvidia/Cosmos-Predict1-14B-Text2World",
    "nvidia/Cosmos-Predict1-14B-Video2World",
    "nvidia/Cosmos-Predict1-4B",
    "nvidia/Cosmos-Predict1-5B-Video2World",
    "nvidia/Cosmos-Predict1-7B-Text2World",
    "nvidia/Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview",
    "nvidia/Cosmos-Predict1-7B-Video2World",
    "nvidia/Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview",
    "nvidia/Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView",
    "nvidia/Cosmos-Predict1-7B-WorldInterpolator",
    "nvidia/Cosmos-Predict2-0.6B-Text2Image",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1",
    "nvidia/Cosmos-Predict2-14B-Text2Image",
    "nvidia/Cosmos-Predict2-14B-Video2World",
    "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned",
    "nvidia/Cosmos-Predict2-2B-Text2Image",
    "nvidia/Cosmos-Predict2-2B-Video2World",
    "nvidia/Cosmos-Predict2.5-2B",
    "nvidia/Cosmos-Reason2-2B",
    "nvidia/Cosmos-Reason2-8B",
    "nvidia/Cosmos-Transfer1-7B",
    "nvidia/Cosmos-Transfer1-7B-4KUpscaler",
    "nvidia/Cosmos-Transfer1-7B-Sample-AV",
    "nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView",
    "nvidia/Cosmos-Transfer2.5-2B",
    "nvidia/Cosmos-UpsamplePrompt1-12B-Text2World",
    "nvidia/CWIP-1.0",
    "nvidia/DeepSeek-V3.1-NVFP4",
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "nvidia/DreamDojo",
    "nvidia/Eagle2.5-8B",
    "nvidia/EgoHand-v1",
    "nvidia/finite-difference-flow-optimization",
    "nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Model",
    "nvidia/Ising-Decoder-ColorCode-1-Fast",
    "nvidia/Ising-Decoder-SurfaceCode-1-Accurate",
    "nvidia/Ising-Decoder-SurfaceCode-1-Fast",
    "nvidia/Kimodo-SMPLX-RP-v1",
    "nvidia/Llama-4-Maverick-17B-128E-Eagle3",
    "nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
    "nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4",
    "nvidia/magpie_tts_multilingual_357m",
    "nvidia/multitalker-parakeet-streaming-0.6b-v1",
    "nvidia/nemotron-3-8b-base-4k",
    "nvidia/nemotron-3-8b-chat-4k-rlhf",
    "nvidia/nemotron-3-8b-chat-4k-sft",
    "nvidia/nemotron-3-8b-chat-4k-steerlm",
    "nvidia/nemotron-3-8b-qa-4k",
    "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "nvidia/Nemotron-4-340B-Base",
    "nvidia/Nemotron-Elastic-12B",
    "nvidia/NV-Embed-v1",
    "nvidia/NV-Llama2-70B-RLHF-Chat",
    "nvidia/NV-Retriever-v1",
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
    "nvidia/omni-dreams-models",
    "nvidia/omnivinci",
    "nvidia/OpenMath-CodeLlama-34b-Python",
    "nvidia/OpenMath-CodeLlama-70b-Python",
    "nvidia/OpenMath-Llama-2-70b",
    "nvidia/parakeet_realtime_eou_120m-v1",
    "nvidia/personaplex-7b-v1",
    "nvidia/RT-DETR-Hand-Detector-v1",
    "nvidia/X-Mobility",
    "OpenMOSS-Team/EasyWAM-Unified-Wan22",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
    "OpenMOSS-Team/MOSS-TTS-GGUF",
    "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
    "OpenMOSS-Team/MOSS-VL-Realtime",
    "tencent/Ex-Omni",
    "tencent/Hy4-flash",
    "tencent/Hy4-flash-FP8",
    "tencent/WeMM-Video-2B-AWQ",
    "tencent/WeMM-Video-35BA3B-AWQ",
    "tencent/WeMM-Video-4B-AWQ",
    "tencent/WeMM-Video-9B-AWQ",
    "tencent/WeVisForensics",
    "zai-org/cogagent-chat-hf",
    "zai-org/cogagent-vqa-hf",
    "zai-org/cogvlm-base-224-hf",
    "zai-org/cogvlm-base-490-hf",
    "zai-org/cogvlm-chat-hf",
    "zai-org/cogvlm-grounding-generalist-hf",
    "zai-org/cogvlm2-llama3-caption",
    "zai-org/cogvlm2-llama3-chinese-chat-19B",
)
_LISTED = {**_ALIASES, "model": {**{repo: repo for repo in _ORG_KEPT}, **_ALIASES["model"]}}
_ALIAS_INDEX = {
    kind: {repo.casefold(): target for repo, target in table.items()}
    for kind, table in _LISTED.items()
}
_ORG_INDEX = {org.casefold(): target for org, target in _ORG_ALIASES.items()}


def upstream_id(kind: str, repo: str) -> str:
    hit = _ALIAS_INDEX[kind].get(repo.casefold())
    if hit is not None:
        return hit
    org, _, name = repo.partition("/")
    target = _ORG_INDEX.get(org.casefold()) if kind == "model" else None
    return f"{target}/{name}" if target and name else repo


# HF pins missing from ModelScope history -> matching ModelScope commit; unlisted pins miss (a pin guards remote code).
_PINNED_COMMITS = {
    (
        "model",
        "ibm-research/dac.speech.v1.0",
        "1ea7f64cd0678415e2d8c32d67b190722cb9b149",
    ): "3268c083add6c07ae3c4a36650863eb2c3bb6717",
    (
        "model",
        "ggml-org/clef-flash-gguf",
        "4a192915ef971886004b5b13294f2b4c7a7fc39d",
    ): "7b759102aa09974f325b52ee533bbbd47126c87b",
    (
        "model",
        "ggml-org/clef-gguf",
        "63840a1a68cb7084c88610cffc328509356b04cb",
    ): "9680d82742044d42d2e77fc4d1a915454e313955",
    (
        "model",
        "ggml-org/kev-0.8b-gguf",
        "e551e319d483ff57e1ff208924b349d397cffc1c",
    ): "9b1e85197d3a20c2e37fa9ac804b9bbfe2a3e392",
    (
        "model",
        "ggml-org/kev-4b-gguf",
        "d924f2e2c3872da8b8aaf3eb4453b4126deceb79",
    ): "444aa7a51432034cc45eda7733f0dab31319b0b3",
    (
        "model",
        "ggml-org/kev-9b-gguf",
        "ec2bbfe6620218aee2e01cc93bb78dee2a96ed58",
    ): "1910374c3e4fbec4e5ce4da7453df3f2b4aa91ad",
    (
        "model",
        "ggml-org/lev-gguf",
        "3e9286a79ae857b4e1de051c92fab6dc574581ce",
    ): "ddc329eeb7f1d073d692c3e63b53f883c4a36884",
    (
        "model",
        "ggml-org/bespoke-nimble-9b-v3-gguf",
        "a72bdbadb355ca3014f7f9d2585380917cc59971",
    ): "39398e9f18dd1268181a0e55aa93015303a3fe0f",
    (
        "model",
        "ggml-org/openjev-gguf",
        "10840f375658dea7afc5ff4711127bca8218b560",
    ): "0e92bdc0c205913a707db422da43a12914078026",
    (
        "model",
        "ggml-org/laya-gguf",
        "22265007700297ba9e128297e82540cf28c5d7d4",
    ): "968a9847e6f99bdc1440441cdb352635e5a68121",
    (
        "model",
        "ggml-org/julia-1-gguf",
        "16fee17949206fbf58da9347daea44d792a81211",
    ): "75a35d1ee9701238e520fce8a3c8c5a546f63c65",
    (
        "model",
        "liquidai/d1-omni-600m-gguf",
        "8eb2d2a1c5404bba8a34a88bd8e159c242197d51",
    ): "34f9c859701c5a1b041644078fc7216cc24f9705",
    (
        "model",
        "liquidai/d1-2b-gguf",
        "e968f6575f0d4fad928312d7af15cad90fd365c0",
    ): "6eb1050a62fcfd4b9349c596f8b04d67e39c1bd1",
    (
        "model",
        "liquidai/d1-omni-2b-gguf",
        "c53ae0b1f92b17065e9158d34019c9ca1ea5bcf8",
    ): "d0a359d1e918b90329fa449baa8519f7c7c2a2ef",
    (
        "model",
        "liquidai/d1-3b-gguf",
        "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563",
    ): "49d07e1e9b33827c7dfbdeaeb3437828d63d7e67",
}


def upstream_commit(kind: str, repo: str, sha: str) -> str:
    return _PINNED_COMMITS.get((kind, repo.casefold(), sha.lower()), sha.lower())


def answered_commit(revision: str, sha: str) -> str:
    """A full-sha request is answered under that commit, so a mapped pin caches as the pin."""
    return revision.lower() if _SHA.fullmatch(revision.lower()) else sha


class UpstreamError(Exception):
    """ModelScope could not answer. Never read as absence."""


class NotFound(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def repo_missing(repo: str) -> NotFound:
    return NotFound("RepoNotFound", not_on_modelscope(repo))


class _TTLCache:
    def __init__(self, max_entries: int = 512):
        self._data: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._max = max_entries

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            return hit[1] if hit is not None and hit[0] > time.monotonic() else None

    def put(self, key, value, ttl: float) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                self._data.clear()
            self._data[key] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache = _TTLCache()
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[httpx.AsyncClient, asyncio.Semaphore]]" = weakref.WeakKeyDictionary()
_clients_lock = threading.Lock()
transport: Optional[httpx.AsyncBaseTransport] = None


def client() -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
    loop = asyncio.get_running_loop()
    with _clients_lock:
        entry = _clients.get(loop)
        if entry is None:
            entry = _clients[loop] = (
                httpx.AsyncClient(
                    base_url = MODELSCOPE,
                    timeout = httpx.Timeout(20.0, connect = 10.0),
                    transport = transport,
                    headers = {"User-Agent": "unsloth-studio"},
                ),
                asyncio.Semaphore(8),
            )
        return entry


async def _get(
    path: str,
    *,
    params = None,
    headers = None,
) -> httpx.Response:
    http, limiter = client()
    async with limiter:
        try:
            return await http.get(path, params = params, headers = headers)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"ModelScope request failed: {type(exc).__name__}") from exc


async def _json(path: str, params = None) -> Any:
    response = await _get(path, params = params)
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise UpstreamError(f"ModelScope answered {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamError("ModelScope answered with invalid JSON") from exc


def _cached(ttl: float):
    def wrap(fn: Callable[..., Awaitable[Any]]):
        async def inner(*args):
            key = (fn.__name__, *args)
            hit = _cache.get(key)
            if hit is not None:
                return hit
            value = await fn(*args)
            _cache.put(key, value, ttl)
            return value

        return inner

    return wrap


def parse_ref_advertisement(body: bytes) -> dict[str, str]:
    refs: dict[str, str] = {}
    pos = 0
    while pos + 4 <= len(body):
        try:
            size = int(body[pos : pos + 4], 16)
        except ValueError:
            break
        if size < 4:
            pos += 4
            continue
        line = body[pos + 4 : pos + size].split(b"\0", 1)[0].decode("ascii", "replace").strip()
        pos += size
        sha, _, ref = line.partition(" ")
        if _SHA.fullmatch(sha) and ref:
            refs[ref] = sha
    return refs


@_cached(60)
async def _refs(kind: str, repo: str) -> dict[str, str]:
    # The REST API names no commit for a branch; git does.
    prefix = "/datasets" if kind == "dataset" else ""
    response = await _get(
        f"{prefix}/{upstream_id(kind, repo)}.git/info/refs",
        params = {"service": "git-upload-pack"},
        headers = {"User-Agent": "git/2.45.0"},
    )
    if response.status_code in (401, 403, 404):
        raise repo_missing(repo)
    if response.status_code != 200:
        raise UpstreamError(f"ModelScope refs answered {response.status_code}")
    refs = parse_ref_advertisement(response.content)
    if not refs:
        raise UpstreamError("ModelScope advertised no refs")
    return refs


async def repo_exists(kind: str, repo: str) -> None:
    await _refs(kind, repo)


async def resolve_revision(kind: str, repo: str, revision: str) -> str:
    if _SHA.fullmatch(revision.lower()):
        return upstream_commit(kind, repo, revision)
    refs = await _refs(kind, repo)
    if revision in _DEFAULT_REVISIONS:
        sha = refs.get("HEAD") or refs.get("refs/heads/master")
    else:
        name = revision.removeprefix("refs/heads/").removeprefix("refs/tags/")
        sha = refs.get(f"refs/heads/{name}") or refs.get(f"refs/tags/{name}")
    if not sha:
        raise NotFound("RevisionNotFound", f"Revision {revision} not found on ModelScope.")
    return sha


async def _listing_missing(kind: str, repo: str, sha: str) -> NotFound:
    # ModelScope answers an unknown commit of a real repo with the same 404 as a missing repo.
    await _refs(kind, repo)
    return NotFound(
        "RevisionNotFound",
        f"ModelScope's copy of {repo} has no commit {sha}. "
        "Switch the model source to Hugging Face in Settings to use this revision.",
    )


async def _file_entries(kind: str, repo: str, sha: str) -> list[dict]:
    source = upstream_id(kind, repo)
    if kind == "model":
        data = await _json(
            f"/api/v1/models/{source}/repo/files", {"Revision": sha, "Recursive": "true"}
        )
        if data is None:
            raise await _listing_missing(kind, repo, sha)
        return (data.get("Data") or {}).get("Files") or []
    entries: list[dict] = []
    for page in range(1, _TREE_MAX_PAGES + 1):
        data = await _json(
            f"/api/v1/datasets/{source}/repo/tree",
            {
                "Revision": sha,
                "Root": "/",
                "Recursive": "True",
                "PageNumber": page,
                "PageSize": _TREE_PAGE,
            },
        )
        if data is None:
            raise await _listing_missing(kind, repo, sha)
        batch = (data.get("Data") or {}).get("Files") or []
        entries.extend(batch)
        if len(batch) < _TREE_PAGE:
            return entries
    raise UpstreamError(f"ModelScope lists more than {len(entries)} files for {repo}")


@_cached(300)
async def files(kind: str, repo: str, sha: str) -> dict[str, dict]:
    """Every file at ``sha`` by path. The one place existence is decided: ModelScope
    answers a missing file with a 500, not a 404."""
    found = {}
    for entry in await _file_entries(kind, repo, sha):
        path = entry.get("Path")
        if entry.get("Type") == "tree" or not isinstance(path, str):
            continue
        if kind == "dataset" and path in _HIDDEN_DATASET_FILES:
            continue
        found[path] = {
            "size": int(entry.get("Size") or 0),
            "sha256": str(entry.get("Sha256") or ""),
            "lfs": bool(entry.get("IsLFS")),
        }
    return found


@_cached(300)
async def detail(kind: str, repo: str) -> Optional[dict]:
    data = await _json(f"/openapi/v1/{kind}s/{upstream_id(kind, repo)}")
    return (data or {}).get("data") or None


@_cached(60)
async def search(kind: str, query: str, author: str, sort: str, page: int) -> tuple[list, int]:
    params: dict[str, Any] = {"page_number": page, "page_size": PAGE_SIZE}
    if query:
        params["search"] = query
    if author:
        params["author"] = author
    if sort != "default":
        params["sort"] = sort
    body = ((await _json(f"/openapi/v1/{kind}s", params)) or {}).get("data") or {}
    return body.get(f"{kind}s") or [], int(body.get("total_count") or 0)


@_cached(30)
async def reachable() -> bool:
    try:
        response = await _get("/openapi/v1/models", params = {"page_size": 1})
    except UpstreamError:
        return False
    return response.status_code < 500


def page_url(kind: str, repo: str) -> str:
    return f"{MODELSCOPE}/{kind}s/{upstream_id(kind, repo)}"


def page_links() -> dict:
    """What the browser needs to build ``page_url`` itself, so hub links skip the backend."""
    return {"base": MODELSCOPE, "aliases": _LISTED, "orgs": _ORG_ALIASES}


def file_url(kind: str, repo: str, sha: str, path: str) -> str:
    from urllib.parse import quote
    return f"{page_url(kind, repo)}/resolve/{sha}/{quote(path)}"


def branch_url(kind: str, repo: str, revision: str, path: str) -> str:
    """``path`` on a named revision, without asking ModelScope; its default branch is ``master``."""
    from urllib.parse import quote

    name = revision.removeprefix("refs/heads/").removeprefix("refs/tags/")
    if revision in _DEFAULT_REVISIONS:
        ref = "master"
    elif _SHA.fullmatch(name.lower()):
        ref = upstream_commit(kind, repo, name)
    else:
        ref = quote(name, safe = "")
    return file_url(kind, repo, ref, path)


SORTS = {"downloads": "downloads", "likes": "likes", "lastModified": "last_modified"}
_EPOCH = "1970-01-01T00:00:00Z"


def _tag_parts(tags) -> tuple[list[str], list[str], Optional[str], list[str]]:
    libraries, hub_tags, model_type, tasks = [], [], None, []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        prefix, _, value = tag.partition(":")
        if not value:
            continue
        if prefix == "library":
            libraries.append("transformers" if value == "transformer" else value)
        elif prefix == "model_type":
            model_type = value
        elif prefix == "task":
            tasks.append(value)
        elif prefix == "license":
            hub_tags.append(tag)
        elif prefix == "custom_tag":
            hub_tags.append(value)
    return libraries, hub_tags, model_type, tasks


def hub_entry(kind: str, item: dict) -> dict:
    repo = str(item.get("id") or "")
    libraries, tags, model_type, tag_tasks = _tag_parts(item.get("tags"))
    tasks = [task for task in (item.get("tasks") or []) if isinstance(task, str)] or tag_tasks
    downloads = int(item.get("downloads") or 0)
    entry = {
        "_id": repo,
        "id": repo,
        "author": repo.split("/", 1)[0],
        "private": bool(item.get("private")),
        "gated": "manual" if item.get("gated") else False,
        "downloads": downloads,
        "downloadsAllTime": downloads,
        "likes": int(item.get("likes") or 0),
        "lastModified": item.get("last_modified") or item.get("created_at") or _EPOCH,
        "createdAt": item.get("created_at") or _EPOCH,
        "tags": list(dict.fromkeys(libraries + tags + tasks)),
    }
    if kind == "dataset":
        entry["cardData"] = {"pretty_name": item.get("display_name")}
        return entry
    params = int(item.get("params") or 0)
    entry["modelId"] = repo
    entry["pipeline_tag"] = tasks[0] if tasks else None
    entry["library_name"] = (
        "transformers" if "transformers" in libraries else next(iter(libraries), None)
    )
    if model_type:
        entry["config"] = {"model_type": model_type}
    if params and "gguf" in libraries:
        entry["gguf"] = {"total": params, **({"architecture": model_type} if model_type else {})}
    elif params and "safetensors" in libraries:
        entry["safetensors"] = {"total": params, "parameters": {"BF16": params}}
    return entry


def matches(entry: dict, filters: list[str], pipeline_tag: str) -> bool:
    tags = {tag.lower() for tag in entry.get("tags", [])}
    pipeline_tag = pipeline_tag.lower()
    if pipeline_tag and pipeline_tag not in tags and entry.get("pipeline_tag") != pipeline_tag:
        return False
    library = str(entry.get("library_name")).lower()
    return all(f.lower() in tags or f.lower() == library for f in filters)


def sibling(path: str, meta: dict) -> dict:
    entry = {"rfilename": path, "size": meta["size"], "blobId": meta["sha256"]}
    if meta["lfs"]:
        entry["lfs"] = {"sha256": meta["sha256"], "size": meta["size"], "pointerSize": 134}
    return entry


def tree_entry(path: str, meta: Optional[dict]) -> dict:
    if meta is None:
        return {"type": "directory", "oid": "", "size": 0, "path": path}
    entry = {"type": "file", "oid": meta["sha256"], "size": meta["size"], "path": path}
    if meta["lfs"]:
        entry["lfs"] = {"oid": meta["sha256"], "size": meta["size"], "pointerSize": 134}
    return entry


def directories(paths) -> set[str]:
    found = set()
    for path in paths:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            found.add("/".join(parts[:index]))
    return found
