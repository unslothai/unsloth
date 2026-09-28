# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A fast_inference model must not hand vLLM top_k=None.

Unsloth's GRPOConfig defaults top_k to None and maps it to -1 only when use_vllm is set on the
config. A notebook that gets vLLM from FastLanguageModel(fast_inference=True) never sets it, so
the trainer turns use_vllm on in __init__, after that guard ran, and TRL >= 0.27 passes the None
straight to vllm.SamplingParams, which raises TypeError. Runs the generated __init__ block.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest


os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)

_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


def _vllm_setter_block():
    pytest.importorskip("trl")
    import unsloth  # noqa: F401
    from trl import GRPOTrainer

    assert GRPOTrainer.__name__ == "UnslothGRPOTrainer", "GRPO patch did not apply"
    # __init__ is wrapped after generation, so read the generated module rather than the method.
    source = inspect.getsource(sys.modules[GRPOTrainer.__module__])
    match = re.search(
        r"^( +)if hasattr\(model, 'vllm_engine'\) and hasattr\(args, 'use_vllm'\):\n(?:\1 {4}.*\n)+",
        source,
        flags = re.MULTILINE,
    )
    assert match is not None, "vLLM setter block not found in the generated trainer"
    return textwrap.dedent(match.group(0))


def _run(block, model, **args):
    args = types.SimpleNamespace(**args)
    exec(block, {"os": os}, {"model": model, "args": args})
    return args


@pytest.mark.parametrize("top_k", [None, 0])
def test_fast_inference_model_gets_a_vllm_safe_top_k(top_k):
    engine_model = types.SimpleNamespace(vllm_engine = object())
    args = _run(_vllm_setter_block(), engine_model, use_vllm = False, top_k = top_k)
    assert args.use_vllm is True
    assert args.top_k == -1


@pytest.mark.parametrize("top_k", [1, 20, -1])
def test_a_user_top_k_is_kept(top_k):
    engine_model = types.SimpleNamespace(vllm_engine = object())
    assert _run(_vllm_setter_block(), engine_model, use_vllm = False, top_k = top_k).top_k == top_k


def test_a_model_without_vllm_keeps_none():
    # HF generate reads None as "use the model's generation_config", so it must survive.
    args = _run(_vllm_setter_block(), types.SimpleNamespace(), use_vllm = False, top_k = None)
    assert args.top_k is None and args.use_vllm is False
