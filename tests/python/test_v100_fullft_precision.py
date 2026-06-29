# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for full finetuning precision on no-bf16 GPUs (V100/T4).

Full finetuning upcasts trainable weights to float32, so the model dtype is
float32 (not bfloat16). The SFTTrainer mixed-precision template in
unsloth/models/rl.py must then:
  - run the forward pass under float16 autocast for normal models,
  - keep FORCE_FLOAT32 models (Gemma3, gpt_oss, ...) in pure float32,
  - never select bf16 on hardware without bf16.

We execute the REAL template block extracted from rl.py source (no heavy unsloth
import) against mocked inputs. See issue #4082.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

RL_PY = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "rl.py"


def _extract_mixed_precision_code() -> str:
    lines = RL_PY.read_text().split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if "mixed_precision = (" in l)
    except StopIteration:
        pytest.skip("mixed_precision template not found in rl.py")
    body, k = [], start + 1
    while lines[k].strip() != ")":
        body.append(lines[k])
        k += 1
    return eval("(\n" + "\n".join(body) + "\n)")  # only string literals + comments


CODE = _extract_mixed_precision_code()


def _restore(mapping, saved):
    """Restore a dict-like to its saved snapshot: pop keys that were absent."""
    for k, v in saved.items():
        if v is None:
            mapping.pop(k, None)
        else:
            mapping[k] = v


def _decide(dtype, *, bf16_supported, force_float32, full_finetuning, mixed_precision, fp16, bf16):
    """Run the template block; return (args.fp16, args.bf16, ACCELERATE_MP, raised).

    Stubs (sys.modules, env vars, torch.cuda.is_bf16_supported) are restored on
    exit so a decision can't leak into later tests in the same process.
    """
    uzu = types.ModuleType("unsloth_zoo.utils")
    uzu._get_dtype = lambda x: x
    uzd = types.ModuleType("unsloth_zoo.device_type")
    uzd.device_is_bf16_supported = lambda: bf16_supported  # device-aware signal stub

    env_keys = (
        "UNSLOTH_FORCE_FLOAT32",
        "UNSLOTH_ENABLE_FULL_FINETUNING",
        "UNSLOTH_MIXED_PRECISION",
        "ACCELERATE_MIXED_PRECISION",
    )
    mod_keys = ("unsloth_zoo", "unsloth_zoo.utils", "unsloth_zoo.device_type")
    saved_env = {k: os.environ.get(k) for k in env_keys}
    saved_mods = {k: sys.modules.get(k) for k in mod_keys}
    orig_bf16 = torch.cuda.is_bf16_supported
    try:
        sys.modules.setdefault("unsloth_zoo", types.ModuleType("unsloth_zoo"))
        sys.modules["unsloth_zoo.utils"] = uzu
        sys.modules["unsloth_zoo.device_type"] = uzd
        for k in env_keys:
            os.environ.pop(k, None)
        os.environ["UNSLOTH_FORCE_FLOAT32"] = "1" if force_float32 else "0"
        os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1" if full_finetuning else "0"
        os.environ["UNSLOTH_MIXED_PRECISION"] = mixed_precision
        torch.cuda.is_bf16_supported = lambda *a, **k: bf16_supported
        args = types.SimpleNamespace(fp16 = fp16, bf16 = bf16, mixed_precision = None)
        emb = types.SimpleNamespace(weight = types.SimpleNamespace(dtype = dtype))
        model = types.SimpleNamespace(
            config = types.SimpleNamespace(dtype = dtype, torch_dtype = dtype),
            get_input_embeddings = lambda: emb,
        )
        raised = None
        try:
            exec(CODE, {"torch": torch, "os": os}, {"args": args, "model": model})
        except TypeError:
            raised = "TypeError"
        return args.fp16, args.bf16, os.environ.get("ACCELERATE_MIXED_PRECISION"), raised
    finally:
        torch.cuda.is_bf16_supported = orig_bf16
        _restore(os.environ, saved_env)
        _restore(sys.modules, saved_mods)


def test_v100_normal_fullft_fp16_explicit():
    # Normal model, full FT (weights upcast to float32), V100, fp16=True.
    fp16, bf16, amp, raised = _decide(
        torch.float32,
        bf16_supported = False,
        force_float32 = False,
        full_finetuning = True,
        mixed_precision = "float32",
        fp16 = True,
        bf16 = False,
    )
    assert raised is None
    assert (fp16, bf16) == (True, False)  # float32 weights + fp16 forward


def test_v100_normal_fullft_precision_unset():
    # Same, but user left precision unset -> must pick fp16, never bf16.
    fp16, bf16, amp, raised = _decide(
        torch.float32,
        bf16_supported = False,
        force_float32 = False,
        full_finetuning = True,
        mixed_precision = "float32",
        fp16 = False,
        bf16 = False,
    )
    assert raised is None
    assert (fp16, bf16) == (True, False)
    assert amp == "fp16"


def test_force_float32_model_fullft_is_pure_float32():
    # FORCE_FLOAT32 model (Gemma3, gpt_oss, ...) in full FT -> pure float32, no autocast.
    fp16, bf16, amp, raised = _decide(
        torch.float32,
        bf16_supported = False,
        force_float32 = True,
        full_finetuning = True,
        mixed_precision = "float32",
        fp16 = True,
        bf16 = False,
    )
    assert raised is None
    assert (fp16, bf16) == (False, False)
    assert amp in (None, "no")


def test_no_bf16_on_volta_in_auto_branch():
    # bf16 model dtype but no bf16 HW, precision unset -> fp16, never bf16.
    fp16, bf16, amp, raised = _decide(
        torch.bfloat16,
        bf16_supported = False,
        force_float32 = False,
        full_finetuning = False,
        mixed_precision = "float32",
        fp16 = False,
        bf16 = False,
    )
    assert bf16 is False


def test_bf16_gpu_unchanged_auto_branch():
    # Regression guard: on a bf16 GPU, a float32 model with unset precision
    # still selects bf16 autocast (behavior must not change for bf16 hardware).
    fp16, bf16, amp, raised = _decide(
        torch.float32,
        bf16_supported = True,
        force_float32 = False,
        full_finetuning = True,
        mixed_precision = "float32",
        fp16 = False,
        bf16 = False,
    )
    assert raised is None
    assert (fp16, bf16) == (False, True)


def test_genuine_bf16_model_with_fp16_still_raises():
    # A real bfloat16 model on bf16 HW with fp16 requested is a genuine mismatch.
    _, _, _, raised = _decide(
        torch.bfloat16,
        bf16_supported = True,
        force_float32 = False,
        full_finetuning = False,
        mixed_precision = "float32",
        fp16 = True,
        bf16 = False,
    )
    assert raised == "TypeError"
