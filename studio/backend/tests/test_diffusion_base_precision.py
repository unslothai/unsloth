# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the DiT base_precision work.

Covers the new precision plumbing the precision PR adds: the ``base_precision``
config validation (dense-vs-prequant + mixed-precision gating), the prequant-repo
heuristic and its trainer alias, the pure ``auto`` precision policy table, the
explicit-mode passthrough of ``_resolve_base_precision``, the fp8 module filter, the
fp8 branch of the compile policy, the ``train_precision_modes`` machine probe, the
family-info precision fields, and the request-model ``base_precision`` field. No GPU /
model load: every helper here is pure or name-based, so the config validation runs on
name matching (``resolve_trainable_family`` is offline) and the torch probe is monkeypatched.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

import core.training.diffusion_train_common as common
from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    _config_from_dict,
    repo_is_prequantized,
    train_precision_modes,
)
from models.training import DiffusionTrainingStartRequest

# A dense (non-prequant) DiT base and a prequant bnb-4bit base. Both resolve a trainer
# family from their names alone, so normalized() runs without a network call.
_FLUX_DENSE = "black-forest-labs/FLUX.1-dev"
_Z_PREQUANT = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"


def _cfg(base_model = _FLUX_DENSE, **kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(base_model = base_model, data_dir = "d", output_dir = "o", **kw)


# ── base_precision validation ─────────────────────────────────────────────────
def test_base_precision_validation():
    # Default normalizes to the nf4 memory floor.
    assert _cfg().normalized().base_precision == "nf4"

    # An unknown mode is rejected by name.
    with pytest.raises(ValueError, match = "base_precision"):
        _cfg(base_precision = "banana").normalized()

    # A dense mode is case/space-insensitive and stored lowered: " FP8 " on a dense base
    # with bf16 compute normalizes cleanly to "fp8".
    norm = _cfg(base_precision = " FP8 ", mixed_precision = "bf16").normalized()
    assert norm.base_precision == "fp8"

    # A dense mode against a prequant (bnb-4bit) base is refused: the repo already ships a
    # 4-bit transformer and cannot serve the dense precisions.
    with pytest.raises(ValueError, match = "dense base repo"):
        _cfg(base_model = _Z_PREQUANT, base_precision = "bf16").normalized()

    # A dense mode with non-bf16 compute is refused: these modes train in bf16 compute.
    with pytest.raises(ValueError, match = "bf16 compute"):
        _cfg(base_precision = "int8", mixed_precision = "fp16").normalized()

    # "auto" is ACCEPTED by normalized() even on a prequant base: the concrete mode is
    # resolved at runtime against the live GPU, not at config validation.
    assert _cfg(base_model = _Z_PREQUANT, base_precision = "auto").normalized().base_precision == "auto"


# ── repo_is_prequantized heuristic + trainer alias ────────────────────────────
@pytest.mark.parametrize(
    "repo, expected",
    [
        ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", True),
        ("some/model-4bit", True),
        ("some/model-int4", True),
        ("some/model-nf4", True),
        ("black-forest-labs/FLUX.1-dev", False),
        ("Tongyi-MAI/Z-Image-Turbo", False),
    ],
)
def test_repo_is_prequantized_cases(repo, expected):
    assert repo_is_prequantized(repo) is expected


def test_repo_is_prequantized_alias_is_same_object():
    # The trainer keeps a module-level alias for callers/tests; it must be the exact same
    # function object as the common heuristic (moved there for config validation).
    assert dit._repo_is_prequantized is repo_is_prequantized


# ── _pick_auto_precision policy table (pure) ──────────────────────────────────
def test_pick_auto_precision_policy_table():
    p = dit._pick_auto_precision

    # A prequant base always resolves to nf4 (it can only serve 4-bit).
    assert p(True, "cuda", 140, 23.8, (10, 0), True) == "nf4"
    # No CUDA -> nf4 (the dense modes need a GPU).
    assert p(False, "cpu", 140, 23.8, (10, 0), True) == "nf4"
    # Missing free-VRAM number -> the safe nf4 mode.
    assert p(False, "cuda", None, 23.8, (10, 0), True) == "nf4"

    # Plenty of free VRAM -> bf16 regardless of fp8 capability: compiled bf16 measured
    # FASTER than torchao float8 at LoRA-training shapes, so fp8 is opt-in only.
    assert p(False, "cuda", 140, 23.8, (10, 0), True) == "bf16"
    assert p(False, "cuda", 140, 23.8, (8, 0), True) == "bf16"
    assert p(False, "cuda", 140, 23.8, (10, 0), False) == "bf16"

    # Middle band (25 > 23.8 * 0.55 * 1.5 = 19.6, but not > 23.8 * 1.5 = 35.7) -> int8.
    assert p(False, "cuda", 25, 23.8, (10, 0), True) == "int8"
    # Too little free VRAM for even int8 -> nf4.
    assert p(False, "cuda", 10, 23.8, (10, 0), True) == "nf4"


# ── _resolve_base_precision passthrough ───────────────────────────────────────
def test_resolve_base_precision_passes_explicit_through():
    # An explicit mode passes straight through without probing the GPU (normalized() already
    # validated it); the spec is only consulted for "auto".
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "bf16")
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "bf16"
    # Device is irrelevant for an explicit mode: still bf16 on cpu.
    assert dit._resolve_base_precision(cfg, spec, "cpu") == "bf16"


# ── _fp8_module_filter ────────────────────────────────────────────────────────
def test_fp8_module_filter():
    lin = nn.Linear(64, 64)
    # A plain feed-forward Linear with divisible dims gets float8 training compute.
    assert dit._fp8_module_filter(lin, "transformer_blocks.0.ff.net.0") is True
    # A LoRA-owned module is skipped (adapters stay high precision).
    assert dit._fp8_module_filter(lin, "transformer_blocks.0.attn.to_q.lora_A.default") is False
    # The output projection is skipped.
    assert dit._fp8_module_filter(lin, "proj_out") is False
    # An in_features not divisible by 16 is rejected (float8 kernels reject the shape).
    assert dit._fp8_module_filter(nn.Linear(30, 64), "transformer_blocks.0.ff.net.0") is False
    # A non-Linear module is never float8.
    assert dit._fp8_module_filter(nn.LayerNorm(64), "transformer_blocks.0.norm") is False


# ── _should_compile fp8 branch ────────────────────────────────────────────────
def test_should_compile_fp8_branch():
    # fp8 is only competitive compiled, so auto arms compile for it on a dense (non-bnb)
    # cuda base.
    cfg = _cfg(compile_transformer = "auto")
    assert dit._should_compile(cfg, False, "cuda", "fp8") is True
    # fp8 forces compile under auto even when the base is (hypothetically) reported as bnb.
    assert dit._should_compile(cfg, True, "cuda", "fp8") is True
    # An explicit "off" still wins over fp8 -- compile stays off.
    assert dit._should_compile(_cfg(compile_transformer = "off"), False, "cuda", "fp8") is False


# ── train_precision_modes machine probe ───────────────────────────────────────
def test_train_precision_modes_no_cuda(monkeypatch):
    # Patch the torch module attribute the function imports so it observes a CPU-only box:
    # no CUDA -> the nf4-only floor with nf4 recommended, and it never raises.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train_precision_modes() == (["nf4"], "nf4")


# ── family_train_infos precision fields ───────────────────────────────────────
def test_family_train_infos_carries_precision_fields(monkeypatch):
    # Pin the machine probe so the DiT families carry a deterministic mode list, while SDXL
    # (no precision selector) stays empty regardless of the probe.
    monkeypatch.setattr(common, "train_precision_modes", lambda: (["nf4", "bf16"], "auto"))
    infos = {i["name"]: i for i in common.family_train_infos()}

    flux = infos["flux.1"]
    assert flux["precision_modes"] == ["nf4", "bf16"]
    assert flux["recommended_precision"] == "auto"
    assert flux["supports_compile"] is True

    sdxl = infos["sdxl"]
    assert sdxl["precision_modes"] == []
    assert sdxl["recommended_precision"] == "nf4"
    assert sdxl["supports_compile"] is False


# ── request model base_precision field ────────────────────────────────────────
def test_request_model_base_precision():
    # The request defaults to the nf4 memory floor.
    req = DiffusionTrainingStartRequest(base_model = "x", data_dir = "d", output_dir = "o")
    assert req.base_precision == "nf4"

    # An allowed dense mode is accepted.
    assert (
        DiffusionTrainingStartRequest(
            base_model = "x", data_dir = "d", output_dir = "o", base_precision = "fp8"
        ).base_precision
        == "fp8"
    )

    # An out-of-Literal value is rejected by pydantic.
    with pytest.raises(Exception):
        DiffusionTrainingStartRequest(
            base_model = "x", data_dir = "d", output_dir = "o", base_precision = "int4"
        )

    # The generic Studio dict path carries base_precision through onto DiffusionLoraConfig.
    cfg = _config_from_dict(
        {
            "base_model": _FLUX_DENSE,
            "data_dir": "d",
            "output_dir": "o",
            "base_precision": "bf16",
        }
    )
    assert cfg.base_precision == "bf16"
