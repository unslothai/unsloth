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

# A dense (non-prequant) DiT base and a prequant bnb-4bit base. Both resolve a family from their names alone, so normalized() runs offline.
_FLUX_DENSE = "black-forest-labs/FLUX.1-dev"
_Z_PREQUANT = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"
# An SDXL base whose name LOOKS prequant: SDXL ignores base_precision, so the dense-mode gates must not fire for it.
_SDXL_PREQUANT_NAME = "some/sdxl-model-bnb-4bit"
# A dense Qwen-Image base: its DiT is corrupted by fp8, so fp8 is denied for training the same way the inference path denies it.
_QWEN_DENSE = "Qwen/Qwen-Image"


def _cfg(base_model = _FLUX_DENSE, **kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(base_model = base_model, data_dir = "d", output_dir = "o", **kw)


# ── base_precision validation ─────────────────────────────────────────────────
def test_base_precision_validation():
    # Default normalizes to the nf4 memory floor.
    assert _cfg().normalized().base_precision == "nf4"

    # An unknown mode is rejected by name.
    with pytest.raises(ValueError, match = "base_precision"):
        _cfg(base_precision = "banana").normalized()

    # A dense mode is case/space-insensitive and stored lowered: " FP8 " on a dense base with bf16 compute normalizes to "fp8".
    norm = _cfg(base_precision = " FP8 ", mixed_precision = "bf16").normalized()
    assert norm.base_precision == "fp8"

    # A dense mode against a prequant (bnb-4bit) base is refused: the repo already ships a 4-bit transformer.
    with pytest.raises(ValueError, match = "dense base repo"):
        _cfg(base_model = _Z_PREQUANT, base_precision = "bf16").normalized()

    # A dense mode with non-bf16 compute is refused: these modes train in bf16 compute.
    with pytest.raises(ValueError, match = "bf16 compute"):
        _cfg(base_precision = "int8", mixed_precision = "fp16").normalized()

    # "auto" is ACCEPTED even on a prequant base: the concrete mode is resolved at runtime, not at config validation.
    assert _cfg(base_model = _Z_PREQUANT, base_precision = "auto").normalized().base_precision == "auto"


def test_normalized_config_keeps_the_canonical_base_and_pins_its_fetch_mirror(monkeypatch):
    from core.inference import diffusion_families

    upstream = "black-forest-labs/FLUX.2-klein-base-9B"
    mirror = "unsloth/FLUX.2-klein-base-9B"
    seen = []

    def _prefer(base, token = None):
        seen.append((base, token))
        return mirror

    monkeypatch.setattr(diffusion_families, "prefer_ungated_mirror", _prefer)
    norm = _cfg(base_model = upstream, hf_token = " token ").normalized()

    assert norm.base_model == upstream
    assert norm.fetch_base_model == mirror
    assert norm.hf_token == "token"
    assert seen == [(upstream, "token")]

    # SDXL has its own trainer and still loads base_model directly, so its revision source must
    # not be redirected until that loader opts into the same fetch field.
    sdxl = _cfg(base_model = "stabilityai/stable-diffusion-xl-base-1.0").normalized()
    assert sdxl.fetch_base_model == sdxl.base_model
    assert seen == [(upstream, "token")]


def test_base_precision_denies_fp8_for_corrupted_family():
    # fp8 corrupts the Qwen-Image DiT, so a dense Qwen base with base_precision="fp8" is refused up front.
    with pytest.raises(ValueError, match = "fp8"):
        _cfg(base_model = _QWEN_DENSE, base_precision = "fp8", mixed_precision = "bf16").normalized()

    # The deny is fp8-specific: int8 and the other dense modes stay allowed for the same Qwen base.
    for mode in ("nf4", "bf16", "int8", "auto"):
        norm = _cfg(
            base_model = _QWEN_DENSE, base_precision = mode, mixed_precision = "bf16"
        ).normalized()
        assert norm.resolved_family == "qwen-image"
        assert norm.base_precision == mode

    # A family the deny does not cover (FLUX) still accepts fp8.
    flux = _cfg(base_model = _FLUX_DENSE, base_precision = "fp8", mixed_precision = "bf16").normalized()
    assert flux.resolved_family == "flux.1"
    assert flux.base_precision == "fp8"


def test_family_train_infos_drops_denied_fp8_for_qwen(monkeypatch, dit_train_host):
    # /info advertises the machine's DiT modes per family, but a family whose DiT a mode corrupts must not offer it.
    monkeypatch.setattr(
        common, "train_precision_modes", lambda: (["nf4", "bf16", "int8", "fp8", "auto"], "auto")
    )
    # family_train_infos reads the live GPU via bf16_unsupported_reason; pin it so this assertion is deterministic.
    monkeypatch.setattr(common, "bf16_unsupported_reason", lambda name: None)
    infos = {i["name"]: i for i in common.family_train_infos()}
    assert "fp8" not in infos["qwen-image"]["precision_modes"]
    assert "int8" in infos["qwen-image"]["precision_modes"]  # int8 is fine on Qwen
    assert "fp8" in infos["flux.1"]["precision_modes"]


def test_resolve_base_precision_explicit_int8_gates_on_torchao(monkeypatch):
    # Explicit int8 has no runtime fallback, so a missing/stub torchao must fail fast rather than load dense with compile disabled.
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "int8")

    monkeypatch.setattr(dit, "has_functional_torchao", lambda: False)  # torchao absent / stub
    with pytest.raises(ValueError, match = "torchao"):
        dit._resolve_base_precision(cfg, spec, "cuda")

    # With a functional torchao the explicit int8 passes straight through.
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "int8"

    # The gate is int8-specific: explicit bf16/fp8 pass through regardless of torchao.
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: False)
    assert dit._resolve_base_precision(_cfg(base_precision = "bf16"), spec, "cuda") == "bf16"
    assert dit._resolve_base_precision(_cfg(base_precision = "fp8"), spec, "cuda") == "fp8"


def test_bf16_unsupported_reason(monkeypatch):
    # The route uses this to fail fast on a non-bf16 GPU BEFORE evicting resident workloads.
    import torch

    from core.training.diffusion_train_common import bf16_unsupported_reason

    # SDXL (own mixed_precision path) and unknown families are always exempt.
    assert bf16_unsupported_reason("sdxl") is None
    assert bf16_unsupported_reason("") is None

    # A DiT family on a pre-Ampere CUDA GPU gives a clear reason: those cards EMULATE bf16 and report is_bf16_supported() True, so the gate is compute capability.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda *a, **k: True
    )  # emulation reports True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))  # Turing
    assert "bfloat16" in (bf16_unsupported_reason("flux.1") or "")

    # A NATIVE bf16-capable GPU (Ampere+) -> no reason.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert bf16_unsupported_reason("qwen-image") is None

    # A CPU-only host (fp32 fallback for import/unit tests) gives no reason even for a DiT family.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bf16_unsupported_reason("z-image") is None


def test_native_bf16_supported_gates_on_capability(monkeypatch):
    # Native bf16 is gated by compute capability (major >= 8), NOT is_bf16_supported(), which counts pre-Ampere emulation.
    import torch

    from core.training.diffusion_train_common import native_bf16_supported

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda *a, **k: True
    )  # emulation reports True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))  # Turing
    assert native_bf16_supported() is False
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))  # Ampere
    assert native_bf16_supported() is True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert native_bf16_supported() is False


def test_training_precision_preflight_error(monkeypatch):
    # The start route calls this BEFORE evicting resident GPU workloads: it folds the bf16-GPU and explicit-int8 torchao requirements together so both fail fast.
    import torch

    from core.training.diffusion_train_common import training_precision_preflight_error

    # Present a NATIVE bf16-capable CUDA GPU so the int8 gate, not the bf16 gate, is exercised.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))

    # The bf16 gate takes precedence: a pre-Ampere GPU rejects any DiT precision.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert "bfloat16" in (training_precision_preflight_error("flux.1", "int8") or "")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))

    # Explicit int8 on a DiT family with a NON-functional torchao gives a clear int8 reason (no fallback, so the child would raise post-eviction).
    monkeypatch.setattr(common, "has_functional_torchao", lambda: False)
    reason = training_precision_preflight_error("qwen-image", "int8")
    assert reason is not None and "int8" in reason and "torchao" in reason

    # The same int8 request is fine once torchao is functional.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    assert training_precision_preflight_error("qwen-image", "int8") is None

    # With a broken torchao only EXPLICIT int8 is gated: nf4/bf16/auto pass, and it never applies to SDXL or an unknown family.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: False)
    assert training_precision_preflight_error("flux.1", "nf4") is None
    assert training_precision_preflight_error("flux.1", "auto") is None
    assert training_precision_preflight_error("sdxl", "int8") is None
    assert training_precision_preflight_error("", "int8") is None

    # On a host with NO accelerator every DiT precision is rejected up front, nf4 included (its 4-bit load needs bitsandbytes). SDXL still passes.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    for dense in ("bf16", "int8", "fp8", "mxfp8"):
        reason = training_precision_preflight_error("flux.1", dense)
        assert reason is not None and "CUDA" in reason
    for mode in ("nf4", "auto"):
        reason = training_precision_preflight_error("flux.1", mode)
        assert reason is not None and "GPU" in reason
    assert training_precision_preflight_error("sdxl", "bf16") is None

    # An accelerator that is not CUDA (XPU here) satisfies the 4-bit load, so nf4/auto pass while the dense CUDA-only precisions stay rejected.
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert training_precision_preflight_error("flux.1", "nf4") is None
    assert training_precision_preflight_error("flux.1", "auto") is None
    assert training_precision_preflight_error("flux.1", "bf16") is not None
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    # mxfp8 needs Blackwell (sm100+): below it the MX GEMM raises at the first step, AFTER a full dense load, so the preflight rejects it UP FRONT.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    reason = training_precision_preflight_error("flux.1", "mxfp8")
    assert reason is not None and "Blackwell" in reason
    assert training_precision_preflight_error("flux.1", "bf16") is None
    assert training_precision_preflight_error("flux.1", "fp8") is None

    # On a Blackwell (sm100+) GPU mxfp8 is accepted, and it never gates a non-DiT (SDXL) family.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    assert training_precision_preflight_error("flux.1", "mxfp8") is None
    assert training_precision_preflight_error("sdxl", "mxfp8") is None


def test_family_train_infos_empties_dit_modes_on_non_bf16(monkeypatch):
    # On a non-bf16 GPU the start route rejects EVERY DiT family, so /info must not advertise an option that always 400s: modes empty, reason in vram_note, compile off. SDXL is exempt.
    from core.training.diffusion_train_common import _DIT_TRAIN_FAMILIES, family_train_infos

    monkeypatch.setattr(common, "bf16_unsupported_reason", lambda name: "no bfloat16 on this GPU")

    infos = {info["name"]: info for info in family_train_infos()}
    dit_seen = False
    for name, info in infos.items():
        if name in _DIT_TRAIN_FAMILIES:
            dit_seen = True
            assert info["precision_modes"] == []
            assert info["vram_note"] == "no bfloat16 on this GPU"
            assert info["recommended_precision"] == "nf4"
            assert info["supports_compile"] is False
    assert dit_seen  # the registry must still expose at least one DiT family to have covered it


def test_family_train_infos_drops_base_specs_on_a_dit_block(monkeypatch, dit_train_host):
    # The per-base overlay wins in resolveDiffusionTrainingFacts, and FamilyFacts renders
    # vram_note only when there are NO chips. So a blocked host that still published base_specs
    # would put the 9B / 18 GB chips back the moment Klein base-9B is selected, and swap the
    # actionable reason (no CUDA, no native bf16) for a size the user cannot act on. Clearing the
    # family chips is not enough on a family whose bases carry their own.
    from core.training.diffusion_train_common import _DIT_TRAIN_FAMILIES, family_train_infos

    unblocked = {info["name"]: info for info in family_train_infos()}
    # At least one DiT family must ship a per-base overlay, or this asserts nothing.
    assert any(unblocked[n]["base_specs"] for n in _DIT_TRAIN_FAMILIES if n in unblocked)

    monkeypatch.setattr(common, "bf16_unsupported_reason", lambda name: "no bfloat16 on this GPU")
    for name, info in (
        (n, i)
        for n, i in ((i["name"], i) for i in family_train_infos())
        if n in _DIT_TRAIN_FAMILIES
    ):
        assert info["base_specs"] == {}, name
        # The reason survives, which is the whole point of dropping the chips.
        assert info["vram_note"] == "no bfloat16 on this GPU", name


def test_base_precision_gates_skip_sdxl():
    # SDXL ignores base_precision, so the dense-mode gates must not fire for it even on a prequant-looking name.
    norm = _cfg(base_model = _SDXL_PREQUANT_NAME, base_precision = "bf16").normalized()
    assert norm.resolved_family == "sdxl"
    assert norm.base_precision == "bf16"

    # The non-bf16-compute gate is also skipped for SDXL (fp16 is a valid SDXL mixed precision).
    norm2 = _cfg(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0",
        base_precision = "int8",
        mixed_precision = "fp16",
    ).normalized()
    assert norm2.resolved_family == "sdxl"

    # The mode-name validity check still runs for SDXL: an unknown mode is rejected.
    with pytest.raises(ValueError, match = "base_precision"):
        _cfg(base_model = _SDXL_PREQUANT_NAME, base_precision = "banana").normalized()

    # The gates STILL fire for a DiT family: a prequant DiT base with a dense mode raises.
    with pytest.raises(ValueError, match = "dense base repo"):
        _cfg(base_model = _Z_PREQUANT, base_precision = "bf16").normalized()


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
    # The trainer keeps a module-level alias for callers/tests; it must be the exact same function object as the common heuristic.
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

    # Plenty of free VRAM gives bf16 regardless of fp8 capability: compiled bf16 measured FASTER than torchao float8 at LoRA shapes.
    assert p(False, "cuda", 140, 23.8, (10, 0), True) == "bf16"
    assert p(False, "cuda", 140, 23.8, (8, 0), True) == "bf16"
    assert p(False, "cuda", 140, 23.8, (10, 0), False) == "bf16"

    # Middle band (30 > 23.8 * 1.15 = 27.4, but not > 23.8 * 1.5 = 35.7) -> int8.
    assert p(False, "cuda", 30, 23.8, (10, 0), True) == "int8"
    # int8 needs torchao at runtime (no fallback), so its band drops to nf4 when torchao is not importable.
    assert p(False, "cuda", 30, 23.8, (10, 0), True, False) == "nf4"
    assert p(False, "cuda", 140, 23.8, (10, 0), True, False) == "bf16"
    # int8 still materialises the full bf16 transformer before quantize_ shrinks it, so free VRAM below the dense transient falls back to nf4.
    assert p(False, "cuda", 25, 23.8, (10, 0), True) == "nf4"
    # Too little free VRAM for any dense load -> nf4.
    assert p(False, "cuda", 10, 23.8, (10, 0), True) == "nf4"


# ── _resolve_base_precision passthrough ───────────────────────────────────────
def test_resolve_base_precision_passes_explicit_through():
    # An explicit mode passes straight through without probing the GPU; the spec is only consulted for "auto".
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "bf16")
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "bf16"

    # The dense modes are CUDA-only: an explicit request on a GPU-less host fails fast, before any model load.
    with pytest.raises(ValueError, match = "CUDA"):
        dit._resolve_base_precision(cfg, spec, "cpu")
    # nf4 stays a passthrough on any device (the bnb load path owns its own errors).
    assert dit._resolve_base_precision(_cfg(base_precision = "nf4"), spec, "cpu") == "nf4"


def test_resolve_auto_requires_bf16_compute():
    # auto may resolve to bf16/int8, which train in bf16 compute, so a non-bf16 mixed_precision pins auto to the nf4 floor.
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "auto", mixed_precision = "fp16")
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "nf4"


def test_resolve_auto_int8_band_gates_on_torchao(monkeypatch):
    # The int8 auto band needs a FUNCTIONAL torchao; without it _resolve_base_precision must fall to nf4.
    import torch

    spec = dit._SPECS["flux.1"]  # dense_bf16_gb = 23.8
    cfg = _cfg(base_precision = "auto", mixed_precision = "bf16")

    class _FakeCuda:
        # Free VRAM in the int8 band (30 > 23.8 * 1.15) but below the bf16 band.
        @staticmethod
        def mem_get_info():
            return (int(30 * 1e9), int(80 * 1e9))

        @staticmethod
        def get_device_capability():
            return (10, 0)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    monkeypatch.setattr(dit, "has_functional_torchao", lambda: False)  # torchao absent / stub
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "nf4"

    # With a functional torchao the same band picks int8.
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "int8"


def test_resolve_auto_uses_klein_variant_size(monkeypatch):
    import torch

    spec = dit._SPECS["flux.2-klein"]

    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return (int(20 * 1e9), int(24 * 1e9))

        @staticmethod
        def get_device_capability():
            return (10, 0)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)

    four_b = _cfg(
        base_model = "black-forest-labs/FLUX.2-klein-base-4B",
        base_precision = "auto",
        mixed_precision = "bf16",
    )
    nine_b = _cfg(
        base_model = "unsloth/FLUX.2-klein-base-9B",
        base_precision = "auto",
        mixed_precision = "bf16",
    )

    assert dit._resolve_base_precision(four_b, spec, "cuda") == "bf16"
    assert dit._resolve_base_precision(nine_b, spec, "cuda") == "nf4"


def test_resolve_auto_int8_band_treats_stub_as_absent(monkeypatch):
    # Simulate the Windows-ROCm torchao STUB (find_spec succeeds but quantize_ is a no-op), so the int8 band must fall to nf4.
    import torch

    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "auto", mixed_precision = "bf16")

    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return (int(30 * 1e9), int(80 * 1e9))

        @staticmethod
        def get_device_capability():
            return (10, 0)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    # The stub scenario: the probe reports no functional torchao.
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: False)
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "nf4"


def _fake_cuda_with_free_gb(monkeypatch, free_gb: float):
    """Point torch.cuda at a GPU reporting ``free_gb`` free, so the auto pick is deterministic."""
    import torch

    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return (int(free_gb * 1e9), int(80 * 1e9))

        @staticmethod
        def get_device_capability():
            return (10, 0)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)


@pytest.mark.parametrize(
    "base_model",
    [
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        # The unsloth mirrors resolve to the same upstream ids, and they are what the Train tab sends.
        "unsloth/FLUX.2-klein-9B",
        "unsloth/FLUX.2-klein-base-9B",
    ],
)
def test_auto_sizes_flux2_klein_9b_off_its_own_weights(monkeypatch, base_model):
    # flux.2-klein covers a 4B and a 9B transformer under one family entry, so the family's
    # dense_bf16_gb (8.1, the 4B) must not size a 9B run: 20 GB free clears 8.1 * 1.5 but the
    # 9B dense weights are 18.2 GB, so "auto" would pick bf16 and the load would OOM before
    # step 1. Every 9B id has to land on nf4 here.
    spec = dit._SPECS["flux.2-klein"]
    _fake_cuda_with_free_gb(monkeypatch, 20.0)
    cfg = _cfg(base_model = base_model, base_precision = "auto", mixed_precision = "bf16")
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "nf4"
    assert dit._dense_bf16_gb(spec, base_model) > 2 * spec.dense_bf16_gb


def test_auto_still_picks_bf16_for_the_klein_4b_default(monkeypatch):
    # The same 20 GB against the family DEFAULT (4B, 8.1 GB dense) still clears the bf16 band:
    # the per-base lookup must narrow only the variant it has a size for.
    spec = dit._SPECS["flux.2-klein"]
    _fake_cuda_with_free_gb(monkeypatch, 20.0)
    cfg = _cfg(
        base_model = "black-forest-labs/FLUX.2-klein-4B",
        base_precision = "auto",
        mixed_precision = "bf16",
    )
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "bf16"


def test_the_klein_4b_bf16_band_edge_does_not_move(monkeypatch):
    # The band edge is dense_gb * 1.5, so a 12 GB card sits right on top of it for the 4B:
    # 8.1 -> 12.15 keeps int8, and the family table's 7.8 -> 11.70 would flip it to bf16 and
    # hand a 12 GB GPU a dense load with no room left. Pin the edge so the per-base lookup can
    # never widen it for a base it has no size for.
    spec = dit._SPECS["flux.2-klein"]
    _fake_cuda_with_free_gb(monkeypatch, 12.0)
    cfg = _cfg(
        base_model = "black-forest-labs/FLUX.2-klein-4B",
        base_precision = "auto",
        mixed_precision = "bf16",
    )
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "int8"


def test_dense_bf16_gb_keeps_every_base_without_an_override_exactly_where_it_was():
    # Only the klein 9B pair has a per-base override. EVERY other base -- including klein's own
    # 4B default -- must come back with the spec's own number untouched, bit for bit: the shared
    # family table is maintained separately (it records klein at 7.8 GB against this spec's 8.1),
    # so reading through to it would quietly move the auto bands of families this PR never
    # touched. An unknown base must fall back rather than raise, or the lookup could fail a run
    # that would otherwise train.
    for name in ("flux.1", "qwen-image", "z-image", "krea-2", "flux.2-dev", "flux.2-klein"):
        spec = dit._SPECS[name]
        for base in ("some/unknown-base", spec.family, ""):
            assert dit._dense_bf16_gb(spec, base) == spec.dense_bf16_gb
    klein = dit._SPECS["flux.2-klein"]
    for base in (
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "unsloth/FLUX.2-klein-4B",
    ):
        assert dit._dense_bf16_gb(klein, base) == klein.dense_bf16_gb


def test_dense_bf16_gb_survives_a_broken_lookup(monkeypatch):
    # The sizing table is an optimisation, never a precondition: a lookup that blows up falls
    # back to the family number instead of failing the run.
    import core.inference.diffusion_auto_policy as ap

    def _boom(*_a, **_kw):
        raise RuntimeError("table unavailable")

    monkeypatch.setattr(ap, "base_repo_bf16_components_gb", _boom)
    spec = dit._SPECS["flux.2-klein"]
    assert dit._dense_bf16_gb(spec, "unsloth/FLUX.2-klein-base-9B") == pytest.approx(
        spec.dense_bf16_gb
    )


def test_has_functional_torchao_rejects_stub(monkeypatch):
    # has_functional_torchao must reject the import stub: the import succeeds against it, but the symbols are no-op stub types.
    import importlib
    import types

    from core._torchao_stub import _STUB_SENTINEL

    real_import_module = importlib.import_module

    stub_quant = types.ModuleType("torchao.quantization")
    stub_quant._unsloth_stub = _STUB_SENTINEL

    def _fake_import(name, *args, **kwargs):
        if name == "torchao.quantization":
            return stub_quant
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    assert common.has_functional_torchao() is False

    # A real module exposing the int8 symbols (no stub sentinel) probes True.
    real_like = types.ModuleType("torchao.quantization")
    real_like.Int8WeightOnlyConfig = object
    real_like.quantize_ = lambda *a, **k: None

    def _fake_import_real(name, *args, **kwargs):
        if name == "torchao.quantization":
            return real_like
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import_real)
    assert common.has_functional_torchao() is True


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
    # fp8 is only competitive compiled, so auto arms compile for it on a dense (non-bnb) cuda base.
    cfg = _cfg(compile_transformer = "auto")
    assert dit._should_compile(cfg, False, "cuda", "fp8") is True
    # fp8 forces compile under auto even when the base is (hypothetically) reported as bnb.
    assert dit._should_compile(cfg, True, "cuda", "fp8") is True
    # An explicit "off" still wins over fp8: compile stays off.
    assert dit._should_compile(_cfg(compile_transformer = "off"), False, "cuda", "fp8") is False


# ── train_precision_modes machine probe ───────────────────────────────────────
def test_train_precision_modes_no_cuda(monkeypatch):
    # Patch the torch module the function imports so it observes a CPU-only box: no CUDA gives the nf4-only floor, and it never raises.
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train_precision_modes() == (["nf4"], "nf4")


def test_train_precision_modes_gates_int8_fp8_on_torchao(monkeypatch):
    # int8/fp8 are only advertised when torchao is FUNCTIONAL: their explicit paths import it with no fallback.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))

    # No functional torchao (absent or stub): bf16 + auto only, int8/fp8 dropped.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: False)
    modes, recommended = train_precision_modes()
    assert modes == ["nf4", "bf16", "auto"]
    assert "int8" not in modes and "fp8" not in modes
    assert recommended == "auto"

    # With a functional torchao on an fp8-capable GPU, int8 + fp8 are advertised again.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    modes2, _ = train_precision_modes()
    assert "int8" in modes2 and "fp8" in modes2


def test_train_precision_modes_gates_dense_on_bf16_support(monkeypatch):
    # The dense modes all train in bf16 compute, so on a CUDA GPU that cannot do bf16 /info must offer ONLY nf4.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))  # Turing, no bf16
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    modes, recommended = train_precision_modes()
    assert modes == ["nf4"]
    assert recommended == "nf4"


# ── family_train_infos precision fields ───────────────────────────────────────
def test_family_train_infos_carries_precision_fields(monkeypatch, dit_train_host):
    # Pin the machine probe so the DiT families carry a deterministic mode list, while SDXL stays empty.
    monkeypatch.setattr(common, "train_precision_modes", lambda: (["nf4", "bf16"], "auto"))
    # Also pin bf16_unsupported_reason (family_train_infos reads the live GPU through it) so this is deterministic.
    monkeypatch.setattr(common, "bf16_unsupported_reason", lambda name: None)
    infos = {i["name"]: i for i in common.family_train_infos()}

    flux = infos["flux.1"]
    assert flux["precision_modes"] == ["nf4", "bf16"]
    assert flux["recommended_precision"] == "auto"
    assert flux["supports_compile"] is True

    sdxl = infos["sdxl"]
    assert sdxl["precision_modes"] == []
    assert sdxl["recommended_precision"] == "nf4"
    # The SDXL trainer regionally compiles its U-Net blocks too, so compile is advertised for every family.
    assert sdxl["supports_compile"] is True


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


def test_assert_trusted_base_model_rejects_local_non_pipeline(tmp_path):
    # A local base_model dir that is NOT a diffusers pipeline is "trusted" but loads via from_pretrained, so the /diffusion/start preflight must reject it before eviction.
    bad = tmp_path / "bare-base"
    bad.mkdir()
    with pytest.raises(ValueError, match = "model_index.json"):
        common._assert_trusted_base_model(str(bad))
    # A real local pipeline dir (model_index.json) is accepted.
    (bad / "model_index.json").write_text("{}")
    common._assert_trusted_base_model(str(bad))  # no raise
    # An untrusted remote base is still rejected by the trust gate.
    with pytest.raises(ValueError, match = "untrusted"):
        common._assert_trusted_base_model("evil/base")


def test_dit_accelerator_missing_reason_and_info_hide_train_without_a_gpu(monkeypatch):
    # Clicking Start on a GPU-less host evicted the Images pipeline, pulled the text encoders, then died in the child: the 4-bit quantizer needs an accelerator. Reject up front.
    import torch

    from core.training.diffusion_train_common import (
        _DIT_TRAIN_FAMILIES,
        dit_accelerator_missing_reason,
        family_train_infos,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert "GPU" in (dit_accelerator_missing_reason("flux.1") or "")
    # SDXL and unknown families keep their own paths.
    assert dit_accelerator_missing_reason("sdxl") is None
    assert dit_accelerator_missing_reason("") is None

    infos = {info["name"]: info for info in family_train_infos()}
    for name, info in infos.items():
        if name in _DIT_TRAIN_FAMILIES:
            assert info["precision_modes"] == []
            assert "GPU" in info["vram_note"]
            assert info["supports_compile"] is False
        else:
            assert info["precision_modes"] != [] or name not in _DIT_TRAIN_FAMILIES

    # Any accelerator clears it (MPS here, which bitsandbytes accepts).
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert dit_accelerator_missing_reason("flux.1") is None


def test_dit_accelerator_gate_survives_a_torch_without_every_probe(monkeypatch):
    """torch.mps.is_available() only exists from torch 2.5 and the supported floor is 2.4.
    Probing the accelerators under one shared try/except turned that AttributeError into
    "no block", so the very hosts the gate exists for (CPU-only) sailed through it."""
    import torch

    from core.training.diffusion_train_common import dit_accelerator_missing_reason

    class _Missing:
        """A torch.mps that predates is_available()."""

    class _Raising:
        @staticmethod
        def is_available():
            raise RuntimeError("driver not initialised")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", _Raising)
    monkeypatch.setattr(torch.backends, "mps", _Missing)
    assert "GPU" in (dit_accelerator_missing_reason("flux.1") or "")

    # A working probe still clears the gate even when its neighbours are broken.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert dit_accelerator_missing_reason("flux.1") is None
