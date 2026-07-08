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
# An SDXL base whose name LOOKS prequant (bnb-4bit): SDXL ignores base_precision, so the
# dense-mode gates must not fire for it even with a dense mode + fp16 compute.
_SDXL_PREQUANT_NAME = "some/sdxl-model-bnb-4bit"
# A dense Qwen-Image base: its DiT is corrupted by fp8 (activation outliers), so fp8 is
# denied for training the same way the inference path denies it.
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


def test_base_precision_denies_fp8_for_corrupted_family():
    # fp8 corrupts the Qwen-Image DiT (activation outliers exceed fp8's range), so a dense
    # Qwen base with base_precision="fp8" is refused up front -- mirroring the inference deny.
    with pytest.raises(ValueError, match = "fp8"):
        _cfg(base_model = _QWEN_DENSE, base_precision = "fp8", mixed_precision = "bf16").normalized()

    # The deny is fp8-specific: int8 (per-token, unaffected) and the other dense modes stay
    # allowed for the same Qwen base.
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


def test_family_train_infos_drops_denied_fp8_for_qwen(monkeypatch):
    # /info advertises the machine's DiT modes per family, but a family whose DiT the mode
    # corrupts must not offer it: with fp8 in the machine list, Qwen-Image drops fp8 while
    # FLUX keeps it, so the UI never surfaces a mode normalized() would reject.
    monkeypatch.setattr(
        common, "train_precision_modes", lambda: (["nf4", "bf16", "int8", "fp8", "auto"], "auto")
    )
    # family_train_infos reads the live GPU via bf16_unsupported_reason; pin it to "bf16 OK" so
    # this positive-path assertion is deterministic across GPU types (a non-bf16 CUDA box would
    # otherwise empty every DiT family's modes). The empty-on-non-bf16 path is covered separately.
    monkeypatch.setattr(common, "bf16_unsupported_reason", lambda name: None)
    infos = {i["name"]: i for i in common.family_train_infos()}
    assert "fp8" not in infos["qwen-image"]["precision_modes"]
    assert "int8" in infos["qwen-image"]["precision_modes"]  # int8 is fine on Qwen
    assert "fp8" in infos["flux.1"]["precision_modes"]


def test_resolve_base_precision_explicit_int8_gates_on_torchao(monkeypatch):
    # Explicit int8 has no runtime fallback, so a missing/stub torchao must fail fast here
    # rather than load dense with compile disabled. Gate the explicit request the same way
    # auto + /info already gate it.
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "int8")

    monkeypatch.setattr(dit, "has_functional_torchao", lambda: False)  # torchao absent / stub
    with pytest.raises(ValueError, match = "torchao"):
        dit._resolve_base_precision(cfg, spec, "cuda")

    # With a functional torchao the explicit int8 passes straight through.
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "int8"

    # The gate is int8-specific: explicit bf16/fp8 pass through regardless of torchao (fp8 has
    # its own graceful fallback; bf16 needs no torchao).
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

    # A DiT family on a pre-Ampere CUDA GPU -> a clear reason. Pre-Ampere cards EMULATE bf16 and
    # report is_bf16_supported() True, so the gate is native compute capability (major >= 8), not
    # is_bf16_supported() -- otherwise the emulation case would slip through and evict-then-fail.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # emulation reports True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))  # Turing
    assert "bfloat16" in (bf16_unsupported_reason("flux.1") or "")

    # A NATIVE bf16-capable GPU (Ampere+) -> no reason.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert bf16_unsupported_reason("qwen-image") is None

    # A CPU-only host (fp32 fallback for import/unit tests) -> no reason even for a DiT family.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bf16_unsupported_reason("z-image") is None


def test_native_bf16_supported_gates_on_capability(monkeypatch):
    # Native bf16 is gated by compute capability (Ampere major >= 8), NOT is_bf16_supported(),
    # which defaults to counting pre-Ampere emulation. A Turing card that emulates bf16 is
    # correctly reported unsupported; Ampere+ is supported; a CPU-only host is always False.
    import torch

    from core.training.diffusion_train_common import native_bf16_supported

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # emulation reports True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))  # Turing
    assert native_bf16_supported() is False
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))  # Ampere
    assert native_bf16_supported() is True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert native_bf16_supported() is False


def test_training_precision_preflight_error(monkeypatch):
    # The start route calls this BEFORE evicting resident GPU workloads: it folds the bf16-GPU
    # requirement together with the explicit-int8 torchao requirement, so both fail fast instead
    # of only surfacing in the trainer child after the GPU has already been freed.
    import torch

    from core.training.diffusion_train_common import training_precision_preflight_error

    # Present a NATIVE bf16-capable CUDA GPU (Ampere+, cap major >= 8) so the int8 gate (not the
    # bf16 gate) is what we exercise. bf16 is gated by capability, not is_bf16_supported() (which
    # counts pre-Ampere emulation).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))

    # The bf16 gate takes precedence: a pre-Ampere GPU (emulated bf16) rejects any DiT precision.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert "bfloat16" in (training_precision_preflight_error("flux.1", "int8") or "")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))

    # Explicit int8 on a DiT family with a NON-functional torchao -> a clear int8 reason
    # (its _int8_quantize_base has no fallback, so the child would otherwise raise post-eviction).
    monkeypatch.setattr(common, "has_functional_torchao", lambda: False)
    reason = training_precision_preflight_error("qwen-image", "int8")
    assert reason is not None and "int8" in reason and "torchao" in reason

    # The same int8 request is fine once torchao is functional.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    assert training_precision_preflight_error("qwen-image", "int8") is None

    # With a broken torchao, only EXPLICIT int8 is gated -- nf4/bf16/auto pass, and the int8
    # gate never applies to a non-DiT (SDXL) or unknown family.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: False)
    assert training_precision_preflight_error("flux.1", "nf4") is None
    assert training_precision_preflight_error("flux.1", "auto") is None
    assert training_precision_preflight_error("sdxl", "int8") is None
    assert training_precision_preflight_error("", "int8") is None

    # On a CUDA-ABSENT host, bf16_unsupported_reason exempts CPU-only, but the DiT trainer's dense
    # precisions still require CUDA (mirroring _resolve_base_precision), so bf16/int8/fp8/mxfp8 for
    # a DiT family are rejected UP FRONT rather than after eviction. nf4/auto (and SDXL) still pass.
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for dense in ("bf16", "int8", "fp8", "mxfp8"):
        reason = training_precision_preflight_error("flux.1", dense)
        assert reason is not None and "CUDA" in reason
    assert training_precision_preflight_error("flux.1", "nf4") is None
    assert training_precision_preflight_error("flux.1", "auto") is None
    assert training_precision_preflight_error("sdxl", "bf16") is None

    # mxfp8 needs a Blackwell (sm100+) GPU: its MX GEMM has no kernel below sm100 and would raise at
    # the first training step, AFTER a full dense-transformer load. On a CUDA GPU that is older than
    # Blackwell the preflight rejects mxfp8 UP FRONT (mirroring _resolve_base_precision) so eviction
    # is skipped; other dense precisions on the same GPU still pass.
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
    # On a non-bf16 GPU the start route rejects EVERY DiT family (even nf4), so /info must not
    # advertise a DiT precision option that always 400s: the modes empty, the reason surfaces in
    # vram_note, compile is off, and the recommendation degrades to nf4. SDXL (non-DiT) is exempt.
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


def test_base_precision_gates_skip_sdxl():
    # SDXL ignores base_precision, so the dense-mode gates (prequant base / non-bf16 compute)
    # must not fire for it: a prequant-looking SDXL name with base_precision="bf16" does not
    # raise, and the mode is still stored lowered.
    norm = _cfg(base_model = _SDXL_PREQUANT_NAME, base_precision = "bf16").normalized()
    assert norm.resolved_family == "sdxl"
    assert norm.base_precision == "bf16"

    # The non-bf16-compute gate is also skipped for SDXL (fp16 is a valid SDXL mixed
    # precision), even with a dense base_precision requested.
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

    # Middle band (30 > 23.8 * 1.15 = 27.4, but not > 23.8 * 1.5 = 35.7) -> int8.
    assert p(False, "cuda", 30, 23.8, (10, 0), True) == "int8"
    # int8 needs torchao at runtime (no fallback), so the int8 band drops to nf4 when
    # torchao is not importable while the bf16 band is unaffected.
    assert p(False, "cuda", 30, 23.8, (10, 0), True, False) == "nf4"
    assert p(False, "cuda", 140, 23.8, (10, 0), True, False) == "bf16"
    # int8 still materialises the full bf16 transformer before quantize_ shrinks it, so
    # free VRAM below the dense-load transient (25 < 27.4) must fall back to nf4 even
    # though the QUANTIZED weights would have fit.
    assert p(False, "cuda", 25, 23.8, (10, 0), True) == "nf4"
    # Too little free VRAM for any dense load -> nf4.
    assert p(False, "cuda", 10, 23.8, (10, 0), True) == "nf4"


# ── _resolve_base_precision passthrough ───────────────────────────────────────
def test_resolve_base_precision_passes_explicit_through():
    # An explicit mode passes straight through without probing the GPU (normalized() already
    # validated it); the spec is only consulted for "auto".
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "bf16")
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "bf16"

    # The dense modes are CUDA-only: an explicit request on a GPU-less host fails fast
    # (before any model load) instead of silently proceeding; /info never advertised it.
    with pytest.raises(ValueError, match = "CUDA"):
        dit._resolve_base_precision(cfg, spec, "cpu")
    # nf4 stays a passthrough on any device (the bnb load path owns its own errors).
    assert dit._resolve_base_precision(_cfg(base_precision = "nf4"), spec, "cpu") == "nf4"


def test_resolve_auto_requires_bf16_compute():
    # auto may resolve to bf16/int8 which train in bf16 compute, so a non-bf16
    # mixed_precision pins auto to the nf4 floor BEFORE any GPU probe (pure, no CUDA
    # needed here) -- mirroring the normalized() rule for explicit dense modes.
    spec = dit._SPECS["flux.1"]
    cfg = _cfg(base_precision = "auto", mixed_precision = "fp16")
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "nf4"


def test_resolve_auto_int8_band_gates_on_torchao(monkeypatch):
    # The int8 auto band needs a FUNCTIONAL torchao at runtime; when torchao is not
    # importable _resolve_base_precision must fall to nf4 instead of picking an int8 that
    # would crash in _int8_quantize_base. Drive the probe into the int8 band and toggle the
    # functional-torchao probe (shared with train_precision_modes, imported into the trainer).
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


def test_resolve_auto_int8_band_treats_stub_as_absent(monkeypatch):
    # Simulate the Windows-ROCm torchao STUB: has_functional_torchao returns False (the
    # stub satisfies find_spec but its quantize_ is a no-op), so the int8 band must fall to
    # nf4 rather than pick an int8 whose quantization silently does nothing.
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


def test_has_functional_torchao_rejects_stub(monkeypatch):
    # has_functional_torchao must reject the Unsloth import stub: even though
    # `from torchao.quantization import quantize_` would succeed against the stub, the
    # symbols are no-op stub types. Simulate a stub torchao.quantization module carrying the
    # stub sentinel and assert the probe returns False.
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


def test_train_precision_modes_gates_int8_fp8_on_torchao(monkeypatch):
    # int8/fp8 are only advertised when torchao is FUNCTIONAL: on a CUDA host WITHOUT a real
    # torchao (or with only the Windows-ROCm stub) /info must not offer int8/fp8, since their
    # explicit paths import torchao with no fallback. bf16 + auto stay advertised.
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
    # The dense modes (bf16/int8/fp8/auto) all train in bf16 compute, which the DiT trainer
    # requires. On a CUDA GPU that cannot do bf16 (T4/V100/RTX 20xx), /info must offer ONLY
    # nf4 -- otherwise the UI advertises a start that evicts resident models and then fails the
    # trainer's bf16 guard.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))  # Turing, no bf16
    monkeypatch.setattr(common, "has_functional_torchao", lambda: True)
    modes, recommended = train_precision_modes()
    assert modes == ["nf4"]
    assert recommended == "nf4"


# ── family_train_infos precision fields ───────────────────────────────────────
def test_family_train_infos_carries_precision_fields(monkeypatch):
    # Pin the machine probe so the DiT families carry a deterministic mode list, while SDXL
    # (no precision selector) stays empty regardless of the probe.
    monkeypatch.setattr(common, "train_precision_modes", lambda: (["nf4", "bf16"], "auto"))
    # Also pin bf16_unsupported_reason (family_train_infos reads the live GPU through it): "bf16 OK"
    # so this positive-path assertion is deterministic across GPU types, not just on CPU-only CI.
    monkeypatch.setattr(common, "bf16_unsupported_reason", lambda name: None)
    infos = {i["name"]: i for i in common.family_train_infos()}

    flux = infos["flux.1"]
    assert flux["precision_modes"] == ["nf4", "bf16"]
    assert flux["recommended_precision"] == "auto"
    assert flux["supports_compile"] is True

    sdxl = infos["sdxl"]
    assert sdxl["precision_modes"] == []
    assert sdxl["recommended_precision"] == "nf4"
    # The SDXL trainer regionally compiles its U-Net blocks too, so compile is advertised
    # for every family; only the precision selector stays DiT-only.
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
    # A local base_model dir that is NOT a diffusers pipeline (no model_index.json) is "trusted"
    # (any existing path passes the trust check), but the spawned trainer loads it via
    # from_pretrained, so it must be rejected in the /diffusion/start preflight BEFORE
    # _free_gpu_for_diffusion_training tears down the resident models -- not fail the child after
    # the eviction.
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
