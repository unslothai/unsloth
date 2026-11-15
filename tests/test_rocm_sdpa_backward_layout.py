# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch 2.12+rocm7.14.1 on gfx942 returns contiguous SDPA gradients where PyTorch's fake kernel
promises the query's layout, so a compiled backward fails Inductor's stride assert (#17548). The
probe compares one real backward with the fake one; only backends that disagree are turned off."""

import pytest

torch = pytest.importorskip("torch")

import unsloth.import_fixes as fixes
from real_accelerator import has_real_cuda


DEVICE = torch.device("cuda" if has_real_cuda() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

# Custom ops with PyTorch's fake kernel for the flash backward (empty_like(query): the query's
# strides). layout_blind's real kernel answers like the ROCm one, with contiguous gradients.
_LIB = torch.library.Library("unsloth_sdpa_probe_test", "DEF")
_LIB.define("layout_blind(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor, Tensor)")
_LIB.define("layout_true(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor, Tensor)")


def _contiguous_grads(q, k, v):
    return tuple(torch.ones(t.shape, dtype = t.dtype, device = t.device) for t in (q, k, v))


def _grads_like_inputs(q, k, v):
    return tuple(torch.ones_like(t) for t in (q, k, v))


for _name, _real in (("layout_blind", _contiguous_grads), ("layout_true", _grads_like_inputs)):
    for _key in ("CPU", "CUDA"):
        _LIB.impl(_name, _real, _key)
    torch.library.register_fake(f"unsloth_sdpa_probe_test::{_name}", _grads_like_inputs, lib = _LIB)


def _blind(q, k, v):
    return torch.ops.unsloth_sdpa_probe_test.layout_blind(q, k, v)


def _true(q, k, v):
    return torch.ops.unsloth_sdpa_probe_test.layout_true(q, k, v)


def _missing(q, k, v):
    raise RuntimeError("USE_FLASH_ATTENTION was not enabled for build.")


def _mismatches(probes):
    return fixes._sdpa_backward_layout_mismatches(DEVICE, DTYPE, probes)


def test_a_kernel_that_ignores_the_query_layout_is_reported():
    assert _mismatches({"flash": _blind, "mem_efficient": _true}) == ["flash"]
    assert _mismatches({"flash": _blind, "mem_efficient": _blind}) == ["flash", "mem_efficient"]


def test_a_backend_missing_from_the_build_is_left_alone():
    assert _mismatches({"flash": _missing, "mem_efficient": _true}) == []


def test_an_import_under_inference_mode_still_probes():
    with torch.inference_mode():
        assert _mismatches({"flash": _blind}) == ["flash"]


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a GPU for the real SDPA kernels")
def test_the_real_sdpa_backwards_here_match_their_fake_kernels():
    """The real flash / efficient aten ops. A healthy CUDA or ROCm build reports nothing, so nothing
    is turned off; a backend this build lacks (flash on Windows CUDA wheels) is skipped."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    assert fixes._sdpa_backward_layout_mismatches(DEVICE, dtype) == []


@pytest.mark.skipif(not has_real_cuda(), reason = "Inductor's GPU path needs a GPU")
def test_inductor_rejects_exactly_what_the_probe_reports():
    """The probe's criterion is the one that stops training: a compiled graph asserts the fake
    strides on the real output and raises the issue's "incorrect fake (aka meta) kernel" error."""
    pytest.importorskip("triton")
    q, k, v = (
        torch.randn(1, 16, 2, 64, device = DEVICE, dtype = DTYPE).transpose(1, 2) for _ in range(3)
    )
    torch._dynamo.reset()
    with pytest.raises(Exception, match = "stride"):
        torch.compile(lambda q, k, v: [g * 2 for g in _blind(q, k, v)])(q, k, v)
    torch._dynamo.reset()
    out = torch.compile(lambda q, k, v: [g * 2 for g in _true(q, k, v)])(q, k, v)
    assert all(torch.equal(g, torch.full_like(g, 2)) for g in out)


@pytest.fixture
def sdpa_flags():
    cuda = torch.backends.cuda
    saved = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled())
    cuda.enable_flash_sdp(True)
    cuda.enable_mem_efficient_sdp(True)
    yield cuda
    cuda.enable_flash_sdp(saved[0])
    cuda.enable_mem_efficient_sdp(saved[1])


@pytest.mark.parametrize(
    "mismatched, expected",
    [
        ([], [True, True, True]),
        (["flash"], [False, True, True]),
        (["mem_efficient"], [True, False, True]),
        (["flash", "mem_efficient"], [False, False, True]),
    ],
)
def test_only_mismatching_backends_are_turned_off(monkeypatch, sdpa_flags, mismatched, expected):
    monkeypatch.delenv("UNSLOTH_ALLOW_ROCM_SDPA_BACKWARD_LAYOUT", raising = False)
    monkeypatch.setattr(fixes, "_rocm_sdpa_backward_layout_mismatches", lambda: list(mismatched))
    fixes.fix_rocm_sdpa_backward_layout()
    cuda = sdpa_flags
    assert [
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
    ] == expected


def test_opt_out_skips_the_probe(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_ROCM_SDPA_BACKWARD_LAYOUT", "1")
    monkeypatch.setattr(
        fixes,
        "_rocm_sdpa_backward_layout_mismatches",
        lambda: pytest.fail("probed despite the opt-out"),
    )
    fixes.fix_rocm_sdpa_backward_layout()


@pytest.mark.parametrize(
    "key", ["TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "UNSLOTH_COMPILE_DISABLE"]
)
def test_nothing_is_probed_without_torch_compile(monkeypatch, key):
    """Eager backward never checks fake strides, so there is nothing to fix."""
    monkeypatch.setenv(key, "1")
    monkeypatch.setattr(
        fixes,
        "_sdpa_backward_layout_mismatches",
        lambda *a, **k: pytest.fail("probed with torch.compile off"),
    )
    assert fixes._rocm_sdpa_backward_layout_mismatches() == []


def test_the_probe_is_rocm_only():
    if getattr(torch.version, "hip", None) or fixes._is_rocm_torch_build():
        pytest.skip("this host is a ROCm one the probe targets")
    assert fixes._rocm_sdpa_backward_layout_mismatches() == []


def test_a_backend_already_turned_off_is_not_launched(monkeypatch, sdpa_flags):
    """fix_rocm_windows_fused_sdpa may have turned a failing backend off; launching it anyway
    would surface its error on a later, unrelated call."""
    seen = {}
    monkeypatch.setattr(fixes, "_is_rocm_torch_build", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("P", (), {"gcnArchName": "gfx942"})(),
    )
    monkeypatch.setattr(
        fixes,
        "_sdpa_backward_layout_mismatches",
        lambda device, dtype, probes: seen.setdefault("probes", sorted(probes)) and [],
    )
    for key in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "UNSLOTH_COMPILE_DISABLE"):
        monkeypatch.delenv(key, raising = False)
    sdpa_flags.enable_flash_sdp(False)
    assert fixes._rocm_sdpa_backward_layout_mismatches() == []
    assert seen["probes"] == ["mem_efficient"]
