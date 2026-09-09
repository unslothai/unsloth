# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for #9396's ROCm capability misread and fatal child probe retry.

Torch is stubbed through ``sys.modules`` and no process is spawned.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_precision as dp
import core.inference.diffusion_transformer_quant as tq


def _target(*, device = "cuda", dtype = "bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _stub_torch(
    monkeypatch,
    *,
    hip = None,
    version_str = "2.10.0+cu128",
    cc = (11, 0),
    current_device = 0,
):
    """A torch stub that answers like a ROCm build when ``hip`` is set.

    ``current_device`` is the ordinal this thread is pinned to; ``torch.cuda.selected`` records
    every ``set_device`` the code under test makes."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.__version__ = version_str
    torch.version = types.SimpleNamespace(hip = hip, cuda = None if hip else "12.8")
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        get_device_capability = lambda *a: cc,
        get_device_name = lambda *a: "AMD Radeon  780M Graphics" if hip else "NVIDIA B200",
        current_device = lambda: current_device,
        selected = [],
    )
    torch.cuda.set_device = torch.cuda.selected.append
    torch.nn = types.SimpleNamespace(
        Embedding = type("Embedding", (), {}),
        ModuleList = type("ModuleList", (list,), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {}, raising = True)
    return torch


def _forbid_probe(monkeypatch):
    """Fail loudly if anything reaches the smoke probe: on ROCm it is what segfaults."""

    def _boom(*args, **kwargs):
        raise AssertionError("the torchao smoke probe must not run on ROCm")

    monkeypatch.setattr(tq, "_scheme_supported", _boom)
    monkeypatch.setattr(tq, "_run_smoke_probe", _boom)


# ── 1. the arch gate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hip, version_str",
    [
        ("7.1.25424", "2.10.0+rocm7.1"),  # the build in the report
        (None, "2.10.0+rocm7.1"),  # AMD wheels that tag only __version__
    ],
)
def test_dense_transformer_unsupported_on_rocm(monkeypatch, hip, version_str):
    _stub_torch(monkeypatch, hip = hip, version_str = version_str)
    assert tq.torch_is_rocm() is True
    assert tq.dense_transformer_supported(_target()) is False


def test_dense_transformer_still_supported_on_cuda(monkeypatch):
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    assert tq.torch_is_rocm() is False
    assert tq.dense_transformer_supported(_target()) is True


@pytest.mark.parametrize("cc", [(11, 0), (11, 5), (9, 4)])  # gfx1103, gfx1151, gfx942
def test_auto_selects_nothing_and_never_probes_on_rocm(monkeypatch, cc):
    """The regression: (11, 0) is gfx1103, not sm_110, and must not clear the sm_100 tier."""
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1", cc = cc)
    _forbid_probe(monkeypatch)
    assert tq.select_transformer_quant_scheme(_target(), "auto") is None
    assert tq.auto_scheme_candidates(_target()) == ()


@pytest.mark.parametrize("scheme", list(tq.TQ_SCHEMES))
def test_explicit_scheme_refused_without_probing_on_rocm(monkeypatch, scheme):
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    _forbid_probe(monkeypatch)
    assert tq.select_transformer_quant_scheme(_target(), scheme) is None


def test_refusal_reason_names_rocm(monkeypatch):
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    reason = tq.dense_transformer_unsupported_reason(_target())
    assert "ROCm" in reason and "AMD" in reason


def test_refusal_reason_unchanged_off_cuda(monkeypatch):
    _stub_torch(monkeypatch, hip = None)
    assert "CUDA GPU in bf16" in tq.dense_transformer_unsupported_reason(_target(device = "cpu"))


# ── 2. the text-encoder gate has the same capability misread ──────────────────


@pytest.mark.parametrize("mode", [dp.TE_QUANT_INT8, dp.TE_QUANT_FP8_DYNAMIC, dp.TE_QUANT_NVFP4])
def test_torchao_text_encoder_modes_unsupported_on_rocm(monkeypatch, mode):
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    assert dp.te_quant_supported(_target(), mode) is False


def test_layerwise_fp8_text_encoder_still_supported_on_rocm(monkeypatch):
    """Plain fp8 is a torch dtype cast with no torchao in it, so ROCm keeps it."""
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    assert dp.te_quant_supported(_target(), dp.TE_QUANT_FP8) is True


@pytest.mark.parametrize("mode", [dp.TE_QUANT_INT8, dp.TE_QUANT_FP8_DYNAMIC, dp.TE_QUANT_NVFP4])
def test_torchao_text_encoder_modes_still_supported_on_cuda(monkeypatch, mode):
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(dp, "is_stubbed", lambda name: False)
    assert dp.te_quant_supported(_target(), mode) is True


# ── 3. a crashed probe child is a verdict, not a reason to retry in-process ───


class _Proc:
    def __init__(self, exitcode):
        self.exitcode = exitcode


@pytest.mark.parametrize("signal_number", sorted(tq._PROBE_CRASH_SIGNALS))
def test_crashed_child_marks_every_scheme_unusable(signal_number):
    verdict = tq._crashed_child_verdict(_Proc(-signal_number), "cuda")
    assert verdict == {scheme: False for scheme in tq.TQ_SCHEMES}


@pytest.mark.parametrize("exitcode", [0, 1, -9, -15, None])
def test_non_crash_exits_still_fall_back_in_process(exitcode):
    """SIGKILL is the OOM killer, SIGTERM is our own timeout teardown; neither is a verdict."""
    assert tq._crashed_child_verdict(_Proc(exitcode), "cuda") is None
    assert tq._crashed_child_verdict(None, "cuda") is None


@pytest.mark.parametrize(
    "status",
    [
        0xC0000005,  # STATUS_ACCESS_VIOLATION
        0xC000001D,  # STATUS_ILLEGAL_INSTRUCTION
        0xC0000094,  # STATUS_INTEGER_DIVIDE_BY_ZERO
        0xC0000374,  # STATUS_HEAP_CORRUPTION
        0xC0000409,  # STATUS_STACK_BUFFER_OVERRUN, where UCRT's abort() lands
    ],
)
def test_a_windows_native_fault_is_a_crash_not_an_inconclusive_exit(monkeypatch, status):
    """A faulting child on Windows is a positive NTSTATUS (0xC0000005 == 3221225477), never
    -SIGSEGV; read as inconclusive, the parent re-runs the proven-fatal probe in the backend."""
    monkeypatch.setattr(tq, "_sys", types.SimpleNamespace(platform = "win32"))
    verdict = tq._crashed_child_verdict(_Proc(status), "cuda")
    assert verdict == {scheme: False for scheme in tq.TQ_SCHEMES}


@pytest.mark.parametrize("exitcode", [0, 1, 3, 42, -15, 0x40000015, 0x80000003])
def test_a_windows_exit_that_is_not_an_error_status_still_falls_back(monkeypatch, exitcode):
    """Below the error severity: a deliberate exit, TERMINATE, and the lesser severities."""
    monkeypatch.setattr(tq, "_sys", types.SimpleNamespace(platform = "win32"))
    assert tq._crashed_child_verdict(_Proc(exitcode), "cuda") is None


@pytest.mark.parametrize("status", [0xC0000005, 0xC0000409])
def test_a_positive_status_off_windows_is_not_a_crash(monkeypatch, status):
    """POSIX encodes a fatal signal negatively, so a large positive code is the child's choice."""
    monkeypatch.setattr(tq, "_sys", types.SimpleNamespace(platform = "linux"))
    assert tq._crashed_child_verdict(_Proc(status), "cuda") is None


def test_crash_verdict_stops_the_in_process_probe(monkeypatch):
    """End of the #9396 chain: the child died, so the parent must not run the same probe."""
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    monkeypatch.setattr(
        tq, "_child_probe_table", lambda device: tq._crashed_child_verdict(_Proc(-11), device)
    )

    def _boom(*args, **kwargs):
        raise AssertionError("a crashed probe child must not be retried in this process")

    monkeypatch.setattr(tq, "_run_smoke_probe", _boom)
    assert tq._scheme_supported(tq.TQ_INT8, "cuda") is False
    # Cached, so a second load answers from memory instead of spawning another child to die.
    assert tq._SMOKE_CACHE[(tq.TQ_INT8, "cuda:0")] is False


def test_child_verdict_is_used_instead_of_a_second_in_process_probe(monkeypatch):
    """A child verdict stored for cuda:0 must prevent a second probe in the backend."""
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(
        tq, "_child_probe_table", lambda device: {s: (s == tq.TQ_INT8) for s in tq.TQ_SCHEMES}
    )

    def _boom(*args, **kwargs):
        raise AssertionError("the child already answered; the parent must not probe again")

    monkeypatch.setattr(tq, "_run_smoke_probe", _boom)
    assert tq._scheme_supported(tq.TQ_INT8, "cuda") is True
    assert tq._scheme_supported(tq.TQ_FP8, "cuda") is False
    assert tq._SMOKE_CACHE[(tq.TQ_INT8, "cuda:0")] is True


@pytest.mark.parametrize("unproven_ok", [True, False])
def test_child_out_of_memory_is_still_not_a_verdict(monkeypatch, unproven_ok):
    """None is the child's allocator failure: not cached, and not turned into "unsupported"."""
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: {s: None for s in tq.TQ_SCHEMES})
    monkeypatch.setattr(tq, "_run_smoke_probe", lambda *a, **k: pytest.fail("no re-probe"))
    assert tq._scheme_supported(tq.TQ_INT8, "cuda", unproven_ok = unproven_ok) is unproven_ok
    assert (tq.TQ_INT8, "cuda:0") not in tq._SMOKE_CACHE


def test_a_scheme_missing_from_the_table_is_still_not_an_answer(monkeypatch):
    """Only a present key is the child's verdict; an absent one falls through as it always did."""
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: {tq.TQ_FP8: True})
    probed: list = []
    monkeypatch.setattr(
        tq, "_run_smoke_probe", lambda scheme, device: probed.append(scheme) or True
    )
    assert tq._scheme_supported(tq.TQ_INT8, "cuda") is True
    assert probed == [tq.TQ_INT8]


# ── 3a. the child must probe the card the verdict is filed under ─────────────


def test_child_is_asked_about_the_pinned_card_not_the_default_one(monkeypatch):
    """A load pinned to GPU 1 caches under cuda:1, so the child has to be asked about cuda:1.

    A freshly spawned child starts on ordinal 0 whatever this thread selected, so handing it a
    bare "cuda" would file the default card's kernel support against the card the load runs on.
    """
    _stub_torch(
        monkeypatch,
        hip = None,
        version_str = "2.10.0+cu128",
        cc = (8, 6),
        current_device = 1,
    )
    asked: list = []
    # Only the pinned card runs the scheme here, which is the mixed-GPU box this guards.
    monkeypatch.setattr(
        tq,
        "_child_probe_table",
        lambda device: asked.append(device) or {s: device == "cuda:1" for s in tq.TQ_SCHEMES},
    )
    monkeypatch.setattr(
        tq, "_run_smoke_probe", lambda *a, **k: pytest.fail("the child already answered")
    )
    assert tq._scheme_supported(tq.TQ_FP8, "cuda") is True
    assert asked == ["cuda:1"]
    assert tq._SMOKE_CACHE[(tq.TQ_FP8, "cuda:1")] is True
    assert (tq.TQ_FP8, "cuda:0") not in tq._SMOKE_CACHE


@pytest.mark.parametrize(
    "device, expected", [("cuda:1", [1]), ("cuda:0", [0]), ("cuda", []), ("cpu", [])]
)
def test_probe_child_selects_the_card_it_was_given(monkeypatch, device, expected):
    """Argument-less CUDA calls inside the probe (synchronize, torchao's capability lookups)
    read the CURRENT device, so the child pins it rather than only placing tensors."""
    torch = _stub_torch(monkeypatch)
    tq._select_probe_card(device)
    assert torch.cuda.selected == expected


# ── 4. the training gate has the same misread, at four entry points ───────────


def _stub_train_torch(
    monkeypatch,
    *,
    hip,
    version_str,
    cc = (11, 5),
):
    """As _stub_torch, plus what the training gates read (bf16 support, mem_get_info)."""
    torch = _stub_torch(monkeypatch, hip = hip, version_str = version_str, cc = cc)
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "native_bf16_supported", lambda: True)
    monkeypatch.setattr(tc, "has_functional_torchao", lambda: True)
    return torch


_TORCHAO_TRAIN_MODES = ("int8", "fp8", "mxfp8")


def test_info_stops_advertising_the_torchao_train_modes_on_rocm(monkeypatch):
    """The Train tab gates its selector on this list, so it is what users are offered."""
    import core.training.diffusion_train_common as tc

    _stub_train_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    modes, recommended = tc.train_precision_modes()
    assert [m for m in modes if m in _TORCHAO_TRAIN_MODES] == []
    # bf16 and the auto ladder are untouched: neither goes near torchao.
    assert "bf16" in modes and "nf4" in modes and recommended == "auto"


def test_info_still_advertises_them_on_cuda(monkeypatch):
    """Regression guard: (11, 5) is a gfx version on ROCm but a real SM level on NVIDIA."""
    import core.training.diffusion_train_common as tc

    _stub_train_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    modes, _ = tc.train_precision_modes()
    assert all(m in modes for m in _TORCHAO_TRAIN_MODES)


@pytest.mark.parametrize("mode", _TORCHAO_TRAIN_MODES)
def test_the_pre_eviction_guard_refuses_torchao_train_modes_on_rocm(monkeypatch, mode):
    """Refused BEFORE the resident model is evicted, like every other doomed precision."""
    import core.training.diffusion_train_common as tc

    _stub_train_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda *a, **k: None, raising = False)
    reason = tc.training_precision_preflight_error("z-image", mode)
    assert reason is not None and "ROCm/AMD GPU" in reason


@pytest.mark.parametrize("mode", _TORCHAO_TRAIN_MODES)
def test_the_child_refuses_the_same_torchao_train_modes_on_rocm(monkeypatch, mode):
    """The child re-check mirrors the parent guard, so a direct client fails the same way."""
    import core.training.diffusion_dit_trainer as dit

    _stub_train_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    cfg = types.SimpleNamespace(base_precision = mode, base_model = "unsloth/Z-Image")
    with pytest.raises(ValueError, match = "ROCm/AMD GPU"):
        dit._resolve_base_precision(cfg, types.SimpleNamespace(dense_bf16_gb = 12.0), "cuda")


def test_auto_never_resolves_to_int8_on_rocm(monkeypatch):
    """auto is the mode /info recommends, so it is the one users actually land on."""
    import core.training.diffusion_dit_trainer as dit

    # free VRAM inside int8's band (> 1.15x dense, < 1.5x dense): the pick auto would make.
    assert dit._pick_auto_precision(False, "cuda", 13.8, 12.0, (11, 5), True, True) == "int8"
    assert dit._pick_auto_precision(False, "cuda", 13.8, 12.0, (11, 5), True, False) == "nf4"

    _stub_train_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    monkeypatch.setattr(dit, "repo_is_prequantized", lambda repo: False)
    monkeypatch.setattr(dit, "trusted_mem_get_info", lambda: (13.8e9, 16e9), raising = False)
    cfg = types.SimpleNamespace(
        base_precision = "auto", base_model = "unsloth/Z-Image", mixed_precision = "bf16"
    )
    resolved = dit._resolve_base_precision(cfg, types.SimpleNamespace(dense_bf16_gb = 12.0), "cuda")
    assert resolved != "int8"
