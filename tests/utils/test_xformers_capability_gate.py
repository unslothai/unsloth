"""The xformers gate probes the real op, not the compute-capability major version."""

import pytest
from real_accelerator import (
    has_real_cuda,
)
import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad


@pytest.mark.parametrize(
    "capability, probe_result, expect_disabled",
    [
        ((8, 9), None, False),
        ((9, 0), None, False),
        ((10, 0), None, False),
        ((12, 0), True, False),
        ((12, 0), False, True),
    ],
)
def test_capability_gate(capability, probe_result, expect_disabled):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return probe_result

    assert (
        ad._xformers_disabled_for_capability(capability, probe = probe, fa3_enabled = lambda: True)
        is expect_disabled
    )
    # Below sm_120 the probe must not run at all (no import-time kernel launch there).
    assert calls["n"] == (0 if capability[0] < 12 else 1)


@pytest.mark.parametrize(
    "capability, fa3_enabled, probe_result, expect_disabled, expect_probed",
    [
        ((9, 0), True, None, False, False),  # sm_90a build (PyPI wheels): FA3 kept, never probed
        ((9, 0), False, True, False, True),  # sm_90 build whose FA2 / cutlass run (#3963)
        ((9, 0), False, False, True, True),  # build with no sm_90 kernel at all: SDPA
        ((8, 0), False, None, False, False),  # FA3 off elsewhere does not probe
    ],
)
def test_hopper_gate_probes_only_when_fa3_was_turned_off(
    capability, fa3_enabled, probe_result, expect_disabled, expect_probed
):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return probe_result

    assert (
        ad._xformers_disabled_for_capability(
            capability, probe = probe, fa3_enabled = lambda: fa3_enabled
        )
        is expect_disabled
    )
    assert calls["n"] == int(expect_probed)


class _Build:
    def __init__(self, arch_list):
        self.build_env = {"TORCH_CUDA_ARCH_LIST": arch_list}


class _Flash3:
    def __init__(
        self,
        version = "2.8.0",
        loaded = True,
    ):
        self.FLASH_VERSION = version
        self._C_flashattention3 = object() if loaded else None


@pytest.mark.parametrize(
    "arch_list, capability, expect",
    [
        # conda-forge xformers 0.0.32.post2 (#3963) and 0.0.34: no sm_90a anywhere.
        ("5.0;6.0;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX", (9, 0), True),
        ("8.0;8.9;10.0;12.0+PTX", (9, 0), True),
        ("7.5 8.0+PTX 9.0a", (9, 0), False),
        ("9.0a+PTX", (9, 0), False),
        (None, (9, 0), False),
        ("8.0;8.9;10.0;12.0+PTX", (8, 0), False),
        ("8.0;8.9;10.0;12.0+PTX", (10, 0), False),
    ],
)
def test_fa3_lacks_sm90a(arch_list, capability, expect):
    from unsloth.models._utils import _xformers_fa3_lacks_sm90a
    assert _xformers_fa3_lacks_sm90a(_Build(arch_list), capability, _Flash3()) is expect


def test_fa3_lacks_sm90a_ignores_external_or_missing_fa3():
    from unsloth.models._utils import _xformers_fa3_lacks_sm90a

    conda = _Build("8.0;8.9;10.0;12.0+PTX")
    # xformers >= 0.0.35 loads FA3 from a separate flash_attn_3 package.
    assert _xformers_fa3_lacks_sm90a(conda, (9, 0), _Flash3(version = "pip_pkg")) is False
    assert _xformers_fa3_lacks_sm90a(conda, (9, 0), _Flash3(loaded = False)) is False
    assert _xformers_fa3_lacks_sm90a(conda, (9, 0), None) is False


def test_fa3_lacks_sm90a_fails_open_on_unreadable_metadata():
    from unsloth.models._utils import _xformers_fa3_lacks_sm90a
    assert _xformers_fa3_lacks_sm90a(None, (9, 0), _Flash3()) is False
    assert _xformers_fa3_lacks_sm90a(object(), (9, 0), _Flash3()) is False


@pytest.mark.skipif(
    not (has_real_cuda() and ad.HAS_XFORMERS),
    reason = "needs a CUDA GPU with a working xformers build",
)
@pytest.mark.skipif(
    has_real_cuda() and torch.cuda.get_device_capability()[0] >= 12,
    reason = "on real sm_120+ the probe legitimately returns False when the build ships no "
    "sm_120 kernel, so asserting True there would be a false failure",
)
def test_probe_shapes_are_valid_on_working_gpu():
    # A probe that raises everywhere would silently disable xformers on working GPUs.
    assert ad._xformers_runs_on_device() is True


@pytest.mark.parametrize(
    "supports_bf16, expected_dtype",
    [(True, torch.bfloat16), (False, torch.float16)],
)
def test_probe_dtype_follows_bf16_support(monkeypatch, supports_bf16, expected_dtype):
    # Pre-Ampere GPUs lack bf16 attention, so the probe dtype must follow SUPPORTS_BFLOAT16.
    captured = {}

    def fake_zeros(
        *args,
        dtype = None,
        **kwargs,
    ):
        captured["dtype"] = dtype
        raise RuntimeError("stop after capturing the probe dtype")

    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", supports_bf16)
    monkeypatch.setattr(ad.torch, "zeros", fake_zeros)
    ad._xformers_runs_on_device()
    assert captured["dtype"] is expected_dtype


def test_probe_syncs_and_fails_on_deferred_async_error(monkeypatch):
    # Kernel launches are async; the probe must synchronize to catch deferred errors.
    _bias = type(
        "B",
        (),
        {
            "BlockDiagonalCausalMask": type(
                "M", (), {"from_seqlens": staticmethod(lambda seqlens: None)}
            )
        },
    )
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)
    monkeypatch.setattr(ad.torch, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(ad, "xformers", type("X", (), {"attn_bias": _bias}))
    monkeypatch.setattr(ad, "xformers_attention", lambda *a, **k: None)

    def deferred_cuda_error():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(ad.torch.cuda, "synchronize", deferred_cuda_error)
    assert ad._xformers_runs_on_device() is False
