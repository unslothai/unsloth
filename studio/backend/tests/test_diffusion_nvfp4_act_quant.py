# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one-pass NVFP4 activation quantiser: kill switch, eligibility, and (on Blackwell) bit identity."""

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_nvfp4_act_quant as aq  # noqa: E402


def _blackwell() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
    except Exception:  # noqa: BLE001
        return False


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(aq.FAST_ACT_QUANT_ENV, "0")
    assert not aq.fast_act_quant_enabled()
    assert aq.install_nvfp4_act_quant_patch() is False
    assert aq.describe()["reason"].endswith("=0")
    monkeypatch.delenv(aq.FAST_ACT_QUANT_ENV)
    assert aq.fast_act_quant_enabled()


def test_eligibility_is_structural():
    s = torch.tensor([0.01])
    x = torch.zeros(4, 32, dtype = torch.bfloat16)
    # CPU tensors, wrong dtype, unswizzled scales, missing per-tensor scale, odd K: all stay on torchao.
    assert not aq._eligible(x, 16, s, True, False)
    assert not aq._eligible(x.float(), 16, s, True, False)
    assert not aq._eligible(x, 16, None, True, False)
    assert not aq._eligible(x, 16, s, False, False)
    assert not aq._eligible(x, 16, s, True, True)


def test_host_scalar_scale_falls_back_to_torchao():
    # torchao accepts a CPU 0-d per-tensor scale with a CUDA activation; the kernel would read a host pointer.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode(allow_non_fake_inputs = True):
        x = torch.empty(4, 32, dtype = torch.bfloat16, device = "cuda")
        assert aq._eligible(x, 16, torch.empty((), device = "cuda"), True, False)
        assert not aq._eligible(x, 16, torch.empty(()), True, False)


def test_install_without_cuda_is_a_noop(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    monkeypatch.setitem(aq._STATE, "installed", False)
    assert aq.install_nvfp4_act_quant_patch() is False
    assert aq.describe()["reason"] == "cuda unavailable"


@pytest.mark.skipif(not _blackwell(), reason = "needs sm_100+ for torchao NVFP4")
def test_bit_identical_to_compiled_torchao():
    pytest.importorskip("triton")
    ok, reason = aq.verify()
    assert ok, reason


def test_fake_tensors_count_as_traced():
    # AOTAutograd traces torchao's subclass dispatch with fake tensors while is_compiling() is False.
    from torch._subclasses.fake_tensor import FakeTensorMode
    assert not aq._traced(torch.zeros(2, 16))
    with FakeTensorMode():
        assert aq._traced(torch.empty(2, 16))


@pytest.mark.skipif(not _blackwell(), reason = "needs sm_100+ for torchao NVFP4")
def test_compiled_nvfp4_linear_launches_the_kernel():
    pytest.importorskip("triton")
    from torch.profiler import ProfilerActivity, profile
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    assert aq.install_nvfp4_act_quant_patch()
    m = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 512))
    m = m.cuda().to(torch.bfloat16)
    quantize_(m, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
    f = torch.compile(m, fullgraph = True)
    x = torch.randn(256, 512, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        f(x)
        with profile(activities = [ProfilerActivity.CUDA]) as p:
            f(x)
            torch.cuda.synchronize()
    assert sum(1 for e in p.events() if "act_quant_kernel" in e.name) == 2
