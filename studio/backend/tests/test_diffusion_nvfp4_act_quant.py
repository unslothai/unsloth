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
