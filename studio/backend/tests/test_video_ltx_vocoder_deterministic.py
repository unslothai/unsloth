# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_ltx2 as L  # noqa: E402


def _cuda_like():
    return types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda"))


@pytest.fixture(autouse = True)
def _flags(monkeypatch):
    monkeypatch.delenv(L.VOCODER_DETERMINISTIC_ENV, raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    before = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = False
    yield
    torch.backends.cudnn.deterministic = before


def test_cuda_call_runs_deterministic_and_restores():
    with L._deterministic_cudnn(_cuda_like()):
        assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.deterministic is False


def test_restored_when_the_call_raises():
    with pytest.raises(RuntimeError):
        with L._deterministic_cudnn(_cuda_like()):
            raise RuntimeError("boom")
    assert torch.backends.cudnn.deterministic is False


def test_a_global_deterministic_setting_is_left_on():
    torch.backends.cudnn.deterministic = True
    with L._deterministic_cudnn(_cuda_like()):
        assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.parametrize("case", ["cpu", "rocm", "rocm_sdk_wheel", "switch_off"])
def test_cpu_rocm_and_kill_switch_keep_the_global_setting(case, monkeypatch):
    mel = _cuda_like()
    if case == "cpu":
        mel = torch.zeros(1)
    elif case == "rocm":
        monkeypatch.setattr(torch.version, "hip", "7.1", raising = False)
    elif case == "rocm_sdk_wheel":
        # AMD SDK / Radeon wheels: torch.version.hip stays unset, the ROCm tag is only in __version__.
        monkeypatch.setattr(torch, "__version__", "2.11.0+rocm7.14.0")
    else:
        monkeypatch.setenv(L.VOCODER_DETERMINISTIC_ENV, "0")
    with L._deterministic_cudnn(mel):
        assert torch.backends.cudnn.deterministic is False


class _Voc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(4, 2, 4, stride = 2)

    def forward(self, mel):
        return self.conv(mel)


def test_the_installed_forward_enters_the_scope(monkeypatch):
    entered = []

    class _Scope:
        def __init__(self, mel):
            entered.append(mel)

        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(L, "_deterministic_cudnn", _Scope)
    voc = _Voc().eval()
    L.install_vocoder_fp32_forward(voc)
    mel = torch.randn(1, 4, 8)
    with torch.no_grad():
        out = voc(mel)
    assert len(entered) == 1 and entered[0] is mel
    assert out.shape == (1, 2, 18)


def test_overlapping_calls_keep_the_flag_until_the_last_exits():
    import threading

    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def call_a():
        with L._deterministic_cudnn(_cuda_like()):
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def call_b():
        a_in.wait(5)
        with L._deterministic_cudnn(_cuda_like()):
            b_in.set()
            a_out.wait(5)
            seen["b_after_a_exit"] = torch.backends.cudnn.deterministic

    threads = [threading.Thread(target = call_a), threading.Thread(target = call_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen["b_after_a_exit"] is True
    assert torch.backends.cudnn.deterministic is False
    assert L._deterministic_depth == 0
