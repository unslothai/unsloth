# SPDX-License-Identifier: AGPL-3.0-or-later
"""head_dim 256 routes to flex only until SDPA reaches cuDNN with a mask (torch 2.14, sm100).
Above 256 no cuDNN build helps, so it routes on every version."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _cfg(head_dim):
    return _Cfg(model_type = "fake", head_dim = head_dim, num_attention_heads = 8)


@pytest.fixture(autouse = True)
def _no_env_override(monkeypatch):
    # The env var short-circuits before the gate, so it must be clear for these.
    monkeypatch.delenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, raising = False)


@pytest.fixture
def cudnn_reaches_256(monkeypatch):
    def _set(value):
        monkeypatch.setattr(u, "_sdpa_reaches_cudnn_at_head_dim_256", lambda: value)

    return _set


def test_head_dim_256_takes_flex_while_sdpa_cannot_reach_cudnn(cudnn_reaches_256):
    cudnn_reaches_256(False)
    assert u._prefers_flex_for_head_dim(_cfg(256)) is True


def test_head_dim_256_stays_on_sdpa_once_cudnn_takes_the_mask(cudnn_reaches_256):
    cudnn_reaches_256(True)
    assert u._prefers_flex_for_head_dim(_cfg(256)) is False


@pytest.mark.parametrize("head_dim", [264, 512])
def test_above_256_takes_flex_on_every_version(head_dim, cudnn_reaches_256):
    for reaches in (False, True):
        cudnn_reaches_256(reaches)
        assert u._prefers_flex_for_head_dim(_cfg(head_dim)) is True


@pytest.mark.parametrize("head_dim", [64, 128])
def test_at_or_below_128_never_takes_flex(head_dim, cudnn_reaches_256):
    for reaches in (False, True):
        cudnn_reaches_256(reaches)
        assert u._prefers_flex_for_head_dim(_cfg(head_dim)) is False


def test_no_head_dim_means_no_routing(cudnn_reaches_256):
    cudnn_reaches_256(False)
    assert u._prefers_flex_for_head_dim(_Cfg(model_type = "fake")) is False


def test_env_var_forces_flex_even_when_cudnn_would_serve_it(monkeypatch, cudnn_reaches_256):
    cudnn_reaches_256(True)
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    assert u._prefers_flex_for_head_dim(_cfg(256)) is True


def test_env_var_keeps_sdpa_even_above_256(monkeypatch, cudnn_reaches_256):
    cudnn_reaches_256(False)
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "0")
    assert u._prefers_flex_for_head_dim(_cfg(512)) is False


def test_gate_is_off_below_torch_2_14(monkeypatch):
    monkeypatch.setattr(u.torch, "__version__", "2.13.0+cu130")
    assert u._sdpa_reaches_cudnn_at_head_dim_256() is False


def test_gate_is_off_on_rocm(monkeypatch):
    monkeypatch.setattr(u.torch, "__version__", "2.14.0+cu130")
    monkeypatch.setattr(u.torch.version, "hip", "6.2.0", raising = False)
    assert u._sdpa_reaches_cudnn_at_head_dim_256() is False


def test_gate_is_off_with_no_cuda(monkeypatch):
    monkeypatch.setattr(u.torch, "__version__", "2.14.0+cu130")
    monkeypatch.setattr(u.torch.version, "hip", None, raising = False)
    monkeypatch.setattr(u.torch.cuda, "is_available", lambda: False)
    assert u._sdpa_reaches_cudnn_at_head_dim_256() is False


def test_gate_is_off_below_blackwell(monkeypatch):
    monkeypatch.setattr(u.torch, "__version__", "2.14.0+cu130")
    monkeypatch.setattr(u.torch.version, "hip", None, raising = False)
    monkeypatch.setattr(u.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(u.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(u.torch.cuda, "get_device_capability", lambda index: (9, 0))
    assert u._sdpa_reaches_cudnn_at_head_dim_256() is False


def test_gate_is_on_for_blackwell_on_torch_2_14(monkeypatch):
    monkeypatch.setattr(u.torch, "__version__", "2.14.0+cu130")
    monkeypatch.setattr(u.torch.version, "hip", None, raising = False)
    monkeypatch.setattr(u.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(u.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(u.torch.cuda, "get_device_capability", lambda index: (10, 0))
    assert u._sdpa_reaches_cudnn_at_head_dim_256() is True


def test_a_mixed_box_falls_back_to_the_weakest_card(monkeypatch):
    monkeypatch.setattr(u.torch, "__version__", "2.14.0+cu130")
    monkeypatch.setattr(u.torch.version, "hip", None, raising = False)
    monkeypatch.setattr(u.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(u.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        u.torch.cuda, "get_device_capability", lambda index: (10, 0) if index == 0 else (9, 0)
    )
    assert u._sdpa_reaches_cudnn_at_head_dim_256() is False
