# SPDX-License-Identifier: AGPL-3.0-or-later
"""The head_dim 256 flex routing is only worth it while SDPA cannot reach cuDNN with a mask.

torch 2.14 raises cuDNN's SDPA head-dim ceiling to 256 on sm100 and lets it accept an explicit
mask, which is the entire reason head_dim 256 needed flex. Kernel names on identical inputs,
head_dim 256 with a mask (scripts/attn_version_gate_probe.py):

    torch 2.13.0+cu130  ->  fmha_cutlass...sm80                     the slow fallback
    torch 2.14.0+cu130  ->  cudnn_..._sdpa_sm100_flash_fprop_...    a real sm100 flash kernel

Timed, Qwen3.5-2B, T=8192, fwd+bwd, reproduced twice to within 1%: masked SDPA drops from
63.892 ms to 9.332 ms, which matches torch 2.13's FlexAttention at 9.255 ms while paying none
of flex's ~4.5 s Inductor compile. So on 2.14 the routing costs a multi-second compile and a
recompile per sequence length and buys nothing.

head_dim > 256 is a different question and is NOT fixed by 2.14: no cuDNN build takes the head
dim, both versions stay on fmha_cutlass, and flex remains the only alternative. These pin that
the gate separates the two bands and that it never fires on an unmeasured device.
"""

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


# ---------------------------------------------------------------------------------------------
# The two bands
# ---------------------------------------------------------------------------------------------


def test_head_dim_256_takes_flex_while_sdpa_cannot_reach_cudnn(cudnn_reaches_256):
    cudnn_reaches_256(False)
    assert u._prefers_flex_for_head_dim(_cfg(256)) is True


def test_head_dim_256_stays_on_sdpa_once_cudnn_takes_the_mask(cudnn_reaches_256):
    cudnn_reaches_256(True)
    assert u._prefers_flex_for_head_dim(_cfg(256)) is False


@pytest.mark.parametrize("head_dim", [264, 512])
def test_above_256_takes_flex_on_every_version(head_dim, cudnn_reaches_256):
    # cuDNN's ceiling stops at 256, so 2.14 does not rescue this band and the gate must not
    # remove it. Gemma 4's global layers live here.
    for reaches in (False, True):
        cudnn_reaches_256(reaches)
        assert u._prefers_flex_for_head_dim(_cfg(head_dim)) is True


@pytest.mark.parametrize("head_dim", [64, 128])
def test_at_or_below_128_never_takes_flex(head_dim, cudnn_reaches_256):
    # cuDNN already serves these masked or not; flex measured 2-3x WORSE here.
    for reaches in (False, True):
        cudnn_reaches_256(reaches)
        assert u._prefers_flex_for_head_dim(_cfg(head_dim)) is False


def test_no_head_dim_means_no_routing(cudnn_reaches_256):
    cudnn_reaches_256(False)
    assert u._prefers_flex_for_head_dim(_Cfg(model_type = "fake")) is False


# ---------------------------------------------------------------------------------------------
# The env var still wins, in both directions, over the gate
# ---------------------------------------------------------------------------------------------


def test_env_var_forces_flex_even_when_cudnn_would_serve_it(monkeypatch, cudnn_reaches_256):
    cudnn_reaches_256(True)
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    assert u._prefers_flex_for_head_dim(_cfg(256)) is True


def test_env_var_keeps_sdpa_even_above_256(monkeypatch, cudnn_reaches_256):
    # The break-even is 74-83 steps; a short run must be able to say no even at head_dim 512.
    cudnn_reaches_256(False)
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "0")
    assert u._prefers_flex_for_head_dim(_cfg(512)) is False


# ---------------------------------------------------------------------------------------------
# The gate itself is deliberately narrow
# ---------------------------------------------------------------------------------------------


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
    # The measurement is sm100-only and cuDNN's ceiling is per architecture, so an A100 or an
    # H100 must keep today's behaviour rather than inherit a conclusion drawn on a B200.
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
