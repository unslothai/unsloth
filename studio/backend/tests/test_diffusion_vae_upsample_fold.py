# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Strip decode upsample fold (diffusion_vae_strips): nearest 2x + conv3x3(p1) as one ConvTranspose2d(k4, s2, p1).
CPU: the fold equals the 2x chain to fp32-summed-weight rounding (fp64 here: ~1e-15), whole map and strip by strip with
the one-row halo; the gate keeps unmeasured devices on the 2x path; the folded kernel follows in-place weight edits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
F = torch.nn.functional
from core.inference import diffusion_vae_strips as st  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(st.UPSAMPLE_FOLD_ENV, raising = False)
    st._FOLD_CACHE.clear()  # noqa: SLF001
    yield
    st._FOLD_CACHE.clear()  # noqa: SLF001


def _chain(
    conv,
    x,
    bias = True,
):
    return F.conv2d(
        F.interpolate(x, scale_factor = 2.0, mode = "nearest"),
        conv.weight,
        conv.bias if bias else None,
        padding = 1,
    )


def _case(
    cin = 6,
    cout = 5,
    h = 9,
    w = 7,
    dtype = torch.float64,
):
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(cin, cout, 3, padding = 1).to(dtype)
    x = torch.randn(1, cin, h, w, dtype = dtype).contiguous(memory_format = torch.channels_last)
    return conv, x


@pytest.mark.parametrize("bias", [True, False])
def test_whole_map_fold_matches_the_2x_chain(bias):
    conv, x = _case()
    y = st._fold_upsample(conv, x, conv.bias if bias else None)  # noqa: SLF001
    ref = _chain(conv, x, bias)
    assert y.shape == ref.shape and y.is_contiguous(memory_format = torch.channels_last)
    assert (y - ref).abs().max().item() < 1e-6  # fp32-summed taps, fp64 data


@pytest.mark.parametrize("h", [1, 2, 5, 13])
def test_strip_fold_matches_the_whole_map(monkeypatch, h):
    """Strips with the one-row halo, including the first / last strip and single-row maps."""
    conv, x = _case(h = h)
    monkeypatch.setattr(
        st, "STRIP_ELEMS", conv.out_channels * 4 * 2 * x.shape[-1]
    )  # 2 input rows per strip
    whole = st._fold_upsample(conv, x, conv.bias)  # noqa: SLF001
    strips = st._fold_upsample(conv, x, conv.bias, full_max = 0)  # noqa: SLF001
    # Not bitwise: the CPU conv backend may block a strip-sized input differently from the whole map and sum the taps
    # in another order (fp64 here, so ~1e-16). A seam or halo bug is off by O(1) and still fails.
    assert (whole - strips).abs().max().item() < 1e-10
    assert (strips - _chain(conv, x)).abs().max().item() < 1e-6


def test_folded_taps_follow_in_place_weight_edits():
    conv, x = _case()
    a = st.folded_upsample_weight(conv)
    assert st.folded_upsample_weight(conv) is a  # cached
    with torch.no_grad():
        conv.weight.mul_(2)  # e.g. a merged LoRA
    b = st.folded_upsample_weight(conv)
    assert b is not a and torch.allclose(b, 2 * a)


def test_gate_keeps_unmeasured_devices_on_the_2x_path(monkeypatch):
    conv, x = _case(dtype = torch.float32)
    up = type("Upsample2D", (), {"conv": conv})()
    calls = st._FOLD_CALLS[0]  # noqa: SLF001
    out, pend = st._upsample(up, x, False)  # noqa: SLF001 - CPU, auto: the 2x path
    assert st._FOLD_CALLS[0] == calls and pend is None  # noqa: SLF001
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    out2, pend2 = st._upsample(up, x, True)  # noqa: SLF001 - forced: folded, bias deferred
    assert st._FOLD_CALLS[0] == calls + 1 and pend2 is conv.bias  # noqa: SLF001
    assert (out2 + conv.bias.view(1, -1, 1, 1) - out).abs().max().item() < 1e-5
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "0")
    st._upsample(up, x, False)  # noqa: SLF001
    assert st._FOLD_CALLS[0] == calls + 1  # noqa: SLF001


def test_auto_arms_only_the_measured_cuda_cards(monkeypatch):
    x = torch.empty(1)
    assert not st._fold_on(x)  # noqa: SLF001 - CPU
    fake = type("T", (), {"is_cuda": True, "device": "cuda:0"})()
    monkeypatch.setattr(torch.version, "hip", None)
    for cap, want in (
        ((8, 0), True),
        ((12, 0), True),
        ((8, 9), True),
        ((7, 5), True),
        ((10, 0), False),
        ((9, 0), False),
        ((8, 6), False),
    ):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d, c = cap: c)
        assert st._fold_on(fake) is want, cap  # noqa: SLF001
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 0))
    for arch, want in (
        ("gfx1151", True),
        ("gfx1151:sramecc-:xnack-", True),
        ("gfx942", False),
        ("gfx1100", False),
        ("", False),
    ):
        props = type("P", (), {"gcnArchName": arch})()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d, p = props: p)
        assert st._fold_on(fake) is want, arch  # noqa: SLF001 - ROCm: only the measured MIOpen arch
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "0")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda d: type("P", (), {"gcnArchName": "gfx1151"})()
    )
    assert not st._fold_on(fake)  # noqa: SLF001 - the kill switch still wins


def test_non_3x3_or_dtype_mismatch_keeps_the_2x_path(monkeypatch):
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    conv, x = _case(dtype = torch.float32)
    assert st._fold_ok(conv, x)  # noqa: SLF001
    assert not st._fold_ok(conv, x.double())  # noqa: SLF001
    c5 = torch.nn.Conv2d(6, 5, 5, padding = 2)
    assert not st._fold_ok(c5, x)  # noqa: SLF001


def test_fold_follows_a_data_swap_like_the_fp32_retry(monkeypatch):
    """Studio's fp32 decode retry (diffusion_vae_upcast) swaps ``.data`` under the same Parameter: version unchanged."""
    from core.inference import diffusion_vae_upcast as upcast

    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    conv, x = _case(dtype = torch.bfloat16)
    up = type("Upsample2D", (), {"conv": conv})()
    st._upsample(up, x, False)  # noqa: SLF001 - folds and caches the bf16 taps
    token = upcast.upcast_keeping_storage(conv)
    try:
        y, _ = st._upsample(up, x.float(), False)  # noqa: SLF001
        assert y.dtype == torch.float32
        assert (y - _chain(conv, x.float())).abs().max().item() < 1e-5
    finally:
        upcast.restore_storage(token)
    assert st._upsample(up, x, False)[0].dtype == torch.bfloat16  # noqa: SLF001
