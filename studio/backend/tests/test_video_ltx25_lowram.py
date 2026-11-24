# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 low-RAM route on fp16-only GPUs: bf16 storage, float32 compute, decided before any download."""

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_te_prequant as tp
import core.inference.video as v
import core.inference.video_ltx25_lowram as lr
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

LTX25 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2.5")
LTX2 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2")
GIB = 1024
_GIB = 1e9 / 1024**3
# route: anonymous (float32 convolution companions, the encoder's float32 dense tensors, runtime) + the larger mapped
# working set (int8 DiT file, or the encoder's int8 codes + connectors)
_DENSE_PART = 2 * 13.1 - 23.9
ROUTE_INT8_TE_GIB = (
    2.82 * 2 + _DENSE_PART * 2 + 10.5 + max(25.11, 13.1 - _DENSE_PART + 6.34)
) * _GIB
ROUTE_DENSE_TE_GIB = (2.82 * 2 + 23.9 * 2 + 10.5 + 25.11) * _GIB
FP32_DENSE_TE_GIB = ((42.0 + 9.2) * 2 + 23.9 * 2) * _GIB


def _target(dtype):
    return DiffusionDeviceTarget(
        device = "cuda",
        dtype = dtype,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
    )


@pytest.fixture
def host(monkeypatch):
    for env in (
        "UNSLOTH_DIFFUSION_SMALL_HOST",
        "UNSLOTH_DIFFUSION_HOST_RAM_CHECK",
        "UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(tp, "te_prequant_sources_confirmed", lambda *a, **k: True)

    def _no_hub(*a, **k):
        raise AssertionError("the route decision must not touch the Hub")

    import huggingface_hub

    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, name, _no_hub)

    def _set(dtype, host_gib):
        monkeypatch.setattr(v, "resolve_diffusion_device_target", lambda *a, **k: _target(dtype))
        monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(host_gib * GIB))
        monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: int(host_gib * GIB * 0.9))

    return _set


def _decide(
    dtype = torch.float16,
    te = "int8",
    fam = LTX25,
    kind = "pipeline",
):
    return lr.decide_lowram(fam, _target(dtype), model_kind = kind, text_encoder_quant = te)


@pytest.mark.parametrize("te", ["int8", "auto", None])
def test_colab_t4_high_ram_host_takes_the_route(host, te):
    host(torch.float16, 51)
    route = _decide(te = te)
    assert route.engaged and route.refusal is None
    assert abs(route.need_mib / GIB - ROUTE_INT8_TE_GIB) < 0.05
    assert (
        v.ltx25_fp16_host_shortfall(
            LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = te
        )
        is None
    )
    v.assert_video_precision_available(LTX25, model_kind = "pipeline", text_encoder_quant = te)
    # the planner and the load see a bfloat16 card
    assert (
        v._ltx25_lowram_target(
            LTX25, _target(torch.float16), kind = "pipeline", text_encoder_quant = te
        ).dtype
        is torch.bfloat16
    )


@pytest.mark.parametrize("host_gib", [31.4, 12.7])
def test_small_host_is_still_refused_before_the_load(host, host_gib):
    host(torch.float16, host_gib)
    route = _decide()
    assert not route.engaged and route.refusal is not None
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", text_encoder_quant = "int8")
    msg = str(err.value)
    assert msg.startswith("Not enough system RAM to load this model: LTX-2.5")
    assert f"about {ROUTE_INT8_TE_GIB:.1f} GB of host memory even on the low-memory route" in msg
    assert (
        "GGUF" in msg
        and "UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0" in msg
        and f"{host_gib} GB is available" in msg
    )
    assert (
        v._ltx25_lowram_target(
            LTX25, _target(torch.float16), kind = "pipeline", text_encoder_quant = "int8"
        ).dtype
        is torch.float16
    )


def test_dense_encoder_is_priced_at_float32(host):
    host(torch.float16, ROUTE_DENSE_TE_GIB + 0.01)
    assert _decide(te = "none").engaged
    host(torch.float16, ROUTE_DENSE_TE_GIB - 0.5)
    route = _decide(te = "none")
    assert not route.engaged and route.refusal is not None


@pytest.mark.parametrize(
    "dtype, fam, kind",
    [
        (torch.bfloat16, LTX25, "pipeline"),
        (torch.float16, LTX25, "gguf"),
        (torch.float16, LTX25, "single_file"),
        (torch.float16, LTX2, "pipeline"),
    ],
)
def test_route_never_engages_off_its_cell(host, dtype, fam, kind):
    host(dtype, 51)
    route = _decide(dtype = dtype, fam = fam, kind = kind)
    assert not route.engaged and route.refusal is None
    target = _target(dtype)
    assert v._ltx25_lowram_target(fam, target, kind = kind) is target


def test_host_that_holds_the_float32_load_is_unchanged(host):
    host(torch.float16, FP32_DENSE_TE_GIB + 1)
    route = _decide(te = "none")
    assert not route.engaged and route.refusal is None and "fits" in route.reason


def test_forced_route_on_a_large_host(host, monkeypatch):
    host(torch.float16, 512)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SMALL_HOST", "1")
    assert _decide().engaged


def test_kill_switch_keeps_the_float32_load(host, monkeypatch):
    host(torch.float16, 51)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SMALL_HOST", "0")
    assert not _decide().engaged
    # the float32 load as before #this change: its own kill switch semantics (no refusal)
    assert v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline") is None
    target = _target(torch.float16)
    assert v._ltx25_lowram_target(LTX25, target, kind = "pipeline") is target


def test_host_check_off_tries_the_route_on_a_small_host(host, monkeypatch):
    host(torch.float16, 31)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", "0")
    route = _decide()
    assert route.engaged and route.refusal is None


def test_unreadable_host_keeps_the_old_load(host, monkeypatch):
    host(torch.float16, 51)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: None)
    route = _decide()
    assert not route.engaged and route.refusal is None


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(16, 32)
        self.lin = torch.nn.Linear(32, 64)
        self.norm = torch.nn.LayerNorm(64)
        self.conv = torch.nn.Conv1d(64, 8, 3, padding = 1)
        self.scale = torch.nn.Parameter(torch.randn(64))

    def forward(self, ids):
        x = self.norm(self.lin(self.emb(ids)) * self.scale)
        return self.conv(x.transpose(1, 2))


def test_upcast_keeps_linear_and_conv_weights_bf16_and_computes_float32():
    torch.manual_seed(0)
    ref = _Tiny().to(torch.bfloat16)
    dense = _Tiny()
    dense.load_state_dict({k: t.float() for k, t in ref.state_dict().items()})
    pipe = type("P", (), {})()
    pipe.transformer = ref
    kept = lr.upcast_compute_(pipe, torch.float32)
    assert set(kept) == {"transformer"}
    assert ref.lin.weight.dtype is torch.bfloat16 and ref.conv.weight.dtype is torch.bfloat16
    assert ref.norm.weight.dtype is torch.float32 and ref.emb.weight.dtype is torch.float32
    assert ref.scale.dtype is torch.float32
    assert kept["transformer"] == 0  # under 1 MiB
    ids = torch.randint(0, 16, (2, 5))
    out = ref(ids)
    assert out.dtype is torch.float32
    assert (
        ref.lin.weight.dtype is torch.bfloat16 and ref.conv.weight.dtype is torch.bfloat16
    )  # cast back
    torch.testing.assert_close(out, dense(ids), rtol = 0, atol = 0)
