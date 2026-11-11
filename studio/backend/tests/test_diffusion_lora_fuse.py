# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unchanged LoRA set is folded into a resident dense denoiser and restored bit-exactly before a change."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")

from core.inference import diffusion as dmod  # noqa: E402

FUSE_ENV = "UNSLOTH_DIFFUSION_LORA_FUSE"
HOST_GUARD_ENV = "UNSLOTH_DIFFUSION_LORA_FUSE_HOST_GUARD"


def _unet_with_lora():
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size = 8,
        in_channels = 4,
        out_channels = 4,
        layers_per_block = 1,
        block_out_channels = (32, 64),
        down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types = ("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim = 24,
        attention_head_dim = 8,
        norm_num_groups = 8,
    ).eval()
    unet.add_adapter(
        LoraConfig(
            r = 4,
            lora_alpha = 4,
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights = False,
        ),
        adapter_name = "px",
    )
    return unet


def _inputs():
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, 4, 8, 8, generator = g),
        torch.tensor([10]),
        torch.randn(1, 3, 24, generator = g),
    )


def _base_weights(unet) -> dict:
    return {n: p.detach().clone() for n, p in unet.named_parameters() if "lora_" not in n}


class _Pipe(types.SimpleNamespace):
    def set_adapters(
        self,
        names,
        adapter_weights = None,
    ):
        self.unet.set_adapters(names, weights = adapter_weights)
        self.calls.append(("set_adapters", tuple(names), tuple(adapter_weights or ())))


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.delenv(FUSE_ENV, raising = False)
    from core.inference import diffusion_memory
    from core.inference.diffusion_small_host import HOST_RAM_LIMIT_ENV

    monkeypatch.delenv(HOST_GUARD_ENV, raising = False)
    monkeypatch.delenv(HOST_RAM_LIMIT_ENV, raising = False)
    # CI runners can sit below the reserve; the RAM tests cap it explicitly.
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 1 << 20)
    monkeypatch.setattr(dmod, "_refuse_lora_without_loader", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "_lora_load_block_reason", lambda pipe: None)
    monkeypatch.setattr(dmod, "invalidate_cfg_replica", lambda *a, **k: None)
    from core.inference import diffusion_lora

    monkeypatch.setattr(diffusion_lora, "lora_unavailable_reason", lambda **k: None)
    monkeypatch.setattr(dmod, "_lora_file_sigs", lambda desired: ("sig",))
    backend = object.__new__(dmod.DiffusionBackend)
    unet = _unet_with_lora()
    pipe = _Pipe(unet = unet, calls = [], _unsloth_loras = (("px", "/x/px.safetensors", 1.0),))
    setattr(pipe, dmod._LORA_FILE_SIGS_ATTR, ("sig",))
    state = types.SimpleNamespace(
        pipe = pipe,
        family = None,
        kind = "pipeline",
        transformer_quant = None,
        offload_policy = dmod.OFFLOAD_NONE,
        hf_token = None,
        speed_optims = (),
        cuda_graphs = (),
    )
    monkeypatch.setattr(
        dmod.DiffusionBackend,
        "_resolve_lora_set",
        lambda self, specs, **k: tuple(("px", "/x/px.safetensors", float(w)) for _i, w in specs),
    )

    def apply(weight):
        dmod.DiffusionBackend._apply_loras(
            backend, state, [("px", weight)], __import__("threading").Event()
        )

    return types.SimpleNamespace(apply = apply, state = state, pipe = pipe, unet = unet)


def _merged_layers(unet) -> int:
    return sum(1 for m in unet.modules() if hasattr(m, "lora_A") and getattr(m, "merged", False))


def test_unchanged_set_is_folded_and_matches_unfused(engine):
    x = _inputs()
    with torch.no_grad():
        unfused = engine.unet(*x).sample
    engine.apply(1.0)  # same set as the last render
    assert _merged_layers(engine.unet) > 0
    with torch.no_grad():
        folded = engine.unet(*x).sample
    torch.testing.assert_close(folded, unfused, rtol = 1e-4, atol = 1e-5)


def test_change_restores_the_original_weights_bit_exactly(engine):
    before = _base_weights(engine.unet)
    engine.apply(1.0)
    assert _merged_layers(engine.unet) > 0
    engine.apply(0.5)  # a strength change: restore, then the weights-only path
    assert _merged_layers(engine.unet) == 0
    after = _base_weights(engine.unet)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert ("set_adapters", ("px",), (0.5,)) in engine.pipe.calls


def test_changed_output_equals_a_never_folded_model(engine):
    x = _inputs()
    reference = _unet_with_lora()
    reference.set_adapters(["px"], weights = [0.5])
    with torch.no_grad():
        expected = reference(*x).sample
    engine.apply(1.0)
    engine.apply(0.5)
    with torch.no_grad():
        got = engine.unet(*x).sample
    assert torch.equal(got, expected)


def test_kill_switch_never_folds(engine, monkeypatch):
    monkeypatch.setenv(FUSE_ENV, "0")
    engine.apply(1.0)
    assert _merged_layers(engine.unet) == 0


def test_offloaded_denoiser_is_not_folded(engine):
    engine.state.offload_policy = dmod.OFFLOAD_MODEL
    engine.apply(1.0)
    assert _merged_layers(engine.unet) == 0


def test_removing_the_set_restores_before_unload(engine):
    before = _base_weights(engine.unet)
    engine.apply(1.0)
    seen = {}

    def unload():
        seen["merged"] = _merged_layers(engine.unet)
        seen["equal"] = all(
            torch.equal(before[k], v) for k, v in _base_weights(engine.unet).items()
        )

    engine.pipe.unload_lora_weights = unload
    dmod.DiffusionBackend._apply_loras(
        object.__new__(dmod.DiffusionBackend), engine.state, [], __import__("threading").Event()
    )
    assert seen == {"merged": 0, "equal": True}


@pytest.mark.parametrize(
    "free_gib, total_gib, mode, expected",
    [
        (80, 96, None, True),  # G4: 75 GiB stay free
        (10, 24, None, False),  # L4 with SDXL resident: 5.6 GiB would stay free
        (80, 96, "host", False),
        (10, 24, "device", True),
    ],
)
def test_backup_lands_on_the_card_only_with_ample_room(
    monkeypatch, free_gib, total_gib, mode, expected
):
    from core.inference import diffusion_lora_fuse as lf

    gib = 1 << 30
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev = None: (free_gib * gib, total_gib * gib)
    )
    if mode is None:
        monkeypatch.delenv(lf.BACKUP_ENV, raising = False)
    else:
        monkeypatch.setenv(lf.BACKUP_ENV, mode)
    assert lf._backup_on_device(torch.device("cuda", 0), int(4.4 * gib)) is expected
    assert lf._backup_on_device(torch.device("cpu"), int(4.4 * gib)) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU (CUDA or ROCm)")
@pytest.mark.parametrize("backup", ["device", "host"])
def test_fold_and_restore_on_the_gpu_in_bf16(engine, monkeypatch, backup):
    # the real placement: bf16 weights on the card, the backup on the card or in pinned host memory
    from core.inference import diffusion_lora_fuse as lf

    monkeypatch.setenv(lf.BACKUP_ENV, backup)
    engine.unet.to("cuda", torch.bfloat16)
    x = tuple(
        t.to("cuda") if t.is_floating_point() is False else t.to("cuda", torch.bfloat16)
        for t in _inputs()
    )
    before = _base_weights(engine.unet)
    with torch.no_grad():
        unfused = engine.unet(*x).sample.float()
    lin = next(
        m
        for m in engine.unet.modules()
        if hasattr(m, "lora_A") and hasattr(m.get_base_layer(), "in_features")
    )
    base_w = lin.get_base_layer().weight.detach().clone()
    a_w, b_w = lin.lora_A["px"].weight.detach().float(), lin.lora_B["px"].weight.detach().float()
    expected = (base_w.float() + (b_w @ a_w) * lin.scaling["px"]).to(base_w.dtype)
    engine.apply(1.0)
    assert _merged_layers(engine.unet) > 0
    # one fp32 multiply-add, one rounding to bf16 (not a bf16 delta added afterwards)
    assert torch.equal(lin.get_base_layer().weight, expected)
    with torch.no_grad():
        folded = engine.unet(*x).sample.float()
    assert (folded - unfused).abs().max().item() < 0.05 * max(1.0, unfused.abs().max().item())
    engine.apply(0.5)
    assert _merged_layers(engine.unet) == 0
    after = _base_weights(engine.unet)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_a_host_backup_that_cuts_into_the_ram_reserve_stays_unfused(engine, monkeypatch):
    from core.inference import diffusion_lora_fuse as lf
    from core.inference.diffusion_small_host import HOST_RAM_LIMIT_ENV

    monkeypatch.setenv(HOST_RAM_LIMIT_ENV, "2048")  # below the 3 GiB reserve alone
    before = _base_weights(engine.unet)
    engine.apply(1.0)
    assert _merged_layers(engine.unet) == 0
    assert not lf.fused(engine.pipe)
    assert all(torch.equal(before[k], v) for k, v in _base_weights(engine.unet).items())
    monkeypatch.setenv(HOST_GUARD_ENV, "0")
    engine.apply(1.0)
    assert _merged_layers(engine.unet) > 0


@pytest.mark.parametrize(
    "available_mib, nbytes_mib, fits",
    [
        (1 << 20, 4 << 10, True),
        (4 << 10, 512, True),
        (4 << 10, 2 << 10, False),
        (None, 8 << 10, True),  # unreadable host: the allocation decides
    ],
)
def test_the_host_backup_keeps_the_small_host_reserve(monkeypatch, available_mib, nbytes_mib, fits):
    from core.inference import diffusion_lora_fuse as lf
    from core.inference import diffusion_small_host

    monkeypatch.delenv(lf.HOST_GUARD_ENV, raising = False)
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (16 << 10, available_mib))
    assert lf._host_backup_fits(nbytes_mib << 20) is fits
    monkeypatch.setenv(lf.HOST_GUARD_ENV, "0")
    assert lf._host_backup_fits(nbytes_mib << 20) is True


@pytest.mark.parametrize("integrated", [True, False])
def test_the_host_backup_is_pageable_on_an_integrated_gpu(monkeypatch, integrated):
    from core.inference import diffusion_lora_fuse as lf
    from core.inference import host_pinning

    monkeypatch.delenv(lf.HOST_GUARD_ENV, raising = False)
    asked = []
    monkeypatch.setattr(
        host_pinning, "device_is_integrated", lambda device = None: asked.append(device) or integrated
    )
    assert lf.HOST_GUARD_ENV == HOST_GUARD_ENV
    cuda = torch.device("cuda", 1)
    assert lf._host_backup_pinned(cuda) is (not integrated)
    assert asked == [cuda]  # the weights' card, not the current device
    assert lf._host_backup_pinned(torch.device("cpu")) is False
    monkeypatch.setenv(lf.HOST_GUARD_ENV, "0")
    assert lf._host_backup_pinned(cuda) is True


def _pinned_backup_charge(monkeypatch, shapes):
    from core.inference import diffusion_lora_fuse as lf

    weights = [torch.empty(s, dtype = torch.bfloat16, device = "meta") for s in shapes]
    monkeypatch.setattr(lf, "_plan", lambda denoisers: [(None, ["px"], w) for w in weights])
    monkeypatch.setattr(lf, "_backup_on_device", lambda device, nbytes: False)
    monkeypatch.setattr(lf, "_host_backup_pinned", lambda device: True)
    charged, pinned = [], []
    monkeypatch.setattr(lf, "_host_backup_fits", lambda nbytes: charged.append(nbytes) or True)
    real_empty = torch.empty

    def empty(
        *args,
        pin_memory = False,
        **kwargs,
    ):
        if pin_memory:
            pinned.append(args[0])
            return real_empty(1, dtype = torch.uint8)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    try:
        assert lf.fuse(types.SimpleNamespace(), [None]) is False  # stops at the stub backup
    finally:
        monkeypatch.setattr(torch, "empty", real_empty)
    # CUDA's caching host allocator rounds every pinned block up to a power of two
    return charged[0], sum(1 << (n - 1).bit_length() for n in pinned)


_FLUX_LIKE = [(3072, 12288)] * 24 + [(3072, 21504), (15360, 3072)] * 4 + [(3072, 3072)] * 8


def test_the_host_guard_is_charged_what_the_pinned_allocator_holds(monkeypatch):
    from core.inference import diffusion_lora_fuse as lf

    monkeypatch.delenv("UNSLOTH_DIFFUSION_LORA_FUSE_POW2_PACK", raising = False)
    charged, held = _pinned_backup_charge(monkeypatch, _FLUX_LIKE)
    assert charged >= held
    exact = sum(-(-r * c * 2 // 64) * 64 for r, c in _FLUX_LIKE)
    assert held <= exact * 1.15


def test_power_of_two_layout_places_every_weight_inside_its_chunk():
    from core.inference import diffusion_lora_fuse as lf

    nbytes = [-(-r * c * 2 // 64) * 64 for r, c in _FLUX_LIKE]
    places, sizes = lf._layout(nbytes, pow2 = True)
    assert all(s & (s - 1) == 0 for s in sizes)
    spans = {}
    for n, (ci, off) in zip(nbytes, places):
        assert off % lf._ALIGN == 0 and off + n <= sizes[ci]
        spans.setdefault(ci, []).append((off, off + n))
    for chunk in spans.values():
        chunk.sort()
        assert all(a[1] <= b[0] for a, b in zip(chunk, chunk[1:]))


def test_power_of_two_packing_kill_switch_keeps_the_exact_slabs(monkeypatch):
    from core.inference import diffusion_lora_fuse as lf

    monkeypatch.setenv("UNSLOTH_DIFFUSION_LORA_FUSE_POW2_PACK", "0")
    assert lf.POW2_PACK_ENV == "UNSLOTH_DIFFUSION_LORA_FUSE_POW2_PACK"
    charged, _held = _pinned_backup_charge(monkeypatch, _FLUX_LIKE)
    assert charged == sum(
        lf._layout([-(-r * c * 2 // 64) * 64 for r, c in _FLUX_LIKE], pow2 = False)[1]
    )
