# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto's frozen fp8 base: picked on sm89+ from 0.9x dense free VRAM up to the bf16 band, and loaded host-first so
the bf16 base is never whole on the device (the conversion streams Linear by Linear)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_frozen_fp8 as ff8


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(32, 32, bias = False)
        self.ff = nn.Linear(32, 64)
        self.odd = nn.Linear(30, 32)  # not 16-aligned: stays bf16
        self.norm = nn.LayerNorm(32)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.proj_out = nn.Linear(32, 32)
        self.to(torch.bfloat16).requires_grad_(False)


def test_convert_frozen_fp8_streams_host_linears_to_the_device():
    model = _Tiny()
    names = {f"blocks.{i}.{n}" for i in range(2) for n in ("to_q", "ff")}
    done, failed = ff8.convert_frozen_fp8(model, names, device = "meta")
    assert (done, failed) == (4, 0)
    for i in range(2):
        b = model.blocks[i]
        assert isinstance(b.to_q, ff8.FrozenFp8Linear) and b.to_q.weight.device.type == "meta"
        assert b.ff.weight.dtype == torch.float8_e4m3fn and b.ff.bias.device.type == "meta"
        # Untargeted modules are left where they were for the caller's final move.
        assert b.odd.weight.device.type == "cpu" and b.norm.weight.device.type == "cpu"


def test_convert_frozen_fp8_without_device_keeps_the_old_in_place_behaviour():
    model = _Tiny()
    done, _ = ff8.convert_frozen_fp8(model, {"blocks.0.to_q"})
    assert done == 1 and model.blocks[0].to_q.weight.device.type == "cpu"


def test_apply_fp8_training_lands_a_host_base_on_the_device(monkeypatch):
    model = _Tiny()
    events = []
    ok = dit._apply_fp8_training(model, events.append, None, device = "meta")
    assert ok is True
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(b.device.type == "meta" for b in model.buffers())
    assert isinstance(model.blocks[0].to_q, ff8.FrozenFp8Linear)
    assert type(model.blocks[0].odd) is nn.Linear and type(model.proj_out) is nn.Linear


def test_apply_fp8_training_falls_back_to_bf16_on_the_device(monkeypatch):
    model = _Tiny()

    def boom(*a, **k):
        raise RuntimeError("no fp8 here")

    monkeypatch.setattr(dit, "_convert_fp8", boom)
    events = []
    assert dit._apply_fp8_training(model, events.append, None, device = "meta") is False
    assert all(p.device.type == "meta" for p in model.parameters())
    assert type(model.blocks[0].to_q) is nn.Linear


def test_fp8_without_a_hosted_file_loads_on_the_host(monkeypatch):
    moved = []

    class _T(nn.Module):
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, *a, **k):
            moved.append(a)
            return self

    monkeypatch.setattr(dit, "_load_fp8_prequant_transformer", lambda *a, **k: None)
    cfg = types.SimpleNamespace(base_model = "x/y", hf_token = None)
    dit._load_dit_transformer(_T, cfg, "cuda", "fp8")
    assert moved == []
    dit._load_dit_transformer(_T, cfg, "cuda", "bf16")
    assert moved == [("cuda",)]


@pytest.mark.parametrize(
    "free, cap, expected",
    [
        (19.0, (8, 9), "bf16"),  # > 1.5x
        (16.4, (8, 9), "fp8"),  # 1.33x
        (11.5, (8, 9), "fp8"),  # 0.93x: streamed fp8, not nf4
        (10.9, (8, 9), "nf4"),  # 0.89x
        (16.4, (8, 6), "int8"),
        (11.5, (8, 6), "nf4"),
    ],
)
def test_pick_auto_precision_fp8_band(free, cap, expected):
    assert dit._pick_auto_precision(False, "cuda", free, 12.3, cap, True, True) == expected


def test_auto_fp8_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIT_AUTO_FP8", "0")
    assert dit._auto_fp8_enabled() is False
    monkeypatch.setenv("UNSLOTH_DIT_AUTO_FP8", "1")
    assert dit._auto_fp8_enabled() is True


@pytest.mark.parametrize(
    "family, expected", [("qwen-image", "int8"), ("moss-image", "int8"), ("z-image", "fp8")]
)
def test_auto_fp8_respects_the_family_train_deny(monkeypatch, family, expected):
    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return (int(1.3 * dit._SPECS[family].dense_bf16_gb * 1e9), int(80 * 1e9))

        @staticmethod
        def get_device_capability():
            return (8, 9)

    monkeypatch.delenv("UNSLOTH_DIT_AUTO_FP8", raising = False)
    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)
    monkeypatch.setattr(dit, "repo_is_prequantized", lambda repo: False)
    monkeypatch.setattr(dit, "_auto_run_compiles", lambda cfg: True)
    monkeypatch.setattr(dit, "_auto_fp8_probe_ok", lambda device: True)
    spec = dit._SPECS[family]
    cfg = types.SimpleNamespace(
        base_precision = "auto", base_model = "some/dense-base", mixed_precision = "bf16"
    )
    assert dit._resolve_base_precision(cfg, spec, "cuda") == expected


@pytest.mark.parametrize(
    "family, ratio, expected",
    [
        ("z-image", 1.6, "fp8"),
        ("flux.2-klein", 1.6, "fp8"),
        ("flux.1", 1.6, "bf16"),
        ("qwen-image-2.1", 1.6, "bf16"),
        ("wan2.2-ti2v-5b", 1.6, "bf16"),
        ("wan2.2-ti2v-5b", 1.3, "int8"),
        ("wan2.2-ti2v-5b", 1.0, "fp8"),
        ("flux.1", 1.3, "fp8"),
        (None, 1.3, "fp8"),
        (None, 1.6, "bf16"),
    ],
)
def test_pick_auto_precision_per_family(family, ratio, expected):
    assert (
        dit._pick_auto_precision(False, "cuda", 10.0 * ratio, 10.0, (8, 9), True, True, family)
        == expected
    )
    # Below sm89 nothing changes.
    old = dit._pick_auto_precision(False, "cuda", 10.0 * ratio, 10.0, (8, 6), True, True, family)
    assert old == ("bf16" if ratio > 1.5 else "int8" if ratio > 1.15 else "nf4")


def test_fp8_over_bf16_needs_a_compiled_run(monkeypatch):
    assert (
        dit._pick_auto_precision(False, "cuda", 16.0, 10.0, (8, 9), True, True, "z-image", False)
        == "bf16"
    )
    # Eager fp8 lost to the eager picks it would replace (L4, Z-Image): an eager run keeps them in every band.
    assert (
        dit._pick_auto_precision(False, "cuda", 13.0, 10.0, (8, 9), True, True, "z-image", False)
        == "int8"
    )
    assert (
        dit._pick_auto_precision(False, "cuda", 10.0, 10.0, (8, 9), True, True, "z-image", False)
        == "nf4"
    )
    assert (
        dit._pick_auto_precision(False, "cuda", 10.0, 10.0, (8, 9), True, True, "z-image", True)
        == "fp8"
    )
    cfg = types.SimpleNamespace(compile_transformer = "off")
    assert dit._auto_run_compiles(cfg) is False
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert dit._auto_run_compiles(types.SimpleNamespace(compile_transformer = "auto")) is False


def _identity_with(tmp_path, effective):
    """A resumable-looking bundle manifest recording ``effective`` as the trained base precision."""
    from core.training.diffusion_checkpoint import CheckpointIdentity

    ident = CheckpointIdentity(
        family = "z-image",
        base_model = "m",
        lora_target_modules = ("to_q",),
        lora_rank = 4,
        lora_alpha = 4,
        precision = "bf16",
        base_precision = "auto",
        resolution = 512,
        base_precision_effective = effective,
    )
    return {"identity": ident.as_dict()}


@pytest.mark.parametrize(
    "found, recorded, expected",
    [
        (False, None, True),  # no readable bundle: the preflight reports it, the pick is free
        (True, "fp8", True),
        (
            True,
            "int8",
            False,
        ),  # was refused after the full load: "resolved base precision (int8 vs fp8)"
        (True, "bf16", False),
        (True, None, False),  # predates the record: trained before Auto picked fp8
    ],
)
def test_auto_fp8_on_resume_follows_the_recorded_base(
    monkeypatch, tmp_path, found, recorded, expected
):
    monkeypatch.delenv("UNSLOTH_DIT_BLOCK_SWAP", raising = False)
    monkeypatch.setattr(dit, "recorded_base_precision", lambda path: (found, recorded))
    cfg = types.SimpleNamespace(resume_from_checkpoint = str(tmp_path), block_swap = None)
    assert dit._auto_fp8_run_ok(cfg) is expected
    assert dit._auto_fp8_run_ok(types.SimpleNamespace(resume_from_checkpoint = None)) is True


def test_recorded_base_precision_reads_the_bundle(monkeypatch, tmp_path):
    from core.training import diffusion_checkpoint as ck

    manifest = _identity_with(tmp_path, "int8")
    monkeypatch.setattr(ck, "resolve_resume_dir", lambda p: tmp_path)
    monkeypatch.setattr(ck, "checkpoint_step", lambda p: 8)
    monkeypatch.setattr(ck, "read_checkpoint", lambda p: manifest)
    assert ck.recorded_base_precision(str(tmp_path)) == (True, "int8")
    monkeypatch.setattr(ck, "read_checkpoint", lambda p: None)
    monkeypatch.setattr(ck, "iter_valid_checkpoints", lambda p: [])
    assert ck.recorded_base_precision(str(tmp_path)) == (False, None)


@pytest.mark.parametrize("swap", ["auto", "4", 4])
def test_auto_never_picks_fp8_with_a_block_swap(monkeypatch, swap):
    """A requested block swap streams a bf16 base only; fp8 was then refused by the swap gate at load."""
    monkeypatch.delenv("UNSLOTH_DIT_BLOCK_SWAP", raising = False)
    cfg = types.SimpleNamespace(block_swap = swap, resume_from_checkpoint = None)
    assert dit._auto_fp8_run_ok(cfg) is False
    assert dit._auto_fp8_run_ok(types.SimpleNamespace(block_swap = "off")) is True
    monkeypatch.setenv("UNSLOTH_DIT_BLOCK_SWAP", "auto")
    assert dit._auto_fp8_run_ok(types.SimpleNamespace(block_swap = None)) is False


def test_resolver_with_a_block_swap_keeps_the_previous_picks(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return (int(1.6 * dit._SPECS["z-image"].dense_bf16_gb * 1e9), int(80 * 1e9))

        @staticmethod
        def get_device_capability():
            return (8, 9)

    monkeypatch.delenv("UNSLOTH_DIT_AUTO_FP8", raising = False)
    monkeypatch.delenv("UNSLOTH_DIT_BLOCK_SWAP", raising = False)
    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)
    monkeypatch.setattr(dit, "repo_is_prequantized", lambda repo: False)
    monkeypatch.setattr(dit, "_auto_run_compiles", lambda cfg: True)
    monkeypatch.setattr(dit, "_auto_fp8_probe_ok", lambda device: True)
    spec = dit._SPECS["z-image"]
    cfg = types.SimpleNamespace(
        base_precision = "auto", base_model = "some/dense-base", mixed_precision = "bf16"
    )
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "fp8"
    cfg.block_swap = "auto"
    assert dit._resolve_base_precision(cfg, spec, "cuda") == "bf16"


def test_a_failing_fp8_probe_keeps_auto_off_fp8(monkeypatch):
    calls = []

    def probe(
        convert,
        transformer,
        device = None,
    ):
        calls.append(device)
        raise RuntimeError("scaled_mm unsupported")

    monkeypatch.setattr(dit, "_probe_lowp", probe)
    monkeypatch.setattr(dit, "_FP8_PROBE_OK", {})
    assert dit._auto_fp8_probe_ok("cuda:0") is False
    assert dit._auto_fp8_probe_ok("cuda:0") is False
    assert calls == ["cuda:0"]  # cached per device


def test_a_failed_conversion_that_cannot_fall_back_to_bf16_asks_for_nf4(monkeypatch):
    """Below the bf16 fit the fp8 base streams host-first; a failed conversion must not move the bf16 rest onto a
    card it never fit (main trained nf4 there), so the caller reloads nf4."""
    model = _Tiny()

    def boom(*a, **k):
        raise RuntimeError("no fp8 here")

    monkeypatch.setattr(dit, "_convert_fp8", boom)
    monkeypatch.setattr(dit, "_bf16_fallback_fits", lambda transformer, device: False)
    with pytest.raises(dit.Fp8FallbackDoesNotFit, match = "no fp8 here"):
        dit._apply_fp8_training(model, [].append, None, device = "meta")
    # Nothing moved: the caller frees the host copy and loads nf4.
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_the_bf16_fallback_fit_counts_only_host_bytes(monkeypatch):
    model = _Tiny()
    host = dit._host_bytes(model)
    assert host == sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    need = int(host * dit._FP8_FALLBACK_MARGIN) + dit._FP8_FALLBACK_RESERVE

    class _FakeCuda:
        free = need

        @classmethod
        def mem_get_info(cls, device = None):
            return (cls.free, 1 << 40)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    assert dit._bf16_fallback_fits(model, "cuda:0") is True
    _FakeCuda.free = need - (1 << 20)
    assert dit._bf16_fallback_fits(model, "cuda:0") is False
    assert dit._bf16_fallback_fits(model, "cpu") is True
