# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host route: a streamed encoder's memory-mapped weights move into anonymous host RAM when the host covers
them after placement, so a new prompt does not page them back in from disk. CPU only."""

from __future__ import annotations

import sys
import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_small_host as sh

KAGGLE_TOTAL = 31 * 1024
# FLUX.1-dev on a Kaggle T4: T5-XXL leaves 6912 MiB memory-mapped; ~13.3 GB is available once the int8 DiT is placed
T5_STREAMED = 6912
KAGGLE_AFTER_PLACEMENT = 13300


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.ENCODER_RESIDENT_ENV, sh.SMALL_HOST_ENV, sh.HOST_RAM_LIMIT_ENV):
        monkeypatch.delenv(name, raising = False)


def _decide(**kw):
    args = dict(host_total_mib = KAGGLE_TOTAL, host_available_mib = KAGGLE_AFTER_PLACEMENT)
    args.update(kw)
    return sh.decide_encoder_resident(T5_STREAMED, **args)


def test_kaggle_t4_copies_t5_into_host_ram():
    ok, why = _decide()
    assert ok, why


@pytest.mark.parametrize("available", [None, 9000, T5_STREAMED + 3072 + 2560 - 1])
def test_short_host_keeps_encoder_memory_mapped(available):
    ok, _why = _decide(host_available_mib = available)
    assert not ok


def test_boundary_keeps_reserve_and_runtime():
    keep = sh.host_reserve_mib(KAGGLE_TOTAL) + sh._ENCODER_RESIDENT_RUNTIME_MIB
    assert _decide(host_available_mib = T5_STREAMED + keep)[0]
    assert not _decide(host_available_mib = T5_STREAMED + keep - 1)[0]


def test_large_host_reserve_scales_with_total():
    # 10% of a 128 GB host is kept: 20 GB available after placement covers T5 on a 31 GB host but not here
    assert _decide(host_total_mib = 31 * 1024, host_available_mib = 20000)[0]
    assert not _decide(host_total_mib = 160 * 1024, host_available_mib = 20000)[0]


def test_spill_and_commit_keep_encoder_memory_mapped():
    assert not _decide(int8_spill = True)[0]
    assert not _decide(host_commit_mib = 8000)[0]
    assert _decide(host_commit_mib = 20000)[0]


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(sh.ENCODER_RESIDENT_ENV, "0")
    assert not _decide()[0]
    monkeypatch.setenv(sh.ENCODER_RESIDENT_ENV, "1")
    assert _decide(host_available_mib = 100)[0]
    assert not sh.decide_encoder_resident(0, host_total_mib = 1, host_available_mib = 1)[0]


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wi = torch.nn.Linear(64, 128, bias = False)
        self.wo = torch.nn.Linear(128, 64, bias = False)

    def forward(self, h):
        return self.wo(torch.relu(self.wi(h)).to(self.wo.weight.dtype))


class _Enc(torch.nn.Module):
    _keep_in_fp32_modules = ["wo"]

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(100, 64)
        self.block = _Block()
        self.proj = torch.nn.Linear(64, 64)
        self.norm = torch.nn.LayerNorm(64)

    @property
    def dtype(self):
        return next(p.dtype for p in self.parameters() if p.is_floating_point())

    def forward(self, ids):
        return self.norm(self.proj(self.block(self.embed(ids))))


def _file_ranges(path) -> list:
    out = []
    with open("/proc/self/maps", encoding = "utf-8") as f:
        for line in f:
            if line.rstrip().endswith(str(path)):
                lo, hi = line.split()[0].split("-")
                out.append((int(lo, 16), int(hi, 16)))
    return out


def _mapped(t, ranges) -> bool:
    ptr = t.untyped_storage().data_ptr()
    return any(lo <= ptr < hi for lo, hi in ranges)


def _mmap_encoder(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.manual_seed(0)
    src = _Enc()
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in src.state_dict().items()}
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    enc = _Enc()
    with safe_open(str(path), framework = "pt") as f:
        for k in f.keys():
            owner, _, pname = k.rpartition(".")
            sub = enc.get_submodule(owner)
            setattr(sub, pname, torch.nn.Parameter(f.get_tensor(k), requires_grad = False))
    return enc, path


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "reads /proc/self/maps")
def test_materialize_moves_streamed_weights_off_the_file(tmp_path):
    enc, path = _mmap_encoder(tmp_path)
    ranges = _file_ranges(path)
    if not ranges or not _mapped(enc.proj.weight, ranges):
        pytest.skip("safetensors did not memory-map this file")
    ids = torch.randint(0, 100, (2, 7))
    sh.prepare_streamed_encoder_(enc, torch.float32)
    # a forward outside group offload re-stores layerwise-cast weights, so the reference is a second copy
    (tmp_path / "ref").mkdir()
    ref, _ref_path = _mmap_encoder(tmp_path / "ref")
    sh.prepare_streamed_encoder_(ref, torch.float32)
    with torch.no_grad():
        before = ref(ids)
    assert _mapped(enc.proj.weight, _file_ranges(path))
    mib = sh.materialize_streamed_encoder_(enc)
    assert mib == sh.streamed_encoder_mib(enc) and mib > 0
    for sub in enc.modules():
        if getattr(sub, "_unsloth_mmap_streamed", False):
            assert sub.weight.dtype == torch.bfloat16
            assert not _mapped(sub.weight, _file_ranges(path))
    with torch.no_grad():
        assert torch.equal(enc(ids), before)


def test_materialize_keeps_ties():
    enc = _Enc()
    for p in enc.parameters():
        p.data = p.data.to(torch.bfloat16)
    sh.prepare_streamed_encoder_(enc, torch.float32)
    tied = torch.nn.Linear(64, 128, bias = False)
    tied.weight = enc.block.wi.weight
    tied._unsloth_mmap_streamed = True
    enc.tied = tied
    sh.materialize_streamed_encoder_(enc)
    assert enc.tied.weight is enc.block.wi.weight


def _route(monkeypatch, tmp_path, available):
    from core.inference import diffusion as d
    from core.inference.diffusion import DiffusionBackend

    enc, _path = _mmap_encoder(tmp_path)
    monkeypatch.setattr(d, "host_ram_mib", lambda: (KAGGLE_TOTAL, available))
    monkeypatch.setattr(d, "host_commit_mib", lambda: None)
    monkeypatch.setattr(
        dm, "_loaded_component_mib", lambda pipe: {"text_encoder_2": (1, "text_encoder")}
    )
    pipe = types.SimpleNamespace(text_encoder_2 = enc)
    decision = sh.SmallHostDecision(True, "test", storage_dtypes = {"text_encoder_2": "bfloat16"})
    lines: list = []
    log = types.SimpleNamespace(
        info = lambda *a, **k: lines.append(a[0] % a[1:] if len(a) > 1 else a[0])
    )
    plan = dm.MemoryPlan(
        requested_mode = dm.MEMORY_MODE_AUTO,
        offload_policy = dm.OFFLOAD_GROUP,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 14647, 15095),
        estimates = {"safe_device_budget_mib": 12599},
    )
    target = types.SimpleNamespace(torch_device = torch.device("cpu"))
    backend = types.SimpleNamespace(
        _raise_if_load_cancelled = lambda token: None,
        _small_host_plan = lambda pipe, plan, fam, logger: plan,
    )
    backend._small_host_encoders_resident = types.MethodType(
        DiffusionBackend._small_host_encoders_resident, backend
    )
    DiffusionBackend._apply_small_host_route(
        backend, pipe, plan, decision, target, torch.float32, None, log, None
    )
    return pipe, lines


def test_route_copies_encoder_when_host_covers_it(monkeypatch, tmp_path):
    pipe, lines = _route(monkeypatch, tmp_path, 1 << 20)
    info = sh.engaged_on(pipe)
    assert info["components"]["text_encoder_2"].startswith("anonymous host RAM")
    assert any("copied in host RAM" in line for line in lines)


def test_route_keeps_encoder_mapped_on_a_short_host(monkeypatch, tmp_path):
    pipe, lines = _route(monkeypatch, tmp_path, 4000)
    info = sh.engaged_on(pipe)
    assert info["components"]["text_encoder_2"].startswith("memory-mapped")
    assert any("stay memory-mapped" in line for line in lines)


def test_route_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setenv(sh.ENCODER_RESIDENT_ENV, "0")
    pipe, _lines = _route(monkeypatch, tmp_path, 1 << 20)
    assert sh.engaged_on(pipe)["components"]["text_encoder_2"].startswith("memory-mapped")
