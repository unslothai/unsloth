# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3: the pageable streamed conditioner keeps the share that fits in anonymous host RAM. CPU only."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import core.inference.diffusion_small_host as sh
import core.inference.video_minimax_h3_te as te

# Colab L4 high-RAM: 53 GiB host, 20.2 GiB available after load, 23.5 GiB streamed conditioner.
L4_TOTAL = 53 * 1024
L4_AFTER_LOAD = 20664
H3_TE_STREAMED = 24064


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (te.H3_TE_RESIDENT_ENV, sh.ENCODER_RESIDENT_ENV, sh.HOST_RAM_LIMIT_ENV):
        monkeypatch.delenv(name, raising = False)


def _keep(total = L4_TOTAL):
    return sh.encoder_resident_keep_mib(total)


def test_keep_matches_the_small_host_reserve():
    assert _keep() == sh.host_reserve_mib(L4_TOTAL) + sh._ENCODER_RESIDENT_RUNTIME_MIB
    # decide_encoder_resident refuses exactly where the share stops being the whole encoder
    need = 6912
    edge = need + _keep(31 * 1024)
    for available in (edge - 1, edge, edge + 1):
        ok, _ = sh.decide_encoder_resident(
            need, host_total_mib = 31 * 1024, host_available_mib = available
        )
        share, _ = sh.encoder_resident_share_mib(
            need, host_total_mib = 31 * 1024, host_available_mib = available
        )
        assert ok == (share == need)


def test_l4_copies_the_share_beside_the_reserve():
    share, why = sh.encoder_resident_share_mib(
        H3_TE_STREAMED, host_total_mib = L4_TOTAL, host_available_mib = L4_AFTER_LOAD
    )
    assert share == L4_AFTER_LOAD - _keep(), why
    assert 0 < share < H3_TE_STREAMED


def test_whole_encoder_when_it_fits():
    share, _ = sh.encoder_resident_share_mib(
        H3_TE_STREAMED, host_total_mib = 177 * 1024, host_available_mib = 120_000
    )
    assert share == H3_TE_STREAMED


@pytest.mark.parametrize("available", [None, 0, 5000, "keep"])
def test_no_room_copies_nothing(available):
    if available == "keep":
        available = _keep()
    share, why = sh.encoder_resident_share_mib(
        H3_TE_STREAMED, host_total_mib = L4_TOTAL, host_available_mib = available
    )
    assert share == 0, why


def test_small_share_is_not_worth_it():
    share, _ = sh.encoder_resident_share_mib(
        H3_TE_STREAMED,
        host_total_mib = L4_TOTAL,
        host_available_mib = _keep() + 500,
        min_share_mib = 1024,
    )
    assert share == 0


def test_commit_headroom_caps_the_share():
    share, _ = sh.encoder_resident_share_mib(
        H3_TE_STREAMED,
        host_total_mib = L4_TOTAL,
        host_available_mib = 40_000,
        host_commit_mib = _keep() + 3000,
    )
    assert share == 3000


def _mmap_tensors(tmp_path, specs):
    from safetensors import safe_open
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(0)
    path = tmp_path / "te.safetensors"
    save_file(
        {k: torch.randint(-127, 127, shape, generator = gen, dtype = torch.int8) for k, shape in specs},
        str(path),
    )
    with safe_open(str(path), framework = "pt", device = "cpu") as handle:
        return path, {k: handle.get_tensor(k) for k, _ in specs}


def _file_mapped(ptr: int, path: Path) -> bool:
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[-1] == str(path):
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if lo <= ptr < hi:
                return True
    return False


class _Leaf(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)


def _fake_encoder(tensors):
    """text_encoder.model with language and vision leaves, one stream-style offload group per leaf."""
    model = torch.nn.Module()
    model.language_model = torch.nn.ModuleList([_Leaf(tensors["l0"]), _Leaf(tensors["l1"])])
    model.visual = torch.nn.ModuleList([_Leaf(tensors["v0"])])
    groups = []
    for leaf in (model.visual[0], model.language_model[0], model.language_model[1]):
        groups.append(
            SimpleNamespace(
                modules = [leaf],
                parameters = [],
                buffers = [],
                onload_leader = leaf,
                cpu_param_dict = {leaf.weight: leaf.weight.data},
            )
        )
    return SimpleNamespace(model = model), groups


SPECS = [("l0", (1024, 1024)), ("l1", (1024, 1024)), ("v0", (1024, 1024))]
LEAF_MIB = 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "reads /proc/self/maps")
def test_copy_leaves_the_mapping_and_keeps_values(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "_H3_TE_RESIDENT_MIN_MIB", 1)
    path, tensors = _mmap_tensors(tmp_path, SPECS)
    ref = {k: v.clone() for k, v in tensors.items()}
    assert all(_file_mapped(t.data_ptr(), path) for t in tensors.values())
    encoder, groups = _fake_encoder(tensors)
    mib = te.keep_h3_te_resident(encoder, groups = groups, host_ram = (L4_TOTAL, 10**6, None))
    assert mib == 3 * LEAF_MIB
    for group in groups:
        ((tensor, host),) = group.cpu_param_dict.items()
        assert host is not None and not _file_mapped(host.data_ptr(), path)
        assert tensor.data_ptr() == host.data_ptr()  # the offloaded buffer reads the copy too
    m = encoder.model
    for name, leaf in (
        ("l0", m.language_model[0]),
        ("l1", m.language_model[1]),
        ("v0", m.visual[0]),
    ):
        assert torch.equal(leaf.weight, ref[name])


def test_share_fills_language_layers_before_the_vision_tower(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "_H3_TE_RESIDENT_MIN_MIB", 1)
    _path, tensors = _mmap_tensors(tmp_path, SPECS)
    encoder, groups = _fake_encoder(tensors)
    before = {id(g): next(iter(g.cpu_param_dict.values())).data_ptr() for g in groups}
    # room for two of the three 1 MiB leaves above the reserve
    mib = te.keep_h3_te_resident(encoder, groups = groups, host_ram = (L4_TOTAL, _keep() + 2, None))
    assert mib == 2 * LEAF_MIB
    moved = [next(iter(g.cpu_param_dict.values())).data_ptr() != before[id(g)] for g in groups]
    assert moved == [False, True, True]  # visual group first in the list, kept on the mapping


def test_kill_switch_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(te.H3_TE_RESIDENT_ENV, "0")
    _path, tensors = _mmap_tensors(tmp_path, SPECS)
    encoder, groups = _fake_encoder(tensors)
    before = [next(iter(g.cpu_param_dict.values())).data_ptr() for g in groups]
    assert te.keep_h3_te_resident(encoder, groups = groups, host_ram = (L4_TOTAL, 10**6, None)) == 0
    assert [next(iter(g.cpu_param_dict.values())).data_ptr() for g in groups] == before


def test_no_room_changes_nothing(tmp_path):
    _path, tensors = _mmap_tensors(tmp_path, SPECS)
    encoder, groups = _fake_encoder(tensors)
    before = [next(iter(g.cpu_param_dict.values())).data_ptr() for g in groups]
    assert te.keep_h3_te_resident(encoder, groups = groups, host_ram = (L4_TOTAL, 5000, None)) == 0
    assert [next(iter(g.cpu_param_dict.values())).data_ptr() for g in groups] == before


def test_unreadable_host_ram_and_odd_input_never_raise():
    assert te.keep_h3_te_resident(None) == 0
    assert (
        te.keep_h3_te_resident(
            SimpleNamespace(model = torch.nn.Linear(2, 2)), host_ram = (None, None, None)
        )
        == 0
    )
    assert te.keep_h3_te_resident(SimpleNamespace(model = object()), groups = [object()]) == 0


def test_pinned_and_subclass_hosts_are_skipped():
    class _Sub(torch.Tensor):
        pass

    leaf = _Leaf(torch.zeros(4))
    sub = torch.zeros(4).as_subclass(_Sub)
    group = SimpleNamespace(
        modules = [leaf], parameters = [], buffers = [], cpu_param_dict = {leaf.weight: sub}
    )
    model = torch.nn.Module()
    model.leaf = leaf
    assert te._streamed_host_tensors(model, [group]) == []


def test_real_group_offload_forward_unchanged(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    from diffusers.hooks import apply_group_offloading
    from safetensors.torch import load_file, save_file

    monkeypatch.setattr(te, "_H3_TE_RESIDENT_MIN_MIB", 0)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))
    path = tmp_path / "m.safetensors"
    save_file(model.state_dict(), str(path))
    model.load_state_dict(load_file(str(path)), assign = True)
    x = torch.randn(3, 64)
    ref = model(x)
    apply_group_offloading(
        model,
        onload_device = torch.device("cpu"),
        offload_device = torch.device("cpu"),
        offload_type = "leaf_level",
    )
    mib = te.keep_h3_te_resident(SimpleNamespace(model = model), host_ram = (L4_TOTAL, 10**6, None))
    assert mib >= 0
    assert torch.equal(model(x), ref)
    if sys.platform.startswith("linux"):
        assert not any(_file_mapped(p.data_ptr(), path) for p in model.parameters())


def test_load_copies_only_for_the_pageable_stream_after_the_vae_copies():
    src = (Path(te.__file__).parent / "video.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "keep_h3_te_resident" in ast.unparse(node)
        and ast.unparse(node.test) == "te_streamed == 'stream_lazy'"
    ]
    assert len(guards) == 1
    assert src.index("keep_h3_te_resident(") > src.index("install_pinned_swap(")
