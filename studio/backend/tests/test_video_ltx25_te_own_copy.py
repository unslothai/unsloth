# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 text encode: when the worker owns its host weights instead of encoding from the checkpoint's mapping."""

import sys

import pytest
import torch

from core.inference import diffusion_memory as dm
from core.inference import diffusion_te_embed_host as eh
from core.inference import video_ltx25_te_worker as worker

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "file mappings via /proc/self/maps"
)
GIB = 1 << 30


class _Encoder(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding.from_pretrained(table, freeze = True)
        self.proj = torch.nn.Linear(table.shape[1], table.shape[1], bias = False)
        self.register_buffer("gain", torch.ones(table.shape[1]))

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, ids):
        return self.proj(self.embed_tokens(ids)) * self.gain


def _host(monkeypatch, available_gib, total_gib):
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: int(available_gib * 1024))
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(total_gib * 1024))


def _sized(nbytes):
    return [torch.empty(int(nbytes), dtype = torch.uint8, device = "meta")]


def test_floor_only_prices_the_fixed_floor(monkeypatch):
    # The L4 host: 52 GB, ~16.8 GB available at the first encode, 10.3 GB of encoder weights to own.
    _host(monkeypatch, 16.75, 52)
    need = _sized(10.26 * GIB)
    assert not worker._host_fits(need, copies = 1)  # 15 % of 52 GB = 7.8 GB reserve
    assert worker._host_fits(need, copies = 1, floor_only = True)  # 4 GiB floor
    _host(monkeypatch, 14.0, 52)
    assert not worker._host_fits(need, copies = 1, floor_only = True)


def test_pin_keeps_the_fraction_reserve(monkeypatch):
    _host(monkeypatch, 26.0, 52)  # 26 - 2 x 10.26 = 5.5 GB, under the 7.8 GB reserve
    assert not worker._pin_fits_twice(_sized(10.26 * GIB))
    _host(monkeypatch, 40.0, 52)
    assert worker._pin_fits_twice(_sized(10.26 * GIB))


def test_unreadable_host_memory_keeps_the_mapping(monkeypatch):
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: None)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: 52 * 1024)
    assert not worker._host_fits(_sized(GIB), copies = 1, floor_only = True)


def test_owned_tensors_skip_the_host_table_and_count_ties_once(tmp_path, monkeypatch):
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    model = _Encoder(torch.randn(64, 16))
    model.tied = torch.nn.Linear(16, 16, bias = False)
    model.tied.weight = model.proj.weight
    ids = {id(t) for t in worker._owned_tensors(model)}
    assert id(model.proj.weight) in ids and id(model.gain) in ids
    assert (
        len(worker._owned_tensors(model)) == 3
    )  # table, proj (tied once), gain: no host table yet
    model.embed_tokens.__dict__[eh.ATTR] = ["hook"]  # engaged host table
    owned = worker._owned_tensors(model)
    assert id(model.embed_tokens.weight) not in {id(t) for t in owned}
    assert len(owned) == 2


@linux_only
def test_own_host_copy_owns_above_the_floor_and_stays_bit_identical(tmp_path, monkeypatch):
    from safetensors import safe_open
    from safetensors.torch import save_file

    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    torch.manual_seed(0)
    path = tmp_path / "enc.safetensors"
    save_file({"table": torch.randn(64, 16), "proj": torch.randn(16, 16)}, str(path))
    handle = safe_open(str(path), framework = "pt", device = "cpu")
    model = _Encoder(handle.get_tensor("table"))
    model.proj.weight = torch.nn.Parameter(handle.get_tensor("proj"), requires_grad = False)
    ids = torch.tensor([[3, 2, 1]])
    with torch.inference_mode():
        ref = model(ids).clone()
    assert eh.keep_on_host(model)
    assert eh.file_mapped(model.proj.weight.data)
    _host(monkeypatch, 4.5, 52)  # above the 4 GiB floor, far below the 7.8 GB fraction
    assert worker._own_host_copy(model)
    assert not eh.file_mapped(model.proj.weight.data)
    assert eh.file_mapped(model.embed_tokens.weight.data)
    with torch.inference_mode():
        assert torch.equal(model(ids), ref)


@linux_only
def test_own_host_copy_declines_below_the_floor(tmp_path, monkeypatch):
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = tmp_path / "enc.safetensors"
    save_file({"proj": torch.randn(16, 16)}, str(path))
    handle = safe_open(str(path), framework = "pt", device = "cpu")
    model = _Encoder(torch.randn(8, 16))
    model.proj.weight = torch.nn.Parameter(handle.get_tensor("proj"), requires_grad = False)
    _host(monkeypatch, 3.9, 52)
    assert not worker._own_host_copy(model)
    assert eh.file_mapped(model.proj.weight.data)
