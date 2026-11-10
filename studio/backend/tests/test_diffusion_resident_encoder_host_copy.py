# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image load: resident text-encoder groups drop their host copy (saved 8.7 GiB pinned RAM on Qwen-Image-2.1);
an eviction copies it back from the card."""

from __future__ import annotations

import pathlib
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_this_tree_is_imported():
    assert pathlib.Path(dm.__file__).resolve().is_relative_to(_BACKEND)


class _Group:
    def __init__(
        self,
        params,
        resident,
        card = "meta",
    ):
        self.cpu_param_dict = {}
        self.offloaded = 0
        for p in params:
            host = p.data
            if resident and card != "cpu":
                p = torch.nn.Parameter(torch.empty_like(host, device = card), requires_grad = False)
            self.cpu_param_dict[p] = host
        self.modules = []
        self.parameters = list(params)
        self.buffers = []
        self._unsloth_resident = resident
        if resident:
            self._unsloth_streamed_hooks = (None, None)

    def offload_(self):
        self.offloaded += 1
        for p, host in self.cpu_param_dict.items():
            p.data = host


def _param(*values):
    return torch.nn.Parameter(torch.tensor(values, dtype = torch.float32), requires_grad = False)


@pytest.fixture
def pinner(monkeypatch):
    monkeypatch.delenv(dm.RESIDENT_HOST_DROP_ENV, raising = False)
    pinned = []

    def pin_memory(self):
        pinned.append(self)
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    monkeypatch.setattr(dm, "_bump_placement_epoch", lambda: None)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: False)

    def run(groups, lazy = True):
        module = torch.nn.Linear(1, 1)
        monkeypatch.setattr(dm, "_offload_groups", lambda m: list(groups))
        pipe = types.SimpleNamespace()
        assert dm._defer_pinning(
            pipe, module, torch.device("cpu"), None, fast = False, lazy_resident = lazy
        )
        assert dm.start_background_pins(pipe) == 1
        dm.finish_background_pins(pipe)
        return pinned

    return run


def test_resident_groups_drop_their_host_copy_and_streamed_ones_still_pin(pinner):
    resident = _Group([_param(1.0, 2.0)], resident = True)
    streamed = _Group([_param(3.0)], resident = False)
    pinned = pinner([resident, streamed])
    assert resident.cpu_param_dict == {}
    assert getattr(resident, dm._HOST_DROPPED_ATTR)
    assert len(pinned) == 1 and pinned[0].tolist() == [3.0]
    assert next(iter(streamed.cpu_param_dict.values())).tolist() == [3.0]


def test_the_kill_switch_pins_resident_groups_too(pinner, monkeypatch):
    monkeypatch.setenv(dm.RESIDENT_HOST_DROP_ENV, "0")
    resident = _Group([_param(1.0, 2.0)], resident = True)
    streamed = _Group([_param(3.0)], resident = False)
    pinned = pinner([resident, streamed])
    assert len(pinned) == 2
    assert len(resident.cpu_param_dict) == 1


def test_only_image_encoder_pinners_drop(pinner):
    resident = _Group([_param(1.0, 2.0)], resident = True)
    pinned = pinner([resident], lazy = False)
    assert len(pinned) == 1 and len(resident.cpu_param_dict) == 1


def test_a_pinned_or_host_reading_group_keeps_its_copy():
    on_host = _Group([_param(1.0)], resident = True, card = "cpu")
    kept, freed = dm.drop_resident_host_copies([on_host])
    assert kept == [on_host] and freed == 0 and len(on_host.cpu_param_dict) == 1


def test_tied_partners_drop_together_or_not_at_all():
    host = torch.tensor([5.0, 6.0])
    shared = torch.nn.Parameter(torch.empty_like(host, device = "meta"), requires_grad = False)
    a = _Group([], resident = True)
    b = _Group([], resident = False)
    a.cpu_param_dict[shared] = host
    b.cpu_param_dict[shared] = host
    a._unsloth_tied_groups, b._unsloth_tied_groups = [b], [a]
    kept, freed = dm.drop_resident_host_copies([a, b])
    assert kept == [a, b] and freed == 0
    b._unsloth_resident = True
    kept, freed = dm.drop_resident_host_copies([a, b])
    assert kept == [] and freed == host.nbytes


def test_an_eviction_copies_the_device_weights_back_before_streaming(monkeypatch):
    monkeypatch.setattr(dm, "_bump_placement_epoch", lambda: None)
    p = _param(7.0, 8.0)
    group = _Group([p], resident = True, card = "cpu")
    p.data.copy_(torch.tensor([7.0, 8.0]))
    group.cpu_param_dict.clear()
    setattr(group, dm._HOST_DROPPED_ATTR, [p])
    live = p.data
    dm._release_group(group)
    assert group.offloaded == 1 and group._unsloth_resident is False
    host = group.cpu_param_dict[p]
    assert host.tolist() == [7.0, 8.0] and host.data_ptr() != live.data_ptr()
    assert p.data.data_ptr() == host.data_ptr()
    assert not getattr(group, dm._HOST_DROPPED_ATTR)


def test_tied_partners_rebuild_one_shared_copy():
    shared = _param(1.0, 2.0)
    a = _Group([], resident = True)
    b = _Group([], resident = True)
    a._unsloth_tied_groups, b._unsloth_tied_groups = [b], [a]
    setattr(a, dm._HOST_DROPPED_ATTR, [shared])
    setattr(b, dm._HOST_DROPPED_ATTR, [shared])
    copied = dm.rebuild_dropped_host_copy(a)
    assert copied == shared.data.nbytes
    assert a.cpu_param_dict[shared] is b.cpu_param_dict[shared]
    assert dm.rebuild_dropped_host_copy(b) == 0


@pytest.mark.parametrize("pin_budget", [None, 0])
def test_cuda_dropped_encoder_streams_bit_identical_after_an_eviction(monkeypatch, pin_budget):
    """Real diffusers leaf-level group offload on CUDA: the deferred encoder pin drops every resident group's host
    copy, an oversized release copies them back in one registered slab, the streamed forward and the restored one
    match the reference bit for bit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only: diffusers group offloading with a copy stream")
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import apply_group_offloading

    monkeypatch.delenv(dm.RESIDENT_HOST_DROP_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)
    if pin_budget is not None:
        # host RAM cannot pin the copy-back now: pageable copies, same values
        monkeypatch.setattr(dm, "_pin_budget_mib", lambda: pin_budget)
    torch.manual_seed(0)
    net = torch.nn.Sequential(*(torch.nn.Linear(256, 256) for _ in range(6)))
    x = torch.randn(4, 256)
    ref = net.to("cuda")(x.cuda()).cpu()
    net.to("cpu")
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "leaf_level",
        use_stream = True,
        record_stream = True,
        non_blocking = True,
        low_cpu_mem_usage = True,
    )
    pipe = types.SimpleNamespace(components = {"text_encoder": net})
    assert dm._defer_pinning(pipe, net, torch.device("cuda"), None, fast = False, lazy_resident = True)
    assert dm._keep_groups_resident(net, 1024, "cuda") > 0
    dm.start_background_pins(pipe)
    dm.finish_background_pins(pipe)
    groups = dm._offload_groups(net)
    dropped = [g for g in groups if getattr(g, dm._HOST_DROPPED_ATTR, None)]
    assert dropped and all(not g.cpu_param_dict for g in dropped)
    assert torch.equal(net(x.cuda()).cpu(), ref)

    restore = dm.release_resident_groups(pipe, 1024)
    assert restore is not None
    assert not any(getattr(g, dm._HOST_DROPPED_ATTR, None) for g in groups)
    assert all(next(m.parameters()).device.type == "cpu" for m in net)
    hosts = [h for g in groups for h in g.cpu_param_dict.values()]
    assert all(h.is_pinned() for h in hosts) == (pin_budget is None)
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    restore()
    assert all(next(m.parameters()).device.type == "cuda" for m in net)
    assert torch.equal(net(x.cuda()).cpu(), ref)
    again = dm.release_resident_groups(pipe, 1024)
    assert [h.data_ptr() for g in groups for h in g.cpu_param_dict.values()] == [
        h.data_ptr() for h in hosts
    ]
    assert torch.equal(net(x.cuda()).cpu(), ref)
    again()
    assert torch.equal(net(x.cuda()).cpu(), ref)


def test_a_group_whose_copy_back_fails_stays_resident_and_the_release_goes_on(monkeypatch):
    monkeypatch.setattr(dm, "_bump_placement_epoch", lambda: None)

    def refuse(groups):
        raise MemoryError("host RAM")

    monkeypatch.setattr(dm, "rebuild_dropped_host_copies", refuse)
    monkeypatch.setattr(
        dm,
        "rebuild_dropped_host_copy",
        lambda g: refuse([g]) if getattr(g, dm._HOST_DROPPED_ATTR, None) else 0,
    )
    dropped = _Group([_param(1.0)], resident = True, card = "cpu")
    dropped.cpu_param_dict.clear()
    setattr(dropped, dm._HOST_DROPPED_ATTR, list(dropped.parameters))
    dropped._unsloth_resident_bytes = 4 << 20
    kept = _Group([_param(2.0)], resident = True, card = "cpu")
    kept._unsloth_resident_bytes = 4 << 20
    module = types.SimpleNamespace(_unsloth_resident_room = 8, _unsloth_resident_device = None)
    monkeypatch.setattr(dm, "_offload_groups", lambda m: [kept, dropped])
    monkeypatch.setattr(dm, "_is_text_encoder_module", lambda pipe, m: True)
    pipe = types.SimpleNamespace(components = {"text_encoder": module})
    restore = dm.release_resident_groups(pipe, 8)
    assert restore is not None
    assert dropped._unsloth_resident is True and dropped.offloaded == 0
    assert kept._unsloth_resident is False and kept.offloaded == 1
