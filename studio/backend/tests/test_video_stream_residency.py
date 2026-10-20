# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-request residency of a block-streamed Wan denoiser (video_stream_residency)."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

import core.inference.diffusion_memory as dm
import core.inference.video_stream_residency as vr


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(vr.VIDEO_DIT_RESIDENT_ENV, raising = False)
    monkeypatch.delenv(vr.VIDEO_DIT_RESIDENT_BLOCKS_ENV, raising = False)
    monkeypatch.delenv(vr.VIDEO_DIT_RESIDENT_MEASURED_ENV, raising = False)
    monkeypatch.delenv(dm.PARTIAL_RESIDENT_ENV, raising = False)


def _applies(**kw):
    args = dict(is_moe = False, offload_policy = "group", device = "cuda:0")
    args.update(kw)
    family = args.pop("family", "wan2.2-ti2v-5b")
    return vr.applies(family, **args)


def test_applies_only_to_group_streamed_single_dit_wan_on_cuda(monkeypatch):
    assert _applies()
    assert _applies(device = "cuda")
    assert not _applies(family = "wan2.2-t2v-a14b", is_moe = True)
    assert not _applies(family = "ltx-2")
    assert not _applies(family = "hunyuanvideo-1.5")
    assert not _applies(offload_policy = "none")
    assert not _applies(offload_policy = "model")
    assert not _applies(offload_policy = "streaming")
    assert not _applies(device = "cpu")
    assert not _applies(device = "mps")
    assert not _applies(modular = True)
    monkeypatch.setenv(vr.VIDEO_DIT_RESIDENT_ENV, "0")
    assert not _applies()


def test_room_shrinks_with_the_clip_and_never_goes_negative():
    common = dict(free_mib = 9000, unused_cache_mib = 500, resident_mib_now = 0, floor_mib = 2500)
    short = vr.room_mib(width = 1280, height = 704, frames = 25, **common)
    long = vr.room_mib(width = 1280, height = 704, frames = 121, **common)
    assert short > long >= 0
    # T4 after load: 9000 + 500 - (2500 + 2048 + estimate(25 frames) = 4869) = 83
    est = dm.estimate_video_runtime_mib(width = 1280, height = 704, num_frames = 25)
    assert short == 9500 - (2500 + dm.DEFAULT_BASE_OVERHEAD_MIB + est)
    # already-resident bytes are demotable, so they count as available
    assert (
        vr.room_mib(width = 1280, height = 704, frames = 25, **{**common, "resident_mib_now": 4000})
        == short + 4000
    )
    assert vr.room_mib(width = 1280, height = 704, frames = 25, **{**common, "free_mib": 0}) == 0


def test_skipped_without_a_floor_or_a_denoiser():
    assert (
        vr.fit_for_request(
            types.SimpleNamespace(), device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 1
        )
        is None
    )
    net = types.SimpleNamespace()
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    assert (
        vr.fit_for_request(pipe, device = "cuda", floor_mib = None, width = 64, height = 64, frames = 1)
        is None
    )


def test_video_request_path_fits_the_residency_before_the_vram_check():
    """The generate preflight re-fits the resident set, gated by applies(), before it reads free memory."""
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "video_stream_residency"
    ]
    names = {c.func.attr for c in calls}
    assert {"applies", "fit_for_request"} <= names
    fit = next(c for c in calls if c.func.attr == "fit_for_request")
    free_reads = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "trusted_mem_get_info"
    ]
    assert any(line > fit.lineno for line in free_reads)


def test_fit_failure_streams_everything_and_never_raises(monkeypatch):
    released = []

    def boom(*a, **k):
        raise RuntimeError("no device")

    monkeypatch.setattr(vr, "resident_mib", boom)
    monkeypatch.setattr(
        dm, "release_resident_groups", lambda pipe, need, *a, **k: released.append(need)
    )
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1 << 30, 1 << 31))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    net = object()
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    assert (
        vr.fit_for_request(pipe, device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 1) is None
    )
    assert released


# CUDA: real diffusers group offloading with a copy stream.


def _streamed_net():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip(reason = "CUDA-only: diffusers group offloading with a copy stream")
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import apply_group_offloading

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(64, 1024)
            self.blocks = torch.nn.ModuleList(
                torch.nn.Linear(1024, 1024) for _ in range(6)
            )  # ~4 MiB each
            self.proj_out = torch.nn.Linear(1024, 64)

        def forward(self, x):
            x = self.proj_in(x)
            for block in self.blocks:
                x = torch.nn.functional.gelu(block(x))
            return self.proj_out(x)

    torch.manual_seed(0)
    net = Net()
    x = torch.randn(3, 64)
    ref = net.to("cuda")(x.cuda()).cpu()
    net.to("cpu")
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
        low_cpu_mem_usage = True,
    )
    # the production top-level group: re-pointed to its host copy on offload, as _apply_group_offload leaves it
    assert dm._skip_top_level_copy_back(net)
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    return torch, net, pipe, x, ref


def _placed(net):
    return [next(b.parameters()).device.type for b in net.blocks]


def _fit(pipe, monkeypatch, room):
    monkeypatch.setattr(vr, "room_mib", lambda **kw: room)
    return vr.fit_for_request(pipe, device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 1)


def test_base_group_offload_uploads_the_top_level_group_every_call():
    """What the residency removes: without it the top-level group is back on the host after every call."""
    torch, net, pipe, x, ref = _streamed_net()
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
        assert next(net.proj_in.parameters()).device.type == "cpu"
        assert _placed(net) == ["cpu"] * 6


def test_top_level_and_block_prefix_stay_resident_bit_identical(monkeypatch):
    torch, net, pipe, x, ref = _streamed_net()
    kept = _fit(pipe, monkeypatch, 1024)
    assert kept and kept >= 24
    for _ in range(3):  # first call traces the prefetch order, later ones prefetch
        assert torch.equal(net(x.cuda()).cpu(), ref)
        assert next(net.proj_in.parameters()).device.type == "cuda"
        assert next(net.proj_out.parameters()).device.type == "cuda"
        assert _placed(net) == ["cuda"] * 6
    assert vr.counts()["resident_mib"] == kept


def test_prefix_is_sized_to_the_room_and_refit_per_request(monkeypatch):
    torch, net, pipe, x, ref = _streamed_net()
    block_mib = 4
    _fit(pipe, monkeypatch, 1 + 3 * block_mib + 1)  # top-level (~0.5 MiB) + 3 blocks
    assert _placed(net) == ["cuda"] * 3 + ["cpu"] * 3
    assert next(net.proj_in.parameters()).device.type == "cuda"
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    # a longer clip: less room, the prefix shrinks from its end
    _fit(pipe, monkeypatch, 1 + block_mib + 1)
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    # a short clip again: grows back
    _fit(pipe, monkeypatch, 1024)
    assert _placed(net) == ["cuda"] * 6
    assert torch.equal(net(x.cuda()).cpu(), ref)
    # no room: everything streams, as plain group offloading
    _fit(pipe, monkeypatch, 0)
    assert _placed(net) == ["cpu"] * 6
    assert next(net.proj_in.parameters()).device.type == "cpu"
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
        assert next(net.proj_in.parameters()).device.type == "cpu"


def test_blocks_kill_switch_keeps_only_the_top_level_group(monkeypatch):
    torch, net, pipe, x, ref = _streamed_net()
    monkeypatch.setenv(vr.VIDEO_DIT_RESIDENT_BLOCKS_ENV, "0")
    _fit(pipe, monkeypatch, 1024)
    assert _placed(net) == ["cpu"] * 6
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
        assert next(net.proj_in.parameters()).device.type == "cuda"


def test_measured_room_and_cover_lookup():
    common = dict(free_mib = 9000, unused_cache_mib = 500, resident_mib_now = 2000)
    assert (
        vr.measured_room_mib(peak_extra_mib = 2000, **common)
        == 11500 - int(2000 * vr.MEASURED_PEAK_MARGIN) - vr.MEASURED_SLACK_MIB
    )
    assert vr.measured_room_mib(peak_extra_mib = 20000, **common) == 0
    net = types.SimpleNamespace(_unsloth_video_peaks = {100: 1500, 300: 2500})
    assert vr._measured_extra_mib(net, 100) == 2500  # every recorded request at least this large
    assert vr._measured_extra_mib(net, 200) == 2500
    assert (
        vr._measured_extra_mib(net, 301) is None
    )  # larger than anything measured: the estimate sizes it
    assert vr._measured_extra_mib(types.SimpleNamespace(), 1) is None


def test_only_a_completed_request_is_recorded(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 5000 << 20)
    net = types.SimpleNamespace(_unsloth_video_pending = (100, 1000 << 20, "cuda"))
    pipe = types.SimpleNamespace(transformer = net)
    assert vr.record_request_peak(pipe) == 4000
    assert net._unsloth_video_peaks == {100: 4000}
    assert net._unsloth_video_pending is None
    assert vr.record_request_peak(pipe) is None  # recorded once per fit


def test_video_request_path_records_the_peak_after_export():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    record = src.index("video_stream_residency.record_request_peak(")
    assert src.index("mp4_bytes = self._encode_mp4(") < record < src.index('"mp4_bytes": mp4_bytes')


def test_measured_peak_widens_the_next_request(monkeypatch):
    torch, net, pipe, x, ref = _streamed_net()
    est = {"room": 1 + 4 + 1}  # top-level + 1 block from the estimate
    monkeypatch.setattr(vr, "room_mib", lambda **kw: est["room"])
    fit = lambda: vr.fit_for_request(
        pipe, device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 1
    )  # noqa: E731
    fit()
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
    assert torch.equal(net(x.cuda()).cpu(), ref)
    vr.record_request_peak(pipe)
    net._unsloth_video_peaks = {64 * 64 * 1: 1}
    fit()  # the measured room on a large card holds every block
    assert _placed(net) == ["cuda"] * 6
    assert torch.equal(net(x.cuda()).cpu(), ref)
    monkeypatch.setenv(vr.VIDEO_DIT_RESIDENT_MEASURED_ENV, "0")
    fit()
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
    assert torch.equal(net(x.cuda()).cpu(), ref)
    # a larger request than anything measured is sized by the estimate
    monkeypatch.delenv(vr.VIDEO_DIT_RESIDENT_MEASURED_ENV)
    vr.fit_for_request(pipe, device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 2)
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
