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


def test_applies_only_to_block_streamed_single_dit_wan_and_hunyuanvideo_on_cuda(monkeypatch):
    assert _applies()
    assert _applies(offload_policy = "streaming")
    assert _applies(device = "cuda")
    assert not _applies(family = "wan2.2-t2v-a14b", is_moe = True)
    assert not _applies(family = "ltx-2")
    assert _applies(family = "hunyuanvideo-1.5")
    assert _applies(family = "hunyuanvideo-1.5-720p", offload_policy = "streaming")
    assert not _applies(family = "minimax-h3")
    assert not _applies(offload_policy = "none")
    assert not _applies(offload_policy = "model")
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
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(1024, 1024) for _ in range(6))
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
    # The production top-level group is re-pointed to its host copy on offload.
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
    _fit(pipe, monkeypatch, 1 + 3 * block_mib + 1)
    assert _placed(net) == ["cuda"] * 3 + ["cpu"] * 3
    assert next(net.proj_in.parameters()).device.type == "cuda"
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    _fit(pipe, monkeypatch, 1 + block_mib + 1)
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    _fit(pipe, monkeypatch, 1024)
    assert _placed(net) == ["cuda"] * 6
    assert torch.equal(net(x.cuda()).cpu(), ref)
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
    assert vr._measured_extra_mib(net, 100) == 2500
    assert vr._measured_extra_mib(net, 200) == 2500
    assert vr._measured_extra_mib(net, 301) is None
    assert vr._measured_extra_mib(types.SimpleNamespace(), 1) is None


def test_only_a_completed_request_is_recorded(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 5000 << 20)
    net = types.SimpleNamespace(_unsloth_video_pending = (100, 1000 << 20, "cuda"))
    pipe = types.SimpleNamespace(transformer = net)
    assert vr.record_request_peak(pipe) == 4000
    assert net._unsloth_video_peaks == {100: 4000}
    assert net._unsloth_video_pending is None
    assert vr.record_request_peak(pipe) is None


def test_video_request_path_records_the_peak_after_export():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    record = src.index("video_stream_residency.record_request_peak(")
    assert src.index("mp4_bytes = self._encode_mp4(") < record < src.index('"mp4_bytes": mp4_bytes')


def test_measured_peak_widens_the_next_request(monkeypatch):
    torch, net, pipe, x, ref = _streamed_net()
    est = {"room": 1 + 4 + 1}
    monkeypatch.setattr(vr, "room_mib", lambda **kw: est["room"])
    fit = lambda: vr.fit_for_request(
        pipe, device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 1
    )  # noqa: E731
    fit()
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
    assert torch.equal(net(x.cuda()).cpu(), ref)
    vr.record_request_peak(pipe)
    net._unsloth_video_peaks = {64 * 64 * 1: 1}
    fit()
    assert _placed(net) == ["cuda"] * 6
    assert torch.equal(net(x.cuda()).cpu(), ref)
    monkeypatch.setenv(vr.VIDEO_DIT_RESIDENT_MEASURED_ENV, "0")
    fit()
    assert _placed(net) == ["cuda"] + ["cpu"] * 5
    assert torch.equal(net(x.cuda()).cpu(), ref)
    monkeypatch.delenv(vr.VIDEO_DIT_RESIDENT_MEASURED_ENV)
    vr.fit_for_request(pipe, device = "cuda", floor_mib = 1, width = 64, height = 64, frames = 2)
    assert _placed(net) == ["cuda"] + ["cpu"] * 5


def _graph_handle(declined = None, **state):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_cuda_graph as cg

    module = torch.nn.Linear(2, 2)
    module.forward = module.forward  # a group offload forward in the instance slot
    placement = cg.OffloadPlacement(module, "group")
    if declined is not None:
        placement.decline(declined)  # what the streamed speed check does for the whole load
    handle = cg.GraphedForward(module, placement = placement)
    for name, value in state.items():
        setattr(handle, name, value)
    return handle


_DROP = "replay measured within 1% of the eager step on this device (102.1 vs 102.7 ms)"


def test_a_dropped_graph_keys_the_request_on_the_eager_step():
    optims = ("compile", "cuda_graph")
    # the speed check dropped the only graph for the load: steps run eager from here on
    dropped = _graph_handle(
        declined = _DROP, _slower = _DROP, capture_error = {"type": "Refused", "msg": _DROP}
    )
    pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (dropped,))
    assert vr.live_speed_optims(pipe, optims) == ("compile",)
    key = vr.calibration_key("wan2.2-ti2v-5b", speed_optims = vr.live_speed_optims(pipe, optims))
    assert vr._keyed_graphs_live(pipe, key)
    poisoned = _graph_handle(poisoned = True)
    assert vr.live_speed_optims(
        types.SimpleNamespace(_unsloth_cuda_graphs = (poisoned,)), optims
    ) == ("compile",)
    # not armed yet, or no cuda_graph to strip: the load's record stands
    assert vr.live_speed_optims(types.SimpleNamespace(), optims) == optims
    assert vr.live_speed_optims(pipe, ("compile",)) == ("compile",)


@pytest.mark.parametrize("attr", ["fallback", "fallback_handle"])
def test_a_dropped_step_graph_with_per_block_graphs_keeps_the_graph_key(attr):
    # the drop arms (or engaged) the per-block fallback, whose graphs run the next requests
    dropped = _graph_handle(
        declined = _DROP,
        _slower = _DROP,
        capture_error = {"type": "Refused", "msg": _DROP},
        **{attr: object()},
    )
    pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (dropped,))
    assert vr.live_speed_optims(pipe, ("compile", "cuda_graph")) == ("compile", "cuda_graph")


def test_a_request_specific_eager_call_keeps_the_graph_key():
    import core.inference.diffusion_cuda_graph as cg

    # every call so far refused one argument: still armed, a later request can capture and replay
    refused = _graph_handle()
    refused.stats.update(eager_calls = 16, refused_object = 16)
    assert cg.never_engaged((refused,)) is not None
    pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (refused,))
    assert vr.live_speed_optims(pipe, ("compile", "cuda_graph")) == ("compile", "cuda_graph")
    # a placement refusal (Refused, but no speed verdict) is per call too
    placed = _graph_handle(capture_error = {"type": "Refused", "msg": "a placement refusal"})
    pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (placed,))
    assert vr.live_speed_optims(pipe, ("compile", "cuda_graph")) == ("compile", "cuda_graph")


def test_a_capture_racing_the_zero_count_keeps_the_request_dirty(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_measured_peaks as mp

    gen = [3]

    def count(dev):
        gen[0] += 1  # another pipeline captures on the card right after the count was read
        return 0

    monkeypatch.setattr(vr, "_live_graphs", count)
    monkeypatch.setattr(vr, "_capture_generation", lambda dev: gen[0])
    monkeypatch.setattr(mp, "open_window", lambda dev: object())
    monkeypatch.setattr(mp, "close_window", lambda dev, token: True)
    module = types.SimpleNamespace(_unsloth_video_graphs = 0, _unsloth_video_graph_gen = 3)
    vr._open_window(module, torch.device("cuda", 0))
    assert not module._unsloth_video_graphs_clean
    assert module._unsloth_video_graph_gen == 3


def test_video_request_key_names_the_live_speed_optims():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    key = src[src.index("calibration = video_stream_residency.calibration_key(") :]
    key = key[: key.index("video_stream_residency.arm_request(")]
    assert "speed_optims = video_stream_residency.live_speed_optims(" in key
    assert 'speed_optims = tuple(getattr(state, "speed_optims"' not in key


def test_a_request_after_a_dropped_graph_is_recorded_under_its_key(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_measured_peaks as mp

    monkeypatch.setattr(vr, "_PEAKS", {})
    live, gen = [0], [0]
    monkeypatch.setattr(vr, "_live_graphs", lambda dev: live[0])
    monkeypatch.setattr(vr, "_capture_generation", lambda dev: gen[0])
    monkeypatch.setattr(vr, "_store_key", lambda key: None)
    monkeypatch.setattr(mp, "open_window", lambda dev: object())
    monkeypatch.setattr(mp, "close_window", lambda dev, token: True)
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 3000 << 20)
    key = vr.calibration_key("wan2.2-ti2v-5b", speed_optims = ("compile",))
    module = types.SimpleNamespace(_unsloth_video_calibration_key = key)
    pipe = types.SimpleNamespace(transformer = module)
    dev = torch.device("cuda", 0)
    vr._open_window(module, dev)
    assert module._unsloth_video_graphs_clean
    live[0], gen[0] = 0, 1  # it captured one graph, which the speed check then dropped (pool freed)
    vr._open_window(module, dev)
    assert module._unsloth_video_graphs_clean
    module._unsloth_video_pending = (100, 1000 << 20, dev)
    assert vr.record_request_peak(pipe) == 2000
    assert vr.measured_extra_mib(key, 100) == 2000
    # a graph alive from an unrecorded capture keeps the next request dirty
    live[0], gen[0] = 1, 2
    vr._open_window(module, dev)
    assert not module._unsloth_video_graphs_clean


def test_every_capture_is_counted_live_before_its_generation_moves():
    root = Path(__file__).resolve().parents[1] / "core" / "inference"
    block = (root / "diffusion_block_graph.py").read_text(encoding = "utf-8")
    note = "note_capture(self._unsloth_capture_device)"
    for at in [i for i in range(len(block)) if block.startswith(note, i)]:
        body = block[block.rindex("\n    def ", 0, at) : at]
        assert "self.cache[full] = entry" in body
        assert "self._unsloth_capture_device = capture_device()" in body
        if "_LIVE_BLOCKS.add(self)" in block[at : block.index("\n    def ", at)]:
            raise AssertionError("a block joins the live set only after its generation moved")
    step = (root / "diffusion_cuda_graph.py").read_text(encoding = "utf-8")
    at = step.index("            note_capture(self._unsloth_capture_device)")
    assert step.rindex("self.cache[key] = entry", 0, at) < at


@pytest.mark.parametrize(
    "reason",
    [
        "no memory for the speed probe's input copies",
        "replay measured slower than the eager step on this device (110.0 vs 100.0 ms)",
    ],
)
def test_a_per_key_drop_keeps_the_graph_key(reason):
    # a probe OOM, or an unpaid replay on a resident placement, drops one key only: another shape can still capture
    per_key = _graph_handle(_slower = reason, capture_error = {"type": "Refused", "msg": reason})
    assert per_key.placement._declined is None  # the placement still records
    pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (per_key,))
    assert vr.live_speed_optims(pipe, ("compile", "cuda_graph")) == ("compile", "cuda_graph")
