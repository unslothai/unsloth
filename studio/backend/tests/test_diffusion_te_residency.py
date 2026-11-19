# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-encoder residency re-decided per render (diffusion_te_residency): the fit decision after each prompt encode,
the re-check before the next encode, the lend path for other loads, and the wiring in the load, the generate path,
dictation loads and the GPU residency free steps. CPU only: live free memory is faked, and the streaming mechanics run
with a CPU onload device."""

import ast
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import diffusion_te_residency as tr  # noqa: E402

MIB = 1 << 20


class _Encoder(torch.nn.Module):
    def __init__(
        self,
        width = 512,
        layers = 4,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(64, width)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(layers))
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
        return self.norm(x)


class _Pipe(SimpleNamespace):
    """Weak-referenceable, as a diffusers pipeline is."""


def _pipe(encoder = None):
    encoder = encoder if encoder is not None else _Encoder()
    pipe = _Pipe(
        text_encoder = encoder,
        transformer = torch.nn.Linear(8, 8),
        _unsloth_measured_reserve = None,
        _unsloth_peak_identity = None,
    )
    pipe.components = {"text_encoder": encoder, "transformer": pipe.transformer}
    return pipe


def _live(free_mib, total_mib = 24_000):
    return patch.object(tr, "live_free_mib", return_value = (free_mib, total_mib))


class _Recorder(tr.EncoderResidency):
    """Moves recorded instead of performed: the decision logic alone."""

    def __init__(
        self,
        mib = 4_600,
        kind = tr.KIND_PLAIN,
    ):
        pipe = _pipe()
        super().__init__(pipe, "cpu", kind)
        self._keep_pipe = pipe
        self.mib = mib
        self.moves = []

    def release(
        self,
        reason,
        short_mib = None,
    ):
        if self.streamed:
            return 0
        self.moves.append(("release", short_mib))
        self.streamed, self.released_mib = True, self.mib
        self._record(action = "release", mib = self.mib, reason = reason)
        return self.mib

    def restore(self, reason):
        self.moves.append(("restore",))
        self.streamed, self.released_mib = False, 0
        self._record(action = "restore", reason = reason)
        return True


# ---- the fit decision ---------------------------------------------------------------------------------------------


def test_slack_is_the_planners_residency_slack():
    from core.inference.diffusion_memory import _resident_dit_slack_mib
    for total in (8_000, 16_000, 22_500, 81_000):
        assert tr.slack_mib(total) == _resident_dit_slack_mib(
            SimpleNamespace(total_mib = total, free_mib = total)
        )


def test_encoders_stay_when_the_render_fits_with_the_slack():
    r = _Recorder()
    with _live(free_mib = 2_400 + tr.slack_mib(24_000)):
        entry = r.after_encode(2_400)
    assert entry["action"] == "keep" and r.moves == [] and not r.streamed


def test_encoders_stream_when_the_render_does_not_fit():
    r = _Recorder()
    want = 2_400 + tr.slack_mib(24_000)
    with _live(free_mib = want - 1):
        entry = r.after_encode(2_400)
    assert r.moves == [("release", 1)]
    assert (
        entry["action"] == "release" and entry["free_mib"] == want - 1 and entry["want_mib"] == want
    )


def test_unknown_peak_or_free_memory_keeps_the_placement():
    r = _Recorder()
    with _live(free_mib = 10):
        assert r.after_encode(None)["action"] == "keep"
    with patch.object(tr, "live_free_mib", return_value = None):
        assert r.after_encode(5_000)["action"] == "keep"
    assert r.moves == []


# ---- the re-check per render --------------------------------------------------------------------------------------


def test_streamed_encoders_return_only_when_they_and_the_render_fit():
    r = _Recorder(mib = 4_600)
    with _live(free_mib = 1_000):
        r.after_encode(6_000)  # a 2048 px render: streamed
    assert r.streamed
    slack = tr.slack_mib(24_000)
    with _live(free_mib = 4_600 + 2_400 + slack - 1):
        assert r.before_encode(2_400)["action"] == "stay_streamed"
    assert r.streamed
    with _live(free_mib = 4_600 + 2_400 + slack):
        r.before_encode(2_400)  # back to 1024 px
    assert not r.streamed and r.moves == [("release", 6_000 + slack - 1_000), ("restore",)]


def test_each_render_is_priced_at_its_own_size_and_batch():
    pipe = _pipe()
    pipe._unsloth_peak_identity = ("flux.3-klein", "label", "default")
    seen = []

    def measured(family, speed, **kw):
        seen.append((family, speed, kw["width"], kw["height"], kw["batch_size"], kw["quant"]))
        return 1_000 * (kw["width"] * kw["height"] * kw["batch_size"]) // (1024 * 1024)

    with patch("core.inference.diffusion_memory.measured_image_runtime_mib", side_effect = measured):
        small = tr.render_need_mib(pipe, width = 1024, height = 1024, batch_size = 1)
        large = tr.render_need_mib(pipe, width = 2048, height = 2048, batch_size = 1)
        batch = tr.render_need_mib(pipe, width = 1024, height = 1024, batch_size = 4)
    assert (small, large, batch) == (1_000, 4_000, 4_000)
    assert seen[0] == ("flux.3-klein", "default", 1024, 1024, 1, "label")


def test_render_need_adds_what_the_request_needs_beyond_the_reserve():
    pipe = _pipe()
    pipe._unsloth_measured_reserve = (2_304, "qwen-image-2.1", "default")
    assert tr.render_need_mib(pipe, width = 1024, height = 1024, extra_mib = 0) == 2_304
    assert tr.render_need_mib(pipe, width = 2048, height = 2048, extra_mib = 3_000) == 5_304
    assert tr.render_need_mib(_pipe(), width = 1024, height = 1024) is None


def test_calibrated_row_scales_with_pixels_and_a_tiled_decode_is_capped():
    act = SimpleNamespace(
        text_encoder_mib = 300, denoise_mib = 1_000, decode_mib = 2_500, tiled_decode_mib = 2_400
    )
    pipe = _pipe()
    assert tr.render_need_mib(pipe, width = 1024, height = 1024, measured_activation = act) == 2_500
    assert tr.render_need_mib(pipe, width = 512, height = 512, measured_activation = act) == 2_500
    assert tr.render_need_mib(pipe, width = 2048, height = 2048, measured_activation = act) == 10_000
    assert (
        tr.render_need_mib(pipe, width = 2048, height = 2048, measured_activation = act, tiled = True)
        == 4_000
    )
    assert (
        tr.render_need_mib(pipe, width = 1024, height = 1024, batch_size = 2, measured_activation = act)
        == 5_000
    )


def test_resident_groups_are_priced_by_the_planners_reserve_only():
    pipe = _pipe()
    pipe._unsloth_measured_reserve = (2_304, "qwen-image-2.1", "default")
    pipe._unsloth_peak_identity = ("qwen-image-2.1", "label", "default")
    setattr(pipe, tr.ATTR, _Recorder(kind = tr.KIND_GROUPS))
    act = SimpleNamespace(text_encoder_mib = 1, denoise_mib = 1, decode_mib = 9_000, tiled_decode_mib = 1)
    with patch("core.inference.diffusion_memory.measured_image_runtime_mib", return_value = 7_000):
        need = tr.render_need_mib(
            pipe, width = 1024, height = 1024, extra_mib = 100, measured_activation = act
        )
    assert need == 2_404


def test_release_slack_per_placement():
    from core.inference.diffusion_memory import _DECODE_SLACK_MIB

    assert _Recorder(kind = tr.KIND_PLAIN).release_slack_mib(24_000) == tr.slack_mib(24_000)
    assert _Recorder(kind = tr.KIND_GROUPS).release_slack_mib(24_000) == _DECODE_SLACK_MIB
    r = _Recorder(kind = tr.KIND_GROUPS)
    with _live(free_mib = 2_304 + _DECODE_SLACK_MIB):
        assert r.after_encode(2_304)["action"] == "keep"
    with _live(free_mib = 2_304 + _DECODE_SLACK_MIB - 1):
        r.after_encode(2_304)
    assert r.moves == [("release", 1)]


def test_lora_forward_is_priced_on_top():
    pipe = _pipe()
    pipe._unsloth_measured_reserve = (2_000, "flux.3-klein", "default")
    pipe._unsloth_loras = ["a"]
    with patch("core.inference.diffusion_lora_memory.request_extra_mib", return_value = 700):
        assert tr.render_need_mib(pipe, width = 1024, height = 1024) == 2_700


# ---- the streaming mechanics (CPU onload device) -------------------------------------------------------------------


def _hooked(module):
    from core.inference.diffusion_vae_oom import _hooked as hooked
    return hooked(module)


def test_release_streams_and_restore_unhooks_with_identical_outputs():
    torch.manual_seed(0)
    pipe = _pipe()
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        ref = pipe.text_encoder(ids).clone()
    r = tr.EncoderResidency(pipe, "cpu", tr.KIND_PLAIN)
    with patch("core.inference.diffusion_vae_oom._host_free_mib", return_value = 1 << 20):
        assert r.release("test") == r.mib
    assert r.streamed and _hooked(pipe.text_encoder) and r.last["action"] == "release"
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(ids), ref)
    assert r.restore("test") and not r.streamed and not _hooked(pipe.text_encoder)
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(ids), ref)
    assert (r.releases, r.restores) == (1, 1)


def test_release_refuses_when_host_ram_is_short():
    pipe = _pipe()
    r = tr.EncoderResidency(pipe, "cpu", tr.KIND_PLAIN)
    with patch(
        "core.inference.diffusion_vae_oom._host_free_mib",
        return_value = r.mib + tr.HOST_MARGIN_MIB - 1,
    ):
        assert r.release("test") == 0
    assert not r.streamed and not _hooked(pipe.text_encoder) and "host RAM" in r.last["why"]


def test_a_failed_stream_puts_the_encoders_back_whole():
    pipe = _pipe()
    r = tr.EncoderResidency(pipe, "cpu", tr.KIND_PLAIN)
    with (
        patch("core.inference.diffusion_vae_oom._host_free_mib", return_value = 1 << 20),
        patch.object(tr, "_apply_stream", side_effect = RuntimeError("boom")),
    ):
        assert r.release("test") == 0
    assert not r.streamed and not _hooked(pipe.text_encoder) and r.last["action"] == "keep"


def test_a_restore_that_runs_out_of_memory_keeps_streaming():
    pipe = _pipe()
    r = tr.EncoderResidency(pipe, "cpu", tr.KIND_PLAIN)
    with patch("core.inference.diffusion_vae_oom._host_free_mib", return_value = 1 << 20):
        r.release("test")
    with patch(
        "core.inference.diffusion_vae_oom._return_to", side_effect = torch.OutOfMemoryError("oom")
    ):
        assert r.restore("test") is False
    assert r.streamed and _hooked(pipe.text_encoder) and r.last["action"] == "stay_streamed"
    ids = torch.randint(0, 64, (1, 3))
    with torch.no_grad():
        pipe.text_encoder(ids)  # still runs, streamed


def test_resident_groups_stream_the_encoders_only():
    pipe = _pipe()
    calls = []

    def fake_release(
        p,
        need,
        logger = None,
        **kw,
    ):
        calls.append((need, kw))

        def restore(strict = False):
            calls.append(("restore", strict))

        return restore

    sizes = iter([4_000, 1_000])
    with (
        patch("core.inference.diffusion_memory.release_resident_groups", side_effect = fake_release),
        patch(
            "core.inference.diffusion_memory.resident_group_mib", side_effect = lambda p: next(sizes)
        ),
        patch.object(tr.EncoderResidency, "resident_mib", return_value = 4_000),
    ):
        r = tr.EncoderResidency(pipe, "cpu", tr.KIND_GROUPS)
        assert r.release("short", short_mib = 2_500) == 3_000
    assert calls[0][0] == 2_500 and calls[0][1]["encoders_only"] is True
    assert r.restore("fits") and calls[-1] == ("restore", True) and not r.streamed


def test_a_partial_group_release_does_not_block_a_later_one_or_a_lend():
    # Qwen-Image-2.1 on an L4: a near-steady render streamed 81 MiB, which then blocked the 8 GiB lend
    pipe = _pipe()
    order = []
    resident = {"mib": 5_265}

    def fake_release(
        p,
        need,
        logger = None,
        **kw,
    ):
        freed = min(need, resident["mib"])
        resident["mib"] -= freed
        tag = len(order)
        order.append(("release", need))

        def restore(strict = False):
            order.append(("restore", tag))

        return restore

    with (
        patch("core.inference.diffusion_memory.release_resident_groups", side_effect = fake_release),
        patch(
            "core.inference.diffusion_memory.resident_group_mib",
            side_effect = lambda p: resident["mib"],
        ),
        patch.object(tr.EncoderResidency, "resident_mib", lambda self: resident["mib"]),
    ):
        r = tr.EncoderResidency(pipe, "cpu", tr.KIND_GROUPS)
        with _live(free_mib = 7_106):
            r.after_encode(6_912)  # short by 62 MiB
        assert r.streamed and r.released_mib == 62
        assert r.lend("an 8 GiB load") == 5_203
        assert resident["mib"] == 0 and r.released_mib == 5_265
        assert r.lend("again") == 0
        assert r.restore("fits")
    assert order == [("release", 62), ("release", 5_203), ("restore", 1), ("restore", 0)]
    assert not r.streamed and r.released_mib == 0


def test_resident_groups_with_a_decode_lend_are_priced_by_the_denoise_phase():
    from core.inference.diffusion_memory import (
        DECODE_LEND_ATTR,
        MEASURED_RESERVE_DENOISE_PHASE_INDEX,
    )

    pipe = _pipe()
    reserve = [6_912, "qwen-image-2.1", "default"] + [None] * 11
    reserve[MEASURED_RESERVE_DENOISE_PHASE_INDEX] = 5_376
    pipe._unsloth_measured_reserve = tuple(reserve)
    setattr(pipe, tr.ATTR, _Recorder(kind = tr.KIND_GROUPS))
    assert tr.render_need_mib(pipe, width = 1024, height = 1024, extra_mib = 10) == 6_922
    setattr(pipe, DECODE_LEND_ATTR, (1_536, 7_000, 1 << 20, False))
    assert tr.render_need_mib(pipe, width = 1024, height = 1024, extra_mib = 10) == 5_386
    # no phase in the reserve (the L4 placement): the whole-render reserve less what it lent to the decode
    reserve[MEASURED_RESERVE_DENOISE_PHASE_INDEX] = None
    pipe._unsloth_measured_reserve = tuple(reserve)
    assert tr.render_need_mib(pipe, width = 1024, height = 1024, extra_mib = 10) == 6_912 - 1_536 + 10


def test_streamed_groups_come_back_at_the_placements_own_fit():
    # Qwen-Image-2.1 on an L4: 12384 MiB free, 5265 MiB released, 5376 MiB denoise; the planner slack never fit
    from core.inference.diffusion_memory import _DECODE_SLACK_MIB

    r = _Recorder(kind = tr.KIND_GROUPS)
    r.streamed, r.released_mib = True, 5_265
    want = 5_265 + 5_376 + 2 * _DECODE_SLACK_MIB
    with _live(free_mib = want - 1):
        assert r.before_encode(5_376)["action"] == "stay_streamed"
    with _live(free_mib = 12_384):
        r.before_encode(5_376)
    assert r.moves == [("restore",)]
    plain = _Recorder(kind = tr.KIND_PLAIN)
    assert plain.restore_slack_mib(24_000) == tr.slack_mib(24_000)


def test_release_resident_groups_can_leave_the_denoiser_alone():
    import inspect

    from core.inference.diffusion_memory import release_resident_groups

    assert "encoders_only" in inspect.signature(release_resident_groups).parameters
    te = SimpleNamespace(_unsloth_resident_room = 100)
    dit = SimpleNamespace(_unsloth_resident_room = 100)
    pipe = _Pipe(components = {"text_encoder": te, "transformer": dit})
    seen = []
    with patch(
        "core.inference.diffusion_memory._offload_groups",
        side_effect = lambda m: seen.append(m) or [],
    ):
        release_resident_groups(pipe, 10, encoders_only = True)
    assert seen and all(m is not dit for m in seen)


# ---- install and kill switch --------------------------------------------------------------------------------------


def _plan(kind = "discrete", device = "cuda"):
    return SimpleNamespace(
        device_memory = SimpleNamespace(is_unified = kind == "unified", device = device)
    )


def test_kill_switch_and_unsupported_targets_install_nothing(monkeypatch):
    pipe = _pipe()
    with (
        patch.object(tr, "_encoder_device", return_value = torch.device("cuda", 0)),
        patch.object(tr, "_placement_kind", return_value = tr.KIND_PLAIN),
    ):
        monkeypatch.setenv(tr.RESIDENCY_ENV, "0")
        assert tr.maybe_install(pipe, _plan()) is None
        monkeypatch.delenv(tr.RESIDENCY_ENV)
        assert tr.maybe_install(pipe, _plan(kind = "unified")) is None
        assert tr.maybe_install(pipe, _plan(device = "mps")) is None
        with patch("core.inference.diffusion_multigpu.current_selection", return_value = object()):
            assert tr.maybe_install(pipe, _plan()) is None
        assert not hasattr(pipe, tr.ATTR)
        installed = tr.maybe_install(pipe, _plan())
    assert installed is not None and tr.of(pipe) is installed and installed.kind == tr.KIND_PLAIN


def test_placement_kind():
    pipe = _pipe()
    assert tr._placement_kind(pipe, torch.device("cpu")) == tr.KIND_PLAIN
    pipe.text_encoder._unsloth_resident_room = 100
    assert tr._placement_kind(pipe, torch.device("cpu")) == tr.KIND_GROUPS
    assert tr._placement_kind(SimpleNamespace(components = {}), torch.device("cpu")) is None


# ---- the lend path ------------------------------------------------------------------------------------------------


def _engine(residency = None):
    from core.inference.diffusion import DiffusionBackend

    pipe = _pipe()
    if residency is not None:
        setattr(pipe, tr.ATTR, residency)
    engine = SimpleNamespace(_generate_lock = threading.Lock(), _state = SimpleNamespace(pipe = pipe))
    return engine, lambda *a: DiffusionBackend.lend_text_encoder_vram(engine, *a)


def test_an_idle_engine_lends_its_encoders():
    r = _Recorder(mib = 4_600)
    engine, lend = _engine(r)
    assert lend("dictation model load") == 4_600 and r.streamed
    assert lend("again") == 0  # already streamed


def test_a_busy_engine_lends_nothing():
    r = _Recorder()
    engine, lend = _engine(r)
    engine._generate_lock.acquire()
    try:
        assert lend("dictation model load") == 0
    finally:
        engine._generate_lock.release()
    assert not r.streamed


def test_a_lend_smaller_than_the_shortfall_is_not_made():
    r = _Recorder(mib = 4_600)
    engine, lend = _engine(r)
    assert lend("chat", 5_000) == 0 and not r.streamed
    assert lend("chat", 4_000) == 4_600 and r.streamed


def test_lend_from_loaded_engine_never_imports_the_engine(monkeypatch):
    monkeypatch.delitem(sys.modules, "core.inference.diffusion", raising = False)
    assert tr.lend_from_loaded_engine("x") == 0
    assert "core.inference.diffusion" not in sys.modules


def test_dictation_load_asks_the_image_engine_first(tmp_path):
    from core.inference import stt_sidecar

    (tmp_path / "model.safetensors").write_bytes(b"\0" * (3 * MIB))
    (tmp_path / "config.json").write_text("{}")
    with patch.object(tr, "lend_from_loaded_engine", return_value = 4_600) as lend:
        assert (
            stt_sidecar._lend_image_text_encoder_vram("cuda", "dictation model load", str(tmp_path))
            == 4_600
        )
        assert stt_sidecar._lend_image_text_encoder_vram("cpu", "dictation model load") == 0
    # priced by its weight files plus the worker's context, so a card with room keeps the encoders
    lend.assert_called_once_with(
        "dictation model load", need_mib = 3 + stt_sidecar._DICTATION_OVERHEAD_MIB
    )
    assert stt_sidecar._dictation_need_mib(str(tmp_path / "missing")) is None
    source = Path(stt_sidecar.__file__).read_text()
    i = source.index("_lend_image_text_encoder_vram(\n")
    assert source.index('device, "dictation model load", str(snapshot_path)') > i
    assert i < source.index('with _lend_diffusion_training_vram(device, "dictation model load")')


def test_a_lend_is_not_made_when_the_other_load_already_fits():
    r = _Recorder(mib = 4_600)
    engine, lend = _engine(r)
    with _live(free_mib = 1_600 + tr.slack_mib(24_000)):
        assert lend("dictation model load", 0, 1_600) == 0 and not r.streamed
    assert r.last["action"] == "keep"
    with _live(free_mib = 1_600 + tr.slack_mib(24_000) - 1):
        assert lend("dictation model load", 0, 1_600) == 4_600 and r.streamed


def test_lend_from_loaded_engine_passes_the_load_size(monkeypatch):
    calls = []

    class _Engine:
        def lend_text_encoder_vram(
            self,
            reason,
            at_least_mib = 0,
            need_mib = None,
        ):
            calls.append((reason, at_least_mib, need_mib))
            return 7

    monkeypatch.setitem(
        sys.modules, "core.inference.diffusion", SimpleNamespace(_diffusion_backend = _Engine())
    )
    assert tr.lend_from_loaded_engine("dictation", need_mib = 1_000) == 7
    assert tr.lend_from_loaded_engine("chat", 3_000) == 7
    assert calls == [("dictation", 0, 1_000), ("chat", 3_000, None)]


def test_a_group_lend_hands_the_freed_memory_back_to_the_driver():
    """The other load is usually another process (dictation worker, llama.cpp): blocks left in this process's
    allocator cache are not free to it."""
    r = _Recorder(mib = 4_600, kind = tr.KIND_GROUPS)
    with (
        patch.object(tr.EncoderResidency, "resident_mib", return_value = 4_600),
        patch("core.inference.diffusion_vae_oom._empty_cache") as empty,
    ):
        assert r.lend("a chat model") == 4_600
    empty.assert_called_once()


def test_lend_and_unload_both_invalidate_cached_gpu_memory():
    from core.inference.diffusion import DiffusionBackend
    from utils import gpu_memory_events

    wrapper = gpu_memory_events.invalidates_gpu_memory("x")(lambda: None).__code__
    assert DiffusionBackend.unload.__code__ is wrapper
    assert DiffusionBackend.lend_text_encoder_vram.__code__ is wrapper
    assert DiffusionBackend.unload.__wrapped__.__name__ == "unload"


class TestFreeStepsLend:
    def _run(
        self,
        *,
        need_mib = None,
        short_mib = 0,
        lent = 4_600,
        fits = (True,),
        busy = False,
    ):
        from core.inference import gpu_arbiter as arb
        from core.inference import gpu_residency as gr

        saved = {n: getattr(arb, n) for n in ("_owner", "_residents", "_media_needs", "_accounts")}
        arb._owner, arb._residents, arb._media_needs, arb._accounts = (
            arb.DIFFUSION,
            {arb.DIFFUSION: frozenset({0})},
            {},
            {},
        )
        log = []
        fits_iter = iter(fits)
        try:
            with (
                patch.dict(arb._EVICTORS, {arb.DIFFUSION: lambda: log.append("evict image")}),
                patch.object(arb, "resident_slots", return_value = [{"kind": "image", "model": "m"}]),
                patch.object(gr, "_loaded_or_loading", return_value = True),
                patch.object(gr, "_media_busy", return_value = (False, busy)),
                patch.object(gr, "foreign_on", return_value = 0),
                patch.object(
                    gr, "can_load_alongside", side_effect = lambda *a: (next(fits_iter, False), {})
                ),
                patch("core.inference.video_minimax_h3.release_h3_native_servers", return_value = 0),
                patch.object(
                    tr,
                    "lend_from_loaded_engine",
                    side_effect = lambda r, at_least_mib = 0: log.append(("lend", at_least_mib))
                    or lent,
                ),
                patch("core.inference.api_monitor.api_monitor"),
            ):
                freed = gr._free_steps(
                    arb.CHAT,
                    [0],
                    "a chat model",
                    [],
                    need_mib = need_mib,
                    short_mib = short_mib,
                    acting = None,
                )
        finally:
            for n, v in saved.items():
                setattr(arb, n, v)
        return freed, log

    def test_priced_load_takes_the_encoders_before_the_image_model(self):
        freed, log = self._run(need_mib = {0: 3_000})
        assert freed == ["image-text-encoders"] and log == [("lend", 0)]

    def test_image_model_still_goes_when_the_encoders_are_not_enough(self):
        freed, log = self._run(need_mib = {0: 9_000}, fits = (False, True))
        assert log == [("lend", 0), "evict image"]

    def test_short_mib_load_lends_only_when_it_covers_the_shortfall(self):
        freed, log = self._run(short_mib = 3_000)
        assert log == [("lend", 3_000)] and freed == ["image-text-encoders"]
        freed, log = self._run(short_mib = 9_000, lent = 0)
        assert log == [("lend", 9_000), "evict image"]

    def test_unsized_load_and_busy_image_model_skip_the_lend(self):
        _, log = self._run()
        assert log == ["evict image"]
        _, log = self._run(need_mib = {0: 3_000}, busy = True)
        assert ("lend", 0) not in log


# ---- wiring in the image engine -----------------------------------------------------------------------------------


def _diffusion_source():
    from core.inference import diffusion
    return Path(diffusion.__file__).read_text()


def test_generate_rechecks_before_and_after_every_encode():
    source = _diffusion_source()
    guard = source.index("te_residency.before_encode(state.pipe, te_residency_need")
    denoise_hook = source.index("def _enter_denoise_phase(")
    after = source.index("te_residency.after_encode(state.pipe, te_residency_need")
    assert guard < denoise_hook < after
    assert source.index("te_residency_need = te_residency.render_need_mib(") < guard
    # inside the per-chunk denoise hook, so every chunk's encode is followed by a check
    block = source[denoise_hook:after]
    assert 'gen.phase = "denoise"' in block


def test_load_installs_the_check_when_no_unified_releaser_runs():
    source = _diffusion_source()
    i = source.index("te_releaser = te_release.maybe_install(pipe, plan, logger = logger)")
    assert "te_residency.maybe_install(pipe, plan, logger = logger)" in source[i : i + 400]
    ast.parse(source)


def test_live_free_memory_uses_the_planners_trusted_reading():
    """Windows ROCm over-reports driver free (a fresh process sees the card near empty): the residency check reads
    the same capped figure the load planner does, so a streamed encoder is not brought back over a full card."""
    from core.inference import diffusion_memory as dm
    from utils.hardware import hardware

    total = 24_000 * MIB
    with (
        patch.object(torch.cuda, "mem_get_info", return_value = (23_900 * MIB, total)),
        patch.object(torch.cuda, "memory_reserved", return_value = 20_000 * MIB),
        patch.object(torch.cuda, "memory_allocated", return_value = 19_000 * MIB),
        patch.object(dm, "_graph_held_cache_bytes", return_value = 0),
        patch.object(dm, "user_budget_reserve_bytes", return_value = 0),
    ):
        with patch.object(hardware, "rocm_windows_free_is_untrusted", return_value = False):
            assert tr.live_free_mib("cuda:0") == (23_900 + 1_000, 24_000)
        with patch.object(hardware, "rocm_windows_free_is_untrusted", return_value = True):
            # capped at what this process has not reserved, plus its reusable cache
            assert tr.live_free_mib("cuda:0") == (4_000 + 1_000, 24_000)


def test_a_resident_load_whose_quantized_encoder_can_stream_renders_under_no_grad():
    """torchao encoder weights cannot move under inference_mode; a resident load that may stream or restore them per
    render takes the no_grad render mode, as offloaded quant loads already do. Plain encoders keep inference_mode."""
    from core.inference import diffusion as d

    quant = pytest.importorskip("torchao.quantization")
    plain = _pipe()
    state = SimpleNamespace(offload_policy = d.OFFLOAD_NONE, pipe = plain, transformer_quant = "int8")
    assert d._torchao_render_needs_no_grad(state) is False  # no residency check installed
    setattr(plain, tr.ATTR, tr.EncoderResidency(plain, "cpu", tr.KIND_PLAIN))
    assert d._torchao_render_needs_no_grad(state) is False  # plain encoder weights

    encoder = torch.nn.Sequential(torch.nn.Linear(32, 32))
    quant.quantize_(encoder, quant.Int8WeightOnlyConfig())
    quantized = _pipe(encoder)
    setattr(quantized, tr.ATTR, tr.EncoderResidency(quantized, "cpu", tr.KIND_PLAIN))
    state = SimpleNamespace(offload_policy = d.OFFLOAD_NONE, pipe = quantized)
    assert d._torchao_render_needs_no_grad(state) is True
