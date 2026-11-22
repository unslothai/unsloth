# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 klein text-encoder placement at 8 / 12 / 16 / 24 GiB, from the sizes its INT8 auto load logs; CPU only.

Also the learned-peak scale a resident encoder records: its streaming window goes on the prompt-encode phase only."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

# Loaded storage of flux.3-klein auto (int8 ConvRot DiT, int8 text encoder 32/36 layers, VAE), MiB.
KLEIN_LOADED = {
    "transformer": (6144, "dit"),
    "text_encoder": (4619, "text_encoder"),
    "vae": (161, "other"),
}
KLEIN_FLAT = dict(model_dense_mib = 11105, companion_dense_mib = 5283, text_encoder_dense_mib = 5092)
_REAL_FREE_GATE = dm._free_reading_untrusted
# (free MiB a ballast leaves, simulated total MiB) per budget
CARDS = {8: (7788, 8188), 12: (11888, 12288), 16: (15976, 16376), 24: (24176, 24576)}


@pytest.fixture
def klein(monkeypatch):
    for env in (
        getattr(dm, "MEASURED_ACTIVATION_ENV", "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION"),
        dm.PARTIAL_RESIDENT_ENV,
        dm.RESIDENT_DIT_ENV,
        dm.TE_RESIDENT_ROOM_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(KLEIN_LOADED), raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    # a measured free reading (not Windows ROCm, whose free half is a ceiling)
    monkeypatch.setattr(dm, "_free_reading_untrusted", lambda: False)
    return types.SimpleNamespace()


def _plan(gib: int):
    free, total = CARDS[gib]
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        runtime_headroom_mib = 8192,
        calibrated_activation = dm.calibrated_image_activation("flux.3-klein"),
        **KLEIN_FLAT,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return dm.refine_plan_from_loaded_weights(
        klein_pipe(), plan, family = "flux.3-klein", speed_mode = "default"
    )


def klein_pipe():
    # the load marks the family's phase table as its seed (2437 MiB worst phase x 1.15 -> 2816)
    return types.SimpleNamespace(_unsloth_phase_seeds = True)


def _summary(plan) -> tuple:
    return (
        plan.offload_policy,
        bool(plan.stream_transformer),
        bool(plan.stream_text_encoders),
        plan.resident_transformer_mib,
        plan.resident_text_encoder_mib,
    )


def test_12gb_streaming_tier_keeps_encoder_groups_beside_the_whole_transformer(klein, monkeypatch):
    plan = _plan(12)
    headroom = plan.estimates["measured_runtime_headroom_mib"]
    assert plan.offload_policy == dm.OFFLOAD_STREAMING and plan.resident_transformer_mib == 6144
    slack = plan.estimates["resident_te_slack_mib"]
    assert slack == 1228
    assert plan.resident_text_encoder_mib == 11888 - slack - 161 - headroom - 6144 > 0
    assert plan.estimates["flat_resident_text_encoder_mib"] == 0
    # weights kept plus the measured peak plus the slack never pass free memory
    assert 6144 + 161 + plan.resident_text_encoder_mib + headroom + slack <= 11888
    monkeypatch.setenv(dm.TE_RESIDENT_ROOM_ENV, "0")
    off = _plan(12)
    assert off.resident_text_encoder_mib is None and off.resident_transformer_mib == 6144


def test_16gb_group_tier_keeps_the_whole_encoder(klein, monkeypatch):
    plan = _plan(16)
    assert plan.offload_policy == dm.OFFLOAD_GROUP and not plan.stream_transformer
    assert plan.stream_text_encoders
    headroom = plan.estimates["measured_runtime_headroom_mib"]
    flat = plan.estimates["flat_resident_text_encoder_mib"]
    budget = plan.estimates["safe_device_budget_mib"]
    assert (
        flat == budget - headroom - dm.DEFAULT_BASE_OVERHEAD_MIB - 161 - 6144 < 4619
    )  # the old room
    assert plan.resident_text_encoder_mib == 4619 > flat
    assert 6144 + 161 + 4619 + headroom + plan.estimates["resident_te_slack_mib"] <= 15976
    monkeypatch.setenv(dm.TE_RESIDENT_ROOM_ENV, "0")
    assert _plan(16).resident_text_encoder_mib == flat


@pytest.mark.parametrize("gib", [8, 24])
def test_8gb_and_24gb_placements_do_not_move(klein, monkeypatch, gib):
    on = _summary(_plan(gib))
    monkeypatch.setenv(dm.TE_RESIDENT_ROOM_ENV, "0")
    assert _summary(_plan(gib)) == on
    if gib == 8:
        # the whole transformer does not fit the slack: partial DiT, every encoder byte streams as before
        assert on[0] == dm.OFFLOAD_STREAMING and (on[3] or 0) < 6144 and on[4] is None


@pytest.mark.parametrize("gib", [12, 16])
def test_windows_rocm_keeps_the_flat_encoder_room(klein, monkeypatch, gib):
    """Windows ROCm over-reports free memory (a ceiling blind to other processes): the encoders keep the flat room,
    exactly the kill-switch placement, and the transformer placement does not move."""
    import utils.hardware as hardware

    wide = _plan(gib)
    monkeypatch.setenv(dm.TE_RESIDENT_ROOM_ENV, "0")
    flat = _plan(gib)
    monkeypatch.delenv(dm.TE_RESIDENT_ROOM_ENV)
    # the real gate, reached through utils.hardware
    monkeypatch.setattr(dm, "_free_reading_untrusted", _REAL_FREE_GATE)
    monkeypatch.setattr(hardware, "rocm_windows_free_is_untrusted", lambda: True)
    untrusted = _plan(gib)
    assert _summary(untrusted) == _summary(flat) != _summary(wide)
    assert untrusted.resident_transformer_mib == wide.resident_transformer_mib
    assert "resident_te_slack_mib" not in untrusted.estimates


def test_the_rocm_gate_reads_the_hardware_helper(monkeypatch):
    import utils.hardware.hardware as hw

    monkeypatch.setattr(dm, "_free_reading_untrusted", _REAL_FREE_GATE)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert dm._free_reading_untrusted() is True
    monkeypatch.setattr(
        hw.sys, "platform", "linux"
    )  # ROCm on Linux reads free memory as a measurement
    assert dm._free_reading_untrusted() is False


def test_an_unimportable_hardware_helper_keeps_the_reading(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "utils.hardware":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert _REAL_FREE_GATE() is False


def test_an_edit_priced_cut_or_a_decode_lend_keeps_the_flat_room():
    """Source guard: the wider room never replaces an edit's encoder cut or a decode lend's phase pricing."""
    import inspect

    src = inspect.getsource(dm._refine_plan_from_loaded_weights)
    block = src[src.index("whole_beside = (") : src.index("if dit_room <= 0 and te_room <= 0")]
    for guard in (
        "not reference_cut",
        "unlend = lend > 0 and wide_te > te_room",
        "dense_mib is None",
    ):
        assert guard in block


# ------------------------------------------------------------------------------------------ learned-peak scale


def _encoder(torch, groups_resident):
    """A Qwen3-like encoder: one large bf16 embedding leaf and two small linears, with leaf offload groups."""
    enc = torch.nn.Module()
    enc.embed_tokens = torch.nn.Embedding(4096, 256)  # 4 MiB fp32
    enc.fc1 = torch.nn.Linear(256, 256, bias = False)  # 0.25 MiB
    enc.fc2 = torch.nn.Linear(256, 256, bias = False)
    groups = []
    for leaf, resident in zip((enc.embed_tokens, enc.fc1, enc.fc2), groups_resident):
        nbytes = sum(p.numel() * p.element_size() for p in leaf.parameters())
        groups.append(
            types.SimpleNamespace(_unsloth_resident = resident, modules = [leaf], _nbytes = nbytes)
        )
    return enc, groups


def test_encoder_window_is_the_largest_leaves_not_the_largest_three_times(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_offload_prefetch as prefetch

    monkeypatch.setattr(prefetch, "prefetch_depth", lambda: 2)
    monkeypatch.delenv(dm.ENCODER_PAD_PHASE_ENV, raising = False)
    enc, groups = _encoder(torch, (True, True, True))
    dit = torch.nn.Linear(4, 4)
    dit_group = types.SimpleNamespace(_unsloth_resident = False)
    table = {id(enc): groups, id(dit): [dit_group]}
    monkeypatch.setattr(dm, "_offload_groups", lambda module: table.get(id(module), []))
    monkeypatch.setattr(dm, "_group_nbytes", lambda g: getattr(g, "_nbytes", 0))
    pipe = types.SimpleNamespace(components = {"transformer": dit, "text_encoder": enc})
    # the whole-render pad is unchanged (3 x the 4 MiB embedding); the phase window is 4 + 0.25 + 0.25 MiB
    assert dm._encoder_stream_window_mib(pipe) == 12
    assert dm.encoder_window_pad_mib(pipe) == (12, 5)
    groups[0]._unsloth_resident = False  # the embedding streams: its window is in the peak already
    assert dm._encoder_stream_window_mib(pipe) == 0
    assert dm.encoder_window_pad_mib(pipe) is None
    groups[0]._unsloth_resident = True
    monkeypatch.setenv(dm.ENCODER_PAD_PHASE_ENV, "0")
    assert dm.encoder_window_pad_mib(pipe) is None


def test_the_encoder_window_pads_the_encode_phase_only():
    from core.inference.diffusion import _padded_peak

    # Qwen-Image-2.1 at 16 GiB with a resident 1.2 GB embedding: raw peak 1771, denoiser window 619, legacy encoder
    # share 3416 (3 x 1187 less the streamed leaves'), encode phase 300 MiB, phase window 1246
    probe = types.SimpleNamespace(encoder_pad = (3416, 1246), encode_peak = 300)
    pad = 619 + 3416
    assert _padded_peak(probe, 1771, pad) == max(1771, 300 + 1246) + 619 == 2390
    # an encode-dominated render keeps the window on top of the encode
    probe.encode_peak = 1700
    assert _padded_peak(probe, 1771, pad) == 1700 + 1246 + 619
    # no encode reading, no share: the whole pad, as before
    assert (
        _padded_peak(types.SimpleNamespace(encoder_pad = (3416, 1246), encode_peak = None), 1771, pad)
        == 1771 + pad
    )
    assert _padded_peak(types.SimpleNamespace(), 1771, pad) == 1771 + pad
    # a share larger than the pad (a patched pad) never takes the pad below zero
    assert (
        _padded_peak(types.SimpleNamespace(encoder_pad = (9999, 0), encode_peak = 10), 1000, 500)
        == 1000
    )


def test_probe_reads_the_encode_phase_once():
    from core.inference.diffusion_measured_peaks import PeakProbe

    probe = PeakProbe("cpu")  # not CUDA: nothing armed
    probe.mark_encode()
    assert getattr(probe, "encode_peak", None) is None
    armed = PeakProbe.__new__(PeakProbe)
    armed.device, armed.base, armed.encodes, armed.encode_peak = None, None, 0, None
    armed.mark_encode()
    assert armed.encode_peak is None


def test_the_render_marks_the_encode_before_any_encoder_release():
    import inspect

    import core.inference.diffusion as diffusion

    src = inspect.getsource(diffusion.DiffusionBackend.generate)
    start = src.index("def _enter_denoise_phase(")
    mark = src.index("peak_probe.mark_encode()", start)
    assert mark < src.index("state.te_releaser.release()", start)
    assert mark < src.index("te_residency.after_encode(state.pipe", start)


# ------------------------------------------------------------------------------- encoder release during the render


def _record_with(tmp_path, monkeypatch, **probe_attrs):
    """One completed render through _record_peak_probe on an empty store; returns the stored whole-render peak."""
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg
    import core.inference.diffusion_measured_peaks as mp

    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    for env in (mp.LEARNED_PEAKS_ENV, mp.PHASE_PEAKS_ENV, dm.ENCODER_PAD_PHASE_ENV):
        monkeypatch.delenv(env, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA L4")
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(diffusion, "_attention_scaling_known", lambda target, engaged = None: True)
    key = dict(family = "qwen-image-2.1", quant = "torchao:Int8Tensor", speed_mode = "default")
    pipe = types.SimpleNamespace(_unsloth_peak_identity = (key["family"], key["quant"], "default"))
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default")
    probe = types.SimpleNamespace(
        device = "cuda:0",
        failed_captures = 0,
        conditioning_hits = 0,
        # Qwen-Image-2.1 at 16 GiB: denoiser window 619, legacy encoder share 3416, encode 300, phase window 1246
        window_pad = 619 + 3416,
        encoder_pad = (3416, 1246),
        encode_peak = 300,
        phase_peaks = {},
        finish = lambda: 1771,
        **probe_attrs,
    )
    diffusion._record_peak_probe(
        state, probe, width = 1024, height = 1024, batch_size = 1, workflow = "txt2img"
    )
    return mp.lookup(**key, width = 1024, height = 1024)


def test_a_render_whose_resident_encoders_were_released_records_nothing(tmp_path, monkeypatch):
    """After the encode reading, te_residency streamed resident encoder groups: the denoise and decode ran net of
    their bytes, so with the encoders' pad off those phases the sample would under-size later loads (Codex P2)."""
    # the control records the encode-phase scale (max(1771, 300 + 1246) + 619 = 2390)
    assert _record_with(tmp_path / "kept", monkeypatch) == 2390
    assert _record_with(tmp_path / "released", monkeypatch, encoder_released = True) is None


def test_note_encoder_release_marks_the_probe():
    from core.inference.diffusion import _note_encoder_release

    probe = types.SimpleNamespace(encode_peak = 300)
    _note_encoder_release(probe, False, False, False)  # the residency check kept them
    assert not getattr(probe, "encoder_released", False) and probe.encode_peak == 300
    _note_encoder_release(probe, True, False, False)
    assert probe.encoder_released is True
    # a whole-encoder release (unified memory) keeps the whole-render pad: no encode phase is reported
    whole = types.SimpleNamespace(encode_peak = 300)
    _note_encoder_release(whole, False, False, True)
    assert whole.encode_peak is None and not getattr(whole, "encoder_released", False)
    already = types.SimpleNamespace(
        encode_peak = 300
    )  # released before the probe started: baseline excludes it
    _note_encoder_release(already, False, True, True)
    assert already.encode_peak == 300
    _note_encoder_release(None, True, False, True)  # no probe: no-op


def test_the_release_count_reads_the_residency_check(monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_te_residency as te_residency

    pipe = types.SimpleNamespace()
    assert diffusion._encoder_releases(pipe) == 0  # no check installed
    residency = types.SimpleNamespace(releases = 0)
    monkeypatch.setattr(te_residency, "of", lambda p: residency if p is pipe else None)
    before = diffusion._encoder_releases(pipe)
    residency.releases += 1  # what EncoderResidency.release does when it streams groups
    assert diffusion._encoder_releases(pipe) > before


def test_the_render_notes_an_encoder_release_after_the_residency_check():
    import inspect

    import core.inference.diffusion as diffusion

    src = inspect.getsource(diffusion.DiffusionBackend.generate)
    start = src.index("def _enter_denoise_phase(")
    count = src.index("releases = _encoder_releases(state.pipe)", start)
    check = src.index("te_residency.after_encode(state.pipe", start)
    note = src.index("_note_encoder_release(", start)
    assert src.index("state.te_releaser.release()", start) < count < check < note
