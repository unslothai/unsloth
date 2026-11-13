# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Released bf16 MiniMax-H3 denoiser placement beside a streamed conditioner."""

import inspect

import pytest


def _family():
    from core.inference.video_families import detect_video_family
    return detect_video_family("MiniMaxAI/MiniMax-H3")


class _Denoiser:
    def parameters(self):
        import torch
        return iter([torch.empty(8, dtype = torch.bfloat16)])

    def buffers(self):
        return iter(())


def _sizes(
    monkeypatch,
    *,
    te_scheme,
    te_streamed,
    denoiser_gb = 66.3,
):
    import torch

    from core.inference import diffusion_prequant
    from core.inference import video as vid

    monkeypatch.setattr(
        diffusion_prequant, "tensor_payload_bytes", lambda t: int(denoiser_gb * 1e9)
    )
    return vid._h3_dense_denoiser_resident_bytes(
        _family(),
        denoiser = _Denoiser(),
        te_scheme = te_scheme,
        dtype = torch.bfloat16,
        rotating = True,
        te_streamed = te_streamed,
    )


def _plan(
    monkeypatch,
    free_gb,
    te_scheme = None,
):
    from core.inference import video as vid

    rotating = _sizes(monkeypatch, te_scheme = te_scheme, te_streamed = False)
    streamed = _sizes(monkeypatch, te_scheme = te_scheme, te_streamed = True)
    return vid._h3_dense_pin_plan(rotating, streamed, int(free_gb * 1e9))


def test_streamed_conditioner_is_sized_at_its_encode_footprint(monkeypatch):
    from core.inference.video_minimax_h3_te import H3_TE_STREAMED_GB

    fam = _family()
    rotating = _sizes(monkeypatch, te_scheme = None, te_streamed = False)
    streamed = _sizes(monkeypatch, te_scheme = None, te_streamed = True)
    assert rotating[0] == streamed[0]
    vae_gb = fam.bf16_components_gb[2]
    expected = int(fam.bf16_components_gb[1] * 1e9) - int(max(H3_TE_STREAMED_GB, vae_gb) * 1e9)
    assert abs((rotating[1] - streamed[1]) - expected) < 1000
    import torch

    from core.inference import video as vid

    default = vid._h3_dense_denoiser_resident_bytes(
        fam, denoiser = _Denoiser(), te_scheme = None, dtype = torch.bfloat16, rotating = True
    )
    assert default == rotating


@pytest.mark.parametrize(
    "free_gb, te_scheme, expected",
    [
        (101.4, None, "pin_stream_te"),
        (101.4, "int8", "pin_stream_te"),
        (178.0, None, "pin"),
        (178.0, "int8", "pin"),
        (79.0, None, None),
        (47.0, None, None),
        (12.0, "int8", None),
    ],
)
def test_bf16_denoiser_placement_by_card(monkeypatch, free_gb, te_scheme, expected):
    assert _plan(monkeypatch, free_gb, te_scheme) == expected


def test_unreadable_or_unsized_keeps_the_rotation():
    from core.inference import video as vid

    assert vid._h3_dense_pin_plan(None, None, 100 * 10**9) is None
    assert vid._h3_dense_pin_plan((1, 1), (1, 1), None) is None
    assert vid._h3_dense_pin_plan((60 * 10**9, 60 * 10**9), None, 100 * 10**9) is None


def test_load_moves_the_denoiser_before_pinning_the_conditioner_and_falls_back_when_refused():
    from core.inference import video as vid

    src = inspect.getsource(vid.VideoBackend)
    start = src.index('if pin_plan == "pin_stream_te":')
    end = src.index('elif pin_plan == "pin":', start)
    block = src[start:end]
    pin_at = block.index("pin_prequantized_module(")
    stream_at = block.index("stream_h3_text_encoder(\n")
    assert pin_at < stream_at
    refused = block[stream_at:]
    assert "_h3_dense_unpin(manager, denoiser, device, logger = logger)" in refused
    assert "pin_plan = None" in refused and "denoiser_pinned = False" in refused
    assert (
        "_h3_group_offload_available()" in src[src.rindex("streamed_sizes = (", 0, start) : start]
    )


def test_unpin_returns_the_denoiser_to_the_rotation():
    from core.inference import video as vid

    calls = []

    class _Module:
        def to(self, device):
            calls.append(("to", device))

    class _Manager:
        def disable_auto_cpu_offload(self):
            calls.append(("disable",))

        def enable_auto_cpu_offload(
            self,
            device = None,
            memory_reserve_margin = None,
        ):
            calls.append(("enable", device, memory_reserve_margin))

    vid._h3_dense_unpin(_Manager(), _Module(), "cuda")
    assert calls == [("to", "cpu"), ("disable",), ("enable", "cuda", "40GB")]


def test_streamed_conditioner_tier_keeps_the_eager_tier_of_the_rotation_it_replaces():
    from core.inference import video as vid

    src = inspect.getsource(vid.VideoBackend)
    start = src.index("pin_plan = _h3_dense_pin_plan(")
    held = src.index("dense_compile_held = bool(denoiser_pinned)", start)
    assert held < src.index("video.h3_denoiser_placement: %.1f GB free is under", start)
    gate = src.index("if effective_speed in (SPEED_DEFAULT, SPEED_MAX) and dense_compile_held:")
    assert src.index("effective_speed = SPEED_EAGER", gate) < src.index(
        "snapshot_backend_flags()", gate
    )


class _Hook:
    def __init__(
        self,
        model,
        device = "cuda:0",
    ):
        self.model = model
        self.hook = type("H", (), {"execution_device": device})()
        self.others = []
        self.offloaded = False

    def add_other_hook(self, other):
        self.others.append(other)

    def offload(self):
        self.offloaded = True


def _demote_fixture(monkeypatch, available_mib):
    import torch

    cm = pytest.importorskip("diffusers.modular_pipelines.components_manager")

    from core.inference import diffusion_memory
    from core.inference import video as vid

    denoiser = torch.nn.Linear(4, 4)
    conditioner_vae = object()
    manager = type("M", (), {})()
    manager.components = {"transformer_1": denoiser, "vae_2": conditioner_vae}
    manager.model_hooks = [_Hook(conditioner_vae)]
    manager._auto_offload_device = "cuda:0"
    manager._offload_strategy = None
    pipe = type("P", (), {})()
    pipe._components_manager = manager
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: available_mib)
    monkeypatch.setattr(
        cm,
        "custom_offload_with_hook",
        lambda name, model, device, offload_strategy = None: _Hook(model, device),
    )
    monkeypatch.setattr(vid, "clear_gpu_cache", lambda: None)
    return vid, pipe, manager, denoiser


def test_a_request_past_the_pin_budget_returns_the_denoiser_to_the_rotation(monkeypatch):
    vid, pipe, manager, denoiser = _demote_fixture(monkeypatch, 200_000)
    assert vid._h3_dense_demote(pipe, denoiser)
    added = manager.model_hooks[-1]
    assert added.model is denoiser and added.offloaded
    assert added in manager.model_hooks[0].others and manager.model_hooks[0] in added.others
    assert vid._h3_dense_demote(pipe, denoiser) and len(manager.model_hooks) == 2


def test_no_host_room_keeps_the_pin_and_the_refusal(monkeypatch):
    vid, pipe, manager, denoiser = _demote_fixture(monkeypatch, 1)
    assert not vid._h3_dense_demote(pipe, denoiser)
    assert len(manager.model_hooks) == 1


def test_request_guard_tries_the_rotation_before_refusing():
    from core.inference import video as vid

    src = inspect.getsource(vid.VideoBackend)
    demote = src.index("_h3_dense_demote(\n")
    refuse = src.index('f"MiniMax-H3 needs about {required_vram_gb:.1f} GB available "')
    assert demote < refuse
    assert "required_vram_gb = rotation_gb" in src[demote:refuse]
