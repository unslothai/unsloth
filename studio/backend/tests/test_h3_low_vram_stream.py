# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 on 24 / 16 / 12 GB cards: the int8 conditioner streams instead of rotating onto the card whole, and a
streamed denoiser keeps as many blocks resident as each request leaves room for."""

from __future__ import annotations

import gc
import types
import weakref

import pytest

torch = pytest.importorskip("torch")


def _h3_family():
    from core.inference.video_families import detect_video_family
    return detect_video_family("minimax-h3")


# ── VRAM floors ─────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("budget_gb", [24.0, 16.0, 12.0])
def test_a_streamed_conditioner_and_denoiser_floor_fits_small_cards(budget_gb):
    """The old floor was the 27.2 GB conditioner + 1.8 = 29.0 GB at every size: 24 / 16 / 12 GB were all refused.
    Usable free memory on such a card is its size minus the CUDA context and desktop (bench: 11.7 GB on a 12 GB card)."""
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_vram_gb

    floor = estimate_h3_diffusers_vram_gb(
        960,
        544,
        124,
        text_encoder_gb = 27.2,
        transformer_gb = 20.3,
        transformer_streamed = True,
        text_encoder_streamed = True,
    )
    assert floor <= budget_gb - 0.6


def test_the_rotating_conditioner_keeps_its_old_floor():
    """Kill switch / unstreamable conditioner: the floor is exactly what it was."""
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_vram_gb

    floor = estimate_h3_diffusers_vram_gb(
        960, 544, 124, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_streamed = True
    )
    assert floor == pytest.approx(29.0)


def test_the_floor_still_grows_with_the_clip():
    """A long, large clip must still be refused on a small card rather than run out of memory mid-render."""
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_vram_gb

    def floor(w, h, f):
        return estimate_h3_diffusers_vram_gb(
            w,
            h,
            f,
            text_encoder_gb = 27.2,
            transformer_gb = 20.3,
            transformer_streamed = True,
            text_encoder_streamed = True,
        )

    assert floor(1344, 768, 345) > floor(1344, 768, 124) > floor(960, 544, 124)
    assert floor(1344, 768, 345) > 24.0


# ── conditioner streaming ───────────────────────────────────────────────────────────────────────────────────────────


def test_the_conditioner_stream_kill_switch(monkeypatch):
    from core.inference.video_minimax_h3_te import h3_te_stream_enabled, stream_h3_text_encoder

    monkeypatch.setenv("UNSLOTH_H3_TE_STREAM", "0")
    assert not h3_te_stream_enabled()
    te = types.SimpleNamespace(model = torch.nn.Linear(2, 2))
    assert stream_h3_text_encoder(object(), te, "cuda") is None
    monkeypatch.delenv("UNSLOTH_H3_TE_STREAM")
    assert h3_te_stream_enabled()


def test_the_conditioner_never_streams_off_cuda():
    from core.inference.video_minimax_h3_te import stream_h3_text_encoder
    te = types.SimpleNamespace(model = torch.nn.Linear(2, 2))
    assert stream_h3_text_encoder(object(), te, "cpu") is None


def test_pinning_in_place_replaces_tensors_and_releases_the_originals():
    """``.data =`` kept each view's base alive, so the conditioner's 27 GB memory-mapped file stayed resident beside
    the pinned copy. The tensors must be REPLACED: originals released, values kept, ties kept, arenas shared."""
    from core.inference.video_minimax_h3_te import pin_module_in_place

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(8, 16)
            self.b = torch.nn.Linear(16, 8, bias = False)
            self.register_buffer("q", torch.arange(64, dtype = torch.int8).view(8, 8))
            self.c = torch.nn.Linear(16, 8, bias = False)
            self.c.weight = self.b.weight  # tied

    m = M()
    # Views of one flat base, like safetensors tensors over a single mapping.
    base = torch.randn(16 * 8 + 16)
    m.a.weight = torch.nn.Parameter(base[: 16 * 8].view(16, 8))
    m.a.bias = torch.nn.Parameter(base[16 * 8 :])
    expect = {k: v.detach().clone() for k, v in m.state_dict().items()}
    base_ref = weakref.ref(base)
    del base
    arenas = []

    def factory(size):
        arenas.append(torch.zeros(size, dtype = torch.uint8))
        return arenas[-1]

    n = pin_module_in_place(m, arena_bytes = 4096, _arena_factory = factory)
    gc.collect()
    assert base_ref() is None, "the original base is still referenced"
    assert n == sum(t.numel() * t.element_size() for t in (m.a.weight, m.a.bias, m.b.weight, m.q))
    for k, v in m.state_dict().items():
        assert torch.equal(v, expect[k]), k
    assert m.c.weight is m.b.weight
    assert len(arenas) == 1
    ptrs = [t.untyped_storage().data_ptr() for t in (m.a.weight, m.a.bias, m.b.weight, m.q)]
    assert set(ptrs) == {arenas[0].untyped_storage().data_ptr()}
    assert isinstance(m.a.weight, torch.nn.Parameter) and not isinstance(m.q, torch.nn.Parameter)


# ── denoiser residency ──────────────────────────────────────────────────────────────────────────────────────────────


class _Group:
    def __init__(self, nbytes):
        self.modules = [
            torch.nn.Linear(1, nbytes // 4, bias = False)
        ]  # nbytes/4 fp32 weights -> nbytes
        self.parameters = []
        self.buffers = []
        self.stream = None
        self.on = 0
        self.off = 0
        self.where = "cpu"

    def onload_(self):
        self.on += 1
        self.where = "cuda"

    def offload_(self):
        self.off += 1
        self.where = "cpu"


def _residency(top_bytes, block_bytes):
    from core.inference.video_minimax_h3_residency import H3Residency, group_payload_bytes

    r = H3Residency.__new__(H3Residency)
    r.top = _Group(top_bytes) if top_bytes else None
    r.blocks = [_Group(b) for b in block_bytes]
    r.device = "cuda"
    r.logger = None
    r.top_bytes = group_payload_bytes(r.top) if r.top is not None else 0
    r.block_bytes = [group_payload_bytes(g) for g in r.blocks]
    r.max_blocks = 0
    return r


def test_residency_spends_the_budget_on_the_top_level_then_a_block_prefix():
    r = _residency(400, [100] * 10)
    assert r.plan(0) == (False, 0)
    assert r.plan(399) == (False, 3)  # the top does not fit; blocks still use what is there
    assert r.plan(400) == (True, 0)
    assert r.plan(750) == (True, 3)
    assert r.plan(10_000) == (True, 10)


def test_residency_demotes_the_tail_for_a_bigger_request_and_promotes_it_back(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = _residency(400, [100] * 10)
    r.fit(1000, initial = True)
    assert r.resident_blocks() == 6 and res.is_resident(r.top)
    assert [g.where for g in r.blocks] == ["cuda"] * 6 + ["cpu"] * 4
    # A resident group's hooks are no-ops: the streaming hook cannot move it.
    r.blocks[0].offload_()
    assert r.blocks[0].where == "cuda"
    r.fit(600)
    assert r.resident_blocks() == 2
    assert [g.where for g in r.blocks] == ["cuda"] * 2 + ["cpu"] * 8
    # demoted groups stream again
    r.blocks[3].onload_()
    assert r.blocks[3].where == "cuda"
    r.blocks[3].offload_()
    r.fit(1000)
    assert r.resident_blocks() == 6
    r.fit(0)
    assert (
        r.resident_bytes() == 0 and all(g.where == "cpu" for g in r.blocks) and r.top.where == "cpu"
    )


def test_a_fit_that_fails_partway_releases_what_it_promoted(monkeypatch):
    """An allocation failure on the third promotion must not leave the top and two blocks resident with no-op hooks
    and no controller (the load then claims to be fully streamed)."""
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = _residency(400, [100] * 10)

    def oom():
        r.blocks[2].where = "cuda"  # partly onloaded
        raise RuntimeError("CUDA out of memory")

    r.blocks[2].onload_ = oom
    with pytest.raises(RuntimeError):
        r.fit(10_000, initial = True)
    assert res.is_resident(r.top) and r.resident_blocks() == 2
    res.release_all(r)
    assert r.resident_bytes() == 0
    assert r.top.where == "cpu" and all(g.where == "cpu" for g in r.blocks)
    # streaming hooks are live again
    r.blocks[0].onload_()
    assert r.blocks[0].where == "cuda"


def test_residency_kill_switch(monkeypatch):
    from core.inference.video_minimax_h3_residency import h3_dit_resident_enabled

    monkeypatch.setenv("UNSLOTH_H3_DIT_RESIDENT", "0")
    assert not h3_dit_resident_enabled()
    monkeypatch.delenv("UNSLOTH_H3_DIT_RESIDENT")
    assert h3_dit_resident_enabled()


def test_the_phase_need_leaves_room_on_a_12gb_card():
    from core.inference.video_minimax_h3_residency import h3_phase_need_gb
    from core.inference.video_minimax_h3_te import H3_TE_STREAMED_GB

    floor = h3_phase_need_gb(960, 544, 124, te_streamed_gb = H3_TE_STREAMED_GB, fragmentation = False)
    assert floor < 11.0
    assert h3_phase_need_gb(1344, 768, 345, te_streamed_gb = H3_TE_STREAMED_GB) > 24.0
    # The residency plan keeps fragmentation slack free; the refusal floor does not ask for it.
    assert h3_phase_need_gb(1344, 768, 124, te_streamed_gb = H3_TE_STREAMED_GB) > h3_phase_need_gb(
        1344, 768, 124, te_streamed_gb = H3_TE_STREAMED_GB, fragmentation = False
    )


def test_held_host_bytes_dedupes_storage():
    from core.inference.video_minimax_h3_residency import h3_held_host_bytes

    base = torch.zeros(1000)
    m = torch.nn.Module()
    m.register_buffer("a", base[:500])
    m.register_buffer("b", base[500:])
    held = h3_held_host_bytes(m, None)
    assert held == {"pinned": 0, "pageable": 4000}


def _tiny_qwen3_vl():
    transformers = pytest.importorskip("transformers")
    config_cls = getattr(transformers, "Qwen3VLConfig", None)
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if config_cls is None or model_cls is None:
        pytest.skip("this transformers has no Qwen3-VL (the H3 conditioner's architecture)")
    config = config_cls(
        text_config = dict(
            vocab_size = 256,
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
        ),
        vision_config = dict(
            depth = 2,
            hidden_size = 16,
            intermediate_size = 32,
            num_heads = 4,
            out_hidden_size = 16,
            deepstack_visual_indexes = [1],
        ),
        image_token_id = 250,
        vision_start_token_id = 251,
        vision_end_token_id = 252,
    )
    torch.manual_seed(0)
    return model_cls(config).eval()


def test_an_image_prompt_encodes_through_the_streamed_conditioner(monkeypatch):
    """Keyframe / reference prompts run the Qwen3-VL vision tower inside the streamed conditioner. Its position
    interpolation reads the offloaded embedding's device (CPU under group offload; transformers 5.5), which raised a
    cuda / cpu device mismatch. The streamed encode must match the whole-module one."""
    import copy
    import inspect

    if not torch.cuda.is_available():
        pytest.skip("group offload streams onto a CUDA device; nothing to stream onto here")
    import core.inference.diffusion_prequant as prequant
    from core.inference.video_minimax_h3_te import stream_h3_text_encoder

    monkeypatch.setattr(prequant, "_unhook_from_manager", lambda *a, **k: True)
    monkeypatch.setattr(
        prequant, "_evict_rotation_hook", lambda manager, onload: (lambda *a, **k: None)
    )
    reference = _tiny_qwen3_vl()
    streamed = copy.deepcopy(reference)
    assert stream_h3_text_encoder(object(), streamed, "cuda", pin = False) == "stream_lazy"

    vision = reference.config.vision_config
    grid = torch.tensor([[1, 4, 4]])
    patches = int(grid.prod())
    pixels = torch.randn(
        patches, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    )
    image_tokens = patches // vision.spatial_merge_size**2
    ids = torch.tensor([[1, 2, 3, 251] + [250] * image_tokens + [252, 4]])
    kwargs = dict(
        input_ids = ids, pixel_values = pixels, image_grid_thw = grid, output_hidden_states = True
    )
    if "mm_token_type_ids" in inspect.signature(type(reference.model).forward).parameters:
        kwargs["mm_token_type_ids"] = (ids == 250).long()
    with torch.inference_mode():
        expected = reference.cuda().model(
            **{k: v.cuda() if torch.is_tensor(v) else v for k, v in kwargs.items()}
        )
        got = streamed.model(
            **{k: v.cuda() if torch.is_tensor(v) else v for k, v in kwargs.items()}
        )
    torch.testing.assert_close(got.hidden_states[-1], expected.hidden_states[-1])


def test_pinning_is_budgeted_on_the_whole_payload_not_one_arena(monkeypatch):
    """The hosted weights are mmap views already counted as available, so pinning them takes the full payload. A host
    with room for one arena but not the 27 GB conditioner must stream it unpinned instead of exhausting RAM at load."""
    import core.inference.diffusion_memory as dm
    from core.inference.video_minimax_h3_te import h3_te_pin_allowed

    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 20 * 1024)
    assert h3_te_pin_allowed()  # one arena fits
    assert h3_te_pin_allowed(5 * 2**30)  # the VAEs
    assert not h3_te_pin_allowed(27 * 2**30)  # the conditioner does not
    monkeypatch.setenv(dm.GROUP_OFFLOAD_PIN_ENV, "1")
    assert h3_te_pin_allowed(27 * 2**30)
