# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image layer-major batch: one block upload per call, rows bit-identical to solo. CPU only."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

_TINY = dict(
    dim = 64, ffn_dim = 96, num_heads = 2, num_layers = 4, text_dim = 24, freq_dim = 16, latent_channels = 8
)


@pytest.fixture(scope = "module")
def moss():
    from core.inference import diffusion_moss_image
    return diffusion_moss_image


@pytest.fixture(autouse = True)
def _layer_major_on(monkeypatch, moss):
    monkeypatch.delenv("UNSLOTH_MOSS_LAYER_MAJOR", raising = False)
    if hasattr(moss, "layer_major_stats"):
        moss.layer_major_stats(reset = True)


def _offload(model):
    from diffusers.hooks import apply_group_offloading

    cpu = torch.device("cpu")
    apply_group_offloading(
        model,
        onload_device = cpu,
        offload_device = cpu,
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    return model


def _tiny(
    moss,
    dtype = torch.float32,
    offloaded = True,
    seed = 0,
):
    torch.manual_seed(seed)
    model = moss.MOSSImageTransformer(**_TINY).to(dtype).eval()
    return _offload(model) if offloaded else model


def _inputs(
    lengths,
    seed = 1,
    dtype = torch.float32,
    longest = None,
):
    gen = torch.Generator().manual_seed(seed)
    batch, width = len(lengths), longest or max(lengths)
    latents = torch.randn(batch, 8, 4, 6, generator = gen).to(dtype)
    text = torch.randn(batch, width, 24, generator = gen).to(dtype)
    mask = torch.zeros(batch, width, dtype = torch.bool)
    for index, length in enumerate(lengths):
        mask[index, :length] = True
        text[index, length:] = 0
    return latents, text, mask


def _run(model, latents, text, mask, **kwargs):
    with torch.no_grad():
        return model(latents, 0.375, text, mask, return_dict = False, **kwargs)[0]


@pytest.fixture
def block_uploads(moss):
    from diffusers.hooks import group_offloading

    count = [0]
    original = group_offloading.ModuleGroup.onload_

    def onload(self, *args, **kwargs):
        if any(isinstance(module, moss._Block) for module in (self.modules or ())):
            count[0] += 1
        return original(self, *args, **kwargs)

    group_offloading.ModuleGroup.onload_ = onload
    try:
        yield count
    finally:
        group_offloading.ModuleGroup.onload_ = original


def _solo_rows(model, latents, text, mask):
    """Each row through its own batch-1 forward at its true text length (what a solo render calls)."""
    rows = []
    for index in range(latents.shape[0]):
        length = int(mask[index].sum()) if mask is not None else text.shape[1]
        rows.append(_run(model, latents[index : index + 1], text[index : index + 1, :length], None))
    return torch.cat(rows)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(7, 3, 5), (5, 5), (2, 9, 4, 9)])
def test_layer_major_rows_are_their_solo_forwards_bit_for_bit(moss, monkeypatch, dtype, lengths):
    model = _tiny(moss, dtype)
    latents, text, mask = _inputs(lengths, dtype = dtype)
    batched = _run(model, latents, text, mask)
    assert moss.layer_major_stats()["forwards"] == 1
    assert torch.equal(batched, _solo_rows(model, latents, text, mask))
    # the kill switch is the per-sample batch: the same rows up to the batch-B modulation GEMM's rounding
    monkeypatch.setenv(moss.LAYER_MAJOR_ENV, "0")
    serial = _run(model, latents, text, mask)
    assert moss.layer_major_stats()["serial"] == {f"{moss.LAYER_MAJOR_ENV}=0": 1}
    tol = 1e-5 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(batched, serial, rtol = 0, atol = tol)


def test_unmasked_refined_and_per_row_timestep_batches_are_solo_rows(moss):
    model = _tiny(moss)
    latents, text, _ = _inputs((6, 6, 6))
    refined = model.refine_text(text).detach()
    times = torch.tensor([0.25, 0.5, 0.75])
    with torch.no_grad():
        plain = model(latents, 0.5, text, return_dict = False)[0]
        pre = model(latents, 0.5, refined_text = refined, skip_layers = (1,), return_dict = False)[0]
        per_t = model(latents, times, text, return_dict = False)[0]
        for index in range(3):
            row = slice(index, index + 1)
            assert torch.equal(
                plain[row], model(latents[row], 0.5, text[row], return_dict = False)[0]
            )
            solo = model(
                latents[row], 0.5, refined_text = refined[row], skip_layers = (1,), return_dict = False
            )
            assert torch.equal(pre[row], solo[0])
            assert torch.equal(
                per_t[row], model(latents[row], times[index], text[row], return_dict = False)[0]
            )
    assert moss.layer_major_stats()["forwards"] == 3


def test_g1_n1_keeps_the_per_sample_forward(moss):
    resident = _tiny(moss, offloaded = False)
    offloaded = _tiny(moss)
    latents, text, mask = _inputs((5,))
    out = _run(offloaded, latents, text, mask)
    stats = moss.layer_major_stats()
    assert stats["forwards"] == 0 and stats["serial"] == {}
    assert torch.equal(out, _run(resident, latents, text, mask))


@pytest.mark.parametrize("target", [0, 1, 3])
def test_g2a_a_neighbour_never_moves_a_row(moss, target):
    """Change one neighbour's prompt (another length), latent and text; the target row stays bit-identical."""
    model = _tiny(moss)
    lengths = [4, 9, 6, 3]
    latents, text, mask = _inputs(lengths, longest = 11)
    before = _run(model, latents, text, mask)
    neighbour = 2 if target != 2 else 1
    gen = torch.Generator().manual_seed(99)
    latents2, text2, mask2 = latents.clone(), text.clone(), mask.clone()
    latents2[neighbour] = torch.randn(latents[neighbour].shape, generator = gen) * 3
    text2[neighbour] = torch.randn(text[neighbour].shape, generator = gen) * 5
    mask2[neighbour] = False
    mask2[neighbour, :11] = True
    after = _run(model, latents2, text2, mask2)
    assert moss.layer_major_stats()["forwards"] == 2
    assert torch.equal(after[target], before[target])
    assert not torch.equal(after[neighbour], before[neighbour])


def test_g3_one_block_upload_per_call_instead_of_per_row(moss, monkeypatch, block_uploads):
    model = _tiny(moss)
    latents, text, mask = _inputs((7, 3, 5, 2))
    _run(model, latents, text, mask)  # the first forward records the group order
    block_uploads[0] = 0
    _run(model, latents, text, mask)
    assert block_uploads[0] == _TINY["num_layers"]
    monkeypatch.setenv("UNSLOTH_MOSS_LAYER_MAJOR", "0")
    block_uploads[0] = 0
    _run(model, latents, text, mask)
    assert block_uploads[0] == 4 * _TINY["num_layers"]


def test_refusals_keep_the_per_sample_forward(moss, monkeypatch):
    assert moss.layer_major_refusal(_tiny(moss, offloaded = False)) == "resident weights"
    model = _tiny(moss)
    assert moss.layer_major_refusal(model) is None
    monkeypatch.setenv(moss.LAYER_MAJOR_ENV, "0")
    assert moss.layer_major_refusal(model) == f"{moss.LAYER_MAJOR_ENV}=0"
    monkeypatch.delenv(moss.LAYER_MAJOR_ENV)

    handle = model.layers[1].register_forward_pre_hook(lambda module, args: None)
    assert moss.layer_major_refusal(model) == "a block runs compiled, graphed or hooked"
    handle.remove()
    model.layers[2]._compiled_call_impl = lambda *args: None
    assert moss.layer_major_refusal(model) == "a block runs compiled, graphed or hooked"
    model.layers[2]._compiled_call_impl = None
    layer = model.layers[3]
    hooked = layer.__dict__["forward"]
    layer.__dict__["forward"] = lambda *args, **kwargs: hooked(
        *args, **kwargs
    )  # a CUDA-graph style wrapper
    assert moss.layer_major_refusal(model) == "a block runs compiled, graphed or hooked"
    layer.__dict__["forward"] = hooked
    assert moss.layer_major_refusal(model) is None

    model.gradient_checkpointing = True
    assert moss.layer_major_refusal(model) == "gradient checkpointing"
    with torch.no_grad():
        assert moss.layer_major_refusal(model) is None


def test_pipeline_rows_with_cfg_are_solo_renders_and_neighbour_invariant(moss, monkeypatch):
    """Via the pipeline (CFG, skip layer, seed list): rows equal solo renders; swapping one leaves others intact."""
    model = _tiny(moss)
    scheduler = moss.MOSSImageScheduler()
    pipe = moss.MOSSImagePipeline(transformer = model, scheduler = scheduler)
    pipe.set_progress_bar_config(disable = True)

    def embeds(rows):
        """One (length, seed) per row: each row's text depends on its own seed only."""
        width = max(length for length, _ in rows)
        text = torch.zeros(len(rows), width, 24)
        mask = torch.zeros(len(rows), width, dtype = torch.bool)
        for index, (length, seed) in enumerate(rows):
            gen = torch.Generator().manual_seed(seed)
            text[index, :length] = torch.randn(length, 24, generator = gen)
            mask[index, :length] = True
        return text, mask

    def render(lengths, seeds):
        text, mask = embeds([(length, 100 + seed) for length, seed in zip(lengths, seeds)])
        neg, neg_mask = embeds([(2, 4)] * len(lengths))
        gens = [torch.Generator().manual_seed(s) for s in seeds]
        with torch.no_grad():
            return pipe(
                prompt_embeds = text,
                prompt_attention_mask = mask,
                negative_prompt_embeds = neg,
                negative_prompt_attention_mask = neg_mask,
                generator = gens,
                height = 64,
                width = 96,
                num_inference_steps = 3,
                guidance_scale = 2.5,
                uncond_skip_layers = (2,),
                output_type = "latent",
            ).images

    batched = render((5, 8, 3), (1, 2, 3))
    assert moss.layer_major_stats()["forwards"] == 6
    for index, (length, seed) in enumerate(((5, 1), (8, 2), (3, 3))):
        assert torch.equal(batched[index], render((length,), (seed,))[0])
    moved = render((5, 2, 3), (1, 7, 3))
    assert torch.equal(moved[0], batched[0]) and torch.equal(moved[2], batched[2])
    assert not torch.equal(moved[1], batched[1])
    monkeypatch.setenv(moss.LAYER_MAJOR_ENV, "0")
    torch.testing.assert_close(render((5, 8, 3), (1, 2, 3)), batched, rtol = 0, atol = 1e-4)


def test_coalescing_waits_for_a_measured_verdict(moss, monkeypatch):
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "moss-image")
    pipe = types.SimpleNamespace(transformer = _tiny(moss), _execution_device = torch.device("cpu"))
    assert coalesce_prompt_key(family, pipe, "a fox") is None
    monkeypatch.setitem(moss.MEASURED_BATCH_WINS, "cpu", {"serves": False})
    assert coalesce_prompt_key(family, pipe, "a fox") is None
    monkeypatch.setitem(moss.MEASURED_BATCH_WINS, "cpu", {"serves": True})
    assert coalesce_prompt_key(family, pipe, "a fox") == ("moss-image layer-major",)
    assert coalesce_prompt_key(family, pipe, "a much longer prompt about a fox") == (
        "moss-image layer-major",
    )
    resident = types.SimpleNamespace(
        transformer = _tiny(moss, offloaded = False), _execution_device = torch.device("cpu")
    )
    assert coalesce_prompt_key(family, resident, "a fox") is None
    monkeypatch.setenv(moss.LAYER_MAJOR_ENV, "0")
    assert coalesce_prompt_key(family, pipe, "a fox") is None
