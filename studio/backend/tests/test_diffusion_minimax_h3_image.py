# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image on CPU with a tiny random release in the native layout: key conversion (round trip and the PR
converter's own functions), the fixed-grid AdaLN table against the dense model, packing of text + reference rows,
the canvas rule, the sampler, VAE encode/decode, the planner tiers, routing / family defaults and the backend load +
generate path."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONVERTER = (
    Path(__file__).with_name("data") / "minimax_h3_image" / "convert_minimax_h3_to_diffusers.py.txt"
)


def _stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    try:
        from diffusers import DiffusionPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a broken diffusers / hub pair
        pytest.skip(f"diffusers pipelines unavailable: {exc}")
    return torch


@pytest.fixture(scope = "module")
def h3i():
    _stack()
    from core.inference import diffusion_minimax_h3_image
    return diffusion_minimax_h3_image


@pytest.fixture(scope = "module")
def release(h3i, tmp_path_factory):
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_release")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    return root, ref, te


@pytest.fixture(scope = "module")
def pipe(h3i, release):
    root, _ref, te = release
    return h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)


# -- vendored source ----------------------------------------------------------------------------------------------


def test_vendored_files_match_the_manifest():
    manifest = json.loads((_BACKEND / "vendor" / "diffusers_minimax_h3_image.json").read_text())
    assert manifest["license"] == "Apache-2.0"
    assert len(manifest["upstream_commit"]) == 40
    for rel, digest in manifest["files"].items():
        data = (_BACKEND / "vendor" / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel
    for rel in manifest["files"]:
        if rel.endswith(".py"):
            text = (_BACKEND / "vendor" / rel).read_text()
            assert "Licensed under the Apache License, Version 2.0" in text
            # Every relative import was rewritten onto the installed diffusers.
            assert "\nfrom ." not in text


def test_classes_resolve_to_diffusers_or_the_vendored_copy(h3i):
    t, v = h3i.transformer_class(), h3i.vae_class()
    assert t.__name__ == "MiniMaxH3ImageTransformer2DModel"
    assert v.__name__ == "AutoencoderKLMiniMaxH3Image"
    assert h3i.classes_source() in ("diffusers", "vendored")
    assert h3i.in_tree_class("MiniMaxH3DiTModel") is t
    assert h3i.in_tree_class("MiniMaxH3ImagePipeline") is h3i.MiniMaxH3ImagePipeline


# -- conversion ---------------------------------------------------------------------------------------------------


def test_native_release_loads_bitwise(h3i, release):
    import torch

    root, ref, _te = release
    model, stats = h3i.load_h3_image_transformer(
        root, dtype = torch.bfloat16, scheme = None, device = "cpu", keep_on_device = True, table = False
    )
    assert stats["layout"] == "native"
    want = {k: v for k, v in ref["transformer"].state_dict().items() if not k.startswith("rope.")}
    got = {k: v for k, v in model.state_dict().items() if not k.startswith("rope.")}
    assert set(got) == set(want)
    for k in want:
        assert got[k].dtype == want[k].dtype, k
        assert torch.equal(got[k], want[k]), k
    vae = h3i.load_h3_image_vae(root)
    vwant = ref["vae"].state_dict()
    vgot = vae.state_dict()
    assert set(vgot) == set(vwant)
    assert all(torch.equal(vgot[k], vwant[k]) for k in vwant)


def _converter():
    loader = importlib.machinery.SourceFileLoader("h3_converter_reference", str(CONVERTER))
    spec = importlib.util.spec_from_loader("h3_converter_reference", loader)
    module = importlib.util.module_from_spec(spec)
    try:
        loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - it imports diffusers internals of its own era
        pytest.skip(f"PR converter not importable here: {exc}")
    return module


def test_key_conversion_matches_the_pr_converter(h3i):
    import torch

    conv = _converter()
    cfg = {
        **conv.MINIMAX_H3_IMAGE_TRANSFORMER_CONFIG,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
    }
    g = torch.Generator().manual_seed(0)
    keys = [
        ("blocks.3.attn.qkv_proj.weight", (2 * 3 * 8, 16)),
        ("blocks.3.mlp.fc1.weight", (32, 16)),
        ("blocks.3.mlp.fc2.weight", (16, 16)),
        ("blocks.3.attn.out_proj.weight", (16, 16)),
        ("blocks.3.attn.q_norm.weight", (8,)),
        ("token_refiner.blocks.1.attn.qkv_proj.weight", (2 * 3 * 8, 16)),
        ("final_layer.adaln_proj.linear.weight", (32, 8)),
        ("time_embedder.proj_in.weight", (8, 4)),
        ("condition_proj.bias", (16,)),
        ("video_patch_proj.weight", (16, 96)),
    ]
    for key, shape in keys:
        tensor = torch.randn(shape, generator = g)
        source = tensor
        if key.endswith(".attn.qkv_proj.weight"):
            source = conv.reorder_interleaved_qkv(tensor, 2, 8)
        want = dict(conv.convert_transformer_key(key, source, cfg))
        got = dict(h3i.convert_native_transformer_tensor(key, tensor, 2, 8))
        assert set(got) == set(want), key
        assert all(torch.equal(got[k], want[k]) for k in want), key
    for dropped in ("rope.inv_freq", "final_layer.video_out.weight"):
        assert h3i.convert_native_transformer_tensor(dropped, torch.zeros(4), 2, 8) == []
    vcfg = {"decoder_num_attention_heads": 2, "decoder_attention_head_dim": 8}
    for key, shape in (
        ("decoder.transformer_blocks.0.attn.to_qkv.weight", (48, 16)),
        ("decoder.transformer_blocks.0.ff.w1.weight", (64, 16)),
        ("encoder.down.2.block.1.nin_shortcut.weight", (4, 4, 1, 1, 1)),
        ("encoder.down.1.downsample.conv.bias", (4,)),
        ("decoder.x_embedder.weight", (16, 24)),
    ):
        tensor = torch.randn(shape, generator = g)
        want = dict(conv.convert_video_vae_key(key, tensor, vcfg))
        got = dict(h3i.convert_native_vae_tensor(key, tensor, 2, 8))
        assert set(got) == set(want), key
        assert all(torch.equal(got[k], want[k]) for k in want), key


def test_pdd_heads_split_and_count_is_checked(h3i):
    import torch

    heads = {"video_out.weight": torch.randn(6, 4, 3), "video_out.bias": torch.randn(6, 4)}
    out = h3i.split_pdd_heads(heads, 6)
    assert sorted(out) == sorted(
        f"pdd_proj_out.{i}.{s}" for i in range(6) for s in ("weight", "bias")
    )
    assert torch.equal(out["pdd_proj_out.5.weight"], heads["video_out.weight"][5])
    with pytest.raises(ValueError):
        h3i.split_pdd_heads(heads, 7)
    with pytest.raises(KeyError):
        h3i.split_pdd_heads({"video_out.weight": heads["video_out.weight"]}, 6)


def test_native_config_maps_and_refuses_the_video_dit(h3i):
    real = json.loads(
        (
            Path(__file__).with_name("data") / "minimax_h3_image" / "transformer_config.json"
        ).read_text()
    )
    cfg = h3i.transformer_config_from_native(real)
    assert cfg["num_layers"] == 50 and cfg["hidden_size"] == 5376 and cfg["ffn_dim"] == 14336
    assert cfg["in_channels"] == 24 and cfg["text_dim"] == 5120 and cfg["num_pdd_heads"] == 6
    assert cfg["freq_dim"] == 256 and cfg["rope_freq_dim"] == 16 and cfg["num_refiner_layers"] == 2
    with pytest.raises(ValueError):
        h3i.transformer_config_from_native({**real, "audio_enabled": True})


# -- AdaLN table ---------------------------------------------------------------------------------------------------


def _forward_inputs(
    h3i,
    timesteps,
    n_ref = 0,
):
    import torch

    g = torch.Generator().manual_seed(5)
    n_text, lh, lw = 7, 8, 6
    tags = torch.full((n_text,), h3i.H3_IMAGE_TEXT_TAG)
    grids = [(4, 4)] * n_ref
    pos, tok, img, txt, nref = h3i.build_packed_layout(tags, grids, lh, lw)
    rows = torch.randn(img.numel(), 96, generator = g)
    text = torch.randn(1, n_text, 48, generator = g).to(torch.bfloat16)
    plan = h3i.row_timestep_plan(
        torch.tensor(timesteps), int(img.numel() + txt.numel()), img[:nref]
    )
    return rows, text, pos, tok, img, txt, plan


@pytest.mark.parametrize("n_ref", [0, 2])
def test_table_adaln_matches_the_dense_projection(h3i, release, n_ref):
    import copy

    import torch

    _root, ref, _te = release
    dense = ref["transformer"]
    table = copy.deepcopy(dense)
    assert h3i.convert_dense_transformer_to_table(table) == len(dense.transformer_blocks)
    steps = h3i.grid_timesteps()[:6]
    rows, text, pos, tok, img, txt, plan = _forward_inputs(h3i, steps, n_ref)
    for i in (0, 3, 5):
        unique, index = plan[i]
        kw = dict(
            hidden_states = rows[None],
            encoder_hidden_states = text,
            timestep = unique,
            timestep_indices = index,
            token_tags = tok,
            position_ids = pos,
            image_indices = img,
            text_indices = txt,
            pdd_step_index = i,
            return_dict = False,
        )
        with torch.no_grad():
            want = dense(**kw)[0]
            got = table(**kw)[0]
        # Same bf16 rows the dense projection emits; only the batch of timesteps it ran over differs.
        assert torch.allclose(got, want, atol = 2e-2, rtol = 0), (i, (got - want).abs().max())
        assert (got - want).abs().mean() < 2e-3


def test_table_rows_are_exactly_the_dense_rows(h3i, release):
    import torch

    _root, ref, _te = release
    dense = ref["transformer"]
    grid = h3i.grid_timesteps()
    temb = h3i.grid_temb((dense.time_proj, dense.time_embedder), grid)
    lin = dense.transformer_blocks[0].adaln_proj.linear
    rows = h3i.dense_adaln_rows(temb, lin.weight, lin.bias, lin.weight.dtype)
    linear = h3i.table_linear(rows)
    onehot = torch.nn.functional.one_hot(torch.arange(len(grid)), len(grid)).to(torch.bfloat16)
    assert torch.equal(linear(onehot), rows)


def test_grid_has_seven_timesteps_and_refuses_off_grid(h3i):
    import torch

    grid = h3i.grid_timesteps()
    assert len(grid) == 7 and grid[0] == 0.0 and abs(grid[-1] - 0.999) < 1e-6
    embedder = h3i._grid_embedder_class()(grid)
    assert embedder(torch.tensor(grid)).argmax(dim = 1).tolist() == list(range(7))
    with pytest.raises(ValueError, match = "fixed 6-step grid"):
        embedder(torch.tensor([0.5]))


# -- recipe --------------------------------------------------------------------------------------------------------


def test_canvas_rule(h3i):
    assert h3i.canvas_size(2048, 2048) == (2048, 2048)
    assert h3i.canvas_size(1024, 1024) == (2048, 2048)
    assert h3i.canvas_size(1024, 1024, 1024) == (1024, 1024)
    h, w = h3i.canvas_size(768, 1344)
    assert (
        h % 32 == 0
        and w % 32 == 0
        and h * w >= 0.98 * 2048 * 2048
        and abs(w / h - 1344 / 768) < 0.03
    )
    assert h3i.canvas_size(4096, 4096) == (4096, 4096)
    with pytest.raises(ValueError):
        h3i.canvas_size(512, 512, 512)
    for w_, h_ in h3i.H3_IMAGE_PRESETS:
        assert w_ % 32 == 0 and h_ % 32 == 0
    for w_, h_ in h3i.H3_IMAGE_16MP_SIZES:
        assert 15.5e6 < w_ * h_ < 17.5e6


def test_reference_short_edge_bounds(h3i):
    assert h3i.bound_short_edge(1500, 1200, 1024, 2048) == (1500, 1200)
    assert h3i.bound_short_edge(600, 400, 1024, 2048) == (1536, 1024)
    assert h3i.bound_short_edge(5000, 3000, 1024, 2048) == (3424, 2048)
    assert h3i.bound_short_edge(2000, 1500, 1024, 1024) == (1376, 1024)


def test_packed_layout_text_references_target(h3i):
    import torch

    tags = torch.tensor([1, 1, 0, 0, 0, 1, 1])
    pos, tok, img, txt, nref = h3i.build_packed_layout(tags, [(4, 6), (8, 2)], 8, 8)
    assert nref == 2 * 3 + 4 * 1
    assert txt.tolist() == list(range(7)) and img[0] == 7 and img[-1] == pos.shape[0] - 1
    assert tok[:7].tolist() == tags.tolist() and bool((tok[7:] == h3i.H3_IMAGE_IMAGE_TAG).all())
    assert pos.dtype == torch.float64
    assert pos[:7, 0].tolist() == list(range(7))
    # One rotary frame slot per reference, then the target, starting where the text ends.
    assert set(pos[7:13, 0].tolist()) == {7.0} and set(pos[13:17, 0].tolist()) == {8.0}
    assert set(pos[17:, 0].tolist()) == {9.0} and pos.shape[0] == 7 + 10 + 16
    plan = h3i.row_timestep_plan(torch.tensor([0.0, 0.5]), pos.shape[0], img[:nref])
    unique, index = plan[1]
    assert unique.tolist() == pytest.approx([0.5, 0.999])
    assert bool((index[img[:nref]] == 1).all()) and bool((index[txt] == 0).all())


def test_patchify_roundtrip(h3i):
    import torch

    x = torch.randn(1, 24, 1, 8, 6)
    rows = h3i.patchify(x, (1, 2, 2))
    assert rows.shape == (12, 96)
    assert torch.equal(h3i.unpatchify(rows, 8, 6, 24, (1, 2, 2)), x[:, :, 0])


def test_scheduler_is_the_velocity_euler_on_the_fixed_grid(h3i):
    import torch

    s = h3i.MiniMaxH3ImageScheduler(sample_sigmas = list(h3i.H3_IMAGE_SIGMAS))
    s.set_timesteps()
    assert s.timesteps.numel() == 6 and float(s.sigmas[-1]) == 0.0
    x, v = torch.randn(4, 3), torch.randn(4, 3)
    out = s.step(v, s.timesteps[0], x)[0]
    assert torch.equal(out, x + (s.sigmas[0] - s.sigmas[1]) * v)
    noisy = s.scale_noise(torch.ones(2), torch.tensor(0.999), torch.zeros(2))
    assert torch.allclose(noisy, torch.full((2,), 0.999))


def test_vae_encode_decode_shapes(h3i, release):
    import torch

    _root, ref, _te = release
    vae = ref["vae"]
    assert vae.spatial_compression_ratio == 16
    x = torch.randn(1, 3, 64, 96)
    post = vae.encode(x, return_dict = False)[0]
    z = post.sample(generator = torch.Generator().manual_seed(0))
    assert z.shape[1] == 24 and z.shape[-2:] == (4, 6)
    y = vae.decode(z.squeeze(2) if z.ndim == 5 else z, return_dict = False)[0]
    assert y.shape == (1, 3, 64, 96)


# -- pipeline --------------------------------------------------------------------------------------------------------


def test_t2i_renders_at_the_requested_size_and_is_seeded(h3i, pipe):
    import numpy as np
    import torch

    a = pipe(
        prompt = "a cat",
        height = 192,
        width = 256,
        generator = torch.Generator().manual_seed(0),
        min_generation_area = 1024,
    )
    b = pipe(
        prompt = "a cat",
        height = 192,
        width = 256,
        generator = torch.Generator().manual_seed(0),
        min_generation_area = 1024,
    )
    c = pipe(
        prompt = "a cat",
        height = 192,
        width = 256,
        generator = torch.Generator().manual_seed(1),
        min_generation_area = 1024,
    )
    assert a.images[0].size == (256, 192)
    assert np.array_equal(np.asarray(a.images[0]), np.asarray(b.images[0]))
    assert not np.array_equal(np.asarray(a.images[0]), np.asarray(c.images[0]))


def test_small_request_renders_on_the_4mp_canvas_and_resizes_back(h3i, pipe):
    import torch

    img = pipe(
        prompt = "a cat", height = 64, width = 96, generator = torch.Generator().manual_seed(0)
    ).images[0]
    assert img.size == (96, 64)
    w, h = pipe._unsloth_h3_last_stats["canvas"]
    assert w * h >= 0.97 * 2048 * 2048 and w % 32 == 0 and h % 32 == 0


def test_edit_packs_labels_vision_tokens_and_reference_rows(h3i, pipe, monkeypatch):
    import numpy as np
    import torch
    from PIL import Image

    if pipe.processor is None:
        pytest.skip("no Qwen3-VL processor in this transformers")
    seen = {}
    real = h3i.qwen3vl_hidden_state

    def spy(encoder, processor, token_ids, vision_inputs, **kw):
        seen["ids"] = list(token_ids)
        seen["grid"] = vision_inputs.get("image_grid_thw")
        return real(encoder, processor, token_ids, vision_inputs, **kw)

    monkeypatch.setattr(h3i, "qwen3vl_hidden_state", spy)
    ref = Image.fromarray((np.random.RandomState(0).rand(40, 60, 3) * 255).astype("uint8"))
    out = pipe(
        prompt = "<Picture 1> a cat",
        image = [ref],
        height = 64,
        width = 64,
        generator = torch.Generator().manual_seed(0),
        min_generation_area = 1024,
    )
    assert out.images[0].size == (64, 64)
    tok = pipe.tokenizer
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    merge = pipe.processor.image_processor.merge_size**2
    assert seen["ids"].count(pad) == int(seen["grid"][0].prod()) // merge
    label = tok("Picture 1: ", add_special_tokens = False)["input_ids"]
    assert seen["ids"][: len(label)] == label


def test_edit_refuses_more_than_sixteen_references(h3i, pipe):
    from PIL import Image
    img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match = "at most 16"):
        pipe.prepare_references([img] * 17)


def test_step_and_guidance_contract(h3i, pipe):
    with pytest.raises(ValueError, match = "6 steps"):
        pipe(prompt = "a cat", num_inference_steps = 20, height = 64, width = 64)
    with pytest.raises(ValueError, match = "guidance"):
        pipe(prompt = "a cat", guidance_scale = 3.5, height = 64, width = 64)
    with pytest.raises(ValueError, match = "prompt"):
        pipe(prompt = "", height = 64, width = 64)


def test_step_callback_sees_bchw_latents_and_interrupt_stops(h3i, pipe):
    import torch

    calls = []

    def cb(p, i, t, kw):
        calls.append((i, tuple(kw["latents"].shape)))
        if i == 1:
            p._interrupt = True
        return kw

    out = pipe(
        prompt = "a cat",
        height = 64,
        width = 64,
        generator = torch.Generator().manual_seed(0),
        min_generation_area = 1024,
        callback_on_step_end = cb,
    )
    assert calls == [(0, (1, 24, 4, 4)), (1, (1, 24, 4, 4))]
    assert out.images == []


# -- quantization / placement ---------------------------------------------------------------------------------------


def test_int8_quantizes_only_the_block_linears(h3i, release):
    import torch

    pytest.importorskip("torchao")
    root, _ref, _te = release
    model, stats = h3i.load_h3_image_transformer(
        root, dtype = torch.bfloat16, scheme = "int8", device = "cpu", keep_on_device = True
    )
    assert stats["quantized_linears"] == 6 * len(model.transformer_blocks)
    block = model.transformer_blocks[0]
    assert (
        type(block.attn.to_q.weight).__name__ != "Parameter"
        or not isinstance(block.attn.to_q.weight.data, torch.Tensor)
        or type(block.attn.to_q.weight.data).__name__ != "Tensor"
    )
    assert type(block.adaln_proj.linear.weight.data).__name__ == "Tensor"
    assert type(model.token_refiner.refiner_blocks[0].attn.to_q.weight.data).__name__ == "Tensor"


@pytest.mark.parametrize(
    "vram, host, te, placement, te_kept",
    [
        (95.0, 177.0, 27.2, "resident", True),  # G4: everything resident
        (39.5, 83.5, 27.2, "dit_resident", True),  # A100 40 GB: DiT resident, encoder streams
        (22.0, 50.0, 27.2, "stream", True),  # L4: blocks stream, the host holds the rest
        (22.0, 30.0, 27.2, "stream", False),  # small host: the encoder loads per prompt
        (79.0, 167.0, 66.7, "dit_resident", True),  # A100 80 GB with the bf16 encoder
    ],
)
def test_planner_tiers(h3i, vram, host, te, placement, te_kept):
    plan = h3i.plan_placement(
        vram_gb = vram, host_gb = host, dit_gb = h3i.H3_IMAGE_DIT_TABLE_INT8_GB, te_gb = te
    )
    assert plan.placement == placement, plan.reasons
    assert plan.te_kept_on_host is te_kept
    if placement == "stream":
        assert 10 <= plan.dit_resident_blocks < 50 and plan.te_resident_layers == 0
    if placement == "dit_resident":
        assert 0 <= plan.te_resident_layers <= 50


def test_auto_scheme_is_bf16_only_when_it_fits(h3i, monkeypatch):
    import torch

    assert h3i.resolve_dit_scheme(None, device = "cpu", vram_gb = None)[0] is None
    assert h3i.resolve_dit_scheme("int8", device = "cpu", vram_gb = None)[0] is None
    if torch.cuda.is_available():
        assert h3i.resolve_dit_scheme(None, device = "cuda", vram_gb = 39.5)[0] == "int8"
        assert h3i.resolve_dit_scheme(None, device = "cuda", vram_gb = 95.0)[0] is None


def test_streamer_runs_blocks_from_the_host(h3i):
    import torch

    blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    x = torch.randn(2, 4)
    want = blocks[2](blocks[1](blocks[0](x)))
    streamer = h3i.ModuleListStreamer(blocks, "cpu", resident = 1)
    assert streamer.streamed == 2
    got = blocks[2](blocks[1](blocks[0](x)))
    assert torch.equal(got, want)
    streamer.remove()


# -- Studio routing and the backend ---------------------------------------------------------------------------------


def test_family_routing_and_defaults():
    from core.inference import diffusion_families as df
    from core.inference import video_families as vf

    fam = df.detect_family("MiniMaxAI/MiniMax-H3-Image")
    assert fam is not None and fam.name == "minimax-h3-image"
    assert df.detect_family_by_pipeline_class("MiniMaxH3ImagePipeline") is fam
    assert df.default_generation_params("MiniMaxAI/MiniMax-H3-Image") == (6, 1.0)
    assert fam.reference and fam.reference_edit and fam.max_condition_images == 16
    assert not fam.uses_guidance and not fam.uses_negative_prompt
    assert fam.te_quant_auto is None and fam.dimension_multiple == 32
    assert df.in_tree_pipeline("MiniMaxH3ImagePipeline")
    # The video registry shares the "minimax-h3" token: it must not claim the image model.
    assert vf.detect_video_family("MiniMaxAI/MiniMax-H3-Image") is None
    assert "Images page" in (
        vf.unsupported_video_checkpoint_reason("MiniMaxAI/MiniMax-H3-Image") or ""
    )
    assert vf.detect_video_family("MiniMaxAI/MiniMax-H3").name == "minimax-h3"


def test_backend_loads_and_generates(h3i, release, tmp_path):
    import base64
    import io

    import torch
    from PIL import Image

    from core.inference.diffusion import DiffusionBackend

    root, _ref, te = release
    te.save_pretrained(str(root / "text_encoder"))
    backend = DiffusionBackend()
    try:
        status = backend.load_pipeline(str(root), model_kind = "pipeline")
        assert backend._state is not None and backend._state.family.name == "minimax-h3-image"
        assert backend._state.pipe._unsloth_h3_report.layout == "native"
        out = backend.generate(prompt = "a cat", width = 256, height = 256, steps = 6, guidance = 1.0, seed = 3)
        assert out["images"], out
        raw = out["images"][0]
        data = raw.get("b64") if isinstance(raw, dict) else raw
        if isinstance(data, str):
            img = Image.open(io.BytesIO(base64.b64decode(data.split(",")[-1])))
            assert img.size == (256, 256)
        assert status
    finally:
        backend.unload()


@pytest.mark.parametrize("layers", [51, 53])
def test_read_layer_hook_equals_hidden_states_50(h3i, release, layers):
    import torch

    import _minimax_h3_image_tiny as tiny

    _root, ref, te = release
    encoder = te if layers == 51 else tiny.make_text_encoder(ref["tokenizer"], layers = layers)
    ids = ref["tokenizer"]("a cat", add_special_tokens = False)["input_ids"]
    with torch.no_grad():
        got = h3i.qwen3vl_hidden_state(encoder, None, ids, {}, layer = 50, device = torch.device("cpu"))
        x = torch.tensor([ids])
        want = encoder.model(
            input_ids = x, attention_mask = torch.ones_like(x), output_hidden_states = True
        ).hidden_states[50]
    assert torch.equal(got, want)


def test_release_drops_components_and_hooks(h3i, release):
    import torch

    root, _ref, te = release
    import copy

    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = copy.deepcopy(te))
    blocks = p.transformer.transformer_blocks
    p._unsloth_h3_streamers.append(h3i.ModuleListStreamer(blocks, "cpu", resident = 0))
    assert any(b._forward_pre_hooks for b in blocks)
    p.release()
    assert p.transformer is None and p.text_encoder is None and p.vae is None
    assert not any(b._forward_pre_hooks for b in blocks)
    assert torch.is_tensor(next(blocks[0].parameters()))


def test_int8_linears_pad_rows_to_a_multiple_of_16_exactly(h3i):
    import torch

    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import _make_quant_config

    torch.manual_seed(0)
    block = torch.nn.Sequential(torch.nn.Linear(256, 64, bias = False)).to(torch.bfloat16)
    quantize_(block, _make_quant_config("int8"))
    x = torch.randn(1, 37, 256, dtype = torch.bfloat16)
    with torch.no_grad():
        want = block[0](x)
    assert h3i.wrap_row_multiple(block, ["0"]) == 1
    seen = []
    block[0].inner.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape[0]))
    with torch.no_grad():
        got = block[0](x)
    assert seen == [48] and got.shape == want.shape
    assert torch.equal(got, want)
