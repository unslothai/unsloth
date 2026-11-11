# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image on CPU with a tiny random release in the native layout: key conversion (round trip and the PR
converter's own functions), the fixed-grid AdaLN table against the dense model, packing of text + reference rows,
the canvas rule, the sampler, VAE encode/decode, the planner tiers, routing / family defaults and the backend load +
generate path."""

from __future__ import annotations

import hashlib
import importlib
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


def test_vendored_files_match_the_manifest():
    manifest = json.loads(
        (_BACKEND / "vendor" / "diffusers_minimax_h3_image_manifest.json").read_text(
            encoding = "utf-8"
        )
    )
    assert manifest["license"] == "Apache-2.0"
    assert len(manifest["upstream_commit"]) == 40
    for rel, digest in manifest["files"].items():
        data = (_BACKEND / "vendor" / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel
    for rel in manifest["files"]:
        if rel.endswith(".py") and rel not in manifest["guard_files"]:
            text = (_BACKEND / "vendor" / rel).read_text(encoding = "utf-8")
            assert "Licensed under the Apache License, Version 2.0" in text
            assert "\nfrom ." not in text


def test_vendored_directory_is_not_importable_as_a_package(h3i, monkeypatch):
    # native_tls appends vendor/ to sys.path; the guard keeps the copy reachable only through _load_vendored().
    monkeypatch.syspath_prepend(str(_BACKEND / "vendor"))
    monkeypatch.delitem(sys.modules, "diffusers_minimax_h3_image", raising = False)
    with pytest.raises(ImportError, match = "_load_vendored"):
        importlib.import_module("diffusers_minimax_h3_image.transformer_minimax_h3_image")
    mod = h3i._load_vendored("transformer_minimax_h3_image")
    assert mod.__name__ == "unsloth_studio_vendor_minimax_h3_image.transformer_minimax_h3_image"


def test_classes_resolve_to_diffusers_or_the_vendored_copy(h3i):
    t, v = h3i.transformer_class(), h3i.vae_class()
    assert t.__name__ == "MiniMaxH3ImageTransformer2DModel"
    assert v.__name__ == "AutoencoderKLMiniMaxH3Image"
    assert h3i.classes_source() in ("diffusers", "vendored")
    assert h3i.in_tree_class("MiniMaxH3DiTModel") is t
    assert h3i.in_tree_class("MiniMaxH3ImagePipeline") is h3i.MiniMaxH3ImagePipeline


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
        ).read_text(encoding = "utf-8")
    )
    cfg = h3i.transformer_config_from_native(real)
    assert cfg["num_layers"] == 50 and cfg["hidden_size"] == 5376 and cfg["ffn_dim"] == 14336
    assert cfg["in_channels"] == 24 and cfg["text_dim"] == 5120 and cfg["num_pdd_heads"] == 6
    assert cfg["freq_dim"] == 256 and cfg["rope_freq_dim"] == 16 and cfg["num_refiner_layers"] == 2
    with pytest.raises(ValueError):
        h3i.transformer_config_from_native({**real, "audio_enabled": True})


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
        (22.0, 39.0, 27.2, "stream", False),  # small host: the encoder loads per prompt
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


@pytest.mark.parametrize("vram, host", [(22.0, 30.0), (23.5, 32.0), (11.0, 40.0)])
def test_a_host_below_the_per_prompt_encoder_peak_is_refused_at_plan_time(h3i, vram, host):
    # Per-prompt loading still holds the whole encoder beside the host-side DiT blocks while it loads.
    with pytest.raises(RuntimeError, match = "free system memory"):
        h3i.plan_placement(
            vram_gb = vram, host_gb = host, dit_gb = h3i.H3_IMAGE_DIT_TABLE_INT8_GB, te_gb = 27.2
        )


def test_a_quantized_dit_without_the_adaln_table_budgets_the_dense_adaln(h3i):
    dense = h3i.H3_IMAGE_DIT_BF16_GB - h3i.H3_IMAGE_DIT_TABLE_BF16_GB
    assert h3i.dit_gb_for("int8") == h3i.H3_IMAGE_DIT_TABLE_INT8_GB
    assert h3i.dit_gb_for("int8", table = False) == pytest.approx(
        h3i.H3_IMAGE_DIT_TABLE_INT8_GB + dense
    )
    assert h3i.dit_gb_for(None, table = False) == pytest.approx(h3i.H3_IMAGE_DIT_BF16_GB)
    # A100 40 GB: the table-form int8 DiT stays resident, the dense-AdaLN one (~48 GB) has to stream.
    args = dict(vram_gb = 39.5, host_gb = 83.5, te_gb = 27.2)
    assert h3i.plan_placement(dit_gb = h3i.dit_gb_for("int8"), **args).placement == "dit_resident"
    assert (
        h3i.plan_placement(dit_gb = h3i.dit_gb_for("int8", table = False), **args).placement == "stream"
    )


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


def test_backend_loads_and_generates(h3i, release, tmp_path, monkeypatch):
    import base64
    import io

    import torch
    from PIL import Image

    from core.inference import diffusion_attention
    from core.inference.diffusion import DiffusionBackend

    # A real CPU load memoizes the SDPA probe process-wide; later tests run against a faked torch.
    monkeypatch.setattr(diffusion_attention, "_SDPA_PROBE_CACHE", {})
    monkeypatch.setenv(h3i.ALLOW_ANY_DEVICE_ENV, "1")

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

    supplied = copy.deepcopy(te)
    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = supplied)
    assert p._unsloth_h3_owned == {"transformer", "vae"}
    held_dit, held_vae = p.transformer, p.vae  # an outside reference that outlives the unload
    blocks = p.transformer.transformer_blocks
    p._unsloth_h3_streamers.append(h3i.ModuleListStreamer(blocks, "cpu", resident = 0))
    assert any(b._forward_pre_hooks for b in blocks)
    p.release()
    assert p.transformer is None and p.text_encoder is None and p.vae is None
    assert not any(b._forward_pre_hooks for b in blocks)
    assert torch.is_tensor(next(blocks[0].parameters()))
    # Owned components are emptied in place even while held; a supplied encoder is the caller's and stays intact.
    assert all(t.numel() == 0 for t in [*held_dit.parameters(), *held_vae.parameters()])
    assert all(
        a.numel() and torch.equal(a, b) for a, b in zip(supplied.parameters(), te.parameters())
    )


def test_freeing_storage_reaches_quantized_weights(h3i):
    import gc
    import weakref

    import torch

    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import _make_quant_config

    block = torch.nn.Sequential(torch.nn.Linear(256, 64)).to(torch.bfloat16)
    quantize_(block, _make_quant_config("int8"))
    weight = block[0].weight
    inner = [weakref.ref(getattr(weight, n)) for n in weight.__tensor_flatten__()[0]]
    assert h3i.free_module_storage(block) == 2
    gc.collect()
    assert weight.numel() == 0 and block[0].bias.numel() == 0
    assert all(ref() is None for ref in inner)


def test_int8_linears_pad_rows_to_the_supported_multiple_exactly(h3i):
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
    assert h3i.ROW_MULTIPLE == 32
    assert seen == [64] and got.shape == want.shape
    assert torch.equal(got, want)


def test_the_family_declares_the_hosted_dits(h3i):
    from core.inference.diffusion_families import (
        detect_family,
        family_prequant_filename,
        family_prequant_repo,
    )

    fam = detect_family("MiniMaxAI/MiniMax-H3-Image")
    assert fam.name == "minimax-h3-image"
    for scheme, filename in (
        ("int8", "MiniMax-H3-Image-INT8-ConvRot.safetensors"),
        ("fp8", "MiniMax-H3-Image-FP8.safetensors"),
    ):
        assert (
            family_prequant_repo(fam, scheme, "MiniMaxAI/MiniMax-H3-Image")
            == "unsloth/MiniMax-H3-Image-FP8"
        )
        assert family_prequant_filename(fam, scheme) == filename


def test_an_unpublished_hosted_dit_falls_back_without_fetching(h3i, monkeypatch):
    import huggingface_hub

    from core.inference import diffusion_prequant

    calls = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(huggingface_hub, "file_exists", lambda *a, **k: calls.append(a) or False)
    monkeypatch.setattr(
        diffusion_prequant, "load_prequantized_transformer", lambda *a, **k: pytest.fail("fetched")
    )
    h3i._hub_file_exists.cache_clear()
    kwargs = dict(
        base = "MiniMaxAI/MiniMax-H3-Image",
        device = "cpu",
        dtype = None,
        hf_token = None,
        cache_dir = None,
        keep_on_device = False,
    )
    assert h3i.load_hosted_prequant_transformer("int8", local_files_only = False, **kwargs) is None
    assert calls == [("unsloth/MiniMax-H3-Image-FP8", "MiniMax-H3-Image-INT8-ConvRot.safetensors")]
    # Offline: the cache alone decides, never the Hub.
    assert h3i.load_hosted_prequant_transformer("fp8", local_files_only = True, **kwargs) is None
    assert len(calls) == 1
    h3i._hub_file_exists.cache_clear()


def test_a_published_hosted_dit_loads_through_the_curve_hook(h3i, monkeypatch):
    import huggingface_hub

    from core.inference import diffusion_prequant

    seen = {}

    def _load(cls, base, source, **kwargs):
        seen.update(cls = cls, base = base, source = source, **kwargs)
        return "model"

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/cache/file")
    monkeypatch.setattr(diffusion_prequant, "load_prequantized_transformer", _load)
    got = h3i.load_hosted_prequant_transformer(
        "int8",
        base = "MiniMaxAI/MiniMax-H3-Image",
        device = "cpu",
        dtype = None,
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        keep_on_device = False,
    )
    assert got == "model"
    assert seen["source"].location == "unsloth/MiniMax-H3-Image-FP8"
    assert seen["source"].filename == "MiniMax-H3-Image-INT8-ConvRot.safetensors"
    assert seen["family"] == "minimax-h3-image" and seen["scheme"] == "int8"


def test_the_prepare_hook_installs_the_curve_form(h3i):
    import torch

    from _minimax_h3_image_tiny import TINY_TRANSFORMER_NATIVE

    cfg = h3i.transformer_config_from_native(TINY_TRANSFORMER_NATIVE)
    model = h3i.transformer_class()(**cfg).eval()
    prepare = h3i.h3_image_prepare_prequant_model()
    prepare(model, {"adaln_form": "curve", "curve_dim": "8", "curve_grid": "17"})
    assert model.transformer_blocks[0].adaln_proj.linear.in_features == 8
    assert model.norm_out.linear.in_features == 8
    assert tuple(model.time_embedder.table.shape) == (17, 8)
    # The pipeline's raw float32 timesteps (0.999 reference rows included) drive the curve directly.
    t = torch.tensor([0.25, 0.999], dtype = torch.float32)
    assert torch.equal(model.time_proj(t), t)
    dense = h3i.transformer_class()(**cfg).eval()
    prepare(dense, {})
    assert dense.transformer_blocks[0].adaln_proj.linear.in_features != 8


@pytest.mark.parametrize(
    "device, where",
    [("mps", "Apple Silicon"), ("cpu", "the CPU"), ("xpu", "this xpu device")],
)
def test_only_cuda_and_rocm_hosts_may_load(h3i, monkeypatch, device, where):
    monkeypatch.delenv(h3i.ALLOW_ANY_DEVICE_ENV, raising = False)
    assert h3i.h3_image_device_refusal("cuda") is None
    assert h3i.h3_image_device_refusal("cuda:1") is None
    refusal = h3i.h3_image_device_refusal(device)
    assert refusal.startswith("MiniMax-H3-Image needs an NVIDIA or AMD GPU") and where in refusal


def test_the_backend_refuses_a_cpu_host_before_loading(h3i, release, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.delenv(h3i.ALLOW_ANY_DEVICE_ENV, raising = False)
    root, _ref, te = release
    if not (root / "text_encoder" / "config.json").is_file():
        te.save_pretrained(
            str(root / "text_encoder")
        )  # a complete local pipeline, so only the device can refuse
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "needs an NVIDIA or AMD GPU"):
        backend.validate_load_request(str(root), model_kind = "pipeline")
    assert backend._state is None


@pytest.mark.parametrize(
    "repo_id",
    [
        "MiniMaxAI/MiniMax-H3-Image",
        "unsloth/MiniMax-H3-Image-FP8",
        "minimaxai/minimax_h3_image",
        "/models/MiniMax-H3-Image",
    ],
)
def test_h3_image_ids_route_to_the_image_family_not_the_video_one(repo_id):
    from core.inference import diffusion_families as df
    from core.inference import video_families as vf

    assert vf.detect_video_family(repo_id) is None
    assert "Images page" in (vf.unsupported_video_checkpoint_reason(repo_id) or "")
    assert df.detect_family(repo_id).name == "minimax-h3-image"


@pytest.mark.parametrize(
    "files",
    [
        {"model_index.json": {"_class_name": "MiniMaxH3ImagePipeline"}},
        {"model_index.json": {"_class_name": "Other", "_minimax_h3": {"partition": "image"}}},
        {
            "modular_model_index.json": {
                "_class_name": "X",
                "_blocks_class_name": "MiniMaxH3ImageBlocks",
            }
        },
        {"transformer_diffusers/config.json": {"_class_name": "MiniMaxH3ImageTransformer2DModel"}},
    ],
)
def test_a_folder_that_says_h3_image_is_never_a_video_pick(tmp_path, files):
    import json

    from core.inference import diffusion_families as df
    from core.inference import video_families as vf

    # Named only "h3", the video family's alias: the folder's own metadata decides.
    root = tmp_path / "h3"
    for name, payload in files.items():
        (root / name).parent.mkdir(parents = True, exist_ok = True)
        (root / name).write_text(json.dumps(payload))
    assert vf.is_minimax_h3_image_folder(str(root))
    assert vf.detect_video_family(str(root)) is None
    assert "Images page" in (vf.unsupported_video_checkpoint_reason(str(root)) or "")
    if (
        "model_index.json" in files
        and files["model_index.json"]["_class_name"] == "MiniMaxH3ImagePipeline"
    ):
        assert df.detect_family_for_pick(str(root)).name == "minimax-h3-image"


def test_the_h3_video_release_and_its_folders_stay_video(tmp_path):
    import json

    from core.inference import video_families as vf

    for repo_id in ("MiniMaxAI/MiniMax-H3", "unsloth/MiniMax-H3-FP8", "unsloth/MiniMax-H3-GGUF"):
        assert vf.detect_video_family(repo_id).name == "minimax-h3"
    root = tmp_path / "MiniMax-H3"
    root.mkdir()
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    assert not vf.is_minimax_h3_image_folder(str(root))
    assert vf.detect_video_family(str(root)).name == "minimax-h3"


def test_the_hosted_files_conditioning_path_runs_dense(h3i, monkeypatch):
    import torch

    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_quant_pad import is_quantized_linear
    from core.inference.diffusion_transformer_quant import _make_quant_config

    torch.manual_seed(0)
    model = torch.nn.Module()
    model.context_embedder = torch.nn.Linear(256, 64)
    model.token_refiner = torch.nn.Module()
    model.token_refiner.refiner_blocks = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
    model.to(torch.bfloat16)
    quantize_(model, _make_quant_config("int8"))
    want = model.context_embedder.weight.dequantize().to(torch.bfloat16)
    monkeypatch.setenv(h3i.QUANT_CONDITIONING_ENV, "1")
    assert h3i.densify_conditioning_path(model) == 0
    monkeypatch.delenv(h3i.QUANT_CONDITIONING_ENV)
    assert h3i.densify_conditioning_path(model) == 2
    assert not is_quantized_linear(model.context_embedder)
    assert not is_quantized_linear(model.token_refiner.refiner_blocks[0])
    assert torch.equal(model.context_embedder.weight, want)
    assert is_quantized_linear(model.transformer_blocks[0])


def _release_copy(root, dst, with_encoder):
    import shutil
    shutil.copytree(
        root, dst, ignore = None if with_encoder else shutil.ignore_patterns("text_encoder")
    )
    return dst


def _bf16_sized(h3i, monkeypatch, encoder_cls):
    """``_module_gb`` reports the 66.7 GB bf16 encoder for the tiny one; the planner calls are recorded."""
    real_gb, real_plan = h3i._module_gb, h3i.plan_placement
    sized = []
    monkeypatch.setattr(
        h3i,
        "_module_gb",
        lambda m: h3i.H3_IMAGE_TE_BF16_GB if isinstance(m, encoder_cls) else real_gb(m),
    )
    monkeypatch.setattr(
        h3i, "plan_placement", lambda **kw: sized.append(kw["te_gb"]) or real_plan(**kw)
    )
    return sized


def test_the_plan_is_redone_when_the_int8_encoder_falls_back_to_bf16(
    h3i, release, tmp_path, monkeypatch
):
    import copy

    root, _ref, te = release
    rel = _release_copy(root, tmp_path / "rel", with_encoder = False)
    monkeypatch.setattr(
        h3i, "load_h3_image_text_encoder", lambda scheme, **kw: (copy.deepcopy(te), None)
    )
    sized = _bf16_sized(h3i, monkeypatch, type(te))
    pipe = h3i.load_minimax_h3_image_pipeline(str(rel), device = "cpu", text_encoder_quant = "int8")
    assert sized == [h3i.H3_IMAGE_TE_INT8_GB, h3i.H3_IMAGE_TE_BF16_GB]
    assert pipe._unsloth_h3_plan.te_gb == h3i.H3_IMAGE_TE_BF16_GB
    assert "planned again" in pipe._unsloth_h3_plan.reasons[0]
    pipe.release()


def _no_hosted_int8(monkeypatch):
    """The hosted int8 encoder is unusable; a bf16 ``from_pretrained`` would be recorded (never allocated)."""
    import transformers
    from core.inference import video_minimax_h3_te

    monkeypatch.setattr(video_minimax_h3_te, "load_h3_quantized_text_encoder", lambda *a, **k: None)
    calls = []

    def _record(*a, **k):
        calls.append(a)
        raise AssertionError("the bf16 encoder was materialised")

    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, "from_pretrained", _record)
    return calls


def test_the_bf16_fallback_is_refused_before_it_allocates_when_the_host_cannot_hold_it(
    h3i, release, tmp_path, monkeypatch
):
    root, _ref, _te = release
    rel = _release_copy(root, tmp_path / "rel", with_encoder = False)
    calls = _no_hosted_int8(monkeypatch)
    # Room for the 27.2 GB int8 encoder, not for the 66.7 GB bf16 one.
    monkeypatch.setenv(h3i.HOST_BUDGET_ENV, "40")
    with pytest.raises(RuntimeError, match = "free system memory"):
        h3i.load_minimax_h3_image_pipeline(str(rel), device = "cpu", text_encoder_quant = "int8")
    assert calls == []


def test_a_local_dense_encoder_is_refused_before_it_allocates_when_the_host_cannot_hold_it(
    h3i, release, tmp_path, monkeypatch
):
    import transformers

    root, _ref, _te = release
    rel = _release_copy(root, tmp_path / "rel", with_encoder = False)
    folder = rel / "text_encoder"
    folder.mkdir(exist_ok = True)
    (folder / "config.json").write_text("{}")
    with open(folder / "model.safetensors", "wb") as f:  # sparse: 66 GB on paper, no disk used
        f.truncate(66 * 10**9)
    calls = []
    monkeypatch.setattr(
        transformers.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        lambda *a, **k: calls.append(a),
    )
    monkeypatch.setenv(h3i.HOST_BUDGET_ENV, "40")
    with pytest.raises(RuntimeError, match = "free system memory"):
        h3i.load_minimax_h3_image_pipeline(str(rel), device = "cpu", text_encoder_quant = "int8")
    assert calls == []


def test_the_per_prompt_encoder_refuses_the_bf16_fallback_before_it_allocates(h3i, monkeypatch):
    calls = _no_hosted_int8(monkeypatch)

    def refuse():
        raise RuntimeError("refused")

    with pytest.raises(RuntimeError, match = "refused"):
        h3i.load_h3_image_text_encoder(
            "int8",
            dtype = None,
            text_encoder_repo = "x/y",
            hf_token = None,
            cache_dir = None,
            local_files_only = True,
            fallback_check = refuse,
        )
    assert calls == []


def test_a_streamed_dit_is_placed_before_the_encoder_is_materialised(
    h3i, release, tmp_path, monkeypatch
):
    import copy

    root, _ref, te = release
    rel = _release_copy(root, tmp_path / "rel", with_encoder = False)
    real_plan = h3i.plan_placement

    def stream(**kw):
        plan = real_plan(**kw)
        plan.placement, plan.dit_resident_blocks = h3i.PLACE_STREAM, 1
        return plan

    seen = []

    def load_te(scheme, **kw):
        blocks = pipe_blocks[0]
        seen.append([bool(b._forward_pre_hooks) for b in blocks])
        return copy.deepcopy(te), "int8"

    pipe_blocks = []
    real_place = h3i._place_streamed_dit

    def place(transformer, *a, **k):
        pipe_blocks.append(transformer.transformer_blocks)
        return real_place(transformer, *a, **k)

    monkeypatch.setattr(h3i, "plan_placement", stream)
    monkeypatch.setattr(h3i, "_place_streamed_dit", place)
    monkeypatch.setattr(h3i, "load_h3_image_text_encoder", load_te)
    pipe = h3i.load_minimax_h3_image_pipeline(str(rel), device = "cpu", text_encoder_quant = "int8")
    assert len(pipe_blocks) == 1 and seen and seen[0][0] is False and all(seen[0][1:])
    assert [getattr(x, "_unsloth_role", None) for x in pipe._unsloth_h3_streamers].count(
        "transformer"
    ) == 1
    pipe.release()


def test_the_plan_is_redone_for_a_local_text_encoder_folder(h3i, release, tmp_path, monkeypatch):
    import transformers

    root, _ref, te = release
    rel = _release_copy(root, tmp_path / "rel", with_encoder = False)
    te.save_pretrained(str(rel / "text_encoder"))
    sized = _bf16_sized(h3i, monkeypatch, transformers.Qwen3VLForConditionalGeneration)
    pipe = h3i.load_minimax_h3_image_pipeline(str(rel), device = "cpu", text_encoder_quant = "int8")
    assert sized == [h3i.H3_IMAGE_TE_INT8_GB, h3i.H3_IMAGE_TE_BF16_GB]
    assert pipe._unsloth_h3_plan.te_gb == h3i.H3_IMAGE_TE_BF16_GB
    pipe.release()


def test_replanning_moves_to_a_tier_that_fits_and_refuses_when_none_does(h3i):
    args = {
        "vram_gb": 95.0,
        "host_gb": 177.0,
        "dit_gb": 21.9,
        "width": 2048,
        "height": 2048,
        "forced": None,
    }
    plan = h3i.plan_placement(te_gb = h3i.H3_IMAGE_TE_INT8_GB, **args)
    assert plan.placement == h3i.PLACE_RESIDENT
    # The whole bf16 encoder no longer fits beside the DiT: the DiT stays, the encoder streams.
    again = h3i.replan_for_encoder(plan, h3i.H3_IMAGE_TE_BF16_GB, args)
    assert again.placement == h3i.PLACE_DIT_RESIDENT and again.te_gb == h3i.H3_IMAGE_TE_BF16_GB
    assert h3i.replan_for_encoder(plan, h3i.H3_IMAGE_TE_INT8_GB + 0.5, args) is plan
    small = dict(args, vram_gb = 6.0)
    with pytest.raises(RuntimeError, match = "no placement fits"):
        h3i.replan_for_encoder(
            h3i.plan_placement(te_gb = 27.2, **small), h3i.H3_IMAGE_TE_BF16_GB, small
        )


def test_the_per_prompt_encoder_refuses_a_bf16_fallback(h3i, release, tmp_path, monkeypatch):
    import copy

    root, _ref, te = release
    rel = _release_copy(root, tmp_path / "rel", with_encoder = False)
    real_plan = h3i.plan_placement

    def low_host(**kw):
        plan = real_plan(**kw)
        plan.te_kept_on_host = False  # the low-host tier: the encoder loads for each prompt
        return plan

    monkeypatch.setattr(h3i, "plan_placement", low_host)
    pipe = h3i.load_minimax_h3_image_pipeline(str(rel), device = "cpu", text_encoder_quant = "int8")
    assert pipe.text_encoder is None and pipe._unsloth_h3_te_loader is not None
    monkeypatch.setattr(
        h3i, "load_h3_image_text_encoder", lambda scheme, **kw: (copy.deepcopy(te), None)
    )
    with pytest.raises(RuntimeError, match = "bf16 encoder it would fall back to"):
        pipe._unsloth_h3_te_loader()
    pipe.release()
