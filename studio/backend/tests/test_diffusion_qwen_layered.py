# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Layered on diffusers: repo detection, decomposition call and RGBA layer flattening."""

from __future__ import annotations

import base64
import io
import types

import pytest
from PIL import Image

from core.inference.diffusion import (
    DiffusionBackend,
    _compile_shape_dims,
    _family_workflows,
    _is_trusted_diffusion_repo,
)
from core.inference.diffusion_families import (
    comfy_flow_shift_for,
    default_generation_params,
    detect_family,
    detect_family_for_pick,
)

from .test_diffusion_backend import (  # noqa: F401
    _FakePipe,
    _FakeTransformer,
    _load_into,
    fake_runtime,
)

_LAYER_COLOURS = ((200, 10, 10, 255), (10, 200, 10, 0), (10, 10, 200, 128), (9, 9, 9, 9))


class _FakeLayeredPipe(_FakePipe):
    """QwenImageLayeredPipeline's call signature: no width / height, ``layers`` and ``resolution``,
    and one LIST of RGBA layers per input in ``images``."""

    calls: list = []

    def __call__(
        self,
        image = None,
        prompt = None,
        negative_prompt = None,
        true_cfg_scale = 4.0,
        layers = 4,
        num_inference_steps = 50,
        sigmas = None,
        guidance_scale = None,
        num_images_per_prompt = 1,
        generator = None,
        latents = None,
        output_type = "pil",
        return_dict = True,
        attention_kwargs = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ("latents",),
        max_sequence_length = 512,
        resolution = 640,
        cfg_normalize = False,
        use_en_prompt = False,
    ):
        self.last_kwargs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "layers": layers,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "resolution": resolution,
        }
        _FakeLayeredPipe.calls.append(self.last_kwargs)
        per_input = [
            [Image.new("RGBA", (32, 32), _LAYER_COLOURS[i % 4]) for i in range(layers)]
            for _ in range(num_images_per_prompt)
        ]
        return types.SimpleNamespace(images = per_input)


class _FakeLayeredPipeline:
    @classmethod
    def from_pretrained(cls, base, **kwargs):
        _FakeLayeredPipeline.base = base
        return _FakeLayeredPipe()


def _png(size = (96, 64), color = (120, 30, 30, 77)) -> str:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def layered(fake_runtime, tmp_path):
    import diffusers

    diffusers.QwenImageLayeredPipeline = _FakeLayeredPipeline
    diffusers.QwenImageTransformer2DModel = _FakeTransformer
    _FakeLayeredPipe.calls = []
    (tmp_path / "qwen-image-layered-UD-Q4_K_XL.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend,
        tmp_path,
        gguf_filename = "qwen-image-layered-UD-Q4_K_XL.gguf",
        base_repo = "Qwen/Qwen-Image-Layered",
        family_override = None,
    )
    return backend


@pytest.mark.parametrize(
    "repo_id",
    [
        "unsloth/Qwen-Image-Layered-GGUF",
        "Qwen/Qwen-Image-Layered",
        "unsloth/qwen_image_layered",
        "/models/qwen-image-layered-UD-Q4_K_XL.gguf",
    ],
)
def test_layered_ids_resolve_to_their_own_family(repo_id):
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == "qwen-image-layered"
    assert fam.pipeline_class == "QwenImageLayeredPipeline"
    assert fam.transformer_class == "QwenImageTransformer2DModel"
    assert fam.base_repo == "Qwen/Qwen-Image-Layered"


def test_layered_filename_in_a_neutral_folder_resolves():
    fam = detect_family_for_pick("/models/misc", "qwen-image-layered-BF16.gguf")
    assert fam is not None and fam.name == "qwen-image-layered"


def test_neighbouring_families_keep_their_detection():
    assert detect_family("unsloth/Qwen-Image-GGUF").name == "qwen-image"
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF").name == "qwen-image-edit"
    assert detect_family("unsloth/Qwen-Image-2.1-GGUF").name == "qwen-image-2.1"
    assert detect_family("unsloth/FLUX.1-dev-Layered-GGUF") is None
    assert detect_family("unsloth/Qwen-Image-2512-Inpaint") is None
    assert detect_family_for_pick("/models/misc", "z-image-layered-Q4.gguf") is None


def test_layered_family_contract():
    fam = detect_family("unsloth/Qwen-Image-Layered-GGUF")
    assert fam.edit is True and _family_workflows(fam) == ["edit"]
    assert fam.condition_image_mode == "RGBA"
    assert fam.cfg_kwarg == "true_cfg_scale"
    # ComfyUI Image to Layers template: 20 steps, cfg 2.5, AuraFlow shift 1, 2 layers at 640
    assert (fam.layer_count, fam.layer_resolution) == (2, 640)
    assert comfy_flow_shift_for(fam, "unsloth/Qwen-Image-Layered-GGUF") == 1.0
    for name in (
        "unsloth/Qwen-Image-Layered-GGUF",
        "Qwen/Qwen-Image-Layered",
        "local/qwen_image_layered",
        "local/qwenimagelayered-q4",
    ):
        assert default_generation_params(name) == (20, 2.5)
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF").layer_count == 0
    assert _is_trusted_diffusion_repo("Qwen/Qwen-Image-Layered")


def test_gguf_transformer_reads_the_layered_config(layered):
    assert layered._state.family.name == "qwen-image-layered"
    # the GGUF supplies weights only; transformer config, RGBA VAE and pipeline come from the base
    assert _FakeTransformer.last["config"] == "Qwen/Qwen-Image-Layered"
    assert _FakeTransformer.last["subfolder"] == "transformer"
    assert _FakeTransformer.last["path"].endswith("qwen-image-layered-UD-Q4_K_XL.gguf")
    assert _FakeLayeredPipeline.base == "Qwen/Qwen-Image-Layered"
    assert layered.status()["workflows"] == ["edit"]


def test_decomposition_call_and_layer_outputs(layered):
    out = layered.generate(
        prompt = "a cup on a table",
        steps = 20,
        guidance = 2.5,
        seed = 7,
        init_image = _png(color = (120, 30, 30, 77)),
    )
    call = layered._state.pipe.last_kwargs
    assert call["layers"] == 2 and call["resolution"] == 640
    assert call["true_cfg_scale"] == 2.5 and call["num_inference_steps"] == 20
    # true CFG needs a negative; Studio sends the empty one, as ComfyUI encodes
    assert call["negative_prompt"] == ""
    assert call["image"].mode == "RGBA" and call["image"].getpixel((0, 0)) == (120, 30, 30, 77)
    assert [im.mode for im in out["images"]] == ["RGBA", "RGBA"]
    assert [im.getpixel((0, 0)) for im in out["images"]] == list(_LAYER_COLOURS[:2])
    assert out["seeds"] == [7, 7] and out["seed"] == 7
    assert out["workflow"] == "edit"


def test_seed_batch_runs_one_decomposition_per_forward(layered):
    out = layered.generate(
        prompt = "a cup on a table",
        steps = 4,
        guidance = 2.5,
        seeds = [3, 9],
        init_image = _png(),
    )
    # the pipeline's per-prompt grouping is only right for one input, so one call per seed
    assert [c["num_images_per_prompt"] for c in _FakeLayeredPipe.calls] == [1, 1]
    assert len(out["images"]) == 4
    assert out["seeds"] == [3, 3, 9, 9]


def test_layered_needs_an_input_image(layered):
    with pytest.raises(ValueError, match = "input image"):
        layered.generate(prompt = "no image", steps = 4, guidance = 2.5, seed = 1)


@pytest.mark.parametrize(
    "size,expected",
    [((96, 64), (768, 512)), ((1000, 500), (896, 448)), ((640, 640), (640, 640))],
)
def test_registered_shape_is_the_decomposition_canvas(size, expected):
    fam = detect_family("unsloth/Qwen-Image-Layered-GGUF")
    src = Image.new("RGBA", size)
    # 640 x 640 area at the input aspect on the 32 px grid, as the pipeline sizes it
    assert _compile_shape_dims("edit", src, 1024, 1024, fam) == expected
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    assert _compile_shape_dims("edit", src, 1024, 1024, kontext) == size


def _saved(root, class_name):
    import json

    root.mkdir(parents = True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    return str(root)


def test_saved_layered_pipeline_resolves_and_a_contradicting_one_is_refused(tmp_path):
    real = _saved(tmp_path / "qwen-image-layered", "QwenImageLayeredPipeline")
    assert detect_family_for_pick(real).name == "qwen-image-layered"
    opaque = _saved(tmp_path / "0f3d1a2b4c5d", "QwenImageLayeredPipeline")
    assert detect_family_for_pick(opaque).name == "qwen-image-layered"
    # a layered name over a plain QwenImagePipeline must not route to the layered pipeline
    wrong = _saved(tmp_path / "x" / "qwen-image-layered", "QwenImagePipeline")
    assert detect_family_for_pick(wrong) is None


def test_gguf_packed_embedding_is_dequantised_and_linears_are_left_packed():
    torch = pytest.importorskip("torch")
    gguf = pytest.importorskip("gguf")
    utils = pytest.importorskip("diffusers.quantizers.gguf.utils")
    from core.inference.diffusion import _dequantize_gguf_outside_linears

    # GGUF keeps BF16 as raw bytes: a (2, 8) table is stored (2, 16) uint8
    table = torch.arange(16, dtype = torch.float32).reshape(2, 8).to(torch.bfloat16)
    raw = table.view(torch.uint8)
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(2, 8)
    model.embed.weight = utils.GGUFParameter(raw, quant_type = gguf.GGMLQuantizationType.BF16)
    linear = utils.GGUFLinear(8, 8, bias = False, compute_dtype = torch.bfloat16)
    linear.weight = utils.GGUFParameter(
        torch.zeros(8, 16, dtype = torch.uint8), quant_type = gguf.GGMLQuantizationType.BF16
    )
    model.proj = linear
    assert model.embed(torch.tensor([1])).shape[-1] == 16

    assert _dequantize_gguf_outside_linears(model, torch.bfloat16) == 1
    assert type(model.embed.weight) is torch.nn.Parameter
    assert model.embed.weight.dtype == torch.bfloat16 and tuple(model.embed.weight.shape) == (2, 8)
    assert torch.equal(model.embed(torch.tensor([1]))[0], table[1])
    # the quantizer's own linears dequantise per forward and stay packed
    assert isinstance(model.proj.weight, utils.GGUFParameter)
    assert _dequantize_gguf_outside_linears(model, torch.bfloat16) == 0


def test_load_planning_reserves_what_the_decomposition_guard_charges():
    # the guard charges layers + 1 canvas frames, which once refused default offloaded renders
    from core.inference import diffusion_memory as dm

    fam = detect_family("unsloth/Qwen-Image-Layered-GGUF")
    assert (dm._QWEN_LAYERED_LAYERS, dm._QWEN_LAYERED_CANVAS) == (
        fam.layer_count,
        fam.layer_resolution,
    )
    hint = "qwen-image-layered qwen-image-layered-Q4_K_M.gguf unsloth/Qwen-Image-Layered-GGUF"
    side = fam.layer_resolution
    extra = (fam.layer_count + 1) * side * side
    planned = dm.estimate_image_runtime_mib(width = None, height = None, family = hint)
    assert planned == dm.estimate_image_runtime_mib(
        width = side, height = side, family = hint, condition_pixels = extra
    )
    verdict = dm.image_activation_verdict(
        device_memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 10000, 24576),
        width = side,
        height = side,
        family = hint,
        source_driven = True,
        condition_pixels = extra,
    )
    assert verdict.action == dm.ACTIVATION_RUN
    assert dm.estimate_image_runtime_mib(width = None, height = None, family = "qwen-image") == 8192
