# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDXL single-file GGUFs (#11391): a stable-diffusion.cpp ``convert`` of a community SDXL checkpoint holds the UNet,
both text encoders and the VAE, with no ``general.architecture``. It lists on the Images page, validates for the
diffusers load, and a denoiser-only SDXL GGUF still gets a precise refusal. The fixture is the real header of
RealVisXL_V4.0 converted to q8_0 (names and shapes, trimmed to index 0 of every stack)."""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest

from core.inference import diffusion_content as dc
from core.inference import diffusion_families as df
from hub.services.models import catalog_classification as cc

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "sdxl_whole_pipeline_gguf_header.json").read_text(
        encoding = "utf-8"
    )
)
_SHAPES: dict = _FIXTURE["shapes"]
_UNET = {k: v for k, v in _SHAPES.items() if k.startswith("model.diffusion_model.")}
_NO_VAE = {k: v for k, v in _SHAPES.items() if not k.startswith("first_stage_model.")}


def _write_gguf(
    path: Path,
    shapes: dict,
    arch: str | None = None,
) -> Path:
    """A GGUF header with ``shapes`` and, like sd.cpp's convert, no metadata unless ``arch`` is given."""

    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = struct.pack("<IIQQ", 0x46554747, 3, len(shapes), 1 if arch else 0)
    if arch:
        out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    for name, shape in shapes.items():
        dims = list(reversed(shape))
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", 0, 0)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(out)
    return path


def test_fixture_is_the_three_part_sd_cpp_layout():
    tops = {k.split(".", 1)[0] for k in _SHAPES}
    assert tops == {"model", "conditioner", "first_stage_model"}
    assert "model.diffusion_model.label_emb.0.0.weight" in _SHAPES


def test_whole_pipeline_gguf_is_an_sdxl_checkpoint(tmp_path):
    path = _write_gguf(tmp_path / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    info = dc.inspect_checkpoint(str(path))
    assert (info.role, info.family, info.page, info.whole_pipeline) == (
        "dit",
        "sdxl",
        "image",
        True,
    )
    assert dc.whole_pipeline_gguf_family(str(path)) == "sdxl"


@pytest.mark.parametrize("shapes", [_UNET, _NO_VAE], ids = ["unet_only", "no_vae"])
def test_partial_sdxl_gguf_is_not_a_whole_pipeline(tmp_path, shapes):
    path = _write_gguf(tmp_path / "sdxl_unet-q8_0.gguf", shapes)
    info = dc.inspect_checkpoint(str(path))
    assert info.family == "sdxl" and not info.whole_pipeline
    assert dc.whole_pipeline_gguf_family(str(path)) is None


def test_whole_pipeline_safetensors_still_reports_its_layout(tmp_path):
    # The flag rides on CheckpointInfo for every format; a .safetensors path never takes the GGUF route.
    header = {k: {"dtype": "F16", "shape": v, "data_offsets": [0, 0]} for k, v in _SHAPES.items()}
    raw = json.dumps(header).encode()
    path = tmp_path / "RealVisXL_V4.0.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert dc.inspect_checkpoint(str(path)).whole_pipeline
    assert dc.whole_pipeline_gguf_family(str(path)) is None


def test_whole_pipeline_gguf_lists_on_the_images_page(tmp_path):
    path = _write_gguf(tmp_path / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    # Before: no architecture key fell through to the name, which reads as a chat model (task None).
    assert cc._gguf_file_task(path, ()) == "text-to-image"


def test_denoiser_only_sdxl_gguf_keeps_its_old_classification(tmp_path):
    path = _write_gguf(tmp_path / "sdxl_unet-q8_0.gguf", _UNET)
    assert cc._gguf_file_task(path, ()) == cc._arch_to_task(None, name_hints = ())


def test_llm_gguf_classification_is_unchanged(tmp_path):
    path = _write_gguf(
        tmp_path / "Qwen3-0.6B-Q8_0.gguf", {"token_embd.weight": [151936, 1024]}, "qwen3"
    )
    assert cc._gguf_file_task(path, ()) == "text-generation"
    # An arch-less GGUF that is not a diffusion checkpoint falls back exactly as before.
    other = _write_gguf(tmp_path / "mystery.gguf", {"blk.0.attn_q.weight": [8, 8]})
    assert cc._gguf_file_task(other, ()) == cc._arch_to_task(None, name_hints = ())


def test_validation_accepts_a_whole_pipeline_gguf(tmp_path):
    from core.inference.diffusion import get_diffusion_backend

    _write_gguf(tmp_path / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    fam = get_diffusion_backend().validate_load_request(
        str(tmp_path), gguf_filename = "RealVisXL_V4.0-q8_0.gguf", model_kind = "gguf"
    )
    assert fam.name == "sdxl" and fam.single_file_is_pipeline


def test_comfy_checkpoints_layout_validates_too(tmp_path):
    from core.inference.diffusion import get_diffusion_backend

    folder = tmp_path / "ComfyUI" / "models" / "checkpoints"
    _write_gguf(folder / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    fam = get_diffusion_backend().validate_load_request(
        str(folder), gguf_filename = "RealVisXL_V4.0-q8_0.gguf", model_kind = "gguf"
    )
    assert fam.name == "sdxl"


@pytest.mark.parametrize("shapes", [_UNET, _NO_VAE], ids = ["unet_only", "no_vae"])
def test_validation_refuses_a_partial_sdxl_gguf_precisely(tmp_path, shapes):
    from core.inference.diffusion import get_diffusion_backend
    _write_gguf(tmp_path / "sdxl_unet-q8_0.gguf", shapes)
    with pytest.raises(ValueError, match = "also carries the text encoders and VAE"):
        get_diffusion_backend().validate_load_request(
            str(tmp_path), gguf_filename = "sdxl_unet-q8_0.gguf", model_kind = "gguf"
        )


def test_validation_refuses_a_hub_sdxl_gguf(tmp_path):
    from core.inference.diffusion import get_diffusion_backend
    with pytest.raises(ValueError, match = "on disk"):
        get_diffusion_backend().validate_load_request(
            "someone/sdxl-gguf",
            gguf_filename = "sdxl-q8_0.gguf",
            model_kind = "gguf",
            family_override = "sdxl",
        )


def test_the_load_branch_routes_whole_pipeline_ggufs_to_the_bundle_loader():
    src = (Path(__file__).parents[1] / "core" / "inference" / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    branch = src[
        src.index('elif kind in ("single_file", "gguf") and fam.single_file_is_pipeline:') :
    ]
    branch = branch[: branch.index("\n                        else:\n")]
    assert "load_whole_pipeline_gguf(" in branch
    assert "_dequantize_gguf_outside_linears(" in branch
    # Auto precision must not take the dense fast path, which would load the base repo's UNet instead.
    assert 'if kind == "gguf" and fam.single_file_is_pipeline:\n' in src


def test_bundle_loader_redirects_only_its_own_file(monkeypatch):
    from core.inference import diffusion_gguf_pipeline as gp

    reads = []

    def real_read(link, *args, **kwargs):
        reads.append(link)
        return {
            "model.diffusion_model.input_blocks.0.0.weight": "unet",
            "conditioner.embedders.0.transformer.x": "te",
            "first_stage_model.decoder.conv_in.weight": "vae",
        }

    single_file = types.ModuleType("diffusers.loaders.single_file")
    single_file.load_single_file_checkpoint = real_read
    utils = types.ModuleType("diffusers.loaders.single_file_utils")
    utils.load_single_file_checkpoint = real_read
    gguf_utils = types.ModuleType("diffusers.quantizers.gguf.utils")
    gguf_utils.GGUFParameter = type("GGUFParameter", (), {})
    gguf_utils.dequantize_gguf_tensor = lambda t: t
    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype: ("gguf", compute_dtype)
    for name, mod in {
        "diffusers": diffusers,
        "diffusers.loaders": types.ModuleType("diffusers.loaders"),
        "diffusers.loaders.single_file": single_file,
        "diffusers.loaders.single_file_utils": utils,
        "diffusers.quantizers": types.ModuleType("diffusers.quantizers"),
        "diffusers.quantizers.gguf": types.ModuleType("diffusers.quantizers.gguf"),
        "diffusers.quantizers.gguf.utils": gguf_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    diffusers.loaders = sys.modules["diffusers.loaders"]
    sys.modules["diffusers.loaders"].single_file = single_file

    seen = {}

    class Unet:
        @classmethod
        def from_single_file(cls, sd, **kwargs):
            seen["unet_sd"], seen["unet_kwargs"] = sd, kwargs
            return "UNET"

    class Pipe:
        @classmethod
        def from_single_file(cls, path, **kwargs):
            seen["pipe_kwargs"] = kwargs
            seen["pipe_read"] = single_file.load_single_file_checkpoint(path)
            seen["other_read"] = single_file.load_single_file_checkpoint("/elsewhere.safetensors")
            return "PIPE"

    out = gp.load_whole_pipeline_gguf(
        Pipe,
        Unet,
        "/m/RealVisXL_V4.0-q8_0.gguf",
        {
            "config": "stabilityai/stable-diffusion-xl-base-1.0",
            "local_files_only": True,
            "torch_dtype": "bf16",
        },
        dtype = "bf16",
    )
    assert out == "PIPE"
    assert list(seen["unet_sd"]) == ["model.diffusion_model.input_blocks.0.0.weight"]
    assert seen["unet_kwargs"]["subfolder"] == "unet"
    assert seen["unet_kwargs"]["quantization_config"] == ("gguf", "bf16")
    assert seen["unet_kwargs"]["local_files_only"] is True
    assert seen["pipe_kwargs"]["unet"] == "UNET"
    assert set(seen["pipe_read"]) == {
        "conditioner.embedders.0.transformer.x",
        "first_stage_model.decoder.conv_in.weight",
    }
    # Another file read during the load reaches the real reader, and the swap is undone afterwards.
    assert reads == ["/m/RealVisXL_V4.0-q8_0.gguf", "/elsewhere.safetensors"]
    assert single_file.load_single_file_checkpoint is real_read


def test_family_detection_for_the_pick_is_sdxl(tmp_path):
    _write_gguf(tmp_path / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    fam = df.detect_family_for_pick(str(tmp_path), "RealVisXL_V4.0-q8_0.gguf")
    assert fam is not None and fam.name == "sdxl"


def test_whole_pipeline_gguf_stages_only_the_base_config(tmp_path, monkeypatch):
    import huggingface_hub

    from core.inference.diffusion import DiffusionBackend

    _write_gguf(tmp_path / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    names = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "tokenizer/vocab.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    listing = types.SimpleNamespace(
        siblings = [types.SimpleNamespace(rfilename = n, size = 10**9) for n in names], sha = "abc"
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: types.SimpleNamespace(model_info = lambda *a, **k: listing),
    )
    _total, files = DiffusionBackend._estimate_download_bytes(
        str(tmp_path),
        "RealVisXL_V4.0-q8_0.gguf",
        "stabilityai/stable-diffusion-xl-base-1.0",
        None,
        kind = "gguf",
        single_file_is_pipeline = True,
    )
    # The GGUF carries the UNet, text encoders and VAE: no base weight is staged.
    assert sorted(files) == ["model_index.json", "tokenizer/vocab.json", "unet/config.json"]


def test_whole_pipeline_gguf_resident_prices_dequantized_components(tmp_path, monkeypatch):
    from core.inference import diffusion_gguf_pipeline as gp

    q8, f16 = object(), object()
    tensors = [
        # quantized UNet linear: stays packed (100 bytes)
        types.SimpleNamespace(
            name = "model.diffusion_model.a.weight",
            shape = [8, 8],
            tensor_type = q8,
            n_bytes = 100,
            n_elements = 64,
        ),
        # quantized UNet conv and a quantized text encoder linear: both dequantized
        types.SimpleNamespace(
            name = "model.diffusion_model.c.weight",
            shape = [2, 2, 2, 2],
            tensor_type = q8,
            n_bytes = 30,
            n_elements = 16,
        ),
        types.SimpleNamespace(
            name = "conditioner.embedders.0.w",
            shape = [8, 8],
            tensor_type = q8,
            n_bytes = 70,
            n_elements = 64,
        ),
    ]
    fake = types.ModuleType("gguf")
    fake.GGMLQuantizationType = types.SimpleNamespace(F32 = f16, F16 = f16, BF16 = f16)
    fake.GGUFReader = lambda path: types.SimpleNamespace(tensors = tensors)
    monkeypatch.setitem(sys.modules, "gguf", fake)
    mib = 1024 * 1024
    for t in tensors:
        t.n_bytes *= mib
        t.n_elements *= mib
    # 100 packed + (16 + 64) * 2 dense = 260 MiB, plus the 5% margin
    assert gp.whole_pipeline_gguf_resident_mib("x.gguf") == int(260 * 1.05)
    assert gp.whole_pipeline_gguf_resident_mib("x.gguf", dense_bytes = 4) == int(420 * 1.05)
    monkeypatch.setattr(fake, "GGUFReader", lambda path: (_ for _ in ()).throw(OSError("bad")))
    assert gp.whole_pipeline_gguf_resident_mib("x.gguf") is None


def test_status_reports_the_header_variant_recipe(tmp_path):
    from core.inference.diffusion import _generation_defaults_for

    # A renamed FLUX.1-dev whose base resolves to schnell: the header variant decides, as the OpenAI route does.
    shapes = {
        "double_blocks.0.img_attn.qkv.weight": [9216, 3072],
        "single_blocks.0.linear1.weight": [21504, 3072],
        "vector_in.in_layer.weight": [3072, 768],
        "img_in.weight": [3072, 64],
        "guidance_in.in_layer.weight": [3072, 256],
    }
    header = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    raw = json.dumps(header).encode()
    (tmp_path / "my_flux.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert dc.content_variant_hint(str(tmp_path), "my_flux.safetensors") == "flux.1-dev"
    got = _generation_defaults_for(
        str(tmp_path), "my_flux.safetensors", "black-forest-labs/FLUX.1-schnell"
    )
    assert (got["steps"], got["guidance"]) == df.default_generation_params("flux.1-dev")
    assert (got["steps"], got["guidance"]) != df.default_generation_params(
        "black-forest-labs/FLUX.1-schnell"
    )
    # SDXL fine-tune with an opaque name: its base names the family.
    _write_gguf(tmp_path / "RealVisXL_V4.0-q8_0.gguf", _SHAPES)
    got = _generation_defaults_for(
        str(tmp_path), "RealVisXL_V4.0-q8_0.gguf", "stabilityai/stable-diffusion-xl-base-1.0"
    )
    assert got == {"steps": 25, "guidance": 7.0}
