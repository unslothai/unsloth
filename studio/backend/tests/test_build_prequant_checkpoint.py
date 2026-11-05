# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ConvRot buildability and filename are checked early, as a wrong answer fails only after the build."""

import importlib.util
import sys
from pathlib import Path

import types

import pytest

from core.inference.diffusion_convrot import rotation_metadata, rotation_metadata_error
from core.inference.diffusion_families import detect_family
from core.inference.video_families import detect_video_family

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_prequant_checkpoint.py"


def _script():
    """The build script as a module. Imported by path: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("build_prequant_checkpoint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_convrot_group_that_rotates_nothing_is_refused_before_anything_is_built():
    build = _script()
    refusal = build.convrot_refusal(4096, (), ("blocks.0.ff.net.0", "blocks.0.attn.to_q"))
    assert refusal is not None
    assert "4096" in refusal and "2" in refusal
    assert rotation_metadata_error(rotation_metadata(4096, ())) is not None
    assert build.convrot_refusal(256, ("blocks.0.attn.to_q",), ()) is None


def test_a_rotated_upload_goes_to_the_name_the_loader_asks_for_not_the_legacy_fallback():
    build = _script()
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert h3 is not None
    # Rotated INT8 is published under the family's declared name, asked for first.
    assert build.upload_destination(h3, "int8", rotated = True) == "MiniMax-H3-INT8-ConvRot.pt"
    assert build.upload_destination(h3, "int8", rotated = False) == "transformer_int8.pt"


def test_a_rotated_upload_with_no_declared_name_is_refused_rather_than_published_over_the_fallback():
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None
    # A v2 artifact under the legacy name would be picked up by older builds.
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(zimage, "int8", rotated = True)
    assert (
        build.upload_destination(
            zimage, "int8", rotated = True, override = "Z-Image-Turbo-INT8-ConvRot.pt"
        )
        == "Z-Image-Turbo-INT8-ConvRot.pt"
    )
    assert build.upload_destination(zimage, "fp8", rotated = False) == "transformer_fp8.pt"


def test_a_safetensors_upload_needs_a_name_the_loader_would_ask_for():
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None
    # Derived names end in .pt, so a safetensors artifact there is unreachable.
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(zimage, "fp8", rotated = False, safetensors = True)
    assert (
        build.upload_destination(
            zimage,
            "fp8",
            rotated = False,
            safetensors = True,
            override = "Z-Image-Turbo-FP8.safetensors",
        )
        == "Z-Image-Turbo-FP8.safetensors"
    )


def test_a_safetensors_build_refuses_a_declared_name_that_reads_as_a_pickle():
    build = _script()
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert h3 is not None
    # A safetensors file under a .pt name fails to load for a misleading reason.
    with pytest.raises(ValueError, match = r"does not end in '\.safetensors'"):
        build.upload_destination(h3, "int8", rotated = False, safetensors = True)


def test_a_rotated_pickle_build_refuses_a_declared_name_that_reads_as_safetensors():
    """A pickle build must not take a declared safetensors name, since loaders dispatch on extension."""
    build = _script()
    fam = types.SimpleNamespace(
        name = "qwen-image-2.1",
        prequant_filenames = (("fp8", "Qwen-Image-2.1-FP8.safetensors"),),
    )
    with pytest.raises(ValueError, match = r"does not end in '\.pt'"):
        build.upload_destination(fam, "fp8", rotated = True, safetensors = False)


def test_an_override_still_has_to_match_the_container_it_is_naming():
    """An override still has to match the container: a pickle published as .safetensors can never open."""
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None

    with pytest.raises(ValueError, match = r"does not end in '\.safetensors'"):
        build.upload_destination(
            zimage, "fp8", rotated = False, safetensors = True, override = "Z-Image-Turbo-FP8.pt"
        )
    with pytest.raises(ValueError, match = r"does not end in '\.pt'"):
        build.upload_destination(
            zimage,
            "fp8",
            rotated = True,
            safetensors = False,
            override = "Z-Image-Turbo-FP8.safetensors",
        )
    assert (
        build.upload_destination(
            zimage, "fp8", rotated = True, override = "Z-Image-Turbo-FP8-ConvRot.pt"
        )
        == "Z-Image-Turbo-FP8-ConvRot.pt"
    )
    assert (
        build.upload_destination(
            zimage,
            "fp8",
            rotated = False,
            safetensors = True,
            override = "Z-Image-Turbo-FP8.safetensors",
        )
        == "Z-Image-Turbo-FP8.safetensors"
    )


def test_a_plain_safetensors_build_derives_the_name_the_loader_now_asks_for_first():
    """Derived names lead with .safetensors, so a family without a declared entry needs no override."""
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None

    name = build.upload_destination(
        zimage,
        "fp8",
        rotated = False,
        safetensors = True,
        upload_repo = "unsloth/Z-Image-Turbo-FP8",
    )
    assert name == "Z-Image-Turbo-FP8.safetensors", name
    from core.inference.diffusion_prequant import derived_prequant_filenames

    assert derived_prequant_filenames("unsloth/Z-Image-Turbo-FP8", "fp8")[0] == name

    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(
            zimage,
            "fp8",
            rotated = True,
            safetensors = True,
            upload_repo = "unsloth/Z-Image-Turbo-FP8",
        )
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(zimage, "fp8", rotated = False, safetensors = True)


def test_the_recorded_base_must_be_the_canonical_id_not_just_the_same_tail(capsys, monkeypatch):
    """The recorded base must be the canonical id, not just the same tail; refuse before any download."""
    # Stub diffusers only when absent: the refusal happens before it is used.
    pytest.importorskip("torchao")
    if importlib.util.find_spec("diffusers") is None:
        monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    build = _script()
    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam is not None and fam.base_repo == "Qwen/Qwen-Image-2.1"

    argv = [
        "--base",
        "./temp/qwen_image_21",
        "--family",
        "qwen-image-2.1",
        "--scheme",
        "int8",
        "--out",
        "/nonexistent/out.pt",
        "--base-model-id",
        "other/Qwen-Image-2.1",
    ]
    assert build.main(argv) == 2
    message = capsys.readouterr().out
    assert "other/Qwen-Image-2.1" in message and "Qwen/Qwen-Image-2.1" in message

    # Accepted means the run gets past this guard, not that it succeeds.
    for accepted in ("Qwen/Qwen-Image-2.1", "  Qwen/Qwen-Image-2.1  "):
        try:
            rc = build.main(argv[:-1] + [accepted])
        except AttributeError as exc:
            assert fam.transformer_class in str(exc), exc
            continue
        assert rc != 2 or "is not" not in capsys.readouterr().out, accepted


def _build_nvfp4(
    monkeypatch,
    tmp_path,
    fam = None,
    extra_argv = (),
    loaded = None,
):
    """Run the build on a tiny aligned + ragged dense model; returns (quantized names, metadata)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    import torchao.quantization as tq

    import core.inference.diffusion_transformer_quant as dtq

    fam = fam or detect_family("Tongyi-MAI/Z-Image-Turbo")
    assert fam is not None

    class _Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aligned = torch.nn.Linear(1024, 1024, dtype = torch.bfloat16)
            self.ragged = torch.nn.Linear(1024, 1000, dtype = torch.bfloat16)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            if loaded is not None:
                loaded.append(kwargs.get("subfolder"))
            return cls()

        def to(self, *args, **kwargs):
            return self

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.__version__ = "0"
    setattr(fake_diffusers, fam.transformer_class, _Dense)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    quantized = []

    def _fake_quantize(model, config, filter_fn):
        quantized.extend(n for n, m in model.named_modules() if n and filter_fn(m, n))

    monkeypatch.setattr(tq, "quantize_", _fake_quantize)
    monkeypatch.setattr(dtq, "_make_quant_config", lambda *a, **k: object())

    build = _script()
    out = tmp_path / "out.pt"
    argv = ["--base", "b", "--family", fam.name, "--scheme", "nvfp4", "--out", str(out)]
    assert build.main(argv + list(extra_argv)) == 0
    return quantized, torch.load(out, weights_only = False)["metadata"]


def test_the_build_skips_ragged_linears_and_records_the_alignment_floor(monkeypatch, tmp_path):
    """A build skips Linears off the GEMM tiling floor and stamps ``require_divisible``."""
    import core.inference.diffusion_transformer_quant as dtq

    quantized, meta = _build_nvfp4(monkeypatch, tmp_path)
    assert quantized == ["aligned"]
    assert meta["require_divisible"] == dtq.divisible_for_scheme("nvfp4") == 16


def test_the_build_stamps_its_denoiser_so_it_cannot_load_as_the_second_expert(
    monkeypatch, tmp_path
):
    """The build reads ``subfolder="transformer"``; published under a transformer_2 name, the loader must refuse it."""
    from core.inference.diffusion_prequant import PREQUANT_FORMAT, _validate_checkpoint

    _, meta = _build_nvfp4(monkeypatch, tmp_path)
    assert meta["component"] == "transformer"
    ckpt = {"format": PREQUANT_FORMAT, "metadata": {**meta, "base_model_id": ""}, "state_dict": {}}
    assert _validate_checkpoint(ckpt, "nvfp4", "", None, component = "transformer")
    assert not _validate_checkpoint(ckpt, "nvfp4", "", None, component = "transformer_2")


def test_the_second_expert_builds_from_its_own_subfolder_and_loads_as_transformer_2(
    monkeypatch, tmp_path
):
    """``--component transformer_2`` reads that subfolder, stamps it, and publishes under the name the loader asks for."""
    from core.inference.diffusion_prequant import PREQUANT_FORMAT, _validate_checkpoint

    wan = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = "wan2.2-t2v-a14b")
    assert wan is not None
    loaded = []
    _, meta = _build_nvfp4(
        monkeypatch, tmp_path, fam = wan, extra_argv = ["--component", "transformer_2"], loaded = loaded
    )
    assert loaded == ["transformer_2"]
    assert meta["component"] == "transformer_2" and meta["family"] == wan.name
    ckpt = {"format": PREQUANT_FORMAT, "metadata": {**meta, "base_model_id": ""}, "state_dict": {}}
    assert _validate_checkpoint(ckpt, "nvfp4", "", None, component = "transformer_2")
    assert not _validate_checkpoint(ckpt, "nvfp4", "", None, component = "transformer")

    build = _script()
    assert (
        build.upload_destination(wan, "nvfp4", rotated = False, component = "transformer_2")
        == "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"
    )
    assert build.upload_destination(wan, "nvfp4", rotated = False) == "transformer_nvfp4.pt"
    with pytest.raises(ValueError, match = "transformer_2"):
        build.upload_destination(
            detect_family("Tongyi-MAI/Z-Image-Turbo"),
            "nvfp4",
            rotated = False,
            component = "transformer_2",
        )


def test_a_default_build_still_reads_the_transformer_subfolder(monkeypatch, tmp_path):
    loaded = []
    _, meta = _build_nvfp4(monkeypatch, tmp_path, loaded = loaded)
    assert loaded == ["transformer"] and meta["component"] == "transformer"
