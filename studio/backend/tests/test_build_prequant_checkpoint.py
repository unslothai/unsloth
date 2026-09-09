# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two decisions ``scripts/build_prequant_checkpoint.py`` makes that a bad answer to is
silent: whether a ConvRot build is buildable at all, and what filename it publishes under.

Both end in an artifact that costs GPU-hours and tens of gigabytes and then cannot be resolved
or cannot be loaded, with nothing at build time saying so, which is why they are pulled out as
pure functions and asserted here rather than left inline in ``main``."""

import importlib.util
import sys
import types
from pathlib import Path

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
    # The group divides no quantized input axis, so the rotation would be empty.
    refusal = build.convrot_refusal(4096, (), ("blocks.0.ff.net.0", "blocks.0.attn.to_q"))
    assert refusal is not None
    assert "4096" in refusal and "2" in refusal
    # And the reason it has to be refused: the artifact it would have written is unloadable.
    assert rotation_metadata_error(rotation_metadata(4096, ())) is not None
    # A non-empty set is built normally.
    assert build.convrot_refusal(256, ("blocks.0.attn.to_q",), ()) is None


def test_a_rotated_upload_goes_to_the_name_the_loader_asks_for_not_the_legacy_fallback():
    build = _script()
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert h3 is not None
    # The rotated INT8 denoiser is published under the family's declared name, which is the one
    # resolve_prequant_source asks for first. transformer_int8.pt is never asked for on this
    # family, so an upload landing there would be invisible.
    assert build.upload_destination(h3, "int8", rotated = True) == "MiniMax-H3-INT8-ConvRot.pt"
    # A plain build goes to the DERIVED name, which is the one resolve_prequant_source asks for
    # first; the legacy transformer_int8.pt stays resolvable as the loader's fallback for the
    # repos that only ever carried it.
    assert (
        build.upload_destination(h3, "int8", rotated = False, repo_id = "unsloth/MiniMax-H3-FP8")
        == "MiniMax-H3-INT8.pt"
    )


def test_a_rotated_upload_with_no_declared_name_is_refused_rather_than_published_over_the_fallback():
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None
    # Publishing a v2 artifact under transformer_<scheme>.pt hands it to every OLDER build as the
    # fallback, which refuses the tag and drops to the dense download. Refuse instead.
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(zimage, "int8", rotated = True)
    # An explicit name is the operator's escape hatch, rotated or not.
    assert (
        build.upload_destination(
            zimage, "int8", rotated = True, override = "Z-Image-Turbo-INT8-ConvRot.pt"
        )
        == "Z-Image-Turbo-INT8-ConvRot.pt"
    )
    assert (
        build.upload_destination(zimage, "fp8", rotated = False, repo_id = "unsloth/Z-Image-Turbo-FP8")
        == "Z-Image-Turbo-FP8.pt"
    )
    # A plain build derives its name from the destination repo, so publishing without one is
    # refused rather than guessed at.
    with pytest.raises(ValueError, match = "--upload-repo"):
        build.upload_destination(zimage, "fp8", rotated = False)


# ── the publishing gate ──────────────────────────────────────────────────────────
def test_a_build_may_not_publish_without_a_second_build_to_verify_against():
    build = _script()
    # An unverified artifact is indistinguishable from a verified one once it is hosted, and every
    # auto pick that resolves the repo then loads it. There is deliberately no escape hatch flag.
    refusal = build.upload_gate_refusal("unsloth/Wan2.2-TI2V-5B-NVFP4", None)
    assert refusal is not None and "--verify-against" in refusal
    assert build.upload_gate_refusal("unsloth/Wan2.2-TI2V-5B-NVFP4", "other.pt") is None
    # Nothing to gate when the build is not publishing.
    assert build.upload_gate_refusal(None, None) is None


def test_verifying_an_artifact_against_itself_is_refused(tmp_path):
    build = _script()
    out = tmp_path / "build_a.pt"
    out.write_bytes(b"x")
    link = tmp_path / "same.pt"
    link.symlink_to(out)
    # A file always matches itself, so accepting this would report a verified build and publish it.
    assert build.verify_target_refusal(str(out), str(out)) is not None
    assert build.verify_target_refusal(str(out), str(link)) is not None
    assert build.verify_target_refusal(str(out), str(tmp_path / "build_b.pt")) is None
    assert build.verify_target_refusal(str(out), None) is None


def test_the_fingerprint_diff_names_every_differing_weight_and_where_it_sits():
    build = _script()
    mine = {"count": 2, "modules": {"blocks.3.attn1.to_q.weight": "aa", "proj_out.weight": "bb"}}
    other = {"count": 2, "modules": {"blocks.3.attn1.to_q.weight": "cc", "proj_out.weight": "bb"}}
    assert build.fingerprint_mismatches(mine, other) == ["blocks.3.attn1.to_q.weight"]
    # A weight one build quantised and the other did not is a difference too: not the same artifact.
    fewer = {"count": 1, "modules": {"proj_out.weight": "bb"}}
    assert build.fingerprint_mismatches(mine, fewer) == ["blocks.3.attn1.to_q.weight"]
    assert build.fingerprint_mismatches(mine, mine) == []
    # The position is what discriminates between a per-shape effect and a global one.
    assert build.parse_key("transformer_2/blocks.12.attn1.to_q.weight") == (
        "transformer_2",
        12,
        "attn1.to_q.weight",
    )
    assert build.parse_key("proj_out.weight") == (None, None, "proj_out.weight")
    described = build.describe_key("blocks.12.attn1.to_q.weight")
    assert "block 12" in described and "attn1.to_q.weight" in described


def test_a_verify_against_mismatch_exits_3(monkeypatch, capsys):
    build = _script()
    mine = {"count": 2, "modules": {"blocks.3.attn1.to_q.weight": "aa", "proj_out.weight": "bb"}}
    monkeypatch.setattr(
        build,
        "_read_metadata",
        lambda path: {
            "fingerprint": {"count": 2, "modules": {**mine["modules"], "proj_out.weight": "zz"}}
        },
    )
    assert build.verify_against("other.pt", mine) == 3
    assert "proj_out.weight" in capsys.readouterr().out
    # The same two builds agreeing is the only thing that returns 0.
    monkeypatch.setattr(build, "_read_metadata", lambda path: {"fingerprint": mine})
    assert build.verify_against("other.pt", mine) == 0
    # An artifact with no block verifies nothing, so it is a refusal rather than a match.
    monkeypatch.setattr(build, "_read_metadata", lambda path: {})
    assert build.verify_against("other.pt", mine) == 3


# ── families and components ──────────────────────────────────────────────────────
def test_a_video_base_the_image_registry_does_not_know_resolves_in_the_video_one():
    build = _script()
    # detect_family answers None for every video family, so without the fallback the builder can
    # only ever bake image DiTs.
    assert detect_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = "wan2.2-t2v-a14b") is None
    fam = build.resolve_build_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = "wan2.2-t2v-a14b")
    assert fam is not None and fam.name == "wan2.2-t2v-a14b"
    assert fam.transformer_class == "WanTransformer3DModel"
    # An image family keeps resolving in the image registry, unchanged.
    assert (
        build.resolve_build_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image").name == "z-image"
    )
    # --modality pins one registry, so a name in the wrong one is refused rather than guessed at.
    assert (
        build.resolve_build_family(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = "wan2.2-t2v-a14b", modality = "image"
        )
        is None
    )
    assert (
        build.resolve_build_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image", modality = "video")
        is None
    )


def test_each_denoiser_component_publishes_under_its_own_name():
    build = _script()
    wan = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert wan is not None
    repo = "unsloth/Wan2.2-T2V-A14B-NVFP4"
    # Both experts share family, scheme, base and key set, so the filename is the only thing
    # standing between expert 2's slot and expert 1's weights.
    assert (
        build.upload_destination(wan, "nvfp4", rotated = False, repo_id = repo)
        == "Wan2.2-T2V-A14B-NVFP4.pt"
    )
    assert (
        build.upload_destination(
            wan, "nvfp4", rotated = False, repo_id = repo, component = "transformer_2"
        )
        == "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"
    )


# ── what a build stamps into the artifact ────────────────────────────────────────
_UINT8 = object()  # the stub torch's uint8, so the fingerprint's dtype view is a no-op here


class _Bytes:
    """Enough of a tensor for the fingerprint: it views the payload as uint8 and hashes it."""

    dtype = _UINT8

    def __init__(self, payload):
        self._payload = payload

    def detach(self):
        return self

    def contiguous(self):
        return self

    def to(self, device):
        return self

    def dim(self):
        return 1

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np
        return np.frombuffer(self._payload, dtype = np.uint8)


class Float8Tensor:
    """A quantized weight as far as the fingerprint is concerned: the class NAME is the key."""

    def __init__(self, tag):
        self.qdata = _Bytes(tag)
        self.scale = _Bytes(b"scale")

    def detach(self):
        return self

    def to(self, device):
        return self


def _fake_state_dict():
    """Two quantized linears plus the dense tensors a real DiT keeps beside them."""
    return {
        "blocks.0.attn1.to_q.weight": Float8Tensor(b"q0"),
        "blocks.1.attn1.to_q.weight": Float8Tensor(b"q1"),
        "blocks.0.norm1.weight": _Bytes(b"norm"),  # dense: not hashed, recorded as skipped
        "blocks.0.attn1.to_q.bias": _Bytes(b"bias"),  # not a weight: not recorded at all
    }


def _stub_build_stack(monkeypatch, state_dict):
    """torch / torchao / diffusers as far as the builder walks them. Returns what was saved."""
    saved: dict = {}

    torch = types.ModuleType("torch")
    torch.__version__ = "2.12.1+cu130"
    torch.bfloat16 = "bfloat16"
    torch.uint8 = _UINT8

    def _save(obj, path):
        saved["ckpt"] = obj
        Path(path).write_bytes(b"x")

    torch.save = _save
    monkeypatch.setitem(sys.modules, "torch", torch)

    torchao = types.ModuleType("torchao")
    torchao.__version__ = "0.17.0"
    quantization = types.ModuleType("torchao.quantization")

    def _quantize_(
        module,
        config,
        filter_fn = None,
    ):
        saved["filter_fn"] = filter_fn

    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    quantization.quantize_ = _quantize_
    quantization.Float8DynamicActivationFloat8WeightConfig = _Config
    quantization.Int8DynamicActivationInt8WeightConfig = _Config
    mx_formats = types.ModuleType("torchao.prototype.mx_formats")

    class _NVFP4Config:
        def __init__(self, use_triton_kernel = True):
            self.use_triton_kernel = use_triton_kernel

    mx_formats.NVFP4DynamicActivationNVFP4WeightConfig = _NVFP4Config
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)
    monkeypatch.setitem(sys.modules, "torchao.prototype", types.ModuleType("torchao.prototype"))
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx_formats)

    class _Transformer:
        @classmethod
        def from_pretrained(cls, base, **kwargs):
            saved["from_pretrained"] = {"base": base, **kwargs}
            return cls()

        def to(self, device):
            return self

        def state_dict(self):
            return state_dict

    diffusers = types.ModuleType("diffusers")
    diffusers.__version__ = "0.39.0"
    diffusers.WanTransformer3DModel = _Transformer
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return saved


def test_a_build_stamps_the_component_the_filter_floor_and_a_fingerprint(monkeypatch, tmp_path):
    build = _script()
    saved = _stub_build_stack(monkeypatch, _fake_state_dict())
    out = tmp_path / "expert2.pt"
    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--family",
            "wan2.2-t2v-a14b",
            "--scheme",
            "nvfp4",
            "--component",
            "transformer_2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    # The component drives which denoiser is loaded ...
    assert saved["from_pretrained"]["subfolder"] == "transformer_2"
    meta = saved["ckpt"]["metadata"]
    # ... and is recorded, because both experts pass every other check the loader makes.
    assert meta["component"] == "transformer_2"
    assert meta["family"] == "wan2.2-t2v-a14b"
    # The GEMM tiling floor the runtime filter applies, so an offline build cannot bake the ragged
    # linears the runtime leaves dense.
    assert meta["require_divisible"] == 16
    # One entry per quantized weight; the dense norm is skipped and the bias is not a weight.
    assert meta["fingerprint"]["algo"] == "md5-packed-v1"
    assert meta["fingerprint"]["count"] == 2
    assert set(meta["fingerprint"]["modules"]) == {
        "blocks.0.attn1.to_q.weight",
        "blocks.1.attn1.to_q.weight",
    }
    assert meta["fingerprint"]["skipped"] == ["blocks.0.norm1.weight"]
    # Different weights hash differently, which is the only property the check rests on.
    digests = set(meta["fingerprint"]["modules"].values())
    assert len(digests) == 2


def test_a_build_refuses_to_publish_before_it_verifies(monkeypatch, tmp_path):
    build = _script()
    _stub_build_stack(monkeypatch, _fake_state_dict())
    out = tmp_path / "build_a.pt"
    argv = [
        "--base",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "--family",
        "wan2.2-t2v-a14b",
        "--scheme",
        "nvfp4",
        "--out",
        str(out),
    ]
    # Refused before the load, not after the hours: the answer is in the arguments.
    assert build.main([*argv, "--upload-repo", "unsloth/Wan2.2-T2V-A14B-NVFP4"]) == 2
    assert not out.exists()
    assert build.main([*argv, "--verify-against", str(out)]) == 2
    assert not out.exists()


def test_a_build_whose_second_run_differs_exits_3_without_uploading(monkeypatch, tmp_path):
    build = _script()
    _stub_build_stack(monkeypatch, _fake_state_dict())
    monkeypatch.setattr(
        build,
        "_read_metadata",
        lambda path: {
            "fingerprint": {"count": 2, "modules": {"blocks.0.attn1.to_q.weight": "deadbeef"}}
        },
    )
    # A refused build must not reach the Hub at all, so an upload here is an import error.
    hub = types.ModuleType("huggingface_hub")

    def _no_upload(*a, **k):
        raise AssertionError("a build that failed verification must not upload")

    hub.HfApi = _no_upload
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    out = tmp_path / "build_b.pt"
    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--family",
            "wan2.2-t2v-a14b",
            "--scheme",
            "nvfp4",
            "--out",
            str(out),
            "--verify-against",
            str(tmp_path / "build_a.pt"),
            "--upload-repo",
            "unsloth/Wan2.2-T2V-A14B-NVFP4",
        ]
    )
    assert code == 3
