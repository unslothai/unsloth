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


# ── GPTQ corrections ─────────────────────────────────────────────────────────────
def test_a_correction_that_does_not_help_is_not_applied_and_says_so():
    build = _script()
    meta = {
        # GPTQ raises the Frobenius weight error on every layer by construction: it trades weight
        # error for output error, which is why the weight error cannot be the do-no-harm test.
        "blocks.0.attn1.to_q": {"err_rtn": 0.096, "err_gptq": 0.125, "damp": 0.01},
        "blocks.1.attn1.to_q": {"err_rtn": 0.095, "err_gptq": 0.121, "damp": 0.1},
    }
    score = {
        "blocks.0.attn1.to_q": {"out_err_rtn": 0.034, "out_err_gptq": 0.006},
        "blocks.1.attn1.to_q": {"out_err_rtn": 0.021, "out_err_gptq": 0.037},
    }
    plan = build.plan_gptq(list(meta), meta, score, mode = "check")
    assert plan["apply"] == ["blocks.0.attn1.to_q"]
    assert plan["counts"] == {
        "applied": 1,
        "skipped_no_gain": 1,
        "skipped_unscored": 0,
        "missing": 0,
    }
    assert plan["layers"]["blocks.1.attn1.to_q"]["reason"] == "no_gain"
    # The weight-space rule is the one the Hessian pass recorded, and on this evidence it admits
    # nothing at all, which is why it is not the default.
    assert build.plan_gptq(list(meta), meta, score, mode = "meta")["counts"]["applied"] == 0
    # A layer with no score is left alone rather than applied on faith.
    plan = build.plan_gptq(["blocks.9.ffn.net.2", *meta], meta, score, mode = "check")
    assert plan["counts"]["skipped_unscored"] == 1
    assert plan["layers"]["blocks.9.ffn.net.2"]["reason"] == "unscored"


def test_a_correction_the_pass_never_wrote_is_counted_not_ignored():
    build = _script()
    meta = {"blocks.0.attn1.to_q": {"err_rtn": 0.09, "err_gptq": 0.12}}
    score = {"blocks.0.attn1.to_q": {"out_err_rtn": 0.03, "out_err_gptq": 0.01}}
    # The correction helps, but the file is not there. Silently leaving it out would make an
    # artifact that claims a calibration it only partly has.
    plan = build.plan_gptq(list(meta), meta, score, has_weight = lambda fqn: False)
    assert plan["apply"] == []
    assert plan["counts"]["missing"] == 1 and plan["counts"]["applied"] == 0
    assert plan["layers"]["blocks.0.attn1.to_q"]["reason"] == "missing"
    # And the errors are still recorded, so the artifact can say what was skipped and why.
    assert plan["layers"]["blocks.0.attn1.to_q"]["out_err_gptq"] == 0.01


def test_the_gptq_layout_of_a_moe_family_resolves_per_expert(tmp_path):
    build = _script()
    root = tmp_path / "gptq"
    (root / "weights" / "transformer_2").mkdir(parents = True)
    (root / "gptq_meta_transformer_2.json").write_text("{}")
    (root / "gptq_score_transformer_2.json").write_text("{}")
    (root / "gptq_meta.json").write_text("{}")
    where = build.gptq_sources(str(root), "transformer_2")
    # Both experts share every name in the model, so the per-expert directory is the only thing
    # keeping expert 1's corrections out of expert 2's artifact.
    assert where["weights"].endswith("weights/transformer_2")
    assert where["meta"].endswith("gptq_meta_transformer_2.json")
    assert where["score"].endswith("gptq_score_transformer_2.json")
    # A single-denoiser family writes the flat layout, and the same directory serves it.
    flat = tmp_path / "flat"
    (flat / "weights").mkdir(parents = True)
    (flat / "gptq_meta.json").write_text("{}")
    (flat / "gptq_check.json").write_text("{}")
    where = build.gptq_sources(str(flat), "transformer")
    assert where["weights"].endswith("flat/weights")
    assert where["meta"].endswith("flat/gptq_meta.json")
    assert where["score"].endswith("flat/gptq_check.json")
    assert build.gptq_weight_filename("blocks.0.ffn.net.0.proj") == "blocks_0_ffn_net_0_proj.pt"


def test_a_calibrated_build_stamps_which_weights_are_corrected(monkeypatch, tmp_path):
    build = _script()
    state = _fake_state_dict()
    saved = _stub_build_stack(monkeypatch, state)
    gptq = tmp_path / "gptq"
    (gptq / "weights").mkdir(parents = True)
    import json as _json

    (gptq / "gptq_meta.json").write_text(
        _json.dumps(
            {
                "prompts": 32,
                "steps_sampled": [0, 12, 25, 37],
                "base_damp": 0.01,
                "grid": "832x480x49f_50s",
                "layers": {
                    "blocks.0.attn1.to_q": {"err_rtn": 0.09, "err_gptq": 0.12, "damp": 0.01}
                },
            }
        )
    )
    (gptq / "gptq_check.json").write_text(
        _json.dumps(
            {"layers": {"blocks.0.attn1.to_q": {"out_err_rtn": 0.03, "out_err_gptq": 0.01}}}
        )
    )
    # One admitted linear with a correction on disk, one without a file at all.
    (gptq / "weights" / "blocks_0_attn1_to_q.pt").write_bytes(b"w")

    # The runtime filter is the admitted set, so the stub has to look like what it inspects:
    # nn.Linear, 16-aligned, at or above the min_features floor.
    class _Linear:
        def __init__(self):
            self.in_features = 1024
            self.out_features = 1024
            self.weight = types.SimpleNamespace(
                shape = (1024, 1024), device = "cuda", dtype = "bfloat16", data = None
            )

    nn = types.ModuleType("torch.nn")
    nn.Linear = _Linear
    sys.modules["torch"].nn = nn
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    module, other = _Linear(), _Linear()
    transformer = sys.modules["diffusers"].WanTransformer3DModel()
    transformer.named_modules = lambda: [
        ("blocks.0.attn1.to_q", module),
        ("blocks.1.attn1.to_q", other),
    ]
    monkeypatch.setattr(
        sys.modules["diffusers"].WanTransformer3DModel,
        "from_pretrained",
        classmethod(lambda cls, base, **kwargs: transformer),
    )
    loaded = types.SimpleNamespace(shape = (1024, 1024), to = lambda *a: "corrected")
    # The idempotency check reads packed NVFP4 payloads off a real GPU; here it stands in for one,
    # so what is asserted is that its verdict reaches the metadata.
    monkeypatch.setattr(
        build,
        "verify_gptq_idempotency",
        lambda modules, load_weight: {
            "checked": len(modules),
            "max_abs": 0.0,
            "max_abs_fqn": None,
            "frac_diff": 0.0,
        },
    )
    sys.modules["torch"].load = lambda path, weights_only = True: loaded
    out = tmp_path / "wan5b.pt"
    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--family",
            "wan2.2-ti2v-5b",
            "--scheme",
            "nvfp4",
            "--out",
            str(out),
            "--gptq-dir",
            str(gptq),
        ]
    )
    assert code == 0
    # The corrected weight replaced the dense one BEFORE quantize_, which is the only order in
    # which the quantiser packs the correction rather than re-deriving it.
    assert module.weight.data == "corrected"
    block = saved["ckpt"]["metadata"]["gptq"]
    assert block["applied"] == 1 and block["missing"] == 1
    assert block["prompts"] == 32 and block["steps_sampled"] == [0, 12, 25, 37]
    assert block["base_damp"] == 0.01 and block["score_mode"] == "check"
    assert set(block["layers"]) == {"blocks.0.attn1.to_q", "blocks.1.attn1.to_q"}
    assert block["layers"]["blocks.0.attn1.to_q"]["applied"] is True
    assert block["layers"]["blocks.1.attn1.to_q"]["reason"] == "missing"
    assert block["idempotency"] == {
        "checked": 1,
        "max_abs": 0.0,
        "max_abs_fqn": None,
        "frac_diff": 0.0,
    }
    # The fingerprint stays a hash of the packed payloads alone, so the block never becomes part
    # of the artifact's identity by itself.
    assert "gptq" not in saved["ckpt"]["metadata"]["fingerprint"]


def test_a_rotated_build_may_not_also_be_a_calibrated_one(monkeypatch, tmp_path):
    build = _script()
    _stub_build_stack(monkeypatch, _fake_state_dict())
    out = tmp_path / "both.pt"
    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--family",
            "wan2.2-ti2v-5b",
            "--scheme",
            "nvfp4",
            "--out",
            str(out),
            "--gptq-dir",
            str(tmp_path),
            "--convrot-groupsize",
            "128",
        ]
    )
    # Refused from the arguments alone: the correction was solved against unrotated activations.
    assert code == 2
    assert not out.exists()


# ── per-layer NVFP4 policies ─────────────────────────────────────────────────────
def test_the_policy_a_build_applies_is_the_one_that_resolves_for_its_base():
    build = _script()
    # auto is the default and must leave every existing invocation building what it builds today:
    # no policy describes an fp8 or int8 artifact, and none resolves for the video families.
    assert build.resolve_build_policy("auto", "fp8", "z-image", "Tongyi-MAI/Z-Image-Turbo") == (
        None,
        None,
    )
    assert build.resolve_build_policy(
        "auto", "nvfp4", "wan2.2-ti2v-5b", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    ) == (None, None)
    policy, refusal = build.resolve_build_policy(
        "auto", "nvfp4", "z-image", "Tongyi-MAI/Z-Image-Turbo"
    )
    assert refusal is None and policy.policy_id == "zimg_f8mod_toq34_v1"
    # off forces the whole-model build even where one would resolve.
    assert build.resolve_build_policy("off", "nvfp4", "z-image", "Tongyi-MAI/Z-Image-Turbo") == (
        None,
        None,
    )
    # A named policy is a pin: the operator says which layer set they measured, and a table that
    # has moved under them is a refusal rather than a different artifact.
    policy, refusal = build.resolve_build_policy(
        "zimg_f8mod_toq34_v1", "nvfp4", "z-image", "unsloth/Z-Image-Turbo"
    )
    assert refusal is None and policy.policy_id == "zimg_f8mod_toq34_v1"
    _, refusal = build.resolve_build_policy(
        "zimg_f8mod_toq34_v1", "nvfp4", "flux.1", "black-forest-labs/FLUX.1-schnell"
    )
    assert refusal is not None and "flux_mod_single_v1" in refusal
    _, refusal = build.resolve_build_policy(
        "flux_mod_single_v1", "nvfp4", "flux.1", "black-forest-labs/FLUX.1-dev"
    )
    assert refusal is not None and "none" in refusal
    _, refusal = build.resolve_build_policy(
        "nope_v1", "nvfp4", "z-image", "Tongyi-MAI/Z-Image-Turbo"
    )
    assert refusal is not None and "unknown --policy" in refusal
    _, refusal = build.resolve_build_policy(
        "zimg_f8mod_toq34_v1", "fp8", "z-image", "Tongyi-MAI/Z-Image-Turbo"
    )
    assert refusal is not None and "nvfp4 build" in refusal


class _PolicyLinear:
    """A Linear as the shared filter and the two quantise passes read one."""

    def __init__(
        self,
        in_features = 1024,
        out_features = 1024,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _StubParameter()


class _StubParameter:
    dtype = "bfloat16"
    shape = (1024, 1024)
    device = "cuda"
    data = None


class _Quantized:
    """A torchao weight subclass as far as the post-pass walk is concerned: the class NAME."""

    def __init__(self, name):
        self.__class__ = type(name, (_Quantized,), {})


def _stub_policy_build(monkeypatch, tmp_path):
    """A two-linear DiT, a tiny policy over it, and a quantize_ that records each pass."""
    import types as _types

    from core.inference import diffusion_nvfp4_policy as policies

    saved = _stub_build_stack(monkeypatch, _fake_state_dict())
    torch = sys.modules["torch"]
    nn = _types.ModuleType("torch.nn")
    nn.Linear = _PolicyLinear
    nn.Parameter = _StubParameter
    torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch.nn", nn)

    modules = {"blocks.0.attn1.to_q": _PolicyLinear(), "blocks.0.ffn.net.0": _PolicyLinear()}
    transformer = sys.modules["diffusers"].WanTransformer3DModel()
    transformer.named_modules = lambda: list(modules.items())
    monkeypatch.setattr(
        sys.modules["diffusers"].WanTransformer3DModel,
        "from_pretrained",
        classmethod(lambda cls, base, **kwargs: transformer),
    )

    tiny = policies.NVFP4Policy(
        policy_id = "tiny_v1",
        version = 3,
        family = "wan2.2-ti2v-5b",
        base_repos = ("wan-ai/wan2.2-ti2v-5b-diffusers",),
        rules = (policies.Rule(suffix = "attn1.to_q", precision = policies.PRECISION_NVFP4, expect = 1),),
        expected_counts = {
            policies.PRECISION_NVFP4: 1,
            policies.PRECISION_FP8: 1,
            policies.PRECISION_BF16: 0,
        },
    )
    monkeypatch.setattr(policies, "NVFP4_POLICIES", (tiny,))

    passes: list = []
    produced = {"nvfp4": "NVFP4Tensor", "fp8": "Float8Tensor"}

    def _quantize_(
        module,
        config,
        filter_fn = None,
    ):
        selected = [fqn for fqn, sub in module.named_modules() if filter_fn(sub, fqn)]
        passes.append({"config": config, "selected": selected})
        for fqn in selected:
            modules[fqn].weight = _Quantized(produced[config.scheme])
        # A pass that ran is not the whole-model call, which records its filter and nothing else.
        saved["filter_fn"] = filter_fn

    sys.modules["torchao.quantization"].quantize_ = _quantize_
    dtq = sys.modules["core.inference.diffusion_transformer_quant"]
    monkeypatch.setattr(
        dtq,
        "_make_quant_config",
        lambda scheme, fast_accum = None: _types.SimpleNamespace(
            scheme = scheme, fast_accum = fast_accum
        ),
    )
    return saved, passes, modules


def _policy_argv(out, *extra):
    return [
        "--base",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--family",
        "wan2.2-ti2v-5b",
        "--scheme",
        "nvfp4",
        "--out",
        str(out),
        *extra,
    ]


def test_a_policy_build_runs_two_passes_and_stamps_what_it_assigned(monkeypatch, tmp_path):
    build = _script()
    saved, passes, modules = _stub_policy_build(monkeypatch, tmp_path)
    out = tmp_path / "policy.pt"
    assert build.main(_policy_argv(out)) == 0
    # NVFP4 first, then fp8, over disjoint sets: the order is what lets the fp8 filter also
    # require a plain Parameter, so no layer can be quantised twice.
    assert [p["config"].scheme for p in passes] == ["nvfp4", "fp8"]
    assert passes[0]["selected"] == ["blocks.0.attn1.to_q"]
    assert passes[1]["selected"] == ["blocks.0.ffn.net.0"]
    assert type(modules["blocks.0.attn1.to_q"].weight).__name__ == "NVFP4Tensor"
    assert type(modules["blocks.0.ffn.net.0"].weight).__name__ == "Float8Tensor"
    ckpt = saved["ckpt"]
    # The tag an older build refuses, rather than loading the mixture as a whole-model artifact.
    assert ckpt["format"] == "unsloth_prequant_transformer_state_dict_v3"
    block = ckpt["metadata"]["nvfp4_policy"]
    assert block["policy_id"] == "tiny_v1" and block["policy_version"] == 3
    assert block["counts"] == {"fp8": 1, "nvfp4": 1}
    assert block["nvfp4_fqns"] == ["blocks.0.attn1.to_q"]
    assert block["activation_scales_baked"] is False and block["gptq"] is False
    # The scheme token does not change: the policy is metadata about an nvfp4 artifact.
    assert ckpt["metadata"]["scheme"] == "nvfp4"
    # The fp8 half bakes an accumulate mode and a granularity in, so both are recorded.
    assert ckpt["metadata"]["fp8_granularity"] == "per_row"
    assert ckpt["metadata"]["fast_accum"] is not None
    assert passes[1]["config"].fast_accum == ckpt["metadata"]["fast_accum"]


def test_a_policy_build_may_not_also_rotate_and_a_family_without_one_may_not_ask(
    monkeypatch, tmp_path
):
    build = _script()
    saved, passes, _ = _stub_policy_build(monkeypatch, tmp_path)
    out = tmp_path / "policy.pt"
    # Both rewrite the weights before quantize_ and both claim the one format tag slot.
    assert build.main(_policy_argv(out, "--convrot-groupsize", "128")) == 2
    assert not out.exists()
    # A policy id that does not resolve for this family and base is refused from the arguments
    # alone, before the hours: the layer set was solved on another checkpoint's weights.
    assert build.main(_policy_argv(out, "--policy", "zimg_f8mod_toq34_v1")) == 2
    assert build.main(_policy_argv(out, "--policy", "no_such_policy_v1")) == 2
    assert not out.exists()
    # off builds the whole-model artifact, which is also what every base without a policy gets:
    # one pass over the filter's set, the v1 tag, and no policy block.
    assert build.main(_policy_argv(out, "--policy", "off")) == 0
    assert [p["config"].scheme for p in passes] == ["nvfp4"]
    assert saved["ckpt"]["format"] == "unsloth_prequant_transformer_state_dict_v1"
    assert "nvfp4_policy" not in saved["ckpt"]["metadata"]


def test_a_calibrated_policy_build_corrects_the_4_bit_layers_only(monkeypatch, tmp_path):
    build = _script()
    saved, passes, modules = _stub_policy_build(monkeypatch, tmp_path)
    import json as _json

    gptq = tmp_path / "gptq"
    (gptq / "weights").mkdir(parents = True)
    layers = {
        "blocks.0.attn1.to_q": {"err_rtn": 0.09, "err_gptq": 0.12},
        "blocks.0.ffn.net.0": {"err_rtn": 0.08, "err_gptq": 0.11},
    }
    (gptq / "gptq_meta.json").write_text(_json.dumps({"prompts": 32, "layers": layers}))
    (gptq / "gptq_check.json").write_text(
        _json.dumps(
            {"layers": {fqn: {"out_err_rtn": 0.03, "out_err_gptq": 0.01} for fqn in layers}}
        )
    )
    for fqn in layers:
        (gptq / "weights" / build.gptq_weight_filename(fqn)).write_bytes(b"w")
    sys.modules["torch"].load = lambda path, weights_only = True: types.SimpleNamespace(
        shape = (1024, 1024), to = lambda *a: "corrected"
    )
    monkeypatch.setattr(
        build,
        "verify_gptq_idempotency",
        lambda modules, load_weight: {
            "checked": len(modules),
            "max_abs": 0.0,
            "max_abs_fqn": None,
            "frac_diff": 0.0,
        },
    )
    out = tmp_path / "policy_gptq.pt"
    assert build.main(_policy_argv(out, "--gptq-dir", str(gptq))) == 0
    block = saved["ckpt"]["metadata"]["gptq"]
    # Both layers have a correction on disk that scores better, but only the NVFP4 one is in
    # scope: correcting a weight that then becomes the source of an fp8 replica raised the error
    # 46 percent in the campaign, and a static policy avoids that by construction.
    assert set(block["layers"]) == {"blocks.0.attn1.to_q"}
    assert block["applied"] == 1
    assert saved["ckpt"]["metadata"]["nvfp4_policy"]["gptq"] is True
