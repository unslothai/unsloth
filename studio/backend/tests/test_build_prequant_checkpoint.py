# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two decisions ``scripts/build_prequant_checkpoint.py`` makes that a bad answer to is
silent: whether a ConvRot build is buildable at all, and what filename it publishes under.

Both end in an artifact that costs GPU-hours and tens of gigabytes and then cannot be resolved
or cannot be loaded, with nothing at build time saying so, which is why they are pulled out as
pure functions and asserted here rather than left inline in ``main``."""

import importlib.util
import os
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
    with pytest.raises(ValueError, match = "--upload-repo"):
        build.upload_destination(zimage, "fp8", rotated = False)


def test_a_build_may_not_publish_without_a_second_build_to_verify_against():
    build = _script()
    refusal = build.upload_gate_refusal("unsloth/Wan2.2-TI2V-5B-NVFP4", None)
    assert refusal is not None and "--verify-against" in refusal
    assert build.upload_gate_refusal("unsloth/Wan2.2-TI2V-5B-NVFP4", "other.pt") is None
    assert build.upload_gate_refusal(None, None) is None


def test_verifying_an_artifact_against_itself_is_refused(tmp_path):
    build = _script()
    out = tmp_path / "build_a.pt"
    out.write_bytes(b"x")
    link = tmp_path / "same.pt"
    link.symlink_to(out)
    assert build.verify_target_refusal(str(out), str(out)) is not None
    assert build.verify_target_refusal(str(out), str(link)) is not None
    assert build.verify_target_refusal(str(out), str(tmp_path / "build_b.pt")) is None
    assert build.verify_target_refusal(str(out), None) is None


def test_the_fingerprint_diff_names_every_differing_weight_and_where_it_sits():
    build = _script()
    mine = {"count": 2, "modules": {"blocks.3.attn1.to_q.weight": "aa", "proj_out.weight": "bb"}}
    other = {"count": 2, "modules": {"blocks.3.attn1.to_q.weight": "cc", "proj_out.weight": "bb"}}
    assert build.fingerprint_mismatches(mine, other) == ["blocks.3.attn1.to_q.weight"]
    fewer = {"count": 1, "modules": {"proj_out.weight": "bb"}}
    assert build.fingerprint_mismatches(mine, fewer) == ["blocks.3.attn1.to_q.weight"]
    assert build.fingerprint_mismatches(mine, mine) == []
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
    monkeypatch.setattr(build, "_read_metadata", lambda path: {"fingerprint": mine})
    assert build.verify_against("other.pt", mine) == 0
    monkeypatch.setattr(build, "_read_metadata", lambda path: {})
    assert build.verify_against("other.pt", mine) == 3


def test_a_video_base_the_image_registry_does_not_know_resolves_in_the_video_one():
    build = _script()
    assert detect_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = "wan2.2-t2v-a14b") is None
    fam = build.resolve_build_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = "wan2.2-t2v-a14b")
    assert fam is not None and fam.name == "wan2.2-t2v-a14b"
    assert fam.transformer_class == "WanTransformer3DModel"
    assert (
        build.resolve_build_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image").name == "z-image"
    )
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
    assert saved["from_pretrained"]["subfolder"] == "transformer_2"
    meta = saved["ckpt"]["metadata"]
    assert meta["component"] == "transformer_2"
    assert meta["family"] == "wan2.2-t2v-a14b"
    assert meta["require_divisible"] == 16
    assert meta["fingerprint"]["algo"] == "md5-packed-v1"
    assert meta["fingerprint"]["count"] == 2
    assert set(meta["fingerprint"]["modules"]) == {
        "blocks.0.attn1.to_q.weight",
        "blocks.1.attn1.to_q.weight",
    }
    assert meta["fingerprint"]["skipped"] == ["blocks.0.norm1.weight"]
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
    assert build.main([*argv, "--upload-repo", "unsloth/Wan2.2-T2V-A14B-NVFP4"]) == 2
    assert not out.exists()
    assert build.main([*argv, "--verify-against", str(out)]) == 2
    assert not out.exists()


def test_a_remote_base_may_not_be_published_under_another_repos_identity(monkeypatch, tmp_path):
    build = _script()
    saved = _stub_build_stack(monkeypatch, _fake_state_dict())
    out = tmp_path / "build.pt"
    # 480p weights, 720p identity: same shapes, same prequant repo, and the 720p family declares
    # its own filename, so nothing after this point can tell the two apart.
    code = build.main(
        [
            "--base",
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            "--base-id",
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
            "--family",
            "hunyuanvideo-1.5-720p",
            "--scheme",
            "nvfp4",
            "--out",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()
    assert "ckpt" not in saved
    refusal = build.base_id_refusal(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    )
    assert refusal is not None and "--base-id" in refusal
    # What --base-id is for: a local mirror declaring the repo it mirrors.
    local = tmp_path / "mirror"
    local.mkdir()
    assert build.base_id_refusal(str(local), "Wan-AI/Wan2.2-T2V-A14B-Diffusers") is None
    code = build.main(
        [
            "--base",
            str(local),
            "--base-id",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--family",
            "wan2.2-t2v-a14b",
            "--scheme",
            "nvfp4",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert saved["ckpt"]["metadata"]["base_model_id"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert saved["from_pretrained"]["base"] == str(local)
    # A remote --base the loader would call the same model keeps working, and no --base-id at all
    # is untouched.
    assert (
        build.base_id_refusal(
            "unsloth/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        )
        is None
    )
    assert build.base_id_refusal("Wan-AI/Wan2.2-T2V-A14B-Diffusers", None) is None


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


def test_a_correction_that_does_not_help_is_not_applied_and_says_so():
    build = _script()
    meta = {
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
    assert build.plan_gptq(list(meta), meta, score, mode = "meta")["counts"]["applied"] == 0
    plan = build.plan_gptq(["blocks.9.ffn.net.2", *meta], meta, score, mode = "check")
    assert plan["counts"]["skipped_unscored"] == 1
    assert plan["layers"]["blocks.9.ffn.net.2"]["reason"] == "unscored"


def test_a_correction_the_pass_never_wrote_is_counted_not_ignored():
    build = _script()
    meta = {"blocks.0.attn1.to_q": {"err_rtn": 0.09, "err_gptq": 0.12}}
    score = {"blocks.0.attn1.to_q": {"out_err_rtn": 0.03, "out_err_gptq": 0.01}}
    plan = build.plan_gptq(list(meta), meta, score, has_weight = lambda fqn: False)
    assert plan["apply"] == []
    assert plan["counts"]["missing"] == 1 and plan["counts"]["applied"] == 0
    assert plan["layers"]["blocks.0.attn1.to_q"]["reason"] == "missing"
    assert plan["layers"]["blocks.0.attn1.to_q"]["out_err_gptq"] == 0.01


def test_the_gptq_layout_of_a_moe_family_resolves_per_expert(tmp_path):
    build = _script()
    root = tmp_path / "gptq"
    (root / "weights" / "transformer_2").mkdir(parents = True)
    (root / "gptq_meta_transformer_2.json").write_text("{}")
    (root / "gptq_score_transformer_2.json").write_text("{}")
    (root / "gptq_meta.json").write_text("{}")
    where = build.gptq_sources(str(root), "transformer_2")
    assert where["weights"].endswith("weights/transformer_2")
    assert where["meta"].endswith("gptq_meta_transformer_2.json")
    assert where["score"].endswith("gptq_score_transformer_2.json")
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
    (gptq / "weights" / "blocks_0_attn1_to_q.pt").write_bytes(b"w")

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
    assert "gptq" not in saved["ckpt"]["metadata"]["fingerprint"]
    # The block is published with the checkpoint, so it names the files and never the build host's
    # directory layout.
    assert block["source"] == "gptq"
    assert block["meta_path"] == "gptq_meta.json" and block["score_path"] == "gptq_check.json"
    assert not any(os.sep in str(block[field]) for field in ("source", "meta_path", "score_path"))


def test_a_calibrated_build_is_refused_against_another_campaigns_base(
    monkeypatch, tmp_path, capsys
):
    """Corrections belong to the model they were solved on. The separately trained HunyuanVideo-1.5
    480p and 720p transformers share every fqn and shape, so the meta's own base is the only thing
    that can tell one campaign's corrections from the other's."""
    build = _script()
    _stub_build_stack(monkeypatch, _fake_state_dict())
    import json as _json

    gptq = tmp_path / "gptq"
    (gptq / "weights").mkdir(parents = True)
    (gptq / "gptq_meta.json").write_text(
        _json.dumps(
            {
                "base_model_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                "layers": {},
            }
        )
    )
    (gptq / "gptq_check.json").write_text(_json.dumps({"layers": {}}))
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
    assert code == 2
    assert not out.exists()
    assert "calibrated on" in capsys.readouterr().out
    assert build.gptq_meta_base({"base": " org/model "}) == "org/model"
    assert build.gptq_meta_base({"layers": {}}) is None
    # The campaign that names this base, and one that names no base at all (it predates the stamp).
    assert (
        build.gptq_base_refusal(
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
            {"base_model_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"},
        )
        is None
    )
    assert build.gptq_base_refusal("Wan-AI/Wan2.2-TI2V-5B-Diffusers", {"layers": {}}) is None


def test_a_calibrated_build_is_refused_under_another_quantiser(monkeypatch, tmp_path, capsys):
    """The corrections lie on the NVFP4 grid and were scored there, so a build that would
    re-quantise them as fp8 is refused rather than published as a measured artifact whose scores
    no longer describe it."""
    build = _script()
    _stub_build_stack(monkeypatch, _fake_state_dict())
    out = tmp_path / "fp8.pt"
    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--family",
            "wan2.2-ti2v-5b",
            "--scheme",
            "fp8",
            "--out",
            str(out),
            "--gptq-dir",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert not out.exists()
    # The refusal itself, not the "cannot read the meta" the empty directory would also produce.
    assert "--gptq-dir cannot be combined with --scheme fp8" in capsys.readouterr().out
    assert build.gptq_scheme_refusal("nvfp4", str(tmp_path)) is None
    assert build.gptq_scheme_refusal("fp8", None) is None
    assert "mxfp8" in (build.gptq_scheme_refusal("mxfp8", str(tmp_path)) or "")


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
    assert code == 2
    assert not out.exists()


def test_a_rotated_second_expert_is_published_under_its_own_declared_name():
    """The declared-name lookup is per COMPONENT: the task-agnostic row is expert 1's artifact."""
    build = _script()
    wan = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert wan is not None
    assert (
        build.upload_destination(wan, "nvfp4", rotated = True, component = "transformer_2")
        == "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"
    )
    assert build.upload_destination(wan, "nvfp4", rotated = True) == "Wan2.2-T2V-A14B-NVFP4.pt"
    # A component the family names no artifact for is refused rather than published over expert 1's.
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(wan, "nvfp4", rotated = True, component = "transformer_3")


def test_two_families_in_one_repo_publish_under_their_declared_names():
    """The derived name is a function of the REPO, so the 480p and 720p variants derive the same
    one and the second build would publish over the first."""
    build = _script()
    repo = "unsloth/HunyuanVideo-1.5-NVFP4"
    p480 = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    p720 = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v")
    assert p480 is not None and p720 is not None and p480.name != p720.name
    assert (
        build.upload_destination(p480, "nvfp4", rotated = False, repo_id = repo)
        == "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4.pt"
    )
    assert (
        build.upload_destination(p720, "nvfp4", rotated = False, repo_id = repo)
        == "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.pt"
    )
    assert build.families_sharing_prequant_repo(p480, "nvfp4", repo) == (p720.name,)
    # A repo one family owns keeps the derived name every hosted layout already uses.
    wan = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert build.families_sharing_prequant_repo(wan, "nvfp4", "unsloth/Wan2.2-TI2V-5B-NVFP4") == ()
    assert (
        build.upload_destination(
            wan, "nvfp4", rotated = False, repo_id = "unsloth/Wan2.2-TI2V-5B-NVFP4"
        )
        == "Wan2.2-TI2V-5B-NVFP4.pt"
    )


def test_a_second_expert_never_takes_the_flat_gptq_layout(tmp_path):
    """The experts share every fqn and shape, so expert 1's flat correction would load into expert
    2, pass the shape check and be baked in."""
    build = _script()
    flat = tmp_path / "flat"
    (flat / "weights").mkdir(parents = True)
    (flat / "gptq_meta.json").write_text("{}")
    (flat / "gptq_check.json").write_text("{}")
    where = build.gptq_sources(str(flat), "transformer_2")
    assert where["weights"].endswith("weights/transformer_2")
    assert where["meta"].endswith("gptq_meta_transformer_2.json")
    assert where["score"] is None
    # The default component is what the flat layout describes, and it still reads it.
    default = build.gptq_sources(str(flat), "transformer")
    assert default["weights"].endswith("flat/weights")
    assert default["meta"].endswith("flat/gptq_meta.json")
    assert default["score"].endswith("flat/gptq_check.json")
