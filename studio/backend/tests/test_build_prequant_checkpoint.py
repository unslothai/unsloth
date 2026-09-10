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


def test_the_policy_a_build_applies_is_the_one_that_resolves_for_its_base():
    build = _script()
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
    assert build.resolve_build_policy("off", "nvfp4", "z-image", "Tongyi-MAI/Z-Image-Turbo") == (
        None,
        None,
    )
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
    assert [p["config"].scheme for p in passes] == ["nvfp4", "fp8"]
    assert passes[0]["selected"] == ["blocks.0.attn1.to_q"]
    assert passes[1]["selected"] == ["blocks.0.ffn.net.0"]
    assert type(modules["blocks.0.attn1.to_q"].weight).__name__ == "NVFP4Tensor"
    assert type(modules["blocks.0.ffn.net.0"].weight).__name__ == "Float8Tensor"
    ckpt = saved["ckpt"]
    assert ckpt["format"] == "unsloth_prequant_transformer_state_dict_v3"
    block = ckpt["metadata"]["nvfp4_policy"]
    assert block["policy_id"] == "tiny_v1" and block["policy_version"] == 3
    assert block["counts"] == {"fp8": 1, "nvfp4": 1}
    assert block["nvfp4_fqns"] == ["blocks.0.attn1.to_q"]
    assert block["activation_scales_baked"] is False and block["gptq"] is False
    assert ckpt["metadata"]["scheme"] == "nvfp4"
    assert ckpt["metadata"]["fp8_granularity"] == "per_row"
    assert ckpt["metadata"]["fast_accum"] is not None
    assert passes[1]["config"].fast_accum == ckpt["metadata"]["fast_accum"]


def test_a_policy_build_may_not_also_rotate_and_a_family_without_one_may_not_ask(
    monkeypatch, tmp_path
):
    build = _script()
    saved, passes, _ = _stub_policy_build(monkeypatch, tmp_path)
    out = tmp_path / "policy.pt"
    assert build.main(_policy_argv(out, "--convrot-groupsize", "128")) == 2
    assert not out.exists()
    assert build.main(_policy_argv(out, "--policy", "zimg_f8mod_toq34_v1")) == 2
    assert build.main(_policy_argv(out, "--policy", "no_such_policy_v1")) == 2
    assert not out.exists()
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
    assert set(block["layers"]) == {"blocks.0.attn1.to_q"}
    assert block["applied"] == 1
    assert saved["ckpt"]["metadata"]["nvfp4_policy"]["gptq"] is True




def _calib_prompts():
    build = _script()
    return build.load_calibration_prompts()


def test_the_calibration_prompts_are_disjoint_from_the_gate_suite():
    """A calibration set scored on its own prompts measures memorisation, so the two are disjoint."""
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "scripts" / "gptq_prompts.py"
    spec = importlib.util.spec_from_file_location("gptq_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calibration = module.CALIBRATION_PROMPTS
    gate = module.GATE_SUITE_PROMPTS
    assert len(calibration) == 32
    assert len(set(calibration)) == 32
    assert len(gate) == 7
    assert not set(calibration) & set(gate)

    def _normalise(text):
        return " ".join("".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split())

    assert not {_normalise(p) for p in calibration} & {_normalise(p) for p in gate}


def test_the_default_calibration_file_is_the_one_the_flag_documents():
    build = _script()
    assert build.DEFAULT_CALIB_PROMPTS.endswith("scripts/gptq_prompts.py")
    assert len(_calib_prompts()) == 32


def test_a_prompt_file_is_read_line_by_line_and_a_repeat_is_refused(tmp_path):
    build = _script()
    path = tmp_path / "prompts.txt"
    path.write_text("# a comment\na red bicycle\n\na blue bicycle\n")
    assert build.load_calibration_prompts(str(path)) == ("a red bicycle", "a blue bicycle")
    repeated = tmp_path / "repeat.txt"
    repeated.write_text("a red bicycle\na red bicycle\n")
    with pytest.raises(ValueError):
        build.load_calibration_prompts(str(repeated))
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    with pytest.raises(ValueError):
        build.load_calibration_prompts(str(empty))


def test_the_step_spec_parses_or_refuses():
    build = _script()
    assert build.parse_step_spec("0,12,25,37") == (0, 12, 25, 37)
    assert build.parse_step_spec(" 4 , 0 ,4") == (0, 4)
    for bad in ("", "0,-3", "first"):
        with pytest.raises(ValueError):
            build.parse_step_spec(bad)


def test_the_calibration_stages_run_hessians_then_gptq_then_the_bake():
    """Hessians on the uncorrected weights, activation scales on the ones that ship."""
    build = _script()
    assert build.calibration_stage_order(32, True) == ("hessians", "gptq", "bake")
    assert build.calibration_stage_order(32, False) == ("hessians", "gptq")
    assert build.calibration_stage_order(0, True) == ("bake",)
    assert build.calibration_stage_order(0, False) == ()


def test_the_calibration_flags_are_refused_for_a_build_that_cannot_honour_them():
    build = _script()
    common = {
        "nvfp4": "nvfp4",
        "gptq_dir": None,
        "convrot_groupsize": 0,
        "available_prompts": 32,
    }
    assert build.calibration_refusal(scheme = "fp8", gptq_prompts = 0, bake = False, **common) is None
    assert "nvfp4" in build.calibration_refusal(scheme = "fp8", gptq_prompts = 0, bake = True, **common)
    assert "nvfp4" in build.calibration_refusal(scheme = "int8", gptq_prompts = 4, bake = False, **common)
    both = dict(common, gptq_dir = "/tmp/gptq")
    assert "--gptq-dir" in build.calibration_refusal(
        scheme = "nvfp4", gptq_prompts = 4, bake = False, **both
    )
    assert build.calibration_refusal(scheme = "nvfp4", gptq_prompts = 0, bake = True, **both) is None
    rotated = dict(common, convrot_groupsize = 64)
    assert "unrotated" in build.calibration_refusal(
        scheme = "nvfp4", gptq_prompts = 4, bake = False, **rotated
    )
    assert "exceeds" in build.calibration_refusal(
        scheme = "nvfp4", gptq_prompts = 64, bake = False, **common
    )
    assert build.calibration_refusal(scheme = "nvfp4", gptq_prompts = 32, bake = True, **common) is None


class _StubPipe:
    """A pipeline as far as the calibration pass drives it: prompts in, nothing out."""

    def __init__(
        self,
        *,
        supports = (
            "prompt",
            "num_inference_steps",
            "width",
            "height",
            "generator",
            "output_type",
            "guidance_scale",
            "callback_on_step_end",
        ),
    ):
        self.calls: list = []
        self._supports = tuple(supports)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            for step in range(int(kwargs.get("num_inference_steps", 0))):
                callback(self, step, 0.0, {})
        return None


def test_every_calibration_render_is_seeded_so_a_second_build_reproduces_it():
    build = _script()
    pipe = _StubPipe()
    armed: list = []
    ran = build.render_calibration(
        pipe,
        ("a red bicycle", "a blue bicycle"),
        steps = 4,
        guidance = 3.5,
        cfg_kwarg = "true_cfg_scale",
        width = 512,
        height = 512,
        seed = 11,
        device = "cpu",
        before_prompt = lambda: armed.append(len(pipe.calls)),
    )
    assert ran == 2 and len(pipe.calls) == 2 and armed == [0, 1]
    first, second = pipe.calls
    assert first["prompt"] == "a red bicycle" and second["prompt"] == "a blue bicycle"
    assert first["num_inference_steps"] == 4 and first["width"] == first["height"] == 512
    assert first["true_cfg_scale"] == 3.5 and "guidance_scale" not in first
    assert first["output_type"] == "latent"
    assert first["generator"].initial_seed() == 11
    assert second["generator"].initial_seed() == 12


def test_a_pipeline_with_no_step_callback_cannot_be_hessian_calibrated():
    build = _script()

    class _NoCallback:
        def __call__(
            self,
            prompt,
            num_inference_steps = 1,
            generator = None,
        ):
            return None

    with pytest.raises(ValueError) as excinfo:
        build.render_calibration(
            _NoCallback(),
            ("a red bicycle",),
            steps = 4,
            guidance = 1.0,
            device = "cpu",
            callback = lambda *a: None,
        )
    assert "callback_on_step_end" in str(excinfo.value)
    assert (
        build.render_calibration(
            _NoCallback(), ("a red bicycle",), steps = 4, guidance = 1.0, device = "cpu"
        )
        == 1
    )


def test_the_gptq_metadata_block_says_which_weights_are_corrected_and_what_made_them():
    build = _script()
    scores = {
        "blocks.0.attention.to_q": {
            "err_rtn": 1.0,
            "err_gptq": 0.5,
            "ratio": 0.5,
            "improved": True,
        },
        "blocks.1.attention.to_q": {
            "err_rtn": 1.0,
            "err_gptq": 1.5,
            "ratio": 1.5,
            "improved": False,
        },
    }
    plan = {
        "apply": ["blocks.0.attention.to_q"],
        "counts": {"applied": 1, "applied_regressed": 0, "skipped_no_gain": 1},
    }
    block = build.gptq_metadata_block(
        prompts = ("a red bicycle", "a blue bicycle"),
        steps_sampled = (0, 2, 4, 6),
        schedule_steps = 8,
        max_regressions = 0,
        plan = plan,
        scores = scores,
        damps = {"blocks.0.attention.to_q": 0.01, "blocks.1.attention.to_q": 0.05},
        seconds = 12.34,
    )
    assert block["source"] == "in-builder"
    assert block["prompts"] == 2 and len(block["prompt_sha256"]) == 64
    assert block["steps_sampled"] == [0, 2, 4, 6] and block["schedule_steps"] == 8
    assert (block["applied"], block["skipped_no_gain"], block["applied_regressed"]) == (1, 1, 0)
    assert block["scored"] == 2 and block["seconds"] == 12.3
    applied = block["layers"]["blocks.0.attention.to_q"]
    skipped = block["layers"]["blocks.1.attention.to_q"]
    assert applied["applied"] is True and applied["reason"] == "applied" and applied["damp"] == 0.01
    assert skipped["applied"] is False and skipped["reason"] == "no_gain"
    assert (skipped["err_rtn"], skipped["err_gptq"]) == (1.0, 1.5)
    assert build.prompt_digest(("a red bicycle",)) != block["prompt_sha256"]


def test_the_baked_scales_record_the_set_and_the_schedule_they_were_measured_on():
    build = _script()
    meta = build.activation_scale_metadata(
        prompts = ("a red bicycle", "a blue bicycle"),
        schedule_steps = 8,
        scales = {"a": 12.0, "b": 4.0, "c": 100.0},
        layers = 3,
    )
    assert meta["prompts"] == 2 and meta["schedule_steps"] == 8 and meta["layers"] == 3
    assert meta["steps_sampled"] == "all"
    assert (meta["min_a_gsf"], meta["max_a_gsf"], meta["scaled"]) == (4.0, 100.0, 3)


def test_a_calibrated_build_is_refused_before_the_dense_download(monkeypatch, tmp_path):
    """Every calibration refusal is decided by the arguments alone, before the dense load."""
    build = _script()
    saved = _stub_build_stack(monkeypatch, _fake_state_dict())
    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--family",
            "wan2.2-t2v-a14b",
            "--scheme",
            "fp8",
            "--out",
            str(tmp_path / "a.pt"),
            "--bake-activation-scales",
        ]
    )
    assert code == 2
    assert "from_pretrained" not in saved  # nothing was downloaded

    code = build.main(
        [
            "--base",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--family",
            "wan2.2-t2v-a14b",
            "--scheme",
            "nvfp4",
            "--out",
            str(tmp_path / "a.pt"),
            "--gptq-prompts",
            "64",
        ]
    )
    assert code == 2
    assert "from_pretrained" not in saved




def test_the_calibration_grid_reads_as_wxhxframes_only_for_a_video_family():
    build = _script()
    assert build.parse_calib_grid("1024", video = False) == (1024, 1024, None)
    assert build.parse_calib_grid("1024x576", video = False) == (1024, 576, None)
    with pytest.raises(ValueError) as excinfo:
        build.parse_calib_grid("832x480x25", video = False)
    assert "frame count" in str(excinfo.value)
    assert build.parse_calib_grid("832x480x25", video = True) == (832, 480, 25)
    assert build.parse_calib_grid("832x480", video = True) == (832, 480, 25)
    assert build.parse_calib_grid(None, video = False) == (1024, 1024, None)
    assert build.parse_calib_grid(None, video = True) == (832, 480, 25)
    for bad in ("", "832x", "832xW", "0x480x25", "1x2x3x4"):
        with pytest.raises(ValueError):
            build.parse_calib_grid(bad, video = True)


def test_a_frame_count_the_family_cannot_render_is_refused_before_the_dense_load():
    build = _script()
    wan = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert build.frame_count_refusal(wan, 25) is None
    assert "k * 4 + 1" in build.frame_count_refusal(wan, 26)
    assert build.frame_count_refusal(detect_family("Tongyi-MAI/Z-Image-Turbo"), None) is None


def test_the_two_registries_are_told_apart_by_type_not_by_a_shared_attribute():
    build = _script()
    assert build.is_video_family(detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
    assert not build.is_video_family(detect_family("Tongyi-MAI/Z-Image-Turbo"))


class _StubVideoPipe:
    """A video pipeline as far as the calibration pass drives it, guider and all."""

    class _Guider:
        def __init__(self) -> None:
            self.guidance_scale = 1.0

    def __init__(self, *, supports = None, guider = False):
        self.calls: list = []
        self.guider = self._Guider() if guider else None
        self._supports = tuple(
            supports
            if supports is not None
            else (
                "prompt",
                "num_inference_steps",
                "width",
                "height",
                "num_frames",
                "generator",
                "output_type",
                "guidance_scale",
                "guidance_scale_2",
            )
        )

    def __call__(self, **kwargs):
        unexpected = sorted(set(kwargs) - set(self._supports))
        assert not unexpected, f"pipeline was passed kwargs it does not take: {unexpected}"
        self.calls.append(kwargs)
        return None


def test_a_video_calibration_renders_a_clip_at_the_grid_it_was_given():
    build = _script()
    pipe = _StubVideoPipe()
    ran = build.render_calibration(
        pipe,
        ("a red fox trotting through falling snow",),
        steps = 20,
        guidance = 5.0,
        width = 832,
        height = 480,
        num_frames = 25,
        seed = 3407,
        device = "cpu",
    )
    assert ran == 1
    (call,) = pipe.calls
    assert (call["width"], call["height"], call["num_frames"]) == (832, 480, 25)
    assert call["num_inference_steps"] == 20 and call["guidance_scale"] == 5.0
    assert call["output_type"] == "latent" and call["generator"].initial_seed() == 3407


def test_a_family_with_no_guidance_kwarg_is_calibrated_through_its_guider():
    """HunyuanVideo-1.5's __call__ takes no guidance at all."""
    build = _script()
    pipe = _StubVideoPipe(
        supports = (
            "prompt",
            "num_inference_steps",
            "width",
            "height",
            "num_frames",
            "generator",
            "output_type",
        ),
        guider = True,
    )
    build.render_calibration(
        pipe,
        ("a candle flame flickering in a dark room",),
        steps = 20,
        guidance = 6.0,
        width = 832,
        height = 480,
        num_frames = 25,
        guidance_via_guider = True,
        device = "cpu",
    )
    assert pipe.guider.guidance_scale == 6.0
    assert "guidance_scale" not in pipe.calls[0]

    with pytest.raises(ValueError) as excinfo:
        build.render_calibration(
            _StubVideoPipe(),
            ("a candle flame",),
            steps = 2,
            guidance = 6.0,
            num_frames = 25,
            guidance_via_guider = True,
            device = "cpu",
        )
    assert "guider" in str(excinfo.value)


def test_the_second_expert_guidance_is_passed_only_when_the_family_names_a_kwarg_for_it():
    build = _script()
    pipe = _StubVideoPipe()
    build.render_calibration(
        pipe,
        ("a herd of horses galloping across a dusty plain",),
        steps = 20,
        guidance = 5.0,
        num_frames = 25,
        cfg2_kwarg = "guidance_scale_2",
        guidance_2 = 4.0,
        device = "cpu",
    )
    assert pipe.calls[0]["guidance_scale_2"] == 4.0
    pipe = _StubVideoPipe()
    build.render_calibration(
        pipe, ("a herd of horses",), steps = 20, guidance = 5.0, num_frames = 25, device = "cpu"
    )
    assert "guidance_scale_2" not in pipe.calls[0]


def test_an_investigation_prompt_module_declaring_CALIB_is_read(tmp_path):
    """The video calibration set is the CALIB list the replayed GPTQ weights were solved on."""
    build = _script()
    module = tmp_path / "prompts.py"
    module.write_text('CALIB = ["a red fox", "a blue whale"]\n')
    assert build.load_calibration_prompts(str(module)) == ("a red fox", "a blue whale")


def test_the_baked_scales_record_the_grid_they_were_measured_at():
    build = _script()
    meta = build.activation_scale_metadata(
        prompts = ("a red fox",),
        schedule_steps = 20,
        scales = {"a": 12.0},
        layers = 1,
        grid = "832x480x25",
    )
    assert meta["grid"] == "832x480x25"


def test_a_whole_model_video_build_bakes_its_scales_through_its_own_pipeline(monkeypatch, tmp_path):
    """A video artifact: every admitted linear at 4 bits, one activation scale each, and the
    top-level baked flag, since a whole-model artifact has no policy block to carry it."""
    build = _script()
    saved = _stub_build_stack(monkeypatch, _fake_state_dict())

    import contextlib

    torch = sys.modules["torch"]
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(empty_cache = lambda: None)

    class _Generator:
        def __init__(self, device = None):
            self._seed = 0

        def manual_seed(self, seed):
            self._seed = seed
            return self

        def initial_seed(self):
            return self._seed

    torch.Generator = _Generator

    class _Linear:
        def __init__(self):
            self.in_features = 1024
            self.out_features = 1024
            self.weight = types.SimpleNamespace(
                shape = (1024, 1024), device = "cuda", dtype = "bfloat16", data = None
            )

    nn = types.ModuleType("torch.nn")
    nn.Linear = _Linear
    torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch.nn", nn)

    transformer = sys.modules["diffusers"].WanTransformer3DModel()
    layers = {"blocks.0.attn1.to_q": _Linear(), "blocks.1.attn1.to_q": _Linear()}
    transformer.named_modules = lambda: list(layers.items())
    monkeypatch.setattr(
        sys.modules["diffusers"].WanTransformer3DModel,
        "from_pretrained",
        classmethod(lambda cls, base, **kwargs: transformer),
    )

    class _Pipe:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list = []
            _Pipe.instances.append(self)

        @classmethod
        def from_pretrained(cls, base, **kwargs):
            return cls(base = base, **kwargs)

        def to(self, device):
            return self

        def set_progress_bar_config(self, disable = True):
            return None

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return None

    sys.modules["diffusers"].WanPipeline = _Pipe

    class _Amax:
        def __init__(self, modules):
            self.modules = dict(modules)

        def attach(self):
            return self

        def detach(self):
            return self

        def unseen(self):
            return []

        def global_scales(self):
            return {fqn: 224.0 for fqn in self.modules}

    from core.inference import diffusion_nvfp4_gptq

    monkeypatch.setattr(diffusion_nvfp4_gptq, "ActivationAmaxAccumulator", _Amax)

    prompts = tmp_path / "prompts.py"
    prompts.write_text('CALIB = ["a red fox", "a blue whale", "a green field"]\n')
    out = tmp_path / "wan5b.pt"
    code = build.main(
        [
            "--base",
            str(tmp_path),  # a local mirror: the id the loader checks comes from --base-id
            "--base-id",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--modality",
            "video",
            "--family",
            "wan2.2-ti2v-5b",
            "--scheme",
            "nvfp4",
            "--out",
            str(out),
            "--bake-activation-scales",
            "--bake-prompts",
            "2",
            "--calib-prompts",
            str(prompts),
        ]
    )
    assert code == 0
    metadata = saved["ckpt"]["metadata"]
    assert metadata["activation_scales_baked"] is True
    assert set(metadata["act_global_scales"]) == set(layers)
    assert metadata["activation_calibration"]["grid"] == "832x480x25"
    assert metadata["activation_calibration"]["prompts"] == 2
    assert metadata["activation_calibration"]["schedule_steps"] == 20
    (pipe,) = _Pipe.instances
    assert pipe.kwargs["transformer"] is transformer
    assert len(pipe.calls) == 2
    call = pipe.calls[0]
    assert (call["width"], call["height"], call["num_frames"]) == (832, 480, 25)
    assert call["num_inference_steps"] == 20 and call["guidance_scale"] == 5.0


def test_a_moe_video_build_calibrates_the_expert_it_was_asked_for(monkeypatch, tmp_path):
    """Both A14B experts share a family, a class and a key set."""
    build = _script()
    _stub_build_stack(monkeypatch, _fake_state_dict())

    import contextlib

    torch = sys.modules["torch"]
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(empty_cache = lambda: None)
    torch.Generator = lambda device = None: types.SimpleNamespace(manual_seed = lambda s: s)

    class _Linear:
        def __init__(self):
            self.in_features = 1024
            self.out_features = 1024
            self.weight = types.SimpleNamespace(
                shape = (1024, 1024), device = "cuda", dtype = "bfloat16", data = None
            )

    nn = types.ModuleType("torch.nn")
    nn.Linear = _Linear
    torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch.nn", nn)

    transformer = sys.modules["diffusers"].WanTransformer3DModel()
    transformer.named_modules = lambda: [("blocks.0.attn1.to_q", _Linear())]
    monkeypatch.setattr(
        sys.modules["diffusers"].WanTransformer3DModel,
        "from_pretrained",
        classmethod(lambda cls, base, **kwargs: transformer),
    )

    built: dict = {}

    class _Pipe:
        def __init__(self, **kwargs):
            built.update(kwargs)

        @classmethod
        def from_pretrained(cls, base, **kwargs):
            return cls(**kwargs)

        def to(self, device):
            return self

        def set_progress_bar_config(self, disable = True):
            return None

        def __call__(self, **kwargs):
            return None

    sys.modules["diffusers"].WanPipeline = _Pipe

    class _Amax:
        def __init__(self, modules):
            self.modules = dict(modules)

        def attach(self):
            return self

        def detach(self):
            return self

        def unseen(self):
            return []

        def global_scales(self):
            return {fqn: 224.0 for fqn in self.modules}

    from core.inference import diffusion_nvfp4_gptq

    monkeypatch.setattr(diffusion_nvfp4_gptq, "ActivationAmaxAccumulator", _Amax)
    prompts = tmp_path / "prompts.py"
    prompts.write_text('CALIB = ["a red fox"]\n')
    code = build.main(
        [
            "--base",
            str(tmp_path),
            "--base-id",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--modality",
            "video",
            "--family",
            "wan2.2-t2v-a14b",
            "--scheme",
            "nvfp4",
            "--component",
            "transformer_2",
            "--out",
            str(tmp_path / "a14b_2.pt"),
            "--bake-activation-scales",
            "--bake-prompts",
            "1",
            "--calib-prompts",
            str(prompts),
        ]
    )
    assert code == 0
    assert built["transformer_2"] is transformer
    assert "transformer" not in built


def test_a_replayed_correction_is_in_place_before_the_scales_are_baked(monkeypatch, tmp_path):
    """--gptq-dir and --bake-activation-scales compose in one order only."""
    build = _script()
    saved = _stub_build_stack(monkeypatch, _fake_state_dict())

    import contextlib
    import json as _json

    torch = sys.modules["torch"]
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(empty_cache = lambda: None)
    torch.Generator = lambda device = None: types.SimpleNamespace(manual_seed = lambda s: s)

    class _Linear:
        def __init__(self):
            self.in_features = 1024
            self.out_features = 1024
            self.weight = types.SimpleNamespace(
                shape = (1024, 1024), device = "cuda", dtype = "bfloat16", data = "dense"
            )

    nn = types.ModuleType("torch.nn")
    nn.Linear = _Linear
    torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch.nn", nn)

    gptq = tmp_path / "gptq"
    (gptq / "weights").mkdir(parents = True)
    (gptq / "gptq_meta.json").write_text(
        _json.dumps(
            {
                "prompts": 32,
                "grid": "832x480x49f_50s",
                "layers": {"blocks.0.attn1.to_q": {"err_rtn": 0.09, "err_gptq": 0.12}},
            }
        )
    )
    (gptq / "gptq_score.json").write_text(
        _json.dumps(
            {"layers": {"blocks.0.attn1.to_q": {"out_err_rtn": 0.03, "out_err_gptq": 0.01}}}
        )
    )
    (gptq / "weights" / "blocks_0_attn1_to_q.pt").write_bytes(b"w")
    torch.load = lambda path, weights_only = True: types.SimpleNamespace(
        shape = (1024, 1024), to = lambda *a: "corrected"
    )

    module = _Linear()
    transformer = sys.modules["diffusers"].WanTransformer3DModel()
    transformer.named_modules = lambda: [("blocks.0.attn1.to_q", module)]
    monkeypatch.setattr(
        sys.modules["diffusers"].WanTransformer3DModel,
        "from_pretrained",
        classmethod(lambda cls, base, **kwargs: transformer),
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

    class _Pipe:
        @classmethod
        def from_pretrained(cls, base, **kwargs):
            return cls()

        def to(self, device):
            return self

        def set_progress_bar_config(self, disable = True):
            return None

        def __call__(self, **kwargs):
            return None

    sys.modules["diffusers"].WanPipeline = _Pipe

    seen: dict = {}

    class _Amax:
        def __init__(self, modules):
            self.modules = dict(modules)
            seen.update({fqn: mod.weight.data for fqn, mod in self.modules.items()})

        def attach(self):
            return self

        def detach(self):
            return self

        def unseen(self):
            return []

        def global_scales(self):
            return {fqn: 224.0 for fqn in self.modules}

    from core.inference import diffusion_nvfp4_gptq

    monkeypatch.setattr(diffusion_nvfp4_gptq, "ActivationAmaxAccumulator", _Amax)
    prompts = tmp_path / "prompts.py"
    prompts.write_text('CALIB = ["a red fox"]\n')
    code = build.main(
        [
            "--base",
            str(tmp_path),
            "--base-id",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--modality",
            "video",
            "--family",
            "wan2.2-ti2v-5b",
            "--scheme",
            "nvfp4",
            "--out",
            str(tmp_path / "wan5b.pt"),
            "--gptq-dir",
            str(gptq),
            "--bake-activation-scales",
            "--bake-prompts",
            "1",
            "--calib-prompts",
            str(prompts),
        ]
    )
    assert code == 0
    assert seen == {"blocks.0.attn1.to_q": "corrected"}
    metadata = saved["ckpt"]["metadata"]
    assert metadata["gptq"]["applied"] == 1
    assert metadata["activation_scales_baked"] is True
    assert set(metadata["act_global_scales"]) == {"blocks.0.attn1.to_q"}
