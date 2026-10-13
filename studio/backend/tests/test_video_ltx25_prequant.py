# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted LTX-2.5 DiT checkpoints (unsloth/LTX-2.5-FP8): which base seeds them, which file each scheme asks for, and
that the metadata the builder bakes is what this runtime accepts. Network-free and torch-free."""

from __future__ import annotations

import logging

import pytest

from core.inference.diffusion_prequant import (
    PREQUANT_FORMAT,
    PREQUANT_FORMAT_ROTATED,
    _validate_checkpoint,
    prequant_format_for,
    prequant_mirror_path,
    resolve_prequant_source,
)
from core.inference.diffusion_transformer_quant import (
    INT8_CONVROT_ENV,
    convrot_prequant_filename,
    convrot_spec_for_scheme,
    exclude_tokens_for_scheme,
    int8_convrot_enabled,
)
from core.inference.video_families import (
    detect_video_family,
    video_family_prequant_repo,
    video_family_prequant_resident_gb,
    video_family_prequant_schemes,
)

PACK = "Lightricks/LTX-2.5-Diffusers"
SINGLE_FILE_RELEASE = "Lightricks/LTX-2.5"
REPO = "unsloth/LTX-2.5-FP8"


@pytest.fixture
def fam():
    family = detect_video_family(PACK)
    assert family is not None and family.name == "ltx-2.5"
    return family


@pytest.fixture(autouse = True)
def _no_convrot_env(monkeypatch):
    monkeypatch.delenv(INT8_CONVROT_ENV, raising = False)


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_the_pack_resolves_the_hosted_repo(fam, scheme):
    assert video_family_prequant_repo(fam, scheme, PACK) == REPO
    # Keyed on the lowercased id, so the spelling a caller passes does not matter.
    assert video_family_prequant_repo(fam, scheme, PACK.lower()) == REPO


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
@pytest.mark.parametrize("base", [SINGLE_FILE_RELEASE, None, "", "someone/LTX-2.5-Diffusers-fork"])
def test_no_other_base_resolves_it(fam, scheme, base):
    # The single-file release, an unnamed base and a fork never seed a checkpoint baked from the pack's DiT.
    assert video_family_prequant_repo(fam, scheme, base) is None
    assert resolve_prequant_source(fam, scheme, base_repo = base) is None


@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp8"])
def test_only_the_hosted_schemes_resolve(fam, scheme):
    assert video_family_prequant_repo(fam, scheme, PACK) is None
    assert set(video_family_prequant_schemes(fam)) == {"int8", "fp8"}


def test_fp8_asks_for_its_own_file_first(fam):
    src = resolve_prequant_source(fam, "fp8", base_repo = PACK)
    assert (src.kind, src.location, src.filename) == ("repo", REPO, "LTX-2.5-FP8.safetensors")
    assert src.fallback_filenames[0] == "LTX-2.5-FP8.pt"


@pytest.mark.parametrize(
    "env, first, second",
    [
        (None, "LTX-2.5-INT8-ConvRot.safetensors", "LTX-2.5-INT8.safetensors"),
        ("1", "LTX-2.5-INT8-ConvRot.safetensors", "LTX-2.5-INT8.safetensors"),
        ("0", "LTX-2.5-INT8.safetensors", "LTX-2.5-INT8.pt"),
    ],
)
def test_int8_asks_for_the_rotated_file_unless_convrot_is_off(fam, monkeypatch, env, first, second):
    if env is not None:
        monkeypatch.setenv(INT8_CONVROT_ENV, env)
    src = resolve_prequant_source(fam, "int8", base_repo = PACK)
    assert (src.location, src.filename, src.fallback_filenames[0]) == (REPO, first, second)
    # The plain file stays in the chain behind the rotated one, so a repo without it still serves int8.
    assert "LTX-2.5-INT8.safetensors" in (src.filename,) + tuple(src.fallback_filenames)


def test_quantizing_at_load_stays_plain_int8():
    # The rotation lives in the hosted file's metadata; there is no runtime ConvRot spec for LTX-2.5, so the default-on
    # flag only picks the hosted file and never rotates a dense DiT.
    assert int8_convrot_enabled("ltx-2.5")
    assert convrot_spec_for_scheme("int8", "ltx-2.5") == (0, ())
    assert convrot_prequant_filename("int8", "ltx-2.5") is None


def test_ltx23_keeps_its_own_hosted_dit():
    ltx2 = detect_video_family("Lightricks/LTX-2.3")
    assert ltx2 is not None and ltx2.name == "ltx-2"
    src = resolve_prequant_source(ltx2, "fp8", base_repo = "Lightricks/LTX-2.3")
    assert (src.location, src.filename) == ("unsloth/LTX-2.3-FP8", "LTX-2.3-FP8.safetensors")
    assert resolve_prequant_source(ltx2, "int8", base_repo = "Lightricks/LTX-2.3") is None
    assert video_family_prequant_repo(ltx2, "fp8", PACK) is None


def test_the_plan_prices_the_hosted_files(fam):
    int8 = video_family_prequant_resident_gb(fam, "int8")
    fp8 = video_family_prequant_resident_gb(fam, "fp8")
    assert int8 == pytest.approx(25.11) and fp8 == pytest.approx(19.06)
    assert fp8 < int8 < fam.bf16_components_gb[0]


def test_only_a_pack_pipeline_load_plans_the_seed(fam, monkeypatch):
    from core.inference import video as vid
    from core.inference import video_denoiser_prequant as vdp

    monkeypatch.setattr(vdp, "hosted_denoiser_refused", lambda *a, **k: False)
    sources = vdp.denoiser_prequant_sources(fam, "int8", PACK)
    assert set(sources) == {"transformer"}
    assert sources["transformer"].location == REPO
    assert vdp.denoiser_prequant_sources(fam, "int8", SINGLE_FILE_RELEASE) is None
    assert vdp.denoiser_prequant_sources(fam, "auto", PACK) is None
    assert vid._planned_denoiser_components(fam, "pipeline") == ("transformer",)
    assert vid._planned_denoiser_components(fam, "single_file") == ()


def test_a_local_mirror_serves_the_staged_files(monkeypatch, tmp_path):
    folder = tmp_path / "unsloth" / "LTX-2.5-FP8"
    folder.mkdir(parents = True)
    for name in ("LTX-2.5-FP8.safetensors", "LTX-2.5-INT8-ConvRot.safetensors"):
        (folder / name).write_bytes(b"x")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PREQUANT_MIRROR", str(tmp_path))
    assert prequant_mirror_path(REPO, "LTX-2.5-INT8-ConvRot.safetensors") == str(
        folder / "LTX-2.5-INT8-ConvRot.safetensors"
    )
    assert prequant_mirror_path(REPO, "LTX-2.5-INT8.safetensors") is None


# What scripts/build_prequant_checkpoint.py wrote into the staged files' unsloth_metadata (fingerprint, version and
# rotation fqn list left out). The loader checks these against this runtime's filter, so a drift in the int8 excludes
# would refuse every hosted file and silently fall back to quantizing at load.
_BAKED = {
    "base_model_id": PACK,
    "family": "ltx-2.5",
    "min_features": 512,
    "torch_dtype": "bfloat16",
    "quant_backend": "torchao",
    "transformer_class": "LTX2VideoTransformer3DModel",
    "component": "transformer",
}
_BAKED_INT8_EXCLUDES = [
    "norm",
    "_mod",
    "modulation",
    "timestep_embed",
    "guidance_embed",
    "time_text_embed",
    "pooled",
    "time_embed",
    "audio",
    "av_cross_attn",
    "adaln",
]


def _meta(scheme: str, rotated: bool = False) -> dict:
    meta = dict(_BAKED, scheme = scheme)
    if scheme == "fp8":
        meta.update(
            exclude_name_tokens = [],
            require_bf16 = True,
            require_divisible = 16,
            fast_accum = True,
            fp8_granularity = "per_row",
        )
    else:
        meta.update(
            exclude_name_tokens = list(_BAKED_INT8_EXCLUDES),
            require_bf16 = False,
            require_divisible = 0,
            fast_accum = None,
        )
    if rotated:
        meta.update(
            activation_rotation = "convrot_hadamard_v1",
            activation_rotation_group = 256,
            activation_rotation_fqns = ["transformer_blocks.0.attn1.to_q"],
        )
    return meta


def test_the_baked_int8_excludes_are_this_runtimes():
    assert list(exclude_tokens_for_scheme("int8", "ltx-2.5")) == _BAKED_INT8_EXCLUDES
    assert list(exclude_tokens_for_scheme("fp8", "ltx-2.5")) == []


@pytest.mark.parametrize(
    "scheme, rotated, fmt",
    [
        ("fp8", False, PREQUANT_FORMAT),
        ("int8", False, PREQUANT_FORMAT),
        ("int8", True, PREQUANT_FORMAT_ROTATED),
    ],
)
def test_the_baked_metadata_passes_the_loader(scheme, rotated, fmt):
    meta = _meta(scheme, rotated)
    # Only the rotated file carries the v2 tag an older loader refuses; FP8 / INT8 stay v1.
    assert prequant_format_for(meta) == fmt
    ckpt = {"format": fmt, "metadata": meta}
    assert _validate_checkpoint(
        ckpt,
        scheme,
        PACK,
        logging.getLogger("t"),
        min_features = 512,
        fast_accum = True if scheme == "fp8" else None,
        component = "transformer",
        metadata_only = True,
    )
    # The same file is refused for any other base, so it can never seed the single-file release or LTX-2.3.
    for other in (SINGLE_FILE_RELEASE, "Lightricks/LTX-2.3"):
        assert not _validate_checkpoint(
            ckpt, scheme, other, logging.getLogger("t"), min_features = 512, metadata_only = True
        )


def _api(siblings = None, missing = False):
    import types

    asked: list = []

    def model_info(repo_id, **_kwargs):
        asked.append(repo_id)
        if missing:
            raise RuntimeError(f"404 Client Error: {repo_id}")
        return types.SimpleNamespace(
            siblings = [
                types.SimpleNamespace(rfilename = n, size = s) for n, s in (siblings or {}).items()
            ]
        )

    return types.SimpleNamespace(token = None, model_info = model_info, asked = asked)


_PUBLISHED = {
    "LTX-2.5-FP8.safetensors": 19_056_002_016,
    "LTX-2.5-INT8.safetensors": 25_105_102_312,
    "LTX-2.5-INT8-ConvRot.safetensors": 25_105_121_000,
    "README.md": 1,
}


@pytest.fixture
def _planning(monkeypatch):
    import core.inference.diffusion_prequant as pq
    from core.inference import video_denoiser_prequant as vdp

    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    monkeypatch.setattr(vdp, "hosted_denoiser_refused", lambda *a, **k: False)


@pytest.mark.parametrize(
    "scheme, env, expected",
    [
        ("int8", None, "LTX-2.5-INT8-ConvRot.safetensors"),
        ("int8", "0", "LTX-2.5-INT8.safetensors"),
        ("fp8", None, "LTX-2.5-FP8.safetensors"),
    ],
)
def test_once_published_the_plan_credits_the_hosted_file(
    fam, monkeypatch, _planning, scheme, env, expected
):
    from core.inference.video import VideoBackend

    if env is not None:
        monkeypatch.setenv(INT8_CONVROT_ENV, env)
    api = _api(_PUBLISHED)
    repo, files = VideoBackend._denoiser_prequant_hub_files(fam, scheme, PACK, api)
    assert (repo, files) == (REPO, [(expected, _PUBLISHED[expected])])
    assert api.asked == [REPO]


def test_until_published_the_plan_keeps_the_dense_shards(fam, _planning):
    # The repo 404s today: the plan keeps transformer/, and the load quantizes at load instead (the hosted files'
    # renders are pixel-identical to that, so nothing but load time and the download changes).
    from core.inference.video import VideoBackend

    assert VideoBackend._denoiser_prequant_hub_files(fam, "int8", PACK, _api(missing = True)) == (
        None,
        [],
    )
    # A listing without the rotated file still serves int8 from the plain one.
    plain = {k: v for k, v in _PUBLISHED.items() if "ConvRot" not in k}
    assert VideoBackend._denoiser_prequant_hub_files(fam, "int8", PACK, _api(plain)) == (
        REPO,
        [("LTX-2.5-INT8.safetensors", plain["LTX-2.5-INT8.safetensors"])],
    )


# ComfyUI's LTX-2.5 templates (video_ltx2_5_t2v / i2v / flf2v) load ltx-2.5-22b-distilled-transformer-comfy-int8-convrot,
# so auto seeds the hosted INT8-ConvRot DiT even on a card where the bf16 DiT fits resident. Any other scheme the auto
# ladder lands on (fp8 where int8 does not run) still yields to a resident bf16 DiT, as for every other video family.
@pytest.mark.parametrize(
    "requested, scheme, expected",
    [
        ("auto", "int8", "int8"),
        (None, "int8", "int8"),
        ("auto", "fp8", None),
        ("fp8", "fp8", "fp8"),
        ("int8", "int8", "int8"),
    ],
)
def test_auto_seeds_int8_where_bf16_fits_like_comfyui(fam, monkeypatch, requested, scheme, expected):
    import types

    import core.inference.video as vid

    backend = vid.VideoBackend()
    backend._state = types.SimpleNamespace(pipe = None)
    monkeypatch.setattr(vid, "_video_auto_denoiser_scheme", lambda fam, **kw: scheme)
    monkeypatch.setattr(vid, "_pipeline_device_mib", lambda pipe, ordinal = None: 0)
    # bf16 fits resident on this card, and so does the artifact.
    monkeypatch.setattr(vid, "_video_seed_stays_resident", lambda fam, **kw: True)
    monkeypatch.setattr(
        vid,
        "resolve_diffusion_device_target",
        lambda **_k: types.SimpleNamespace(device = "cuda", dtype = None, ordinal = None),
    )
    assert (
        backend._video_planned_auto_denoiser_scheme(
            fam, base = PACK, kind = "pipeline", transformer_quant = requested, speed_mode = None
        )
        == expected
    )


def test_only_ltx25_int8_skips_the_resident_bf16_rule(fam):
    import core.inference.video as vid

    assert fam.profile.auto_prequant_when_resident == frozenset({"int8"})
    assert not vid._auto_seed_yields_to_resident_bf16("int8", "auto", fam)
    assert vid._auto_seed_yields_to_resident_bf16("fp8", "auto", fam)
    # LTX-2.3 and the other video families keep a resident bf16 DiT under auto.
    for repo in ("Lightricks/LTX-2", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"):
        other = detect_video_family(repo)
        assert other.name != fam.name
        assert vid._auto_seed_yields_to_resident_bf16("int8", "auto", other)
    assert vid._auto_seed_yields_to_resident_bf16("int8", "auto")
