# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) MiniMax-H3-Image: the GGUF repo and its quant ladder, the assets sd-server is handed (release
image VAE, the H3 Qwen3-VL-32B GGUF with its vision tower), the --rng the card uses, and the routing rules: a GGUF
loads only natively (also on CPU / Apple Silicon, where the pipeline is refused), the pipeline stays on diffusers,
and no single-file load exists on either engine."""

from __future__ import annotations

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import (
    detect_family,
    detect_family_for_pick,
    family_gguf_loadable,
    family_sd_cpp_supported,
    sd_cpp_companion_only_repo_ids,
    sd_cpp_text_encoders_for,
)
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    build_sd_cpp_command,
    build_sd_cpp_server_command,
    text_encoder_flags_for_family,
)
from core.inference.sd_cpp_backend import (
    SdCppDiffusionBackend,
    _family_rng_kwargs,
    sd_cpp_binary_runs_family,
)
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

from .test_diffusion_engine_router import (  # noqa: F401 -- the autouse fixture resets router state
    _clean_env_and_state,
    _set_binary,
    _set_device,
    _set_runnable,
)
from .test_sd_cpp_backend import _FakeEngine

_REPO = "unsloth/MiniMax-H3-Image-GGUF"
_GGUF = "minimax_h3_image_pruned-UD-Q4_K_XL.gguf"
_VAE = ("MiniMaxAI/MiniMax-H3-Image", "vae/diffusion_pytorch_model.safetensors")
_TE = ("unsloth/MiniMax-H3-GGUF", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "llm")
# The repo's 14 files, in bytes (Hub listing).
_LADDER = {
    "Q2_K": 7748879072,
    "UD-Q2_K_XL": 8676733664,
    "Q3_K_S": 8768254688,
    "Q3_K_M": 10422256352,
    "UD-Q3_K_XL": 11359142624,
    "Q4_K_S": 11543087840,
    "Q4_K_M": 12721119968,
    "UD-Q4_K_XL": 13607428832,
    "Q5_K_S": 13934375648,
    "Q5_K_M": 14599107296,
    "UD-Q5_K_XL": 15472470752,
    "Q6_K": 16595710688,
    "UD-Q6_K_XL": 17584980704,
    "Q8_0": 22026760928,
}


def _h3i():
    fam = detect_family(_REPO)
    assert fam is not None and fam.name == "minimax-h3-image"
    return fam


def _binary(tmp_path, name, *, marker):
    path = tmp_path / name
    # "MiniMax-H3" alone is the video model's name: only the image name may admit the build.
    body = b"MiniMax-H3-Image" if marker else b"MiniMax-H3 P-H3-Image"
    path.write_bytes(b"ELF-ish padding " * 64 + body + b" trailing")
    return str(path)


def test_the_family_row_carries_the_native_assets():
    fam = _h3i()
    assert family_sd_cpp_supported(fam) and family_gguf_loadable(fam)
    assert (
        not fam.pipeline_only and fam.native_transformer_files_only and not fam.single_file_loadable
    )
    assert fam.sd_cpp_vae == _VAE and fam.sd_cpp_vae_format is None
    assert sd_cpp_text_encoders_for(fam, _REPO, _GGUF) == (_TE,)
    assert text_encoder_flags_for_family(fam.name) == ("--llm",)
    assert fam.sd_cpp_arch_marker == "MiniMax-H3-Image"
    # The PDD grid is sd.cpp's own H3-Image schedule: no scheduler, shift or sigmas from Studio.
    assert fam.sd_cpp_scheduler is None and fam.sd_cpp_flow_shift is None
    assert fam.sd_cpp_sampling_method == "euler" and fam.sd_cpp_rng == "cpu"
    # References run through the arch-gated reference call (the encoder GGUF holds the vision tower).
    assert fam.sd_cpp_reference and fam.reference_edit and fam.max_condition_images == 16


@pytest.mark.parametrize("quant", sorted(_LADDER))
def test_every_shipped_file_names_the_image_family_not_the_video_one(quant):
    from core.inference.video_families import detect_video_family

    gguf = f"minimax_h3_image_pruned-{quant}.gguf"
    assert detect_family_for_pick(_REPO, gguf) is _h3i()
    assert detect_video_family(_REPO) is None
    assert detect_video_family(f"{_REPO}/{gguf}") is None


def test_the_default_quant_is_the_dynamic_q4_and_every_label_parses():
    from hub.utils.gguf import extract_quant_label, pick_best_gguf

    files = [f"minimax_h3_image_pruned-{q}.gguf" for q in _LADDER]
    assert [extract_quant_label(f) for f in files] == list(_LADDER)
    assert pick_best_gguf(files) == _GGUF
    # Sizes rise with the bits, so the size-based fit walks the ladder in quality order.
    assert list(_LADDER) == sorted(_LADDER, key = _LADDER.get)


def test_the_asset_specs_fetch_the_release_vae_and_the_h3_encoder():
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(_REPO, _GGUF, _h3i())
    assert specs == [(_REPO, _GGUF, "diffusion_model"), (*_VAE, "vae"), _TE]


def test_the_video_repo_lending_its_encoder_stays_a_model():
    companions = sd_cpp_companion_only_repo_ids()
    assert "unsloth/minimax-h3-gguf" not in companions
    assert "minimaxai/minimax-h3-image" not in companions


def test_the_card_rng_reaches_both_builders():
    files = SdCppModelFiles(
        diffusion_model = "/m/h3i.gguf", vae = "/m/vae.safetensors", llm = "/m/te.gguf"
    )
    rng = _family_rng_kwargs(_h3i())
    assert rng == {"rng": "cpu"}
    server = build_sd_cpp_server_command("sd-server", files, host = "127.0.0.1", port = 1, **rng)
    cli = build_sd_cpp_command(
        "sd-cli", files, SdCppGenParams(prompt = "x", rng = "cpu"), output_path = "/o/x.png"
    )
    for cmd in (server, cli):
        assert cmd[cmd.index("--rng") + 1] == "cpu"
    # Other families keep sd.cpp's default.
    z = detect_family("unsloth/Z-Image-Turbo-GGUF")
    assert _family_rng_kwargs(z) == {}
    assert "--rng" not in build_sd_cpp_server_command("sd-server", files, host = "127.0.0.1", port = 1)
    assert "--rng" not in build_sd_cpp_command(
        "sd-cli", files, SdCppGenParams(prompt = "x"), output_path = "/o/x.png"
    )


def test_the_marker_is_the_image_model_name(tmp_path):
    fam = _h3i()
    assert sd_cpp_binary_runs_family(_binary(tmp_path, "new", marker = True), fam)
    # A build with only the H3 video model (and P-H3-Image) cannot load the checkpoint.
    assert not sd_cpp_binary_runs_family(_binary(tmp_path, "old", marker = False), fam)


@pytest.mark.parametrize("backend", ["cuda", "cpu", "mps"])
def test_a_gguf_goes_native_on_every_host(monkeypatch, tmp_path, backend):
    _set_device(monkeypatch, backend)
    _set_runnable(monkeypatch)
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli", marker = True))
    fam = _h3i()
    assert r.predict_engine(fam, model_kind = "gguf") == ENGINE_SD_CPP
    r.select_and_activate_engine(fam, model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_SD_CPP
    if backend == "cuda":
        # The pipeline repo stays on diffusers where it runs today.
        assert r.predict_engine(fam, model_kind = "pipeline") == ENGINE_DIFFUSERS


def test_a_build_without_h3_image_is_refused_not_sent_to_diffusers(monkeypatch, tmp_path):
    _set_device(monkeypatch, "cpu")
    _set_runnable(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli-old", marker = False))
    fam = _h3i()
    fallbacks = []
    with pytest.raises(ValueError, match = "predates 'minimax-h3-image' support"):
        r.select_and_activate_engine(
            fam, model_kind = "gguf", before_fallback = lambda: fallbacks.append(1)
        )
    assert fallbacks == []


def test_an_explicit_diffusers_pick_of_the_gguf_is_refused():
    with pytest.raises(ValueError, match = "Diffusers cannot load a minimax-h3-image GGUF"):
        r.refuse_engine_pick(_h3i(), engine = "diffusers", model_kind = "gguf")


def _cpu_target(monkeypatch):
    import types

    import core.inference.diffusion as diffusion_module
    monkeypatch.setattr(
        diffusion_module,
        "resolve_diffusion_device_target",
        lambda *a, **k: types.SimpleNamespace(device = "cpu", backend = "cpu"),
    )


def test_the_gguf_passes_validation_on_a_cpu_host(monkeypatch, tmp_path):
    from core.inference.diffusion import DiffusionBackend

    # Never the in-tree pipeline module (it needs diffusers): the GGUF route has no device refusal.
    monkeypatch.delenv("UNSLOTH_H3_IMAGE_ALLOW_ANY_DEVICE", raising = False)
    _set_device(monkeypatch, "cpu")
    _set_runnable(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli", marker = True))
    _cpu_target(monkeypatch)
    fam = DiffusionBackend().validate_load_request(_REPO, gguf_filename = _GGUF, model_kind = "gguf")
    assert fam.name == "minimax-h3-image"


def test_the_pipeline_refusal_points_at_the_gguf_repo(monkeypatch):
    try:
        from core.inference import diffusion_minimax_h3_image as h3i
    except Exception as exc:  # noqa: BLE001 - a broken diffusers / hub pair
        pytest.skip(f"diffusers pipelines unavailable: {exc}")
    monkeypatch.delenv(h3i.ALLOW_ANY_DEVICE_ENV, raising = False)
    assert h3i.H3_IMAGE_GGUF_REPO == _REPO
    for device in ("cpu", "mps"):
        refusal = h3i.h3_image_device_refusal(device)
        assert (
            refusal.startswith("MiniMax-H3-Image needs an NVIDIA or AMD GPU") and _REPO in refusal
        )


def test_a_gguf_is_refused_before_eviction_when_no_build_can_take_it(monkeypatch, tmp_path):
    from core.inference.diffusion import DiffusionBackend

    _set_device(monkeypatch, "cpu")
    _set_runnable(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli-old", marker = False))
    _cpu_target(monkeypatch)
    with pytest.raises(ValueError, match = r"upstream master-1088 or newer"):
        DiffusionBackend().validate_load_request(_REPO, gguf_filename = _GGUF, model_kind = "gguf")


@pytest.mark.parametrize(
    "filename", ["minimax_h3_image_pruned_bf16.safetensors", "dit.safetensors"]
)
def test_no_single_file_load_on_either_engine(filename):
    from core.inference.diffusion import DiffusionBackend
    with pytest.raises(ValueError, match = "no single-file transformer load"):
        DiffusionBackend().validate_load_request(
            "MiniMaxAI/MiniMax-H3-Image", gguf_filename = filename, model_kind = "single_file"
        )


def test_the_loose_file_listing_still_offers_no_h3_image_file():
    from hub.utils.comfy_models import _single_file_loadable

    fam = _h3i()
    assert not _single_file_loadable(fam, None, "minimax_h3_image_pruned_bf16.safetensors")
    assert not _single_file_loadable(fam, None, _GGUF)
