# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A single ``.safetensors`` checkpoint loads from ANY Hub repo (ComfyUI repacks, Kijai, community fp8
files), while everything that could unpickle or execute repo content stays repo-gated: other weight
suffixes, full pipelines, and ``base_repo``."""

import json
import pickle
import types

import pytest

from core.inference.diffusion import DiffusionBackend, _resolve_base_repo
from core.inference.diffusion_families import detect_family
from core.inference.diffusion_single_file_trust import (
    assert_safetensors_file,
    is_hub_safetensors_single_file,
    single_file_load_allowed,
)
from core.inference.video import VideoBackend
from core.inference.video_families import resolve_video_base_repo

UNTRUSTED_IMAGE_REPO = "Comfy-Org/z_image_turbo"
IMAGE_FILE = "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"
UNTRUSTED_VIDEO_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
VIDEO_FILE = "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
VIDEO_FAMILY = "wan2.2-ti2v-5b"
PICKLE_NAMES = (
    "split_files/diffusion_models/dit.pt",
    "split_files/diffusion_models/dit.pth",
    "split_files/diffusion_models/dit.bin",
    "split_files/diffusion_models/dit.ckpt",
    "split_files/diffusion_models/dit.pkl",
    "dit.safetensors.pt",
    "dit.npz",
)


def _write_safetensors(
    path,
    header = None,
    data = b"\x00\x00\x80\x3f",
):
    header = (
        header
        if header is not None
        else {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    )
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + data)
    return path


# -- filename rule ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [IMAGE_FILE, VIDEO_FILE, "model.safetensors", "a/b/MODEL.SafeTensors"],
)
def test_safetensors_repo_relative_names_qualify(name):
    assert is_hub_safetensors_single_file(name)


@pytest.mark.parametrize(
    "name",
    [
        *PICKLE_NAMES,
        None,
        "",
        ".safetensors",
        "../outside.safetensors",
        "a/../../outside.safetensors",
        "/abs/path.safetensors",
        "C:/x.safetensors",
        "a\\b.safetensors",
        " model.safetensors",
        "a//b.safetensors",
        "model.gguf",
        "model_index.json",
        "pipeline.py",
    ],
)
def test_other_names_do_not_qualify(name):
    assert not is_hub_safetensors_single_file(name)


def test_trust_decision_only_widens_single_file_kind():
    assert single_file_load_allowed(False, "single_file", IMAGE_FILE)
    assert not single_file_load_allowed(False, "pipeline", None)
    assert not single_file_load_allowed(False, "pipeline", IMAGE_FILE)
    assert not single_file_load_allowed(False, "single_file", "dit.pt")
    # A trusted repo keeps every kind it had.
    assert single_file_load_allowed(True, "pipeline", None)


# -- image validator ---------------------------------------------------------------------------------


def test_image_untrusted_repo_safetensors_single_file_is_accepted():
    fam = DiffusionBackend().validate_load_request(UNTRUSTED_IMAGE_REPO, gguf_filename = IMAGE_FILE)
    assert fam.name == "z-image"


@pytest.mark.parametrize("name", PICKLE_NAMES)
def test_image_untrusted_repo_pickle_formats_are_refused(name):
    with pytest.raises(ValueError, match = "single .safetensors checkpoint"):
        DiffusionBackend().validate_load_request(
            UNTRUSTED_IMAGE_REPO, gguf_filename = name, model_kind = "single_file"
        )


def test_image_untrusted_repo_traversal_name_is_refused():
    with pytest.raises(ValueError, match = "single .safetensors checkpoint"):
        DiffusionBackend().validate_load_request(
            UNTRUSTED_IMAGE_REPO, gguf_filename = "../../etc/x.safetensors"
        )


def test_image_untrusted_repo_pipeline_load_is_still_refused():
    with pytest.raises(ValueError, match = "full pipelines"):
        DiffusionBackend().validate_load_request(UNTRUSTED_IMAGE_REPO)
    with pytest.raises(ValueError, match = "full pipelines"):
        DiffusionBackend().validate_load_request(UNTRUSTED_IMAGE_REPO, model_kind = "pipeline")


def test_image_untrusted_base_repo_is_still_refused():
    """The base supplies configs and companions through from_pretrained, so it stays repo-gated even beside an admitted
    single file (including the untrusted repo naming itself as base)."""
    for base in ("evil/companions", UNTRUSTED_IMAGE_REPO):
        with pytest.raises(ValueError, match = "base_repo"):
            DiffusionBackend().validate_load_request(
                UNTRUSTED_IMAGE_REPO, gguf_filename = IMAGE_FILE, base_repo = base
            )


def test_image_untrusted_card_tag_never_becomes_the_base(monkeypatch):
    """No request passes the untrusted repo as config/base: its own base_model card tag is dropped for the family base."""
    monkeypatch.setattr(
        "core.inference.diffusion._hf_base_model", lambda repo_id, token: UNTRUSTED_IMAGE_REPO
    )
    fam = detect_family("z-image-turbo")
    assert _resolve_base_repo(UNTRUSTED_IMAGE_REPO, None, fam, None) == fam.base_repo
    assert fam.base_repo.lower() != UNTRUSTED_IMAGE_REPO.lower()


# -- video validator ---------------------------------------------------------------------------------


def test_video_untrusted_repo_safetensors_single_file_is_accepted():
    fam = VideoBackend().validate_load_request(
        UNTRUSTED_VIDEO_REPO, gguf_filename = VIDEO_FILE, family_override = VIDEO_FAMILY
    )
    assert fam.name == VIDEO_FAMILY
    # Base resolution for a single-file pick never yields the pick's own repo.
    assert resolve_video_base_repo(fam, None) == fam.base_repo
    assert fam.base_repo.lower() != UNTRUSTED_VIDEO_REPO.lower()


@pytest.mark.parametrize("name", PICKLE_NAMES)
def test_video_untrusted_repo_pickle_formats_are_refused(name):
    with pytest.raises(ValueError, match = "single .safetensors checkpoint"):
        VideoBackend().validate_load_request(
            UNTRUSTED_VIDEO_REPO,
            gguf_filename = name,
            model_kind = "single_file",
            family_override = VIDEO_FAMILY,
        )


def test_video_untrusted_repo_pipeline_load_is_still_refused():
    with pytest.raises(ValueError, match = "full pipelines"):
        VideoBackend().validate_load_request(
            UNTRUSTED_VIDEO_REPO, model_kind = "pipeline", family_override = VIDEO_FAMILY
        )


def test_video_untrusted_base_repo_is_still_refused():
    for base in ("evil/companions", UNTRUSTED_VIDEO_REPO):
        with pytest.raises(ValueError, match = "base_repo"):
            VideoBackend().validate_load_request(
                UNTRUSTED_VIDEO_REPO,
                gguf_filename = VIDEO_FILE,
                family_override = VIDEO_FAMILY,
                base_repo = base,
            )


# -- header check ------------------------------------------------------------------------------------


def test_header_check_accepts_a_real_safetensors_file(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch = pytest.importorskip("torch")
    path = tmp_path / "real.safetensors"
    safetensors_torch.save_file({"a": torch.zeros(2, 3), "b": torch.ones(4)}, str(path))
    assert_safetensors_file(path)
    assert_safetensors_file(_write_safetensors(tmp_path / "min.safetensors"))


def test_header_check_refuses_a_pickle_named_safetensors(tmp_path):
    path = tmp_path / "evil.safetensors"
    path.write_bytes(pickle.dumps({"w": [1.0]}))
    with pytest.raises(ValueError, match = "not a valid safetensors checkpoint"):
        assert_safetensors_file(path)


def test_header_check_refuses_a_torch_save_named_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "evil.safetensors"
    torch.save({"w": torch.zeros(2)}, str(path))
    with pytest.raises(ValueError, match = "not a valid safetensors checkpoint"):
        assert_safetensors_file(path)


@pytest.mark.parametrize(
    "writer",
    [
        lambda p: p.write_bytes(b""),
        lambda p: p.write_bytes(b"\x01\x02"),
        # Length prefix far past the end of the file.
        lambda p: p.write_bytes((1 << 40).to_bytes(8, "little") + b"{}"),
        # Not JSON.
        lambda p: p.write_bytes((4).to_bytes(8, "little") + b"\xff\xfe\x00{"),
        # JSON but not an object.
        lambda p: p.write_bytes((2).to_bytes(8, "little") + b"[]"),
        # Tensor data past the end of the data section.
        lambda p: _write_safetensors(
            p, {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        ),
        # Missing dtype.
        lambda p: _write_safetensors(p, {"w": {"shape": [1], "data_offsets": [0, 4]}}),
        # Only metadata, no tensors.
        lambda p: _write_safetensors(p, {"__metadata__": {"format": "pt"}}, data = b""),
    ],
)
def test_header_check_refuses_malformed_files(tmp_path, writer):
    path = tmp_path / "bad.safetensors"
    writer(path)
    with pytest.raises(ValueError, match = "safetensors checkpoint"):
        assert_safetensors_file(path)


# -- catalog classification ---------------------------------------------------------------------------


def _repo_info(repo_id, names):
    files = [types.SimpleNamespace(file_name = n) for n in names]
    return types.SimpleNamespace(repo_id = repo_id, revisions = [types.SimpleNamespace(files = files)])


def test_cached_untrusted_single_file_repo_is_listed_but_untrusted_pipeline_is_not(monkeypatch):
    from hub.services.models import catalog_classification as cc

    single = _repo_info(UNTRUSTED_IMAGE_REPO, [IMAGE_FILE])
    monkeypatch.setattr(cc, "_repo_has_pipeline_index", lambda info, selected = None: False)
    assert cc._untrusted_repo_single_files_loadable(single)
    assert cc._cached_repo_task(single) == "text-to-image"

    pickles_only = _repo_info(UNTRUSTED_IMAGE_REPO, ["split_files/diffusion_models/dit.ckpt"])
    assert not cc._untrusted_repo_single_files_loadable(pickles_only)
    assert cc._cached_repo_task(pickles_only) is None

    # An untrusted PIPELINE repo still has no task: the loader refuses it.
    monkeypatch.setattr(cc, "_repo_has_pipeline_index", lambda info, selected = None: True)
    assert not cc._untrusted_repo_single_files_loadable(single)
    assert cc._cached_repo_task(single) is None
