# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI models folders in the local inventory.

A ComfyUI ``models/diffusion_models`` holds many loose ``.safetensors`` side by side with no
``config.json``; each denoiser must list as its own row (text encoders, VAEs, LoRAs and shards
not), a pick of one file must load that file, and registering a ComfyUI root (or its
``models/``) must reach those folders, plus what ``extra_model_paths.yaml`` adds.

The header classifier is stubbed by name here, so these pin the listing contract, not the
classifier. No GPU, no network.
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from core.inference import diffusion_content
from hub.services.models import local_inventory
from hub.utils import comfy_models
from storage import studio_db

# What the stubbed header check calls each test file. Anything not named is a denoiser.
_NOT_DIT = {
    "qwen_3_4b.safetensors": diffusion_content.ROLE_TEXT_ENCODER,
    "clip_l.safetensors": diffusion_content.ROLE_TEXT_ENCODER,
    "ae.safetensors": diffusion_content.ROLE_VAE,
    "wan2.2_vae.safetensors": diffusion_content.ROLE_VAE,
    "pixel_style_lora.safetensors": diffusion_content.ROLE_LORA,
}


@pytest.fixture(autouse = True)
def _stub_header_check(monkeypatch):
    def inspect(path: str):
        name = Path(path).name
        if name in _NOT_DIT:
            return diffusion_content.CheckpointInfo(_NOT_DIT[name])
        if name == "notes.safetensors":
            # unreadable / unknown layout and no family word: the name fallback refuses it
            return diffusion_content.CheckpointInfo(diffusion_content.ROLE_UNKNOWN)
        if name == "flux1-dev-unknown-layout.safetensors":
            # unknown layout but a family word: the name fallback offers it
            return diffusion_content.CheckpointInfo(diffusion_content.ROLE_UNKNOWN)
        if name == "unsupported_arch_dit.safetensors":
            # a DiT the header recognises as no supported family: not a loadable pick
            return diffusion_content.CheckpointInfo(diffusion_content.ROLE_DIT)
        # a DiT whose header agrees with its name; a renamed file reads as FLUX.1
        from core.inference.diffusion_families import detect_family

        fam = detect_family(name)
        return diffusion_content.CheckpointInfo(
            diffusion_content.ROLE_DIT, family = fam.name if fam else "flux.1", page = "image"
        )

    monkeypatch.setattr(diffusion_content, "inspect_checkpoint", inspect)


@pytest.fixture(autouse = True)
def _registrable_tmp(monkeypatch):
    from hub.storage import scan_folders
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [])


def _st(path: Path, keys = ("x.weight",)) -> Path:
    """A real (tiny) safetensors file: 8-byte LE header length + JSON header + data."""
    path.parent.mkdir(parents = True, exist_ok = True)
    header = json.dumps(
        {
            k: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
            for i, k in enumerate(keys)
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0" * len(keys))
    return path


def _appledouble(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00" + b"\0" * 24)
    return path


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"GGUF" + b"\0" * 28)
    return path


def _comfy_diffusion_folder(folder: Path) -> dict[str, Path]:
    return {
        "flux": _st(folder / "flux1-dev.safetensors"),
        "zimage": _st(folder / "z_image_turbo_bf16.safetensors"),
        "renamed": _st(folder / "my_favourite_model.safetensors"),
        "te": _st(folder / "qwen_3_4b.safetensors"),
        "vae": _st(folder / "ae.safetensors"),
        "lora": _st(folder / "pixel_style_lora.safetensors"),
        "shard": _st(folder / "diffusion_pytorch_model-00001-of-00002.safetensors"),
        "adapter": _st(folder / "adapter_model.safetensors"),
        "appledouble": _appledouble(folder / "._flux1-dev.safetensors"),
        "unknown_no_family": _st(folder / "notes.safetensors"),
        "dit_no_family": _st(folder / "unsupported_arch_dit.safetensors"),
        "unknown_family": _st(folder / "flux1-dev-unknown-layout.safetensors"),
        "gguf": _gguf(folder / "qwen-image-Q4_K_M.gguf"),
    }


def _rows_by_name(rows) -> dict[str, object]:
    return {Path(row.path).name: row for row in rows}


_EXPECTED_DIT_FILES = {
    "flux1-dev.safetensors",
    "flux1-dev-unknown-layout.safetensors",
    "z_image_turbo_bf16.safetensors",
    "my_favourite_model.safetensors",
}


def test_loose_checkpoints_keep_only_offered_whole_files(tmp_path):
    _comfy_diffusion_folder(tmp_path)
    names = {p.name for p in comfy_models.loose_diffusion_checkpoints(tmp_path)}
    assert names == _EXPECTED_DIT_FILES


def test_loose_checkpoints_skip_a_folder_that_is_one_model(tmp_path):
    _comfy_diffusion_folder(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert comfy_models.loose_diffusion_checkpoints(tmp_path) == []


def test_hub_scan_lists_each_dit_file_of_a_multi_file_folder(tmp_path):
    _comfy_diffusion_folder(tmp_path)
    rows = _rows_by_name(local_inventory._scan_custom_folder(tmp_path))
    files = {name for name in rows if name.endswith(".safetensors")}
    assert files == _EXPECTED_DIT_FILES
    for name in _EXPECTED_DIT_FILES:
        row = rows[name]
        assert row.artifact_kind == "single_file_checkpoint"
        # The row's load id IS the file: the picker loads exactly it (dir + filename, single_file).
        assert row.load_id == str(tmp_path / name)
        assert not row.capabilities.can_chat and not row.capabilities.can_train
    # The GGUF beside them still lists as before.
    assert "qwen-image-Q4_K_M.gguf" in rows


def test_compat_scan_lists_each_dit_file_of_a_multi_file_folder(tmp_path):
    _comfy_diffusion_folder(tmp_path)
    rows = _rows_by_name(models_route._scan_models_dir(tmp_path))
    assert {n for n in rows if n.endswith(".safetensors")} == _EXPECTED_DIT_FILES
    assert "qwen-image-Q4_K_M.gguf" in rows
    assert str(tmp_path) not in {row.path for row in rows.values()}


def test_compat_scan_keeps_the_bare_single_file_folder_row(tmp_path):
    # Exactly one checkpoint and nothing else: unchanged, the folder row the loader rescues.
    _st(tmp_path / "flux1-dev.safetensors")
    [row] = models_route._scan_models_dir(tmp_path)
    assert row.path == str(tmp_path)


def test_hub_scan_respects_the_per_folder_limit(tmp_path):
    for i in range(5):
        _st(tmp_path / f"flux1-dev-{i}.safetensors")
    rows = local_inventory._scan_models_dir(tmp_path, limit = 3)
    assert len(rows) == 3


def _comfy_root(root: Path) -> Path:
    models = root / "models"
    _comfy_diffusion_folder(models / "diffusion_models")
    _st(models / "unet" / "wan" / "wan2.2_ti2v_5B_fp16.safetensors")
    _st(models / "checkpoints" / "flux1-schnell-fp8.safetensors")
    _st(models / "text_encoders" / "qwen_3_4b.safetensors")
    _st(models / "clip" / "clip_l.safetensors")
    _st(models / "vae" / "ae.safetensors")
    (models / "loras").mkdir(parents = True)
    (root / "custom_nodes").mkdir()
    return models


def test_comfy_layout_from_root_and_from_models_dir(tmp_path):
    models = _comfy_root(tmp_path)
    for folder in (tmp_path, models):
        layout = comfy_models.comfy_layout(folder)
        assert layout is not None
        assert layout.models_dir == models
        assert layout.dit_dirs == (
            models / "diffusion_models",
            models / "unet",
            models / "checkpoints",
        )
        assert layout.text_encoder_dirs == (models / "text_encoders", models / "clip")
        assert layout.vae_dirs == (models / "vae",)


def test_plain_folders_are_not_comfy(tmp_path):
    _comfy_diffusion_folder(tmp_path / "diffusion_models")
    # A lone diffusion_models folder, with no other ComfyUI role folder beside it, is not a models dir.
    assert comfy_models.comfy_layout(tmp_path) is None
    assert comfy_models.comfy_layout(tmp_path / "diffusion_models") is None
    assert comfy_models.comfy_dit_scan_roots(tmp_path / "nowhere") == ()


def test_extra_model_paths_yaml(tmp_path, monkeypatch):
    root = tmp_path / "ComfyUI"
    root.mkdir()
    shared = tmp_path / "shared"
    (shared / "dits").mkdir(parents = True)
    (shared / "more_dits").mkdir(parents = True)
    (shared / "encoders").mkdir(parents = True)
    absolute_vae = tmp_path / "elsewhere" / "vae"
    absolute_vae.mkdir(parents = True)
    no_base = root / "local_unet"
    no_base.mkdir()
    monkeypatch.setenv("COMFY_TEST_SHARED", str(shared))
    (root / "extra_model_paths.yaml").write_text(
        "a111:\n"
        "  base_path: ../shared\n"
        "  is_default: true\n"
        "  diffusion_models: |\n"
        "    dits\n"
        "    more_dits\n"
        "\n"
        "    missing_dir\n"
        "  text_encoders: encoders\n"
        "  loras: loras\n"
        f"  vae: {absolute_vae}\n"
        "env_section:\n"
        "  base_path: $COMFY_TEST_SHARED\n"
        "  clip: encoders\n"
        "no_base:\n"
        "  unet: local_unet\n"
        "empty_section:\n"
    )
    extra = comfy_models.read_extra_model_paths(root / "extra_model_paths.yaml")
    assert extra["diffusion_models"] == [
        shared / "dits",
        shared / "more_dits",
        shared / "missing_dir",
    ]
    assert extra["unet"] == [no_base]
    assert "loras" not in extra

    layout = comfy_models.comfy_layout(root)
    assert layout is not None and layout.models_dir is None
    # Missing folders dropped; the same folder listed twice (text_encoders + clip) kept once.
    assert layout.dit_dirs == (shared / "dits", shared / "more_dits", no_base)
    assert layout.text_encoder_dirs == (shared / "encoders",)
    assert layout.vae_dirs == (absolute_vae,)


def test_extra_model_paths_beside_a_models_dir_merges(tmp_path):
    models = _comfy_root(tmp_path)
    extra_dits = tmp_path / "extra_dits"
    _st(extra_dits / "qwen_image_bf16.safetensors")
    (tmp_path / "extra_model_paths.yaml").write_text(
        f"comfyui:\n  diffusion_models: {extra_dits}\n"
    )
    for folder in (tmp_path, models):
        layout = comfy_models.comfy_layout(folder)
        assert layout.dit_dirs[-1] == extra_dits
        assert models / "diffusion_models" in layout.dit_dirs


@pytest.mark.parametrize(
    "text",
    ["not: [valid", "- just\n- a list\n", "", "a: 1\n", "x:\n  diffusion_models: 5\n"],
)
def test_broken_extra_model_paths_yaml_is_ignored(tmp_path, text):
    (tmp_path / "extra_model_paths.yaml").write_text(text)
    assert comfy_models.read_extra_model_paths(tmp_path / "extra_model_paths.yaml") == {}
    assert comfy_models.comfy_layout(tmp_path) is None


def test_extra_model_paths_refuses_system_folders(tmp_path, monkeypatch):
    from hub.storage import scan_folders

    denied = tmp_path / "system"
    (denied / "dits").mkdir(parents = True)
    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [str(denied)])
    (tmp_path / "extra_model_paths.yaml").write_text(f"x:\n  diffusion_models: {denied / 'dits'}\n")
    assert comfy_models.comfy_layout(tmp_path) is None


_COMFY_FILES = _EXPECTED_DIT_FILES | {
    "wan2.2_ti2v_5B_fp16.safetensors",
    "flux1-schnell-fp8.safetensors",
}


def test_hub_scan_of_a_comfy_root_reaches_its_denoiser_folders(tmp_path):
    models = _comfy_root(tmp_path)
    for folder in (tmp_path, models):
        rows = local_inventory._scan_custom_folder(folder)
        names = {Path(r.path).name for r in rows if r.path.endswith(".safetensors")}
        assert names == _COMFY_FILES, folder
        # text_encoders/, clip/ and vae/ are never offered as models
        assert not any(Path(r.path).parent.name in ("text_encoders", "clip", "vae") for r in rows)


def test_compat_inventory_of_a_registered_comfy_root(tmp_path):
    _comfy_root(tmp_path / "ComfyUI")
    rows = models_route.collect_local_models(
        tmp_path / "models_root",
        custom_folders = [{"path": str(tmp_path / "ComfyUI")}],
        sources = models_route._CompatLocalInventorySources(
            tmp_path / "e" / "active", tmp_path / "e" / "legacy", tmp_path / "e" / "default", (), ()
        ),
    )
    names = {Path(r.path).name for r in rows if r.path.endswith(".safetensors")}
    assert names == _COMFY_FILES
    assert all(r.source == "custom" for r in rows if r.path.endswith(".safetensors"))
    # the role folders are containers: the publisher/model scan must not list them as models
    role_rows = [r.path for r in rows if Path(r.path).parent.name == "models"]
    assert role_rows == []


def test_split_local_checkpoint_path(tmp_path):
    from core.inference.diffusion import split_local_checkpoint_path

    files = _comfy_diffusion_folder(tmp_path)
    assert split_local_checkpoint_path(str(files["flux"])) == (
        str(tmp_path),
        "flux1-dev.safetensors",
    )
    assert split_local_checkpoint_path(str(tmp_path)) is None
    assert split_local_checkpoint_path(str(files["gguf"])) is None
    assert split_local_checkpoint_path(str(tmp_path / "missing.safetensors")) is None
    assert split_local_checkpoint_path("black-forest-labs/FLUX.1-dev") is None


def test_media_pick_of_one_file_in_a_multi_file_folder(tmp_path):
    from core.inference.diffusion import resolve_local_single_file
    from core.inference.media_locality import normalized_pick
    from core.inference.media_model_index import MediaModelPick

    files = _comfy_diffusion_folder(tmp_path)
    # The folder-level rescue cannot choose among several files ...
    assert resolve_local_single_file(str(tmp_path)) is None
    # ... but a pick naming the file loads exactly that file.
    pick = normalized_pick(MediaModelPick("x", str(files["zimage"])))
    assert (pick.model_path, pick.gguf_filename, pick.model_kind) == (
        str(tmp_path),
        "z_image_turbo_bf16.safetensors",
        "single_file",
    )


def test_diffusion_validation_accepts_a_named_file_in_a_multi_file_folder(tmp_path):
    from core.inference.diffusion import get_diffusion_backend

    _comfy_diffusion_folder(tmp_path)
    backend = get_diffusion_backend()
    fam = backend.validate_load_request(
        str(tmp_path), gguf_filename = "flux1-dev.safetensors", model_kind = "single_file"
    )
    assert fam.name == "flux.1"
    fam = backend.validate_load_request(
        str(tmp_path), gguf_filename = "z_image_turbo_bf16.safetensors", model_kind = "single_file"
    )
    assert fam.name == "z-image"


def test_renamed_file_row_takes_its_family_from_the_header(tmp_path, monkeypatch):
    from hub.services.models import catalog_classification

    files = _comfy_diffusion_folder(tmp_path)
    monkeypatch.setattr(
        diffusion_content,
        "inspect_checkpoint",
        lambda path: diffusion_content.CheckpointInfo(
            diffusion_content.ROLE_DIT, family = "flux.1", page = "image"
        )
        if Path(path).name == "my_favourite_model.safetensors"
        else diffusion_content.CheckpointInfo(diffusion_content.ROLE_UNKNOWN),
    )
    [row] = [
        r for r in local_inventory._scan_models_dir(tmp_path) if r.path == str(files["renamed"])
    ]
    assert "flux.1" in catalog_classification._local_family_needles(row)
    assert catalog_classification._local_model_task(row) == "text-to-image"
