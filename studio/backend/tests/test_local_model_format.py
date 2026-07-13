# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for local GGUF ``model_format`` classification (PR #6364 follow-up).

Suffixless GGUF folders (custom folders / LM Studio) carry no ``-GGUF`` name
hint, so the scanners must surface ``model_format = "gguf"`` for the UI to route
them through the GGUF load path. The rule, shared by ``_dir_model_format`` and
``_scan_models_dir``: a directory is GGUF-format when it holds ``.gguf`` files
and no non-GGUF weights (``.safetensors`` / ``.bin``); a stray ``config.json``
must not disqualify it.

No GPU/network: only file names and sizes are inspected.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Keep runnable without optional logging deps (mirrors the sibling tests).
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0")
    return path


def test_dir_model_format_gguf_only(tmp_path):
    d = tmp_path / "model"
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_gguf_with_config_is_still_gguf(tmp_path):
    # A config.json alongside the .gguf must not flip it to non-GGUF.
    d = tmp_path / "model"
    _touch(d / "config.json")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_mixed_weights_is_not_gguf(tmp_path):
    # Real safetensors weights present -> not a GGUF folder.
    d = tmp_path / "model"
    _touch(d / "model.safetensors")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) is None


def test_dir_model_format_no_gguf(tmp_path):
    d = tmp_path / "model"
    _touch(d / "config.json")
    _touch(d / "model.safetensors")
    assert models_route._dir_model_format(d) is None


def test_dir_model_format_ignores_tokenizer_bin(tmp_path):
    # A companion tokenizer.bin is not a weight file, so a GGUF folder shipping
    # one is still GGUF (not misread as a plain .bin checkpoint).
    d = tmp_path / "model"
    _touch(d / "tokenizer.bin")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_weight_bin_is_not_gguf(tmp_path):
    # A real PyTorch weight .bin alongside a .gguf means mixed weights -> None.
    d = tmp_path / "model"
    _touch(d / "pytorch_model.bin")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) is None


def test_scan_models_dir_classifies_gguf_with_config(tmp_path):
    root = tmp_path / "models"
    # GGUF repo that also ships a config.json (the regression case).
    _touch(root / "gguf_repo" / "config.json")
    _touch(root / "gguf_repo" / "model-Q4_K_M.gguf")
    # A plain safetensors checkpoint stays non-GGUF.
    _touch(root / "st_repo" / "config.json")
    _touch(root / "st_repo" / "model.safetensors")
    # A standalone .gguf file is GGUF.
    _touch(root / "loose.gguf")

    fmt = {Path(m.path).name: m.model_format for m in models_route._scan_models_dir(root)}

    assert fmt["gguf_repo"] == "gguf"
    assert fmt["st_repo"] is None
    assert fmt["loose.gguf"] == "gguf"


def test_scan_models_dir_classifies_root_gguf_with_config(tmp_path):
    # Custom scan folders can point directly at a GGUF repo, not only at a
    # parent directory that contains model repos.
    root = tmp_path / "SuffixlessRepo"
    _touch(root / "config.json")
    _touch(root / "model-Q4_K_M.gguf")

    [row] = models_route._scan_models_dir(root)

    assert row.path == str(root)
    assert row.model_format == "gguf"


def test_scan_models_dir_surfaces_diffusers_pipeline_folder(tmp_path):
    # A standard diffusers PIPELINE folder keeps its weights/configs in component subdirs
    # (transformer/, vae/, ...) and carries only model_index.json at the root. The Images/Video
    # load path accepts such a local pipeline dir, so the scan must surface it -- otherwise the
    # weights-in-subdirs layout is missed and it never reaches task tagging / the On Device
    # picker. It is not a GGUF, so model_format stays None (task tagging classifies it later).
    root = tmp_path / "models"
    pipe = root / "my-pipeline"
    _touch(pipe / "model_index.json")
    _touch(pipe / "transformer" / "config.json")
    _touch(pipe / "transformer" / "diffusion_pytorch_model.safetensors")
    _touch(pipe / "vae" / "diffusion_pytorch_model.safetensors")

    rows = {Path(m.path).name: m for m in models_route._scan_models_dir(root)}

    assert "my-pipeline" in rows
    assert rows["my-pipeline"].model_format is None


def test_scan_models_dir_surfaces_root_diffusers_pipeline(tmp_path):
    # A custom scan folder can point DIRECTLY at a diffusers pipeline (not a parent of repos).
    # Its weights live in component subdirs under a root model_index.json, so _is_model_directory
    # rejects the root; without admitting it the scan would surface the component subdirs
    # (transformer/, vae/) as bogus models and hide the real pipeline. Treat the root as one model.
    root = tmp_path / "my-local-pipeline"
    _touch(root / "model_index.json")
    _touch(root / "transformer" / "config.json")
    _touch(root / "transformer" / "diffusion_pytorch_model.safetensors")
    _touch(root / "vae" / "diffusion_pytorch_model.safetensors")

    rows = models_route._scan_models_dir(root)

    assert [r.path for r in rows] == [str(root)]
    assert rows[0].model_format is None


# ── Images picker task tag for local (non-GGUF) diffusers models ──────────────
from models.models import LocalModelInfo  # noqa: E402


def _local(
    path,
    *,
    model_format = None,
    model_id = None,
    display_name = "m",
    id = "m",
):
    return LocalModelInfo(
        id = id,
        display_name = display_name,
        path = str(path),
        source = "models_dir",
        model_id = model_id,
        model_format = model_format,
    )


def test_local_task_tags_diffusers_pipeline_dir(tmp_path):
    # A local diffusers pipeline (top-level model_index.json) is an image model even
    # though its model_format is not "gguf": tag it so the Images picker keeps it.
    d = tmp_path / "my-local-pipeline"
    _touch(d / "model_index.json")
    _touch(d / "unet" / "diffusion_pytorch_model.safetensors")
    assert models_route._local_model_task(_local(d)) == "text-to-image"


def test_local_task_tags_diffusers_by_family_id(tmp_path):
    # A single-file / safetensors image checkpoint ships no model_index.json; fall back
    # to the model id resolving to a known diffusion family.
    d = tmp_path / "flux-checkpoint"
    _touch(d / "flux1-dev.safetensors")
    assert (
        models_route._local_model_task(_local(d, model_id = "black-forest-labs/FLUX.1-dev"))
        == "text-to-image"
    )


def test_local_task_none_for_plain_llm(tmp_path):
    # A plain non-GGUF LLM checkpoint (no pipeline, no image family) stays untagged.
    d = tmp_path / "llama"
    _touch(d / "config.json")
    _touch(d / "model.safetensors")
    assert models_route._local_model_task(_local(d, model_id = "meta-llama/Llama-3.1-8B")) is None


def test_local_task_tags_video_pipeline_dir(tmp_path):
    # A local diffusers pipeline whose id resolves to a VIDEO family (LTX / Wan / Hunyuan)
    # must be tagged text-to-video so it surfaces in the Video On-Device picker, mirroring the
    # cached-repo path -- not text-to-image, where the image loader would reject it.
    d = tmp_path / "wan-local"
    _touch(d / "model_index.json")
    _touch(d / "transformer" / "diffusion_pytorch_model.safetensors")
    assert (
        models_route._local_model_task(_local(d, model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
        == models_route._VIDEO_GEN_TASK
    )


def test_local_task_tags_video_single_file_checkpoint(tmp_path):
    # A video-family dir holding a bare single-file .safetensors (no model_index.json) is loadable
    # (the route loads it as a single_file), so it must be tagged text-to-video and surfaced, not
    # left task=null and hidden.
    d = tmp_path / "ltx-loose"
    _touch(d / "ltx-2.safetensors")  # loose weights, no model_index.json
    assert (
        models_route._local_model_task(_local(d, model_id = "Lightricks/LTX-2"))
        == models_route._VIDEO_GEN_TASK
    )


def test_local_task_tags_single_file_by_checkpoint_filename(tmp_path):
    # A generically named folder holding one loadable checkpoint whose FILENAME identifies the
    # family (the parent dir does not) is loadable -- the route resolves the sole file via
    # resolve_local_single_file -- so tag it from the filename or the task-scoped picker hides it.
    d = tmp_path / "downloads"
    _touch(d / "qwen-image-2509.safetensors")  # family only in the filename, no model_index.json
    m = _local(d, id = str(d), display_name = "downloads")
    assert models_route._local_is_diffusers(m) is True
    assert models_route._local_model_task(m) == "text-to-image"


def test_local_task_tags_video_single_file_by_checkpoint_filename(tmp_path):
    # Same, for a video family whose token lives only in the sole checkpoint's filename.
    d = tmp_path / "clips"
    _touch(d / "ltx-2.3-distilled.safetensors")  # ltx family only in the filename
    m = _local(d, id = str(d), display_name = "clips")
    assert models_route._local_model_task(m) == models_route._VIDEO_GEN_TASK


def test_local_task_ignores_family_token_in_parent_path(tmp_path):
    # model.id is the full on-disk path for a scanned On-Device model, and the family-token
    # matcher treats any path segment as a hint. A family token in a PARENT dir (e.g.
    # /models/qwen-image/misc) must NOT tag an unrelated single-file as text-to-image: that
    # would surface it in the Images picker and evict the GPU owner before from_single_file
    # fails on the unrelated weights. Detection is scoped to the leaf name, not the raw path.
    d = tmp_path / "misc"
    _touch(d / "unrelated.safetensors")  # one non-family single file, no model_index.json
    m = _local(d, id = "/models/qwen-image/misc", display_name = "misc")
    assert models_route._local_is_diffusers(m) is False
    assert models_route._local_model_task(m) is None
    # Regression guard: a leaf name that itself carries a family hint is still tagged.
    d2 = tmp_path / "z-image-turbo"
    _touch(d2 / "model.safetensors")
    m2 = _local(d2, id = str(d2), display_name = "z-image-turbo")
    assert models_route._local_is_diffusers(m2) is True
