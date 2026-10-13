# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth/LTX-2.5-GGUF: the LTX-2.5 DiT as single-file GGUF quants.

Layout (the unsloth/LTX-2.3-GGUF one): the dev DiT at the repo root, the distilled DiT under ``distilled/``, static
quants plus ``UD-Q*_K_XL`` dynamic ones. Header: ``general.architecture = ltxv`` (the ComfyUI-GGUF tag shared with 2.3),
bare Comfy keys, ``model_version = 2.5.0``. Every file must load as the ``ltx-2.5`` family with the pack as its base
and the variant's own defaults, never down the 2.3 assembly.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.inference.video_families import (
    LTX25_FAMILY,
    LTX2_FAMILY,
    default_video_generation_params,
    detect_video_family,
)

REPO = "unsloth/LTX-2.5-GGUF"
PACK = "Lightricks/LTX-2.5-Diffusers"
QUANTS = (
    "Q2_K",
    "Q3_K_M",
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "BF16",
    "UD-Q2_K_XL",
    "UD-Q3_K_XL",
    "UD-Q4_K_XL",
    "UD-Q5_K_XL",
)
DEV_FILES = [f"ltx-2.5-22b-dev-{q}.gguf" for q in QUANTS]
DISTILLED_FILES = [f"distilled/ltx-2.5-22b-distilled-{q}.gguf" for q in QUANTS]


def test_the_family_curates_the_unsloth_gguf_repo():
    fam = detect_video_family(PACK)
    assert fam.name == LTX25_FAMILY
    assert fam.gguf_repo == REPO
    assert detect_video_family(REPO).name == LTX25_FAMILY
    # 2.3 keeps its own repo.
    assert detect_video_family("", override = LTX2_FAMILY).gguf_repo == "unsloth/LTX-2.3-GGUF"


@pytest.mark.parametrize("filename", DEV_FILES + DISTILLED_FILES)
def test_every_planned_file_routes_to_ltx25_with_its_variant_defaults(filename):
    from core.inference.video import _detect_load_family, _ltx25_dev_load
    from hub.utils.gguf import extract_quant_label

    fam = _detect_load_family(REPO, filename, None)
    assert fam.name == LTX25_FAMILY
    dev = filename in DEV_FILES
    assert _ltx25_dev_load(fam, "gguf", filename, REPO) is dev
    expected = (30, 3.0) if dev else (8, 1.0)
    assert (
        default_video_generation_params(
            None, filename, REPO, fallback = (fam.default_steps, fam.default_guidance), family = fam.name
        )
        == expected
    )
    quant = filename.rsplit("-", 1)[-1][: -len(".gguf")]
    if "UD-" in filename:
        quant = "UD-" + quant
    assert extract_quant_label(filename) == quant


@pytest.mark.parametrize("filename", [DEV_FILES[2], DISTILLED_FILES[9]])
def test_a_hub_pick_validates_on_the_pack_base_without_touching_the_hub(filename, monkeypatch):
    from core.inference.video import get_video_backend
    from core.inference.video_families import resolve_video_base_repo

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    fam = get_video_backend().validate_load_request(REPO, gguf_filename = filename, model_kind = "gguf")
    assert fam.name == LTX25_FAMILY
    assert resolve_video_base_repo(fam, None, REPO) == PACK


def test_a_25_file_in_the_23_repo_and_the_reverse_follow_the_file():
    from core.inference.video import _detect_load_family

    assert _detect_load_family("unsloth/LTX-2.3-GGUF", DISTILLED_FILES[2], None).name == LTX25_FAMILY
    assert (
        _detect_load_family(REPO, "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf", None).name
        == LTX2_FAMILY
    )


def _write_gguf(path, *, v25: bool = True) -> None:
    """A tiny GGUF with the planned header: arch ltxv, model_version / config KVs, bare Comfy DiT + connector keys."""
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), "ltxv")
    writer.add_quantization_version(2)
    writer.add_file_type(15)
    writer.add_string("model_version", "2.5.0" if v25 else "2.3.0")
    writer.add_string("config", '{"transformer": {"_class_name": "AVTransformer3DModel"}}')
    writer.add_string("gemma_source_checkpoint", '{"ltx_version": "2.5.0"}')
    tensors = {
        "scale_shift_table": (2, 64),
        "patchify_proj.weight": (64, 128),
        "audio_adaln_single.linear.weight": (64, 32),
        "transformer_blocks.0.attn1.to_q.weight": (64, 64),
        "transformer_blocks.0.ff.net.0.proj.weight": (128, 64),
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": (64, 64),
    }
    if v25:
        tensors["keyframes_abs_pos_embedding"] = (4, 64)
    else:
        tensors["transformer_blocks.0.ff.net.0.proj.bias"] = (128,)
    for name, shape in tensors.items():
        writer.add_tensor(name, np.zeros(shape, dtype = np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_the_planned_header_reads_as_ltx25_even_renamed(tmp_path):
    from core.inference.diffusion_content import inspect_checkpoint
    from core.inference.video import _detect_load_family, get_video_backend
    from utils.models.gguf_metadata import read_gguf_architecture

    root = tmp_path / "quants"
    (root / "distilled").mkdir(parents = True)
    renamed = root / "distilled" / "model-Q4_K_M.gguf"
    _write_gguf(renamed)
    _write_gguf(root / DEV_FILES[2])
    assert read_gguf_architecture(str(renamed)) == "ltxv"
    info = inspect_checkpoint(str(renamed))
    assert info.family == LTX25_FAMILY and info.layout == "gguf"
    # No version token anywhere in the path: the header decides, never the 2.3 alias "ltxv".
    assert _detect_load_family(str(root), "distilled/model-Q4_K_M.gguf", None).name == LTX25_FAMILY
    backend = get_video_backend()
    for name in ("distilled/model-Q4_K_M.gguf", DEV_FILES[2]):
        fam = backend.validate_load_request(str(root), gguf_filename = name, model_kind = "gguf")
        assert fam.name == LTX25_FAMILY


def test_a_23_header_never_loads_as_ltx25(tmp_path):
    from core.inference.diffusion_content import inspect_checkpoint

    path = tmp_path / "model-Q4_K_M.gguf"
    _write_gguf(path, v25 = False)
    assert inspect_checkpoint(str(path)).family == LTX2_FAMILY


def test_picker_rows_keep_dev_and_distilled_apart_and_rank_ud_first():
    from hub.utils.gguf import gguf_variant_key, pick_best_gguf

    keys = {gguf_variant_key(f) for f in DEV_FILES + DISTILLED_FILES}
    assert len(keys) == len(DEV_FILES) + len(DISTILLED_FILES)
    # The automatic default reads root rows only (gguf_variants._default_variant_candidates), as for LTX-2.3:
    # the dev UD-Q4_K_XL, then Q4_K_M.
    assert pick_best_gguf(DEV_FILES) == "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf"
    assert (
        pick_best_gguf([f for f in DEV_FILES if "UD-" not in f]) == "ltx-2.5-22b-dev-Q4_K_M.gguf"
    )
    assert pick_best_gguf(DISTILLED_FILES) == "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"


def test_unsupported_message_names_both_25_artifacts():
    from core.inference.video_families import unsupported_ltx_message

    text = unsupported_ltx_message("2.6", "x")
    assert PACK in text and REPO in text and "unsloth/LTX-2.3-GGUF" in text
