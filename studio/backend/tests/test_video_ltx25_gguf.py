# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth/LTX-2.5-GGUF: the LTX-2.5 DiT as single-file GGUF quants.

Dev DiT at the root, distilled under ``distilled/``; header arch ``ltxv`` (shared with 2.3) with
``model_version = 2.5.0``. Every file must load as ``ltx-2.5``, never down the 2.3 assembly.
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
            None,
            filename,
            REPO,
            fallback = (fam.default_steps, fam.default_guidance),
            family = fam.name,
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
    assert (
        _detect_load_family("unsloth/LTX-2.3-GGUF", DISTILLED_FILES[2], None).name == LTX25_FAMILY
    )
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
    # Within a checkpoint UD-Q4_K_XL beats Q4_K_M; default_gguf_scope picks the checkpoint.
    assert pick_best_gguf(DEV_FILES) == "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf"
    assert pick_best_gguf([f for f in DEV_FILES if "UD-" not in f]) == "ltx-2.5-22b-dev-Q4_K_M.gguf"
    assert pick_best_gguf(DISTILLED_FILES) == "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"


DISTILLED_DEFAULT = "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"
DISTILLED_DEFAULT_KEY = "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL"


def _rows(files):
    import types

    from hub.utils.gguf import gguf_variant_key
    return [types.SimpleNamespace(filename = f, quant = gguf_variant_key(f)) for f in files]


@pytest.mark.parametrize("order", ["dev_first", "distilled_first"])
def test_bare_repo_id_defaults_to_the_distilled_ud_q4_k_xl(order):
    """The bare-repo /gguf-variants default is distilled/...-UD-Q4_K_XL, whatever the listing order."""
    from hub.services.models.gguf_variants import _default_variant_candidates
    from hub.utils.gguf import pick_best_gguf

    files = DEV_FILES + DISTILLED_FILES if order == "dev_first" else DISTILLED_FILES + DEV_FILES
    rows = _rows(files)
    assert pick_best_gguf(_default_variant_candidates(rows, REPO)) == DISTILLED_DEFAULT
    assert pick_best_gguf(_default_variant_candidates(rows, REPO.lower())) == DISTILLED_DEFAULT
    # Only dev downloaded: the scope falls back to the root rows rather than answering nothing.
    assert pick_best_gguf(_default_variant_candidates(_rows(DEV_FILES), REPO)) == (
        "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf"
    )


def test_every_bare_id_resolver_agrees_on_distilled():
    """_match_variant(None, ...) (OpenAI-style auto-download) and the shared scope used by local_model_resolver and the
    media index name the same distilled checkpoint; an explicit dev variant still loads dev."""
    from core.inference.openai_auto_download import _match_variant, preferred_quant
    from hub.utils.gguf import default_gguf_scope, gguf_variant_key

    sizes = {gguf_variant_key(f): 1 for f in DEV_FILES + DISTILLED_FILES}
    assert _match_variant(None, sizes, REPO) == DISTILLED_DEFAULT_KEY
    assert _match_variant("UD-Q4_K_XL", sizes, REPO) == "UD-Q4_K_XL"
    assert _match_variant("Q8_0", sizes, REPO) == "Q8_0"
    # An HF cache folder names the repo as well as its id does.
    cache_dir = "/cache/hub/models--unsloth--LTX-2.5-GGUF"
    assert preferred_quant(default_gguf_scope(tuple(sizes), cache_dir)) == DISTILLED_DEFAULT_KEY


def test_local_resolver_and_media_index_put_distilled_first(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from core.inference import local_model_resolver as resolver
    from core.inference import media_model_index as index_mod

    folder = tmp_path / "models--unsloth--LTX-2.5-GGUF"
    (folder / "distilled").mkdir(parents = True)
    for name in ("Q4_K_M", "UD-Q4_K_XL"):
        (folder / f"ltx-2.5-22b-dev-{name}.gguf").write_bytes(b"GGUF" + b"\0" * 64)
        (folder / "distilled" / f"ltx-2.5-22b-distilled-{name}.gguf").write_bytes(
            b"GGUF" + b"\0" * 64
        )
    info = SimpleNamespace(path = str(folder), id = REPO, model_id = REPO, display_name = None)
    entry = resolver._local_gguf_entry(REPO, info)
    assert entry is not None and entry.variants[0] == DISTILLED_DEFAULT_KEY

    monkeypatch.setattr(index_mod, "_loader_can_open", lambda *_: True)
    picks: dict = {}
    assert index_mod._add_gguf_picks(picks, info, (REPO,), folder, folder)
    # The index keys are lowercased.
    assert picks[REPO.lower()].gguf_filename.replace("\\", "/") == DISTILLED_DEFAULT
    assert picks[f"{REPO}:UD-Q4_K_XL".lower()].gguf_filename == "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf"


def test_ltx23_and_other_repos_keep_the_root_default():
    from core.inference.openai_auto_download import _match_variant
    from core.inference.video_families import gguf_default_subdir_for_repo
    from hub.services.models.gguf_variants import _default_variant_candidates
    from hub.utils.gguf import pick_best_gguf

    assert gguf_default_subdir_for_repo(REPO) == "distilled"
    assert gguf_default_subdir_for_repo("unsloth/LTX-2.3-GGUF") is None
    files23 = [f.replace("2.5", "2.3") for f in DISTILLED_FILES + DEV_FILES]
    assert pick_best_gguf(_default_variant_candidates(_rows(files23), "unsloth/LTX-2.3-GGUF")) == (
        "ltx-2.3-22b-dev-UD-Q4_K_XL.gguf"
    )
    other = {"distilled/model-Q4_K_M": 1, "Q4_K_M": 1}
    assert _match_variant(None, other, "unsloth/Qwen3-8B-GGUF") == "Q4_K_M"
    assert _match_variant(None, other, None) == "Q4_K_M"


def test_picker_labels_the_root_rows_dev_and_the_subfolder_rows_distilled(tmp_path):
    """Root rows read "<quant> · dev" on every listing path; 2.3 and plain repos keep bare root rows."""
    from hub.utils.gguf import (
        GgufVariantInfo,
        _apply_gguf_display_labels,
        gguf_variant_key,
        list_local_gguf_variants,
    )

    def labels(files, *repo):
        rows = [GgufVariantInfo(filename = f, quant = gguf_variant_key(f), size_bytes = 1) for f in files]
        _apply_gguf_display_labels(rows, *repo)
        return {r.filename: r.display_label for r in rows}

    got = labels(DEV_FILES + DISTILLED_FILES, REPO)
    for q in QUANTS:
        assert got[f"ltx-2.5-22b-dev-{q}.gguf"] == f"{q} · dev"
        # Several rows share the distilled/ scope, so the existing rule adds the file stem to tell them apart.
        assert got[f"distilled/ltx-2.5-22b-distilled-{q}.gguf"].startswith(f"{q} · distilled")
    files23 = [f.replace("2.5", "2.3") for f in DEV_FILES + DISTILLED_FILES]
    got23 = labels(files23, "unsloth/LTX-2.3-GGUF")
    assert got23["ltx-2.3-22b-dev-UD-Q4_K_XL.gguf"] is None
    assert got23["distilled/ltx-2.3-22b-distilled-UD-Q4_K_XL.gguf"].startswith(
        "UD-Q4_K_XL · distilled"
    )
    assert labels(DEV_FILES)["ltx-2.5-22b-dev-Q8_0.gguf"] is None

    # A cached snapshot is listed by directory: the repo is read off the models--org--repo segment.
    snap = tmp_path / "models--unsloth--LTX-2.5-GGUF" / "snapshots" / "abc"
    (snap / "distilled").mkdir(parents = True)
    for name in (
        "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf",
        "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf",
    ):
        (snap / name).write_bytes(b"GGUF" + b"\0" * 64)
    variants, _ = list_local_gguf_variants(str(snap))
    assert {v.quant: v.display_label for v in variants} == {
        "UD-Q4_K_XL": "UD-Q4_K_XL · dev",
        DISTILLED_DEFAULT_KEY: "UD-Q4_K_XL · distilled",
    }


def test_a_plain_local_clone_keeps_the_distilled_default_and_dev_labels(tmp_path):
    """A plain local copy (``/models/unsloth/LTX-2.5-GGUF``) is recognised by its trailing org/repo."""
    from hub.utils.gguf import default_gguf_subdir, gguf_root_label, list_local_gguf_variants

    clone = tmp_path / "models" / "unsloth" / "LTX-2.5-GGUF"
    (clone / "distilled").mkdir(parents = True)
    for name in (
        "ltx-2.5-22b-dev-UD-Q4_K_XL.gguf",
        "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf",
    ):
        (clone / name).write_bytes(b"GGUF" + b"\0" * 64)
    assert default_gguf_subdir(str(clone)) == "distilled"
    assert gguf_root_label(str(clone)) == "dev"
    assert default_gguf_subdir("C:\\models\\unsloth\\LTX-2.5-GGUF") == "distilled"
    variants, _ = list_local_gguf_variants(str(clone))
    assert {v.quant: v.display_label for v in variants} == {
        "UD-Q4_K_XL": "UD-Q4_K_XL · dev",
        DISTILLED_DEFAULT_KEY: "UD-Q4_K_XL · distilled",
    }
    # Unrelated folders and bare names keep the root default.
    assert default_gguf_subdir(str(tmp_path / "models" / "other" / "Repo")) is None
    assert default_gguf_subdir("LTX-2.5-GGUF") is None


def test_unsupported_message_names_both_25_artifacts():
    from core.inference.video_families import unsupported_ltx_message
    text = unsupported_ltx_message("2.6", "x")
    assert PACK in text and REPO in text and "unsloth/LTX-2.3-GGUF" in text
