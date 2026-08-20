# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A calibration imatrix is not a model, so no GGUF surface may offer one.

unsloth/Qwen3.8-27B-GGUF publishes imatrix_unsloth.gguf beside the weights (most
repos use the .dat / .gguf_file spellings the Hub does not list as GGUF). Carrying
a real .gguf suffix, it reached the chat model picker as an unlabelled ~13 MB row
called "GGUF", downloadable and then unloadable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import _is_companion_gguf_path
from hub.utils.gguf import (
    is_imatrix_filename,
    list_gguf_variants,
    list_local_gguf_variants,
    list_partial_gguf_variants_from_state,
    pick_best_gguf,
)
from hub.utils.gguf_plan import build_gguf_variant_plans, is_main_gguf_variant_path
from utils.models.model_config import _is_imatrix_path, detect_gguf_model

IMATRIX_NAMES = [
    "imatrix_unsloth.gguf",
    "imatrix_unsloth.dat",
    "imatrix_unsloth.gguf_file",
    "imatrix.gguf",
    "IMATRIX_UNSLOTH.GGUF",
    "Qwen3.8-27B.imatrix",
    "Qwen3.8-27B-imatrix.gguf",
    "quants/imatrix_unsloth.gguf",
]

MODEL_NAMES = [
    "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
    # The word inside a name is not the file: only a leading or trailing token is.
    "Qwen3-Imatrix-Tuned-Q4_K_M.gguf",
    "imatrixed-model-Q8_0.gguf",
]


@pytest.mark.parametrize("name", IMATRIX_NAMES)
def test_every_published_imatrix_spelling_is_recognized(name):
    assert is_imatrix_filename(name)
    assert _is_imatrix_path(name)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_real_weights_are_not_mistaken_for_an_imatrix(name):
    assert not is_imatrix_filename(name)
    assert not _is_imatrix_path(name)


@pytest.mark.parametrize("name", IMATRIX_NAMES + MODEL_NAMES)
def test_the_three_copies_of_the_predicate_agree(name):
    # hub cannot be imported from utils and core avoids importing it, so the rule is
    # spelled three times; a drift here is a file one surface hides and another loads.
    expected = is_imatrix_filename(name)
    assert _is_imatrix_path(name) is expected
    if name.lower().endswith(".gguf"):
        assert _is_companion_gguf_path(name) is expected


def test_remote_listing_drops_the_imatrix_row(monkeypatch):
    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = "Qwen3.8-27B-UD-Q4_K_XL.gguf", size = 17_559_178_144),
            SimpleNamespace(rfilename = "imatrix_unsloth.gguf", size = 13_642_656),
        ]
    )
    api = SimpleNamespace(model_info = lambda *a, **k: info)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token = None: api)

    variants, _has_vision, _siblings = list_gguf_variants("unsloth/Qwen3.8-27B-GGUF")
    assert [v.quant for v in variants] == ["UD-Q4_K_XL"]


def test_the_imatrix_is_neither_planned_nor_downloaded():
    siblings = [
        SimpleNamespace(rfilename = "Qwen3.8-27B-UD-Q4_K_XL.gguf", size = 17_559_178_144),
        SimpleNamespace(rfilename = "imatrix_unsloth.gguf", size = 13_642_656),
    ]
    plans = build_gguf_variant_plans(siblings)

    assert list(plans) == ["ud-q4_k_xl"]
    plan = plans["ud-q4_k_xl"]
    # Not a main file and not a companion folded into the plan: llama-quantize reads an
    # imatrix, llama-server never does, so no variant needs it on disk.
    assert plan.target_filenames == ("Qwen3.8-27B-UD-Q4_K_XL.gguf",)
    assert plan.download_size_bytes == 17_559_178_144
    assert not is_main_gguf_variant_path("imatrix_unsloth.gguf", "ud-q4_k_xl")


def test_local_listing_and_load_path_skip_the_imatrix(tmp_path):
    (tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(b"GGUF" + b"0" * 64)
    imatrix = tmp_path / "imatrix_unsloth.gguf"
    imatrix.write_bytes(b"GGUF" + b"0" * 8)
    (tmp_path / "config.json").write_text("{}", encoding = "utf-8")

    variants, _has_vision = list_local_gguf_variants(os.fspath(tmp_path))
    assert [v.quant for v in variants] == ["UD-Q4_K_XL"]

    assert pick_best_gguf(["imatrix_unsloth.gguf", "Qwen3.8-27B-UD-Q4_K_XL.gguf"]) == (
        "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    )
    # Pointed at directly it is not a model either, so a folder scan cannot route it.
    assert detect_gguf_model(os.fspath(imatrix)) is None
    assert detect_gguf_model(os.fspath(tmp_path)) == os.fspath(
        tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    )


def _state_sources(monkeypatch, manifests, markers, manifest_for):
    from hub.utils import download_manifest

    monkeypatch.setattr(
        download_manifest,
        "iter_variant_manifests",
        lambda *_a, **_k: iter([(v, Path(f"{v}.json")) for v in manifests]),
    )
    monkeypatch.setattr(
        download_manifest,
        "iter_variant_markers",
        lambda *_a, **_k: iter([(v, Path(f"{v}.marker")) for v in markers]),
    )
    monkeypatch.setattr(
        download_manifest, "read_manifest", lambda _t, _r, variant, **_k: manifest_for(variant)
    )


def test_interrupted_imatrix_state_does_not_come_back_as_a_row(monkeypatch):
    # An older build offered the imatrix as a variant, so a cancelled download of it can
    # still be on disk. Skipping only its expected file left main_filename unset, and the
    # synthetic "<variant>.gguf" fallback put the row back at zero bytes on exactly the
    # offline path this listing serves.
    manifest = SimpleNamespace(
        expected_files = [SimpleNamespace(path = "imatrix_unsloth.gguf", size = 13_642_656)]
    )
    _state_sources(
        monkeypatch,
        manifests = ["imatrix_unsloth"],
        markers = [],
        manifest_for = lambda _variant: manifest,
    )

    variants, _has_vision = list_partial_gguf_variants_from_state("unsloth/Qwen3.8-27B-GGUF")
    assert variants == []


def test_a_marker_only_imatrix_variant_is_dropped_but_a_real_quant_survives(monkeypatch):
    # A cancel marker carries no manifest at all, so the filtered-file check above cannot
    # see it; the stored variant key is the only evidence. A real quant in the same state
    # must keep its synthetic row, or an interrupted download becomes unresumable.
    _state_sources(
        monkeypatch,
        manifests = [],
        markers = ["imatrix_unsloth", "UD-Q4_K_XL"],
        manifest_for = lambda _variant: None,
    )

    variants, _has_vision = list_partial_gguf_variants_from_state("unsloth/Qwen3.8-27B-GGUF")
    assert [v.quant for v in variants] == ["UD-Q4_K_XL"]
    assert variants[0].filename == "UD-Q4_K_XL.gguf"


def test_an_imatrix_only_folder_offers_no_model(tmp_path):
    (tmp_path / "imatrix_unsloth.gguf").write_bytes(b"GGUF" + b"0" * 8)

    assert list_local_gguf_variants(os.fspath(tmp_path)) == ([], False)
    assert detect_gguf_model(os.fspath(tmp_path)) is None
