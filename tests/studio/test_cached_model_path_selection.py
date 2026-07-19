# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Variant-file selection guards for the cached-model-path endpoint.

The Copy path / Reveal endpoint must resolve a quant label to the same file
the variant menus offer: MTP drafters, mmproj vision adapters, and big-endian
builds are excluded, and directory layouts (``BF16/model-00001-of-....gguf``)
resolve their label from the snapshot-relative path, not the basename.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or run from "
        "the repository checkout.",
        allow_module_level = True,
    )

_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"
if str(_STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(_STUDIO_BACKEND))

pytest.importorskip("fastapi")
pytest.importorskip("huggingface_hub")

try:
    from routes import models as routes_models
except Exception as exc:
    pytest.skip(f"studio backend import unavailable: {exc}", allow_module_level = True)


def test_plain_quant_label_resolves():
    assert routes_models._main_variant_gguf_label("Model-Q8_0.gguf") == "Q8_0"


def test_mtp_drafter_in_subdir_is_excluded():
    assert routes_models._main_variant_gguf_label("MTP/Model-Q8_0-MTP.gguf") is None


def test_mtp_drafter_root_prefix_is_excluded():
    assert routes_models._main_variant_gguf_label("mtp-Model-Q8_0.gguf") is None


def test_mmproj_adapter_is_excluded():
    assert routes_models._main_variant_gguf_label("mmproj-Model-F16.gguf") is None


def test_directory_layout_quant_resolves_from_parent_dir():
    assert routes_models._main_variant_gguf_label("BF16/Model-00001-of-00002.gguf") == "BF16"


def test_big_endian_build_is_excluded():
    assert routes_models._main_variant_gguf_label("Model-Q8_0-BE.gguf") is None


def test_non_gguf_file_is_excluded():
    assert routes_models._main_variant_gguf_label("config.json") is None


def test_normalized_quant_label_ignores_separators():
    assert routes_models._normalized_quant_label("UD-Q4_K_XL") == "udq4kxl"
    assert routes_models._normalized_quant_label("Q8-0") == routes_models._normalized_quant_label(
        "Q8_0"
    )
