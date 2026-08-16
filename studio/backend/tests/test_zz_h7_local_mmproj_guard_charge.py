# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training guard must not charge for a projector the load will not open.

``_estimate_gguf_required_gb`` grew a ``disable_vision`` gate on the REMOTE
branch's ``include_mmproj``. The LOCAL branch charges ``gguf_mmproj_file``
unconditionally, so a load that skips the projector is still refused over its
bytes -- the exact failure the remote gate's own comment describes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement  # noqa: F401,E402  (installs the import stubs)

from routes.inference import _estimate_gguf_required_gb  # noqa: E402

_MODEL_BYTES = 4 * 1024**3
_MMPROJ_BYTES = 1024**3


def _local_vision_config(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00")
    mmproj = tmp_path / "mmproj-F16.gguf"
    mmproj.write_bytes(b"\x00")
    return SimpleNamespace(
        gguf_file = str(model),
        gguf_mmproj_file = str(mmproj),
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_hf_repo = None,
        gguf_variant = None,
        is_vision = True,
    )


@pytest.fixture
def _sizes(monkeypatch, tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend

    def _size(path):
        return _MMPROJ_BYTES if "mmproj" in str(path) else _MODEL_BYTES

    monkeypatch.setattr(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(_size))
    return _local_vision_config(tmp_path)


def test_a_local_projector_is_not_charged_when_vision_is_off(_sizes):
    charged = _estimate_gguf_required_gb(_sizes, disable_vision = False)
    skipped = _estimate_gguf_required_gb(_sizes, disable_vision = True)

    assert charged is not None and skipped is not None
    # The 1 GiB projector must drop out of the budget.
    assert charged - skipped == pytest.approx(
        1.0, abs = 0.01
    ), f"guard charges the projector either way: {charged} vs {skipped}"


def test_a_local_projector_is_not_charged_under_an_extras_opt_out(_sizes):
    """The same hole for the pre-existing spelling, which the launch also honours."""
    charged = _estimate_gguf_required_gb(_sizes)
    skipped = _estimate_gguf_required_gb(_sizes, llama_extra_args = ["--no-mmproj"])

    assert charged - skipped == pytest.approx(
        1.0, abs = 0.01
    ), f"guard charges the projector either way: {charged} vs {skipped}"
