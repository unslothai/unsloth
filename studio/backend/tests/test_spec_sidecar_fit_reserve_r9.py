# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ninth-round cells for the DSpark/DFlash sidecar reserve.

A draft pinned to any card of a Studio-chosen Vulkan list is colocated with the target (only
--split-mode none narrows it to the main card), and a DFlash draft with its own embeddings but
no head borrows nothing on a build whose loader ties the head to them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_FLOOR = 1024 + 8 * 1024
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)


@pytest.mark.parametrize(
    "extras, pin, reserved",
    [
        ([], "Vulkan1", True),
        ([], "Vulkan0", True),
        (["--split-mode", "none"], "Vulkan1", False),
        (["--split-mode", "none"], "Vulkan0", True),
    ],
)
def test_a_pin_on_any_studio_chosen_vulkan_card_is_on_the_target(tmp_path, extras, pin, reserved):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO, vulkan = True)
    cmd = _r2._placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        gpu_ids = [0, 1],
        extra_args = [*extras, "--device-draft", pin],
    )["cmd"]
    assert (_r4._margin(cmd) >= _FLOOR) is reserved


def _draft(
    monkeypatch,
    tmp_path,
    *,
    tensors,
    arch = "dflash",
):
    path = tmp_path / "draft.gguf"
    path.write_bytes(b"draft")
    import utils.models.gguf_metadata as gm
    import utils.models.model_config as mc

    monkeypatch.setattr(mc, "colocated_split_shards", lambda p: ([p], True))
    monkeypatch.setattr(gm, "_gguf_shard_has_named_tensor", lambda shard, name: name in tensors)
    monkeypatch.setattr(gm, "read_gguf_architecture", lambda shard: arch)
    return str(path)


@pytest.mark.parametrize(
    "tensors, arch, ties, borrows",
    [
        # Own embeddings, no head: tied on a build that ties it, borrowed on an older one.
        ({"token_embd.weight"}, "dflash", True, False),
        ({"token_embd.weight"}, "dflash", False, True),
        # Only DFlash's loader ties the head.
        ({"token_embd.weight"}, "eagle3", True, True),
        # Neither tensor: the target's are borrowed either way.
        (set(), "dflash", True, True),
        ({"token_embd.weight", "output.weight"}, "dflash", False, False),
    ],
)
def test_a_tied_head_draft_borrows_only_on_a_build_without_the_fallback(
    monkeypatch, tmp_path, tensors, arch, ties, borrows
):
    path = _draft(monkeypatch, tmp_path, tensors = tensors, arch = arch)
    monkeypatch.setattr(_lc, "_dflash_ties_head_to_embeddings", lambda bin_path: ties)
    assert _lc._sidecar_borrows_target_tensors(path, "/fake/llama-server") is borrows


@pytest.mark.parametrize("lib_holds_marker, ties", [(True, True), (False, False)])
def test_the_tied_head_fallback_is_read_off_libllama(tmp_path, lib_holds_marker, ties):
    server = tmp_path / "llama-server"
    server.write_bytes(b"\x7fELF server")
    lib = tmp_path / "libllama.so.0"
    body = b"\x00 unsupported DFlash hidden activation: \x00" if lib_holds_marker else b"\x00"
    lib.write_bytes(b"\x7fELF " + body)
    assert _lc._dflash_ties_head_to_embeddings(str(server)) is ties


def test_an_unknown_binary_keeps_the_tied_head_draft_borrowing(monkeypatch, tmp_path):
    path = _draft(monkeypatch, tmp_path, tensors = {"token_embd.weight"})
    assert _lc._sidecar_borrows_target_tensors(path, None) is True
