# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shipped prebuilds (b12588 and later) size a borrowed DSpark / DFlash draft in their own
fitter (common_fit_draft_target in libllama-common.so.1), so Studio must not keep a second
reserve for it in --fit-target. Read through the same layout the bundles ship: llama-server
links libllama-server-impl.so, which links libllama-common.so.1."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import _sidecar_fit_helpers as h  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402

# How the symbol sits in an unstripped libllama-common.so.1 (mangled constructor name).
SYMBOL = b"\0_ZN23common_fit_draft_targetC2ERKSt6vectorINSt7__cxx1112basic_stringIcEEE\0"


def _bundle(
    tmp_path,
    common_payload: bytes,
    common_name = "libllama-common.so.1",
):
    server = h.elf(tmp_path / "llama-server", b"\0libllama-server-impl.so\0")
    h.elf(tmp_path / "libllama-server-impl.so", b"\0" + common_name.encode() + b"\0libllama.so.1\0")
    h.elf(tmp_path / common_name, common_payload)
    return server


def test_the_shipped_bundle_layout_is_recognised(tmp_path):
    assert lc._fit_measures_borrowed_drafts(str(_bundle(tmp_path, SYMBOL)))


def test_a_soname_1_common_library_without_the_fitter_keeps_the_reserve(tmp_path):
    server = _bundle(
        tmp_path, b"failed to measure the memory of the extra model, fitting without it"
    )
    assert not lc._fit_measures_borrowed_drafts(str(server))


def test_the_older_trace_string_is_still_recognised_through_soname_1(tmp_path):
    assert lc._fit_measures_borrowed_drafts(str(_bundle(tmp_path, h.NATIVE_MARKER)))


@pytest.mark.parametrize("name", ["libllama-common.so.0", "libllama-common.so.1"])
def test_either_soname_is_followed(tmp_path, name):
    assert lc._fit_measures_borrowed_drafts(str(_bundle(tmp_path, SYMBOL, common_name = name)))


def test_a_stray_unlinked_library_does_not_speak_for_the_build(tmp_path):
    server = h.elf(tmp_path / "llama-server", b"\0libllama-server-impl.so\0")
    h.elf(tmp_path / "libllama-server-impl.so", b"\0libllama.so.1\0")
    h.elf(tmp_path / "libllama-common.so.1", SYMBOL)
    assert not lc._fit_measures_borrowed_drafts(str(server))
