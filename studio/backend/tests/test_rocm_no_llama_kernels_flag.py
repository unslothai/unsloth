# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system flags a GPU the installed llama.cpp prebuilt has no kernels for.

The GGUF load already skips it, but the chat GPU picker offered it. Hermetic: the prebuilt's
targets and the arch per device are stubbed, so this never reads the host.
"""

from __future__ import annotations

import pytest

import main
from core.inference.llama_cpp import LlamaCppBackend


def _rows():
    return [{"index": i, "index_kind": "physical"} for i in range(3)]


@pytest.fixture
def r9700_host(monkeypatch):
    # 2x R9700 (gfx1201) plus the Ryzen 9950X iGPU (gfx1036).
    monkeypatch.setattr(
        LlamaCppBackend,
        "_rocm_arch_by_physical_id",
        staticmethod(lambda: {0: "gfx1201", 1: "gfx1201", 2: "gfx1036"}),
    )
    return monkeypatch


def _set_targets(monkeypatch, archs):
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(lambda binary = None: archs)
    )


def test_uncovered_igpu_is_flagged(r9700_host):
    _set_targets(r9700_host, frozenset({"gfx1200", "gfx1201"}))
    rows = _rows()
    main._flag_rocm_gpus_without_llama_kernels(rows)
    assert [r.get("no_llama_kernels", False) for r in rows] == [False, False, True]


def test_unknown_targets_fail_open(r9700_host):
    # A source build has no install marker; the load gate keeps every GPU then, so the picker does too.
    _set_targets(r9700_host, None)
    rows = _rows()
    main._flag_rocm_gpus_without_llama_kernels(rows)
    assert not any("no_llama_kernels" in r for r in rows)


def test_covered_host_and_non_physical_rows_carry_no_flag(r9700_host):
    _set_targets(r9700_host, frozenset({"gfx1201", "gfx1036"}))
    rows = _rows()
    main._flag_rocm_gpus_without_llama_kernels(rows)
    assert not any("no_llama_kernels" in r for r in rows)

    _set_targets(r9700_host, frozenset({"gfx1201"}))
    rows = [{"index": 2, "index_kind": "vulkan"}]
    main._flag_rocm_gpus_without_llama_kernels(rows)
    assert "no_llama_kernels" not in rows[0]


def test_probe_failure_fails_open(r9700_host):
    _set_targets(r9700_host, frozenset({"gfx1201"}))

    def _boom():
        raise RuntimeError("HIP unavailable")

    r9700_host.setattr(LlamaCppBackend, "_rocm_arch_by_physical_id", staticmethod(_boom))
    rows = _rows()
    main._flag_rocm_gpus_without_llama_kernels(rows)
    assert not any("no_llama_kernels" in r for r in rows)
