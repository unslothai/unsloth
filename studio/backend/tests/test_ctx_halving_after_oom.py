# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A context Studio picked that still runs out of GPU memory is halved; a typed one is not."""

from __future__ import annotations

import pytest

import core.inference.llama_cpp as lc

from . import _sidecar_fit_helpers as h

FITS = 131072
TARGET_OOM = [
    "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 112679.75 MiB on device 0: cudaMalloc failed: out of memory",
    "E graph_reserve: failed to allocate compute buffers",
    "E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers",
    "E cmn  common_init_: failed to create context with model '/m/model.gguf'",
]


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _ctx(cmd):
    return int(cmd[cmd.index("-c") + 1])


def _load(
    tmp_path,
    *,
    extra_args = None,
    auto_derived = True,
    n_ctx = 1048576,
):
    # A card the planner's estimate says holds the full 1M: the estimate is what runs low.
    backend, gguf = h.cuda_backend(tmp_path, memory = [(0, 400_000, 400_000)])
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, **kw: h.GIB
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda n_ctx, *a, **kw: n_ctx * 1024
    backend._context_length = 1048576
    draft = h.sidecar(tmp_path)
    cmds = h.spawns(
        backend,
        gguf,
        healthy = lambda i, cmd: _ctx(cmd) <= FITS,
        crash = lambda i, cmd: None if _ctx(cmd) <= FITS else 1,
        stdout = lambda i, cmd: () if _ctx(cmd) <= FITS else TARGET_OOM,
        n_ctx = n_ctx,
        max_seq_length_auto_derived = auto_derived,
        speculative_type = "dspark",
        dspark_draft_path = draft,
        extra_args = extra_args,
    )
    return backend, cmds, draft


def test_a_studio_context_that_does_not_fit_is_halved_and_keeps_the_drafter(tmp_path):
    backend, cmds, draft = _load(tmp_path)
    assert cmds, "no llama-server launch"
    first = _ctx(cmds[0])
    assert first > FITS, f"the planner already picked {first}; the test needs an OOM first"
    assert cmds.loaded is True
    last = cmds[-1]
    assert _ctx(last) <= FITS
    # Halving only: every relaunch is at most half the one before it.
    sizes = [_ctx(c) for c in cmds]
    assert any(b <= a // 2 for a, b in zip(sizes, sizes[1:]))
    assert draft in last and "--spec-default" not in last
    assert backend._effective_context_length == _ctx(last)


def test_a_typed_context_is_never_halved(tmp_path):
    _, cmds, _ = _load(tmp_path, extra_args = ["-c", "1048576"])
    assert cmds
    assert all(_ctx(c) == 1048576 for c in cmds)


def test_the_helper_needs_a_context_memory_failure_and_stays_above_the_floor():
    oom = "\n".join(TARGET_OOM)
    is_oom = lc.LlamaCppBackend._is_gpu_memory_start_failure
    assert lc._halved_ctx_after_oom(["-c", "1048576"], oom, is_oom) == 524288
    assert lc._halved_ctx_after_oom(["-c", "100001"], oom, is_oom) == 49920
    assert lc._halved_ctx_after_oom(["-c", str(lc._CTX_HALVE_FLOOR)], oom, is_oom) is None
    assert lc._halved_ctx_after_oom(["-ngl", "99"], oom, is_oom) is None
    assert lc._halved_ctx_after_oom(["-c", "1048576"], "Segmentation fault", is_oom) is None
    # A memory error that is not the context (no context-creation line) is left alone.
    assert lc._halved_ctx_after_oom(["-c", "1048576"], TARGET_OOM[0], is_oom) is None


def test_the_context_limits_follow_the_launch_that_came_up(tmp_path):
    """A drafter whose context never fits: the halvings fail, the no-drafter fallback comes up at
    its own -c, and the advertised limits are that launch's, not the last halving's."""
    backend, gguf = h.cuda_backend(tmp_path, memory = [(0, 400_000, 400_000)])
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, **kw: h.GIB
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda n_ctx, *a, **kw: n_ctx * 1024
    backend._context_length = 1048576
    draft = h.sidecar(tmp_path)
    drafter_oom = [
        f"I common_speculative_init_result: loading draft model '{draft}'",
        "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000.00 MiB on device 0: cudaMalloc failed: out of memory",
        f"E cmn  common_init_: failed to create context with model '{draft}'",
    ]
    cmds = h.spawns(
        backend,
        gguf,
        healthy = lambda i, cmd: draft not in cmd,
        crash = lambda i, cmd: 1 if draft in cmd else None,
        stdout = lambda i, cmd: drafter_oom if draft in cmd else (),
        n_ctx = 1048576,
        max_seq_length_auto_derived = True,
        speculative_type = "dspark",
        dspark_draft_path = draft,
    )
    assert cmds.loaded is True
    last = cmds[-1]
    assert draft not in last
    # Not vacuous: the drafter launches were halved before the fallback.
    assert min(_ctx(c) for c in cmds if draft in c) < _ctx(cmds[0])
    assert backend._effective_context_length == _ctx(last)
    assert backend._max_context_length == _ctx(last)
