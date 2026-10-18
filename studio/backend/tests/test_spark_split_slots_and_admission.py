# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paired-Spark serving against the single-server planner and admission queue.

A layer split launches with ``--pipeline-groups``, which llama-server refuses unless
``--parallel`` is a multiple of it, so the planner's slot reduction must step in whole groups.
A replica pair serves twice the primary's slots, so route admission must not cap it at one
server's.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_matrix = _load("_matrix_for_spark_split_slots", "test_auto_offload_ctx_platform_matrix.py")
_placement = _matrix._placement

from core.inference import spark_serving  # noqa: E402
from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from routes import inference as route  # noqa: E402

GIB = 1024**3
# One GB10 as the planner sees it: CUDA, unified, ~112 GiB free.
_GB10 = _matrix.Accelerator("gb10", False, ((0, 115_000, 124_000),))


def _split_launch(
    tmp_path,
    monkeypatch,
    extra_args,
    *,
    model_gib = 80,
    n_parallel = 32,
):
    backend, gguf = _matrix.cell_backend(
        tmp_path, monkeypatch, _matrix.PLATFORMS[0], _GB10, model_fraction = 1.0
    )
    backend._get_gguf_size_bytes = lambda _path: int(model_gib * GIB)
    backend.probe_server_capabilities = lambda _binary = None: {"supports_kv_unified": True}
    _placement._install_slot_scaled_compute(backend)
    result = _matrix._launch(
        backend, gguf, n_ctx = 32768, n_parallel = n_parallel, extra_args = tuple(extra_args)
    )
    return int(_matrix._flag(result["cmd"], "--parallel"))


def test_slot_reduction_keeps_a_pipeline_groups_multiple(tmp_path, monkeypatch):
    extra = spark_serving.layer_split_extra_args("10.0.0.2", 50052, pipeline_groups = 2)
    slots = _split_launch(tmp_path, monkeypatch, extra)
    # Reduced (the planner prices one Spark), but never to a count the server refuses.
    assert 1 < slots < 32
    assert slots % 2 == 0


def test_slot_reduction_without_groups_is_unchanged(tmp_path, monkeypatch):
    # Same load, no --pipeline-groups: still steps one slot at a time.
    assert _split_launch(tmp_path, monkeypatch, []) == 23


def test_pipeline_groups_slot_step_reads_argv_then_env():
    _pipeline_groups_slot_step = llama_cpp._pipeline_groups_slot_step
    assert _pipeline_groups_slot_step([]) == 1
    assert _pipeline_groups_slot_step(["--pipeline-groups", "2"]) == 2
    assert _pipeline_groups_slot_step(["--pipeline-groups=4"]) == 4
    assert _pipeline_groups_slot_step([], {"LLAMA_ARG_PIPELINE_GROUPS": "2"}) == 2
    assert _pipeline_groups_slot_step(["--pipeline-groups", "x"]) == 1


def test_slots_that_fit_skips_counts_the_groups_cannot_divide():
    backend = LlamaCppBackend()
    backend._estimate_compute_buffer_bytes = lambda **_kw: 1
    backend._estimate_kv_cache_bytes = lambda *_a, n_parallel = 1, **_kw: n_parallel * GIB
    backend._select_gpus_split_aware = lambda total, *_a, **_kw: (
        ([0], False) if total <= 23 * GIB + 1 else (None, True)
    )
    args = (32, 4096, [(0, 1)], {}, 0, None, 1.0, 0, 1)
    assert backend._slots_that_fit_on_gpu(*args)[2] == 23
    assert backend._slots_that_fit_on_gpu(*args, slot_step = 2)[2] == 22


class _FakeBackend:
    effective_parallel_slots = 4
    _kv_cache_context_total = 32768
    context_length = 8192


def _with_replicas(monkeypatch, healthy):
    routed = _FakeBackend()
    router = SimpleNamespace(healthy_backends = lambda: list(healthy))
    monkeypatch.setattr(spark_serving, "_STATE", SimpleNamespace(router = router))
    monkeypatch.setattr(
        spark_serving,
        "route_base_url",
        lambda backend: "http://127.0.0.1:1" if backend is routed else None,
    )
    return routed


def test_admission_counts_every_healthy_replica(monkeypatch):
    backend = _with_replicas(monkeypatch, ["main", "peer"])
    assert route._openai_llama_admission_capacity(None, backend) == 8
    assert route._openai_llama_admission_budget(backend) == 65536


def test_admission_shrinks_back_when_the_peer_is_down(monkeypatch):
    backend = _with_replicas(monkeypatch, ["main"])
    assert route._openai_llama_admission_capacity(None, backend) == 4
    assert route._openai_llama_admission_budget(backend) == 32768


def test_admission_off_spark_is_one_server(monkeypatch):
    _with_replicas(monkeypatch, ["main", "peer"])
    other = _FakeBackend()  # not the backend the router fronts
    assert route._openai_llama_admission_capacity(None, other) == 4
    assert route._openai_llama_admission_budget(other) == 32768
    monkeypatch.setattr(spark_serving, "_STATE", None)
    monkeypatch.undo()
    assert route._openai_llama_admission_capacity(None, _FakeBackend()) == 4


@pytest.mark.parametrize("total,ctx,expected", [(None, 8192, 8192), (None, None, None)])
def test_admission_budget_fallbacks_unchanged(total, ctx, expected):
    backend = SimpleNamespace(_kv_cache_context_total = total, context_length = ctx)
    assert route._openai_llama_admission_budget(backend) == expected
