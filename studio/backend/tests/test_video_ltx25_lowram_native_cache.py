# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The LTX-2.5 low-RAM route upcasts its native int8 build to float32 compute: the native prequant cache must hold
the storage-dtype build, so a later bf16 load reads back exactly what it would have built."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
pytest.importorskip("accelerate")

TESTS = Path(__file__).resolve().parent
BACKEND = TESTS.parent
for p in (str(BACKEND), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_video_native_prequant_cache as W  # noqa: E402
from test_video_native_prequant_cache import base_dir, cache_root  # noqa: E402,F401

import core.inference.video as v  # noqa: E402
import core.inference.video_ltx25_lowram as lr  # noqa: E402
from core.inference import diffusion_native_cache as NC  # noqa: E402


def _cached(base_dir, fam, spec):
    path = NC.lookup(fam, spec, base_repo = base_dir, component = "transformer")
    assert path is not None
    return NC.load(
        path, W.diffusers.WanTransformer3DModel, base_dir, component = "transformer", spec = spec
    )


def _float32_tensors(module):
    return sorted(k for k, t in module.state_dict().items() if t.dtype == torch.float32)


def _load_as_route(base_dir, monkeypatch):
    """The route's order in load_pipeline: write the cache unless deferred, then upcast; a deferred write runs after."""
    fam, spec = W._family(), W._spec()
    fresh = W._fresh_build(base_dir, "transformer")
    deferred = v._native_cache_write_deferred(lowram_engaged = True)
    if not deferred:
        assert NC.save(fresh, fam, spec, base_repo = base_dir, component = "transformer")
    lr.upcast_compute_(types.SimpleNamespace(transformer = fresh), torch.float32)
    if deferred:
        assert NC.save(fresh, fam, spec, base_repo = base_dir, component = "transformer")
    return fam, spec


def test_route_cache_round_trips_into_a_bf16_load(base_dir, monkeypatch):
    monkeypatch.setattr(v, "_native_cache_background", lambda: True)
    fam, spec = _load_as_route(base_dir, monkeypatch)
    # a later load on a bf16 card reads the cache: bit-identical to the build it would make itself
    cached = _cached(base_dir, fam, spec)
    W._assert_same_module(W._fresh_build(base_dir, "transformer"), cached)
    assert _float32_tensors(cached) == _float32_tensors(W._fresh_build(base_dir, "transformer"))


def test_writing_after_the_upcast_would_poison_the_cache(base_dir):
    # the hazard the ordering avoids: the upcast build's float32 tensors land in the entry
    fam, spec = W._family(), W._spec()
    fresh = W._fresh_build(base_dir, "transformer")
    lr.upcast_compute_(types.SimpleNamespace(transformer = fresh), torch.float32)
    assert NC.save(fresh, fam, spec, base_repo = base_dir, component = "transformer")
    cached = _cached(base_dir, fam, spec)
    assert cached is not None
    poisoned = set(_float32_tensors(cached)) - set(
        _float32_tensors(W._fresh_build(base_dir, "transformer"))
    )
    assert poisoned  # bf16 norms / scales stored as float32
