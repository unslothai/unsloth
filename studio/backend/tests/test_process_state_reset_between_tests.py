# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Process-wide state one test changes does not leak into the next test on the same xdist worker.

1. The diffusers attention backend.

diffusers' ``ModelMixin.set_attention_backend`` also sets the process-wide active backend, which every
processor without a backend of its own dispatches to. ``test_diffusion_flux3_strided_attn.py`` set
``_native_cudnn`` that way, and every later CPU forward on the xdist worker failed with "No viable backend
for scaled_dot_product_attention" (13 tests in test_diffusion_minimax_h3_image.py, 3 in
test_diffusion_flux3_qk_rope.py). The two tests below run in order on one worker: the first leaves another
backend active the way diffusers sets it, the second fails unless conftest put it back in between.
"""

from __future__ import annotations

import pytest

dispatch = pytest.importorskip("diffusers.models.attention_dispatch")

pytestmark = pytest.mark.xdist_group("process_state_reset_between_tests")

_SEEN: dict = {}


def test_a_test_leaves_another_backend_active():
    registry = dispatch._AttentionBackendRegistry
    _SEEN["inherited"] = registry._active_backend
    _SEEN["other"] = (
        dispatch.AttentionBackendName.NATIVE
        if registry._active_backend == dispatch.AttentionBackendName._NATIVE_CUDNN
        else dispatch.AttentionBackendName._NATIVE_CUDNN
    )
    # set_active_backend is what ModelMixin.set_attention_backend calls.
    registry.set_active_backend(_SEEN["other"])
    assert registry.get_active_backend()[0] == _SEEN["other"]


def test_the_next_test_starts_on_the_backend_it_inherited():
    assert _SEEN, "test_a_test_leaves_another_backend_active must run first, on this worker"
    assert dispatch._AttentionBackendRegistry._active_backend == _SEEN["inherited"]


def test_a_test_leaves_dynamo_suppressing_errors():
    import torch._dynamo

    cfg = torch._dynamo.config
    _SEEN["suppress_errors"] = cfg.suppress_errors
    _SEEN["recompile_limit"] = getattr(cfg, "recompile_limit", None)
    cfg.suppress_errors = (
        not cfg.suppress_errors
    )  # the trainer's compile step sets it this way, no restore
    if hasattr(cfg, "recompile_limit"):
        cfg.recompile_limit = cfg.recompile_limit + 56


def test_the_next_test_starts_with_dynamo_as_it_found_it():
    import torch._dynamo

    assert (
        "suppress_errors" in _SEEN
    ), "test_a_test_leaves_dynamo_suppressing_errors must run first, on this worker"
    cfg = torch._dynamo.config
    assert cfg.suppress_errors == _SEEN["suppress_errors"]
    assert getattr(cfg, "recompile_limit", None) == _SEEN["recompile_limit"]
