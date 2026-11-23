# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A DSpark drafter resolved before the launch (route discovery, a resident load) is rechecked against the
llama-server this launch uses: an Inkling 2 layout that build refuses is never put on the command line."""

from __future__ import annotations

import pytest

from . import _sidecar_fit_helpers as h
from . import test_dspark_dual_schema as dual
from ._sidecar_fit_helpers import placement
from utils.models.drafters import dspark_schema as ds


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    monkeypatch.setattr(ds, "_default_llama_server_binary", lambda: None)
    ds._scan_cache.clear()
    ds._remote_cache.clear()


@pytest.mark.parametrize(
    "fixture, loads, emitted",
    [
        ("inkling2_markov2_q8_0", "fused", False),
        ("inkling2_fused_q8_0", "fused", True),
        ("inkling2_fused_q8_0", "markov2", False),
        ("inkling2_markov2_q8_0", "fused,markov2", True),
        # A generic DSpark drafter is never filtered, whatever the build loads.
        ("generic_qwen3_4b_bf16", "markov2", True),
    ],
)
def test_a_carried_drafter_is_rechecked_against_the_launch_binary(
    tmp_path, monkeypatch, fixture, loads, emitted
):
    # /fake/llama-server has no readable library, so the env setting is what the build loads.
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, loads)
    backend, gguf = h.cuda_backend(tmp_path)
    drafter = dual._drafter(tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf", fixture)
    cmd = placement._launch(
        backend, gguf, n_ctx = 4096, speculative_type = "dspark", dspark_draft_path = str(drafter)
    )["cmd"]
    assert (str(drafter) in cmd) is emitted
