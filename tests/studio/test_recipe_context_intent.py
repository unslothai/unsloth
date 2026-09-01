# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The context intent the recipe load gates compare, executed rather than restated.

`contextIntent` lives inside a React hook module that cannot be imported on its own, so
it is sliced out of the real source and run. Restating the predicate here would pass
just as happily with the source deleted.

The rule it encodes: an unpinned MLX load sends 0, so a positive `requested_context_length`
from MLX is an explicit pin. llama.cpp is ambiguous, because a same-model reload echoes the
resolved n_ctx while the control is still Auto (see `resolve-ctx-pin-seed.ts`). Reading a
GGUF echo as a pin makes an unpinned recipe reload the resident model on every run, and the
restoration snapshot then replays that value as a pin the user never set.
"""

from __future__ import annotations

import textwrap

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

RECIPES = source_path("studio/frontend/src/features/recipe-studio/hooks/use-recipe-executions.ts")

TEMP = WORKDIR / "temp" / "recipe_context_intent"

SOURCES = (RECIPES,)


def _harness_source() -> str:
    return "// @ts-nocheck\nexport " + slice_between(
        read(RECIPES),
        "function contextIntent(",
        "\nasync function isLocalModelAlreadyLoaded(",
    )


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script, sources = SOURCES)


def test_only_mlx_reads_a_positive_context_echo_as_a_pin():
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { contextIntent } from "./harness.ts";
            console.log(JSON.stringify({
              mlxPinned: contextIntent(32768, true),
              mlxAuto: contextIntent(0, true),
              mlxUnset: contextIntent(null, true),
              ggufEcho: contextIntent(32768, false),
              ggufAuto: contextIntent(0, false),
            }));
            """
        )
    )
    # MLX: a pinned resident and an unpinned recipe are different loads.
    assert out["mlxPinned"] == 32768
    assert out["mlxAuto"] is None
    assert out["mlxUnset"] is None
    # GGUF: the echo says nothing, so an unpinned recipe must not force a reload.
    assert (
        out["ggufEcho"] is None
    ), "a positive GGUF echo is the resolved n_ctx of an Auto load, not a pin"
    assert out["ggufAuto"] is None


def test_both_load_gates_ask_the_backend_before_comparing_intent():
    """The predicate is only correct if both callers pass the flag; neither may drop it."""
    source = read(RECIPES)
    assert "contextIntent(requestedContextLength, residentIsMlx)" in source
    assert "contextIntent(status.requested_context_length, residentIsMlx)" in source
    assert "contextIntent(left.requestedContextLength, left.isMlx)" in source
    assert "contextIntent(right.requestedContextLength, right.isMlx)" in source
