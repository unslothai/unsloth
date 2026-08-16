# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two things ``disable_vision`` promises that the argv alone cannot deliver.

1. The child env. llama.cpp reads ``LLAMA_ARG_MMPROJ`` / ``LLAMA_ARG_MMPROJ_URL``
   straight into ``params.mmproj.path`` / ``.url`` (common/arg.cpp, the
   ``{"-mm", "--mmproj"}`` and ``{"-mmu", "--mmproj-url"}`` options), and
   ``--no-mmproj`` only clears the ``-hf`` AUTO-download (``opts.download_mmproj``
   is gated on ``params.mmproj.path.empty()``), never an explicit path. So merely
   not emitting ``--mmproj`` leaves an inherited projector loading: the server is
   multimodal while Studio reports ``effective_is_vision=False``, and the fit
   budget that just spent the projector's bytes on context is short by them.

2. The download-manager lock. A load that opens no projector must not be refused
   because the cached snapshot lacks one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement  # noqa: F401,E402  (installs the import stubs)

from test_metal_paravirtual_guard import (  # noqa: E402
    _load_model_source,
    _load_model_tree,
    _names,
)


# ── 1. the inherited-projector env scrub ────


def _vision_off_env_scrub(*, disable_vision: bool) -> dict:
    """Run load_model's real ``disable_vision`` env scrub and report the child's env."""
    keep = [
        node
        for node in ast.walk(_load_model_tree())
        if isinstance(node, ast.If) and "_dv_mmproj_var" in _names(node)
    ]
    assert len(keep) == 1, "the vision-off projector env scrub left load_model"
    scope = {
        "disable_vision": disable_vision,
        "env": {
            "LLAMA_ARG_MMPROJ": "/inherited/proj.gguf",
            "LLAMA_ARG_MMPROJ_URL": "https://example.invalid/proj.gguf",
            "LLAMA_ARG_THREADS": "8",
        },
    }
    exec(ast.unparse(ast.Module(body = keep, type_ignores = [])), scope)
    return scope["env"]


def test_turning_vision_off_drops_the_inherited_projector():
    env = _vision_off_env_scrub(disable_vision = True)
    assert "LLAMA_ARG_MMPROJ" not in env
    assert "LLAMA_ARG_MMPROJ_URL" not in env
    # ...and nothing else in the inherited env is touched here.
    assert env == {"LLAMA_ARG_THREADS": "8"}


def test_leaving_vision_on_keeps_the_inherited_projector():
    """The scrub is scoped to the opt-out: a deliberate LLAMA_ARG_MMPROJ is how a user
    points llama-server at a projector Studio's own discovery never finds."""
    env = _vision_off_env_scrub(disable_vision = False)
    assert env["LLAMA_ARG_MMPROJ"] == "/inherited/proj.gguf"
    assert env["LLAMA_ARG_MMPROJ_URL"] == "https://example.invalid/proj.gguf"


def test_the_vision_off_scrub_lands_before_the_spawn():
    """The env is built once and handed to Popen, so a scrub after the spawn would leave
    the projector loaded on the server it just told the UI has none."""
    src = _load_model_source()
    scrub = 'for _dv_mmproj_var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")'
    assert src.index(scrub) < src.index("_spawn_and_wait(")


# ── 2. the download-manager lock ────


def _blocks(intent) -> bool:
    """``_with_gguf_load_marker``'s guard, with the registry and cache stubbed out."""
    import core.inference.llama_cpp as llama_cpp

    seen: dict = {}

    def _fake_blocks(
        hf_repo,
        hf_variant,
        *,
        require_mmproj = False,
        hf_token = None,
    ):
        seen["require_mmproj"] = require_mmproj
        # The shape this item is about: another variant of the same repo is
        # downloading and the cached copy has the weights but no projector.
        return bool(require_mmproj)

    original = llama_cpp._hub_download_blocks_gguf_load
    llama_cpp._hub_download_blocks_gguf_load = _fake_blocks
    try:

        @llama_cpp._with_gguf_load_marker
        def _load(
            self,
            intent,
            load_cancel_event = None,
        ):
            return "loaded"

        try:
            _load(object(), intent)
        except RuntimeError:
            return True
        return False
    finally:
        llama_cpp._hub_download_blocks_gguf_load = original


def test_a_vision_off_load_is_not_refused_over_an_uncached_projector():
    from core.inference.llama_cpp import GgufLoadIntent
    assert (
        _blocks(
            GgufLoadIntent(
                model_identifier = "m",
                hf_repo = "unsloth/some-vlm-GGUF",
                hf_variant = "Q4_K_M",
                is_vision = True,
                disable_vision = True,
            )
        )
        is False
    )


def test_a_vision_on_load_still_waits_for_the_projector():
    from core.inference.llama_cpp import GgufLoadIntent
    assert (
        _blocks(
            GgufLoadIntent(
                model_identifier = "m",
                hf_repo = "unsloth/some-vlm-GGUF",
                hf_variant = "Q4_K_M",
                is_vision = True,
                disable_vision = False,
            )
        )
        is True
    )


def test_the_route_preflight_uses_the_same_gate():
    """The two lock checks have to agree: /load's preflight runs first, so a gate only
    llama_cpp.py applied would still 409 before load_model was ever entered."""
    import inspect

    import routes.inference as routes_inference

    src = inspect.getsource(routes_inference)
    marker = "gguf_intent is not None and gguf_intent.disable_vision"
    assert marker in src
