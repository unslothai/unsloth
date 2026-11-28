# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Skip gate for tests of transformers 5's weight loading (``transformers.core_model_loading``).

On transformers 4, ``import unsloth`` (tests/conftest.py imports it) registers a stub module of that name so PEFT
imports; ``pytest.importorskip`` alone then passes in the pytest process. Spawned gloo ranks do not import unsloth:
they hit the real ImportError while re-importing the test module and exit without results, so the test failed
instead of skipping."""

import pytest

_STUB = "__unsloth_stub__"  # unsloth/import_fixes.py _UNSLOTH_STUB_SENTINEL


def require_core_model_loading():
    """transformers 5's core_model_loading module, else skip (module level or inside a test)."""
    try:
        import transformers.core_model_loading as module
    except ImportError:
        module = None
    if module is None or getattr(module, _STUB, False):
        pytest.skip(
            "needs transformers 5 weight loading (transformers.core_model_loading)",
            allow_module_level = True,
        )
    return module
