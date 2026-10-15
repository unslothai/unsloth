# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's spelling of the Windows sentencepiece rule in unsloth/import_fixes.py.

Studio cannot reach that one by importing unsloth: that runs unsloth/__init__.py, whose GPU
branch pulls torch, Triton, transformers and the model stack into processes built to stay
light, and can open a competing GPU context. So the rule lives here, stdlib only, importable
at the very top of any Studio process. tests/test_windows_no_sentencepiece.py drives both
spellings through the same cases.
"""

import os
import sys

DISABLE_SENTENCEPIECE_VARIABLE = "UNSLOTH_DISABLE_SENTENCEPIECE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def sentencepiece_should_be_disabled():
    """Windows defaults to disabled, elsewhere only when asked; an unrecognised value is never fatal."""
    value = (os.environ.get(DISABLE_SENTENCEPIECE_VARIABLE) or "").strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return sys.platform == "win32"


def disable_sentencepiece_on_windows():
    """Sets a None sys.modules sentinel so import fails as if uninstalled, and the DLL is never loaded."""
    if not sentencepiece_should_be_disabled():
        return False
    if "sentencepiece" in sys.modules:
        # Replacing a live or already-disabled module would break whoever holds it.
        return sys.modules["sentencepiece"] is None
    if "transformers" in sys.modules:
        # Too late: transformers cached "installed", so a sentinel would make working tokenizers raise.
        return False
    sys.modules["sentencepiece"] = None
    return True
