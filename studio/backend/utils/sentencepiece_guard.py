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
    """Windows by default, every other platform only when asked.

    An unrecognised value falls back to the platform default rather than raising: this runs at
    the top of a process, where a typo in an environment variable must not be fatal.
    """
    value = (os.environ.get(DISABLE_SENTENCEPIECE_VARIABLE) or "").strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return sys.platform == "win32"


def disable_sentencepiece_on_windows():
    """Make ``import sentencepiece`` fail the way an uninstalled package does.

    On Windows the compiled extension is never handed to the loader, so a code integrity
    policy has no file to refuse and the user gets no Bad Image dialog; a probe to find out
    whether this machine would refuse it is itself that dialog. A ``None`` entry is CPython's
    documented sentinel and makes the import raise ImportError, the ordinary "not installed"
    state.

    Must run before transformers is imported, which reads availability during its own import.
    Returns True only when this call is what made it absent.
    """
    if not sentencepiece_should_be_disabled():
        return False
    if "sentencepiece" in sys.modules:
        # Already imported by something earlier, or already disabled by an earlier call.
        # Replacing a live module would break whoever is holding it.
        return sys.modules["sentencepiece"] is None
    sys.modules["sentencepiece"] = None
    return True
