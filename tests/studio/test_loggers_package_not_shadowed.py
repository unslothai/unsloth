# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stub of loggers with no __path__ stays in sys.modules and breaks later submodule imports."""

import importlib
import sys


def test_the_loggers_entry_left_in_sys_modules_is_still_a_package():
    loggers = sys.modules.get("loggers")
    if loggers is None:
        return
    assert hasattr(loggers, "__path__"), (
        "a test module replaced `loggers` with a non-package stub and left it in "
        "sys.modules; give the stub __path__ = [<studio/backend/loggers>] so submodule "
        "imports still resolve"
    )


def test_a_real_submodule_still_imports_through_whatever_stub_is_installed():
    """__path__ is only worth asserting if it actually reaches the real files."""
    if sys.modules.get("loggers") is None:
        return
    assert importlib.import_module("loggers.media_progress") is not None
