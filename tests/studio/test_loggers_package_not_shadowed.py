# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`loggers` must still be a PACKAGE after every test module has been imported.

Test modules that exercise a single backend file routinely stub `loggers` so they need
not pull structlog and the rest of the chain in. The stub is fine; leaving a stub with no
`__path__` in `sys.modules` is not, because that entry outlives the module that installed
it and pytest imports every test module during collection. Any later test that reaches a
real submodule then dies at COLLECTION with

    ModuleNotFoundError: No module named 'loggers.media_progress'; 'loggers' is not a package

which reads like a missing dependency rather than one test file shadowing a package for
the whole session. tests/studio/load_freeze/test_load_orchestrator.py did exactly this
(from #5669) and took tests/studio/test_mlx_context_platform_matrix.py down with it, so
the whole directory could not be collected in one pytest process.

Asserting this at RUN time is what makes it a guard: collection has finished by then, so
every module-level stub in the session is already installed.
"""

import importlib
import sys


def test_the_loggers_entry_left_in_sys_modules_is_still_a_package():
    loggers = sys.modules.get("loggers")
    if loggers is None:
        return  # nothing in this session touched it, so nothing can have shadowed it
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
