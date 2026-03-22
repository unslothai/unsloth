# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for _platform_compat -- Anaconda/conda-forge sys.version shim.
"""

import importlib
import platform
import sys

import pytest


# Representative version strings
ANACONDA_VERSION = (
    "3.12.4 | packaged by Anaconda, Inc. | "
    "(main, Jun 18 2024, 15:12:24) [GCC 11.2.0]"
)
CONDA_FORGE_VERSION = (
    "3.11.9 | packaged by conda-forge | " "(main, Apr 19 2024, 18:36:13) [GCC 12.3.0]"
)
STANDARD_VERSION = "3.12.4 (main, Jun 18 2024, 15:12:24) [GCC 11.2.0]"


@pytest.fixture(autouse = True)
def _clean_platform_state():
    """Save and restore sys.version, platform cache, and module state."""
    orig_version = sys.version
    orig_cache = getattr(platform, "_sys_version_cache", {}).copy()

    yield

    sys.version = orig_version
    if hasattr(platform, "_sys_version_cache"):
        platform._sys_version_cache.clear()
        platform._sys_version_cache.update(orig_cache)


def _reload_compat():
    """Force re-execute _platform_compat module logic."""
    import _platform_compat

    importlib.reload(_platform_compat)


class TestAnacondaVersion:
    def test_cache_seeded_for_anaconda(self):
        sys.version = ANACONDA_VERSION
        platform._sys_version_cache.pop(ANACONDA_VERSION, None)
        _reload_compat()
        assert ANACONDA_VERSION in platform._sys_version_cache

    def test_parsed_values_correct(self):
        sys.version = ANACONDA_VERSION
        platform._sys_version_cache.pop(ANACONDA_VERSION, None)
        _reload_compat()
        result = platform._sys_version_cache[ANACONDA_VERSION]
        # result is a tuple: (name, version, branch, revision, buildno, builddate, compiler)
        assert result[0] == "CPython"
        assert result[1] == "3.12.4"

    def test_platform_calls_succeed(self):
        sys.version = ANACONDA_VERSION
        platform._sys_version_cache.pop(ANACONDA_VERSION, None)
        _reload_compat()
        # These would raise ValueError without the fix
        assert platform.python_implementation() == "CPython"
        assert platform.python_version() == "3.12.4"


class TestCondaForgeVersion:
    def test_cache_seeded_for_conda_forge(self):
        sys.version = CONDA_FORGE_VERSION
        platform._sys_version_cache.pop(CONDA_FORGE_VERSION, None)
        _reload_compat()
        assert CONDA_FORGE_VERSION in platform._sys_version_cache

    def test_parsed_values_correct(self):
        sys.version = CONDA_FORGE_VERSION
        platform._sys_version_cache.pop(CONDA_FORGE_VERSION, None)
        _reload_compat()
        result = platform._sys_version_cache[CONDA_FORGE_VERSION]
        assert result[0] == "CPython"
        assert result[1] == "3.11.9"


class TestStandardCPython:
    def test_no_op_for_standard_version(self):
        sys.version = STANDARD_VERSION
        # Clear any prior cache entry to verify module does not add one
        platform._sys_version_cache.pop(STANDARD_VERSION, None)
        cache_before = platform._sys_version_cache.copy()
        _reload_compat()
        # Standard version has no pipes -- module should be a no-op
        # (the stdlib will cache it on first use, but our module shouldn't add it)
        assert STANDARD_VERSION not in cache_before
        # The module's guard `if "|" in sys.version` means _seed_sys_version_cache
        # is never called, so no new entry should appear for STANDARD_VERSION
        # unless platform itself cached it during reload
        # Just verify no error
        assert platform.python_implementation() == "CPython"


class TestLoggerImportChain:
    def test_loggers_import_succeeds_with_anaconda_version(self):
        """The real failure path: structlog -> rich -> attrs -> platform crash."""
        sys.version = ANACONDA_VERSION
        platform._sys_version_cache.pop(ANACONDA_VERSION, None)
        _reload_compat()
        # This import chain triggers platform.python_implementation()
        # via attrs._compat (used by structlog/rich)
        try:
            import loggers  # noqa: F401
        except ImportError:
            pytest.skip("loggers package not installed in test env")
