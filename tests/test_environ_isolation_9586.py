# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for issue #9586 channel 5: os.environ leaking out of a test.

The medium is what makes this one distinct. Neither `git status` nor any stray-file
sweep can see a process-global environment write, so the containment needs its own
tests rather than riding on the file-residue ones.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_SHARED = Path(__file__).resolve().parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from environ_isolation import contained_environ  # noqa: E402


# The exact pair _install_bnb_windows_rocm() writes, kept as literals rather than
# imported: the point is the observed leak, and importing studio.install_python_stack
# to read the constant would drag the installer into this test for a string.
_LEAKED_BY_THE_INSTALLER = {
    "BNB_ROCM_VERSION": "72",
    "UNSLOTH_BNB_ROCM_VERSION_SOURCE": "detected",
}


def test_the_installer_pair_does_not_outlive_the_block():
    """The observed instance, by name.

    `_install_bnb_windows_rocm()` sets both so the worker subprocess inherits them,
    which is right for a real install. It then READS both a few lines above the write
    to decide whether to re-detect, so a value surviving into a later test in the same
    worker flips that branch and the test passes for the wrong reason.
    """
    before = dict(os.environ)
    for key in _LEAKED_BY_THE_INSTALLER:
        assert key not in before, f"{key} leaked into this test from somewhere earlier"

    with contained_environ():
        os.environ.update(_LEAKED_BY_THE_INSTALLER)
        assert os.environ["BNB_ROCM_VERSION"] == "72"

    assert dict(os.environ) == before


def test_a_key_added_inside_is_removed():
    before = dict(os.environ)
    with contained_environ():
        os.environ["UNSLOTH_TEST_9586_ADDED"] = "1"
    assert "UNSLOTH_TEST_9586_ADDED" not in os.environ
    assert dict(os.environ) == before


def test_a_key_changed_inside_is_put_back(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TEST_9586_CHANGED", "original")
    with contained_environ():
        os.environ["UNSLOTH_TEST_9586_CHANGED"] = "overwritten"
    assert os.environ["UNSLOTH_TEST_9586_CHANGED"] == "original"


def test_a_key_deleted_inside_is_put_back(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TEST_9586_DELETED", "original")
    with contained_environ():
        del os.environ["UNSLOTH_TEST_9586_DELETED"]
    assert os.environ["UNSLOTH_TEST_9586_DELETED"] == "original"


def test_the_body_raising_still_restores():
    """Containment cannot depend on the test passing."""
    before = dict(os.environ)
    with pytest.raises(RuntimeError):
        with contained_environ():
            os.environ["UNSLOTH_TEST_9586_RAISED"] = "1"
            raise RuntimeError("boom")
    assert "UNSLOTH_TEST_9586_RAISED" not in os.environ
    assert dict(os.environ) == before


def test_restoration_writes_through_to_the_real_environment():
    """os.environ is mutated in place, not rebound.

    Subprocesses read the process environment through putenv, not through whatever
    object this module happens to hold, so a restore that rebound os.environ would
    leave every child still seeing the leaked value.
    """
    environ_before = os.environ
    with contained_environ():
        os.environ["UNSLOTH_TEST_9586_INPLACE"] = "1"
    assert os.environ is environ_before
    assert os.getenv("UNSLOTH_TEST_9586_INPLACE") is None


def test_the_containment_fixture_is_autouse(request):
    """Pins the wiring, not the mechanism.

    An ordering-based test ("a later test cannot see it") is not a pin: under xdist
    the two halves can land on different workers, where it passes for free.
    """
    assert "_contain_environ" in request.fixturenames
