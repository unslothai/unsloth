# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in for the test modules that exercise KV preemption.

``UNSLOTH_LLAMA_ADMISSION_PREEMPT`` is off unless an operator sets it, so a module that
tests pausing and resuming has to ask for it the way an operator would::

    pytestmark = pytest.mark.usefixtures("preemption_opted_in")

Deliberately NOT autouse: switching it on for the whole suite would hide the real default
from every test that is supposed to see it, which is the regression the default exists to
prevent. ``tests/conftest.py`` imports the fixture so the marker resolves from anywhere,
including the few modules a sibling imports as a top-level module rather than through the
package, where a relative import of this file would not resolve at all.

A test that sets the variable itself still wins, since ``monkeypatch.setenv`` in a test
body runs after the fixtures that test depends on; that is what keeps the explicit ``=0``
and ``=1`` cases in these modules meaningful.
"""

import pytest

from core.inference.llama_preemption import PREEMPT_ENV


@pytest.fixture
def preemption_opted_in(monkeypatch):
    """Run this module as an install that opted in. Undone after every test."""
    monkeypatch.setenv(PREEMPT_ENV, "1")
    yield
