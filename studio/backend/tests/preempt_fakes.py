# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in for the test modules that exercise KV preemption."""

import pytest

from core.inference.llama_preemption import PREEMPT_ENV


@pytest.fixture
def preemption_opted_in(monkeypatch):
    """Run this module as an install that opted in. Undone after every test."""
    monkeypatch.setenv(PREEMPT_ENV, "1")
    yield
