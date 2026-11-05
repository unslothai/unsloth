# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A hub source one test applies does not decide the next test's endpoint.

``apply_hub_settings()`` writes ``UNSLOTH_STUDIO_HUB_SOURCE`` and the endpoints straight into
``os.environ``. A test that left ModelScope selected made ``_probe_endpoint()`` answer
huggingface.co for every later test on the xdist worker, and
``test_the_probe_url_is_built_safely`` then failed for a mirror it had set itself. The two tests
below run in order on one worker: the first switches to the source the process did not inherit,
the way the settings code writes it, the second fails unless conftest put it back in between.
"""

from __future__ import annotations

import os

import pytest

from utils import hub_settings

pytestmark = pytest.mark.xdist_group("hub_source_env_reset_between_tests")

# Read when the first test starts, not at collection: importing the app later in collection runs
# apply_hub_settings(), which writes the variable, and conftest restores what each test started with.
_SEEN: dict = {}


def test_a_test_leaves_another_source_selected():
    _SEEN["inherited"] = os.environ.get(hub_settings.SOURCE_ENV)
    _SEEN["other"] = (
        hub_settings.HUGGINGFACE
        if hub_settings.active_source() == hub_settings.MODELSCOPE
        else hub_settings.MODELSCOPE
    )
    # Assigned, not monkeypatched: apply_hub_settings() writes it the same way.
    os.environ[hub_settings.SOURCE_ENV] = _SEEN["other"]
    assert hub_settings.active_source() == _SEEN["other"]


def test_the_next_test_starts_on_the_source_it_inherited():
    assert _SEEN, "test_a_test_leaves_another_source_selected must run first, on this worker"
    assert os.environ.get(hub_settings.SOURCE_ENV) == _SEEN["inherited"]
    assert hub_settings.active_source() != _SEEN["other"]
