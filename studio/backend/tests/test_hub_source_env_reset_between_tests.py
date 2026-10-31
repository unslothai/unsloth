# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A hub source one test applies does not decide the next test's endpoint.

``apply_hub_settings()`` writes ``UNSLOTH_STUDIO_HUB_SOURCE`` and the endpoints straight into
``os.environ``. A test that left ModelScope selected made ``_probe_endpoint()`` answer
huggingface.co for every later test on the xdist worker, and
``test_the_probe_url_is_built_safely`` then failed for a mirror it had set itself. The two tests
below run in order on one worker: the first writes the source the way the settings code does, the
second fails unless conftest put it back in between.
"""

from __future__ import annotations

import os

import pytest

from utils import hub_settings

pytestmark = pytest.mark.xdist_group("hub_source_env_reset_between_tests")


def test_a_test_leaves_modelscope_selected():
    # Assigned, not monkeypatched: apply_hub_settings() writes it the same way.
    os.environ[hub_settings.SOURCE_ENV] = hub_settings.MODELSCOPE
    assert hub_settings.active_source() == hub_settings.MODELSCOPE


def test_the_next_test_starts_on_the_source_it_inherited():
    assert os.environ.get(hub_settings.SOURCE_ENV) != hub_settings.MODELSCOPE
    assert hub_settings.active_source() == hub_settings.HUGGINGFACE
