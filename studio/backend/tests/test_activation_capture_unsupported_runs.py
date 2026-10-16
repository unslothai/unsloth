# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decision and embedding runs never wire the capture callback, so their config must not claim it."""

import pytest

from core.training.training import _build_training_worker_config


def _config(**extra):
    return _build_training_worker_config(
        {"model_name": "m", "enable_activation_capture": True, **extra}
    )


@pytest.mark.parametrize("kind", ["is_embedding", "is_decision"])
def test_capture_is_off_for_runs_without_the_callback(kind):
    assert _config(**{kind: True})["enable_activation_capture"] is False


def test_capture_stays_on_for_a_text_run():
    assert _config()["enable_activation_capture"] is True
