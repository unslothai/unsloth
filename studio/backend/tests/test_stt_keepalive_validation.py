# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The keep-alive endpoint feeds a timer, so invalid durations must not reach it."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from routes import inference as inference_routes


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_keepalive_rejects_non_finite_or_negative_seconds(value):
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            inference_routes.stt_keep_alive(
                engine = None,
                model = None,
                keep_alive_seconds = value,
                current_subject = "test",
            )
        )
    assert raised.value.status_code == 422
