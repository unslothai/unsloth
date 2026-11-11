# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.flows.nodes.logic import random_number
from core.flows.registry import FlowNodeError

from .flows_player_fakes import FakeCtx, run


def test_integer_random_number_stays_inside_fractional_bounds():
    for seed in range(50):
        ctx = FakeCtx("logic.random_number", params = {"min": 1.9, "max": 2.1}, seed = seed)
        assert run(random_number(ctx))["value"] == 2
        ctx = FakeCtx("logic.random_number", params = {"min": -2.5, "max": -1.5}, seed = seed)
        assert run(random_number(ctx))["value"] == -2


def test_integer_random_number_refuses_a_range_without_whole_numbers():
    ctx = FakeCtx("logic.random_number", params = {"min": 1.2, "max": 1.8})
    with pytest.raises(FlowNodeError, match = "No whole number"):
        run(random_number(ctx))
