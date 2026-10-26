# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Auto load relaunched as an ordinary server keeps the parking it launched with. The first
launch records that parking was asked for before ``/props`` is read for proof, so a server whose
confirmation read failed still counts as one that may park. The relaunch, the same argv, must
not drop that evidence by deriving it from the failed confirmation: the replacement still parks,
and a stream it parks needs the grace and the neighbours need the budget."""

from __future__ import annotations

from core.inference import llama_cpp as llama_mod
import pytest

from .test_an_auto_load_with_an_unsized_pool_does_not_keep_an_oversized_parking_budget import (
    _Backend,
)
from .test_an_auto_relaunch_takes_the_ordinary_parking_bound import (
    _EXACT_BUDGET,
    _relaunch,
    roomy_host,  # noqa: F401 - fixture
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def test_an_unconfirmed_parking_launch_still_parks_after_the_relaunch(roomy_host):  # noqa: F811
    backend = _Backend()
    # Launched with --preempt-ram, but the confirmation read found no proof.
    backend._server_preempts_kv = False
    backend._launched_parking_mib = _EXACT_BUDGET
    ordinary = llama_mod._host_bounded_parking_mib(_EXACT_BUDGET, backend)
    relaunch, launched = _relaunch(backend, exact_env = True)
    assert relaunch(None, True) is True
    assert len(launched) == 1
    assert launched[0][-2:] == ["--preempt-ram", str(ordinary)]
    assert backend._server_launched_parking is True
    assert backend._launched_parking_mib == ordinary


def test_a_relaunch_with_parking_turned_off_records_none(roomy_host):  # noqa: F811
    backend = _Backend()
    relaunch, launched = _relaunch(backend, exact_env = True)
    relaunch.__globals__["_preempt_ram_disabled_in"] = lambda *a, **k: True
    assert relaunch(None, True) is True
    assert backend._server_launched_parking is False
    assert backend._server_preempts_kv is False
    assert backend._launched_parking_mib is None
