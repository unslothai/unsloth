# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``UNSLOTH_LLAMA_ADMISSION_PREEMPT=0`` switches the server's parking off as well. A build the
probe confirms has ``--preempt-ram`` may predate unslothai/llama.cpp#197 and park on an implicit
8192 MiB, so naming no budget is not enough: the zero has to be written, as it is for a probe
that read nothing."""

import pytest

from core.inference import llama_cpp as llama_mod
from core.inference import llama_preemption as preemption


@pytest.fixture(autouse = True)
def _opted_out(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    monkeypatch.setenv(preemption.PREEMPT_ENV, "0")
    preemption.note_server_parking(False)
    yield
    preemption.note_server_parking(False)


def test_a_confirmed_parking_build_is_told_not_to_park():
    env: dict = {}
    args = ["llama-server", "--kv-unified"]
    assert llama_mod._stand_down_child_parking(env, args, server_supports = True) == []
    assert env == {"LLAMA_ARG_PREEMPT_RAM": "0"}
    assert args == ["llama-server", "--kv-unified"]
    assert llama_mod._preempt_ram_disabled_in(args, env = env)


def test_a_build_without_the_flag_is_still_left_alone():
    env: dict = {}
    assert llama_mod._stand_down_child_parking(env, ["llama-server"], server_supports = False) == []
    assert env == {}
    env = {}
    assert llama_mod._stand_down_child_parking(env, ["llama-server"]) == []
    assert env == {}
