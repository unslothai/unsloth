# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""create_skill writes the manifest a sibling read_skill (or any other call) can read in the
same round. Run in turn, as before calls could overlap, the read finds the skill; overlapped,
the read can start first and report it missing. Both tool loops share this guard."""

from __future__ import annotations

import json

from core.inference import studio_tool_loop as loop_mod


def _call(tool, **arguments):
    return {"function": {"name": tool, "arguments": json.dumps(arguments)}}


def test_creating_a_skill_beside_a_read_of_it_runs_in_order():
    create = _call("create_skill", name = "round-race", description = "d", instructions = "i")
    read = _call("read_skill", name = "round-race")
    assert loop_mod.round_needs_order([create, read])
    assert loop_mod.round_needs_order([read, create])
    assert loop_mod.round_needs_order([create, _call("web_search", query = "q")])


def test_reads_alone_still_overlap():
    assert not loop_mod.round_needs_order(
        [_call("read_skill", name = "a"), _call("read_skill", name = "b")]
    )
    assert not loop_mod.round_needs_order(
        [_call("create_skill", name = "solo", description = "d", instructions = "i")]
    )
