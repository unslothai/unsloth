# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request that says `parallel_tool_calls: false` gets its provider round run one call at a time.

The GGUF loop already honoured the field; the provider loop read only the process switch, so a
provider that returned several calls had them executed together against the caller's opt-out.
"""

from __future__ import annotations

import inspect

from core.inference import openai_codex_tool_loop
from core.inference.studio_tool_loop import ToolLoopPolicy
import routes.inference as inference
import routes.managed_engine_chat as managed_engine_chat

from .test_studio_tool_loop import _events, _run
from .test_tool_calls_within_one_turn_overlap import _two_calls, rendezvous  # noqa: F401


class TestTheOptOutReachesTheLoop:
    def test_false_runs_the_round_in_order(self, rendezvous):  # noqa: F811
        ends = _events(_run(_two_calls(), parallel_tool_calls = False), "tool_end")
        assert len(ends) == 2
        assert all("ALONE" in (end.get("result") or "") for end in ends), (
            "the caller turned parallel calls off, but the round still overlapped: "
            f"{[end.get('result') for end in ends]}"
        )
        assert [end.get("tool_call_id") for end in ends] == ["call_a", "call_b"]

    def test_unset_and_true_still_overlap(self, rendezvous):  # noqa: F811
        for value in (None, True):
            ends = _events(_run(_two_calls(), parallel_tool_calls = value), "tool_end")
            assert all("TOGETHER" in (end.get("result") or "") for end in ends), value

    def test_the_policy_defaults_to_the_old_behaviour(self):
        policy = ToolLoopPolicy(
            tools = [],
            max_calls = 1,
            timeout = 1,
            permission_mode = "off",
            confirm_calls = False,
            bypass_permissions = False,
            rag_scope = None,
        )
        assert policy.parallel_tool_calls is None


class TestEveryProviderRouteCarriesIt:
    def test_the_routes_and_the_codex_wrapper_pass_the_request_field(self):
        external = " ".join(inspect.getsource(inference).split())
        assert "parallel_tool_calls = payload.parallel_tool_calls," in external
        managed = " ".join(inspect.getsource(managed_engine_chat).split())
        assert 'parallel_tool_calls = getattr(payload, "parallel_tool_calls", None),' in managed
        codex = " ".join(inspect.getsource(openai_codex_tool_loop).split())
        assert "parallel_tool_calls = policy.parallel_tool_calls," in codex
        assert (
            openai_codex_tool_loop.CodexToolPolicy(
                tools = [],
                max_calls = 1,
                timeout = 1,
                permission_mode = "auto",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ).parallel_tool_calls
            is None
        )
