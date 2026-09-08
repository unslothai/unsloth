# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Surfaces that took an optimistic lease and then ran outside the mechanism."""

from __future__ import annotations

import ast
import pathlib


ROUTES = pathlib.Path(__file__).resolve().parent.parent / "routes" / "inference.py"
LLAMA_CPP = pathlib.Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"


def _routes_source() -> str:
    return ROUTES.read_text(encoding = "utf-8")


def _function(source: str, name: str) -> ast.AST:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone; this test needs rewriting rather than deleting")


def _calls(node: ast.AST, name: str) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (getattr(child.func, "id", None) == name or getattr(child.func, "attr", None) == name)
    )


class TestTheNonStreamingToolBranch:
    """Streaming got both halves; non-streaming got neither."""

    def test_it_re_arms_after_a_queued_admission(self):
        source = _routes_source()
        assert (
            source.count("if not _gguf_preempt_policy_hold.bound:") == 2
        ), "both branches wait for a lease, so both have to arm once they have one"

    def test_it_disarms_on_every_exit(self):
        """Arming registers a charge; the exit that drops it is not optional."""
        handler = _function(_routes_source(), "produce_openai_chat_completions")
        armed = _calls(handler, "_openai_llama_preemption_arm")
        disarmed = _calls(handler, "_openai_llama_preemption_disarm")
        assert disarmed >= armed, f"{armed} arm site(s), {disarmed} disarm site(s)"
        # Two non-streaming finallys close a GGUF chat -- the plain one and the tool one
        # -- and both have to drop the charge before returning the tokens. Only the plain
        # one did.
        assert (
            _routes_source().count(
                """                _openai_llama_preemption_disarm(
                    llama_backend = llama_backend,
                    gen_id = completion_id,
                )
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)"""
            )
            == 2
        ), "the non-streaming tool branch drops its charge before returning the tokens"


class TestDisarmFollowsTheStreamTeardown:
    """Unregistering is a logical release and may not precede the physical one."""

    def test_the_streaming_tool_finally_disarms_last(self):
        source = _routes_source()
        close = source.index("Error closing GGUF tool stream generator during cleanup")
        disarm = source.index(
            "_openai_llama_preemption_disarm(\n                            llama_backend"
        )
        assert close < disarm, (
            "the disarm runs before the drain and the generator close, so the charge is "
            "given away while the stream can still be decoding"
        )


class TestTheAnthropicSurfaces:
    def test_the_non_streaming_branch_arms_and_disarms(self):
        source = _routes_source()
        assert (
            source.count("_arm_anthropic(reservation, raw = raw)") == 2
        ), "streaming and non-streaming both take a lease, so both arm"
        assert (
            source.count("gen_id = message_id,") >= 3
        ), "and both drop the charge again on the way out"

    def test_the_server_tool_generator_is_handed_the_policy(self):
        """`_arm_anthropic` registers this request as an ordinary preemptible DECODING participant."""
        source = _routes_source()
        run_tool_gen = source[source.index("def _run_tool_gen():") :]
        run_tool_gen = run_tool_gen[: run_tool_gen.index("if payload.stream:")]
        for kwarg in (
            "preempt_event = _anthropic_preempt_signal,",
            "preempt_policy = _anthropic_preempt_policy,",
            "on_tokens = _anthropic_observe_tokens,",
        ):
            assert kwarg in run_tool_gen, f"the Anthropic tool loop is missing {kwarg}"

    def test_the_raw_passthrough_is_counted_and_never_chosen(self):
        """It streams llama-server's bytes straight through, so there is no Studio generator
        holding the conversation and nothing to resume from.
        """
        source = _routes_source()
        assert (
            source.count("raw = True,") == 2
        ), "both client-tool passthrough branches are raw holders"
        arm = source[source.index("def _arm_anthropic(") :]
        arm = arm[: arm.index("async def _admitted_anthropic_stream")]
        assert "_openai_llama_count_raw_holder(" in arm
        assert "if raw:" in arm


class TestTheRespawnRetryKeepsItsControls:
    """`_respawn_if_dead()` re-opens the same generation against a replacement server."""

    def test_the_connect_error_retry_forwards_the_preemption_arguments(self):
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        retry = source[
            source.index(
                "yield from self.generate_chat_completion(\n                    retry_messages,"
            ) :
        ]
        retry = retry[: retry.index("_allow_respawn_retry = False")]
        for kwarg in (
            "admission_output_allowance = admission_output_allowance,",
            '{"preempt_event": preempt_event}',
            "preempt_policy = preempt_policy,",
            "on_tokens = on_tokens,",
        ):
            assert kwarg in retry, f"the respawn retry drops {kwarg}"


class TestTheRoundBoundaryPublishesTheNewCharge:
    """The re-cost is what grows the lease; reading it first published the old figure."""

    def test_the_recost_runs_before_note_tokens(self):
        source = _routes_source()
        body = source[source.index("def _gguf_recost(conversation) -> None:") :]
        body = body[: body.index("# Active tool names gating the bare-rehearsal strip")]
        recost = body.index("_openai_llama_admission_recost(")
        publish = body.index(".note_tokens(")
        assert recost < publish, (
            "the sweep publishes `lease.tokens` and must read it after the re-cost that "
            "grew it, not before"
        )
        assert body.index("_gguf_observe_tokens(0)") > recost


class TestTheParallelToolClosureBindsItsOwnCall:
    """In an overlapped round the tool's worker runs while the loop has moved on."""

    def test_every_per_call_value_is_a_default_argument(self):
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        head = source.index("def _invoke_tool(\n                            _output_callback,")
        body = source[head : source.index("_tool_stream = stream_tool_execution(", head)]
        for bound in (
            "_decision = decision,",
            "_call_position = _call_index,",
            "_compact_flag = _compact_after_execution,",
            "_compacted_tokens = _compacted_turn_tokens,",
        ):
            assert bound in body, f"{bound} is not bound at closure definition"
        # And nothing in the body still reads the loop variables it shadows.
        for leaked in ("decision.tool_name", "decision.tool_call_id", "[_call_index + 1 :]"):
            assert f" {leaked}" not in body.replace(
                f"_{leaked}", ""
            ), f"the closure still reads the outer {leaked}"


class TestTheResumeClearsItsSignalBeforeItIsSelectable:
    """`on_resumed` is what makes this participant a candidate again."""

    def test_the_tool_loop_clears_before_calling_on_resumed(self):
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        block = source[
            source.index("_resumed = yield from _await_resume(preempt_policy, cancel_event)") :
        ]
        block = block[: block.index("if not _resumed:")]
        assert block.index("preempt_event.clear()") < block.index(
            "preempt_policy.on_resumed()"
        ), "the clear must not run after the participant becomes selectable again"
