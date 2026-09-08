# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every local chat surface that takes an admission lease must also arm preemption."""

import ast
import pathlib

ROUTES = pathlib.Path(__file__).resolve().parent.parent / "routes" / "inference.py"
PREEMPTION = (
    pathlib.Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_preemption.py"
)

CHAT_HANDLER = "produce_openai_chat_completions"


def _handler() -> ast.AST:
    tree = ast.parse(ROUTES.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == CHAT_HANDLER:
            return node
    raise AssertionError(f"{CHAT_HANDLER} not found in {ROUTES}")


def _calls(node: ast.AST, name: str) -> int:
    return sum(
        1
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == name
    )


class TestPreemptionIsArmedWhereALeaseIsTaken:
    def test_all_three_local_chat_surfaces_arm(self):
        handler = _handler()
        armed = _calls(handler, "_openai_llama_preemption_arm")
        assert armed >= 3, (
            f"only {armed} arm site(s) in {CHAT_HANDLER}. The tool loop, plain streaming "
            "and non-streaming surfaces must each arm; a surface that takes a lease and "
            "does not arm decodes with no preemption, and the absence is invisible to "
            "every behavioural test."
        )

    def test_arming_is_always_paired_with_disarming(self):
        handler = _handler()
        armed = _calls(handler, "_openai_llama_preemption_arm")
        disarmed = _calls(handler, "_openai_llama_preemption_disarm")
        assert disarmed >= armed, (
            f"{armed} arm site(s) but only {disarmed} disarm site(s). A registration that "
            "is never dropped grows the ledger forever, and once it reads full the next "
            "chat waits for room that cannot arrive."
        )

    def test_every_arming_surface_registers_a_residency_probe(self):
        """Without one, `controller.refresh_residency()` is a no-op."""
        handler = _handler()
        probes = _calls(handler, "get_preemption_controller")
        armed = _calls(handler, "_openai_llama_preemption_arm")
        source = ROUTES.read_text(encoding = "utf-8")
        assert source.count("set_residency_probe") >= armed, (
            f"{armed} arm site(s) but only {source.count('set_residency_probe')} probe "
            "registration(s); a surface that arms without one can livelock its own resume"
        )
        assert probes >= armed

    def test_anthropic_arms_too(self):
        """It drives `generate_chat_completion`, so it can be paused and resumed."""
        source = ROUTES.read_text(encoding = "utf-8")
        assert "_anthropic_preempt_signal = PreemptSignal()" in source
        assert "preempt_policy = _anthropic_preempt_policy" in source
        assert "on_tokens = _anthropic_observe_tokens" in source
        assert "gen_id = message_id" in source

    def test_the_unpausable_surfaces_are_counted_rather_than_ignored(self):
        """A holder the controller cannot see is worse than one it cannot pause."""
        source = ROUTES.read_text(encoding = "utf-8")
        assert "_openai_llama_count_raw_holder" in source
        assert source.count("_openai_llama_count_raw_holder") >= 2, "defined but never called"
        preemption = PREEMPTION.read_text(encoding = "utf-8")
        assert "STREAMING_RAW" in preemption
        # In _HOLDS_KV so it counts, out of _PREEMPTABLE so it is never chosen.
        #
        # Read with the ast rather than by splitting on `frozenset({`: the formatter
        # rewrites that to `frozenset(\n    {`, and a string-split version of this test
        # would then find nothing and pass while asserting about an empty string.
        module = ast.parse(preemption)
        sets = {}
        for node in ast.walk(module):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ("_HOLDS_KV", "_PREEMPTABLE"):
                    sets[target.id] = {
                        n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)
                    }
        assert sets.keys() == {"_HOLDS_KV", "_PREEMPTABLE"}, "constants renamed or moved"
        assert "STREAMING_RAW" in sets["_HOLDS_KV"]
        assert "STREAMING_RAW" not in sets["_PREEMPTABLE"]

    def test_the_signal_reaches_the_generator(self):
        """Arming alone pauses nobody: the stream has to be given the event to notice."""
        source = ROUTES.read_text(encoding = "utf-8")
        assert "preempt_event = _plain_preempt_signal" in source
        assert "preempt_policy = _plain_preempt_policy" in source


class TestEveryLeaseIsAccountedFor:
    """The closing invariant, and the one that would have caught the original gap."""

    def test_the_numbers_add_up(self):
        source = ROUTES.read_text(encoding = "utf-8")
        tree = ast.parse(source)

        def count(name):
            return sum(
                1
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
            )

        reserves = count("_openai_llama_admission_reserve")
        arms = count("_openai_llama_preemption_arm")
        counted = count("_openai_llama_count_raw_holder")
        # The reserve call inside the helper's own definition is not a surface.
        assert reserves >= 7
        assert arms + counted >= reserves, (
            f"{reserves} admission reserve(s), but only {arms} armed and {counted} counted. "
            "A surface that takes a lease and does neither is invisible to the preemptor "
            "while occupying its cache."
        )

    def test_counting_is_paired_with_dropping(self):
        """A counted holder that is never dropped is worse than one never counted."""
        source = ROUTES.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        counted = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_openai_llama_count_raw_holder"
        )
        disarms = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_openai_llama_preemption_disarm"
        )
        assert disarms >= counted
