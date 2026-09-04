# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every local chat surface that takes an admission lease must also arm preemption.

WHY THIS TEST EXISTS

`_openai_llama_preemption_arm` had exactly ONE call site while
`_openai_llama_admission_reserve` had seven. The one was inside the tool-loop branch, so an
ordinary chat with no tools took a lease and then decoded with no preemption at all.
Measured on the plain streaming surface at a confirmed `-c 16384`, four chats, 3000-token
prompts:

    armed 0   paused 0   gave-up 0   ctx_exceeded 0   sub_batch 0

Not even `not-armed`, which is logged whenever arming is attempted and declined. Nothing was
engaged. Every earlier result showing preemption working came from tool runs, because those
were the only ones that armed it.

Nothing in the behavioural suite could catch that: each surface behaved correctly on its own
terms, and the missing feature is only visible as an absence. So this test is structural, and
it asserts the two halves that have each been forgotten once.

  1. Arming. Three local GGUF chat surfaces -- tool loop, plain streaming, non-streaming.
  2. Disarming. Arming REGISTERS a charge with the controller and every exit has to
     unregister it. The tool path shipped without this and the ledger only ever grew: once
     it believed the cache was full the next chat waited for room that could not arrive,
     observed as one chat of four hanging 2400s while llama-server sat idle with every slot
     released. There are more disarm sites than arm sites because a surface has several
     exits, so this asserts a floor, not equality.

Every surface that takes a lease is now accounted for, in one of two ways.

ARMED, because it drives `generate_chat_completion` and so has a Studio-side generator
holding the conversation to resume from: the GGUF tool loop, plain streaming, non-streaming,
and Anthropic.

COUNTED BUT NEVER CHOSEN, because it streams upstream bytes straight to the client and has
no conversation to resume: the Responses surface and the two llama-server passthroughs.
Aborting one of those is a cancel, not a pause. They are registered as `STREAMING_RAW` so
their cells appear in the ledger, because a holder the controller cannot see makes the
watermark fire late by exactly its size and picks a victim from among the chats it CAN see
to make room a passthrough is quietly holding.

`/v1/completions` takes no lease at all and is out of scope for both.
"""

import ast
import pathlib

ROUTES = pathlib.Path(__file__).resolve().parent.parent / "routes" / "inference.py"
PREEMPTION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "core" / "inference" / "llama_preemption.py"
)

CHAT_HANDLER = "produce_openai_chat_completions"


def _handler() -> ast.AST:
    tree = ast.parse(ROUTES.read_text())
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
        """Without one, `controller.refresh_residency()` is a no-op.

        That call is the only thing in the resume wait loop that re-reads the cache and
        reclaims an idle slot's cells. A pause hands back the lease and marks the
        participant PAUSED; it does not touch llama-server, which keeps the idle slot's
        prompt cache for prefix reuse. So the ledger reads free while the cells are held,
        a resume is granted room that does not physically exist, and its prefill does not
        fit. Nothing decodes, so the token sweep never runs, so nothing is reclaimed, so
        nothing can decode.

        Observed with the probe registered on the tool branch alone: 90 seconds with not
        one log line of any kind, four chats waiting, broken only when two hit the give-up
        timeout and freed their cells on the way out. The run still completed 4 of 4,
        which is exactly what made it easy to miss.
        """
        handler = _handler()
        probes = _calls(handler, "get_preemption_controller")
        armed = _calls(handler, "_openai_llama_preemption_arm")
        source = ROUTES.read_text()
        assert source.count("set_residency_probe") >= armed, (
            f"{armed} arm site(s) but only {source.count('set_residency_probe')} probe "
            "registration(s); a surface that arms without one can livelock its own resume"
        )
        assert probes >= armed

    def test_anthropic_arms_too(self):
        """It drives `generate_chat_completion`, so it can be paused and resumed.

        It was left out of the first pass with the surfaces that cannot be. Being armed is
        the difference between a chat that waits its turn and one that either serialises
        everybody or dies with them.
        """
        source = ROUTES.read_text()
        assert "_anthropic_preempt_signal = PreemptSignal()" in source
        assert "preempt_policy = _anthropic_preempt_policy" in source
        assert "on_tokens = _anthropic_observe_tokens" in source
        assert "gen_id = message_id" in source

    def test_the_unpausable_surfaces_are_counted_rather_than_ignored(self):
        """A holder the controller cannot see is worse than one it cannot pause.

        The raw passthrough fills real cells. Uncounted, the watermark fires late by
        exactly its size and then evicts a chat to make room the passthrough is holding.
        Counted and unpreemptable is the truth about it.
        """
        source = ROUTES.read_text()
        assert "_openai_llama_count_raw_holder" in source
        assert source.count("_openai_llama_count_raw_holder") >= 2, "defined but never called"
        preemption = PREEMPTION.read_text()
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
        """Arming alone pauses nobody: the stream has to be given the event to notice.

        `_openai_llama_preemption_arm` returning a policy and the controller setting a
        signal are both no-ops unless `generate_chat_completion` is passed that same
        signal. Asserted by name because the failure mode is a surface that arms and then
        never checks, which looks identical in the logs to one that is simply never chosen
        as a victim.
        """
        source = ROUTES.read_text()
        assert "preempt_event = _plain_preempt_signal" in source
        assert "preempt_policy = _plain_preempt_policy" in source


class TestEveryLeaseIsAccountedFor:
    """The closing invariant, and the one that would have caught the original gap.

    `_openai_llama_admission_reserve` had seven call sites and
    `_openai_llama_preemption_arm` had one. Nothing said those numbers had to relate, so
    six surfaces charged the cache and were invisible to the thing that manages it.

    Every reserve must now be paired with one of two things, and the choice is a real
    design decision rather than a formality:

      * ARM, for a surface with a Studio-side generator holding the conversation, which can
        therefore be paused and resumed.
      * COUNT, for one that streams upstream bytes straight out and cannot. Aborting it is
        a cancel; but leaving it out of the ledger makes the watermark fire late by exactly
        its size, and then evicts a chat to free room the uncounted holder is using.

    A new surface that does neither fails here, which is the point.
    """

    def test_the_numbers_add_up(self):
        source = ROUTES.read_text()
        tree = ast.parse(source)

        def count(name):
            return sum(
                1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == name
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
        """A counted holder that is never dropped is worse than one never counted.

        An over-counted ledger only ever grows, and once it reads full the next chat waits
        for room that cannot arrive.
        """
        source = ROUTES.read_text()
        tree = ast.parse(source)
        counted = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_openai_llama_count_raw_holder"
        )
        disarms = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_openai_llama_preemption_disarm"
        )
        assert disarms >= counted
