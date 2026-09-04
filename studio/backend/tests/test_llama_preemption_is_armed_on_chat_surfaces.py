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

Deliberately NOT covered: Responses, Anthropic, and the two passthrough surfaces. They take
leases too and the plan defers them; when one is armed, raise the number here.
"""

import ast
import pathlib

ROUTES = pathlib.Path(__file__).resolve().parent.parent / "routes" / "inference.py"

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
