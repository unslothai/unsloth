# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The token recount must price the same system prompt the completion would send.

``createOpenAIStreamAdapter`` appends a Canvas instruction to the outbound system
prompt whenever the Canvas pill is on -- the render_html wording when the model can
call the tool, the fenced-HTML fallback otherwise. Neither is a tool schema, so the
server cannot add it back from the flags ``buildLocalTokenCountExtras`` sends: a count
that skips it reports fewer tokens than the next completion actually spends.

``studio/frontend`` carries no JS test runner, so the builder and both instruction
constants are sliced verbatim out of chat-adapter.ts and run under ``node``.
"""

from __future__ import annotations

import math
import textwrap

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

ADAPTER = source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")

TEMP = WORKDIR / "temp" / "token_count_prompt_parity"

SOURCES = (ADAPTER,)


def _canvas_constants() -> str:
    return slice_between(
        read(ADAPTER),
        "export const CANVAS_TOOL_INSTRUCTION =",
        "/**\n * The OpenAI-form messages",
    )


def _outbound_builder() -> str:
    return slice_between(
        read(ADAPTER),
        "export async function buildOutboundMessagesForTokenCount(",
        "/**\n * The tool flags a completion would send",
    )


def _instruction(name: str) -> str:
    """The JS string literal assigned to ``name``, as Python text."""
    text = read(ADAPTER)
    start = text.index(f"export const {name} =")
    opening = text.index('"', start)
    closing = text.index('";', opening + 1)
    return text[opening + 1 : closing]


HARNESS = """
// @ts-nocheck
// Fixtures the sliced builder reads through. Everything below the PRELUDE marker is
// copied verbatim out of studio/frontend/src/features/chat/api/chat-adapter.ts.
const state: any = {
  params: { systemPrompt: "", systemVariables: "" },
  artifactsEnabled: false,
  supportsTools: false,
};

const useChatRuntimeStore: any = { getState: () => state };

export function seed(patch: any): void {
  Object.assign(state, patch);
}

function isAnthropicRefusalMessage(_message: any): boolean {
  return false;
}

function toOpenAIMessages(message: any): any[] {
  return [{ role: message.role, content: message.text }];
}

function resolveSystemPromptVariables(prompt: string, _variables: string): string {
  return prompt;
}

async function resolveProjectInstructions(_threadId: any): Promise<string> {
  return "";
}

// A stand-in for the server-side tokenizer: proportional to the rendered prompt, so a
// dropped instruction shows up as a smaller total rather than a missing symbol.
export function estimateTokens(messages: any[]): number {
  return messages.reduce(
    (total: number, m: any) => total + Math.ceil(String(m.content ?? "").length / 4) + 4,
    0,
  );
}

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""


def _estimate(contents: list[str]) -> int:
    return sum(math.ceil(len(content) / 4) + 4 for content in contents)


def _harness_source() -> str:
    return HARNESS + _canvas_constants() + _outbound_builder()


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script)


def _count_script(seed_patch: str) -> str:
    return textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{
          buildOutboundMessagesForTokenCount,
          estimateTokens,
          seed,
        }} from "./harness.ts";
        seed({seed_patch});
        const outbound = await buildOutboundMessagesForTokenCount(
          [{{ role: "user", text: "draw me a bar chart" }}],
          "thread-a",
        );
        console.log(JSON.stringify({{
          system: outbound[0]?.role === "system" ? outbound[0].content : null,
          inputTokens: estimateTokens(outbound),
        }}));
        """
    )


USER_TURN = "draw me a bar chart"
SYSTEM_PROMPT = "You are a helpful assistant."


def test_the_recount_prices_the_canvas_tool_instruction():
    """Canvas on, tool-capable model: the request carries the render_html wording after
    the user's system prompt, so the count has to carry it too."""
    instruction = _instruction("CANVAS_TOOL_INSTRUCTION")
    out = _run(
        _count_script(
            "{ artifactsEnabled: true, supportsTools: true, params: { systemPrompt: "
            + f'"{SYSTEM_PROMPT}"'
            + ', systemVariables: "" } }'
        )
    )
    expected_system = f"{SYSTEM_PROMPT}\n\n{instruction}"
    assert out.get("system") == expected_system
    assert out.get("inputTokens") == _estimate([expected_system, USER_TURN]), (
        "the recount must price the Canvas instruction the completion sends, or the "
        "bar reports a chat fits when the next completion is larger"
    )


def test_the_recount_prices_the_canvas_fallback_without_tool_support():
    """No tool support means no render_html, and the request sends the fenced-HTML
    fallback instead. It is shorter, but it is still in the prompt."""
    instruction = _instruction("CANVAS_FALLBACK_INSTRUCTION")
    out = _run(_count_script("{ artifactsEnabled: true, supportsTools: false }"))
    assert out.get("system") == instruction, (
        "with no system prompt to append to, the fallback instruction becomes the "
        "leading system turn, exactly as addSystemInstruction does"
    )
    assert out.get("inputTokens") == _estimate([instruction, USER_TURN])


def test_canvas_off_adds_nothing():
    """The pill is off by default; the count must not invent a prompt the request has
    no reason to send."""
    out = _run(_count_script("{ artifactsEnabled: false, supportsTools: true }"))
    assert out.get("system") is None
    assert out.get("inputTokens") == _estimate([USER_TURN])


def test_the_request_path_sends_the_same_constants():
    """The adapter and the recount must read one source of truth, or the count drifts
    the moment either wording is edited."""
    src = read(ADAPTER)
    assert "? CANVAS_TOOL_INSTRUCTION\n          : CANVAS_FALLBACK_INSTRUCTION" in src, (
        "createOpenAIStreamAdapter must build artifactInstruction from the shared "
        "constants the token recount prices"
    )
    for name in ("CANVAS_TOOL_INSTRUCTION", "CANVAS_FALLBACK_INSTRUCTION"):
        assert src.count(f"export const {name} =") == 1
        assert src.count(name) == 3, f"{name} must have exactly one declaration and two uses"
