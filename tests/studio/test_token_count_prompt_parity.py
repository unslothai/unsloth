# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The token recount must price the same system prompt the completion would send.

``createOpenAIStreamAdapter`` appends a Canvas instruction to the outbound system
prompt whenever the Canvas pill is on -- the render_html wording when the model can
call the tool, the fenced-HTML fallback otherwise. Neither is a tool schema, so the
server cannot add it back from the flags ``buildLocalTokenCountExtras`` sends: a count
that skips it reports fewer tokens than the next completion actually spends.

The same applies to the reasoning settings: llama-server layers a request's
``chat_template_kwargs`` over the load-time ``--chat-template-kwargs``, so a count that
sends none of them renders the template in whatever mode the model was LOADED in.

``studio/frontend`` carries no JS test runner, so the builders, both instruction
constants and the shared effort clamp are sliced verbatim out of the studio sources and
run under ``node``.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

ADAPTER = source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
CAPABILITIES = source_path("studio/frontend/src/features/chat/provider-capabilities.ts")

TEMP = WORKDIR / "temp" / "token_count_prompt_parity"

SOURCES = (ADAPTER, CAPABILITIES)


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
        "/**\n * The reasoning fields a completion would send",
    )


def _reasoning_builder() -> str:
    """buildLocalTokenCountReasoning plus the clamp it shares with the request build."""
    clamp = slice_between(
        read(CAPABILITIES),
        "export function clampReasoningEffortToLevels(",
        "\n/**",
    )
    builder = slice_between(
        read(ADAPTER),
        "export function buildLocalTokenCountReasoning(",
        "/**\n * The tool flags a completion would send",
    )
    return clamp + "\n" + builder


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
  supportsReasoning: false,
  reasoningStyle: "enable_thinking",
  reasoningEnabled: true,
  reasoningEffort: "high",
  reasoningEffortLevels: ["low", "medium", "high"],
  supportsPreserveThinking: false,
  preserveThinking: false,
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
    return HARNESS + _canvas_constants() + _outbound_builder() + _reasoning_builder()


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
WITH_PROMPT = (
    '{ artifactsEnabled: true, supportsTools: true, params: { systemPrompt: "'
    + SYSTEM_PROMPT
    + '", systemVariables: "" } }'
)


@pytest.mark.parametrize(
    ("seed_patch", "constant", "prompt"),
    [
        # Canvas on against a tool-capable model: the request carries the render_html
        # wording appended to the user's system prompt, so the count has to carry it too.
        pytest.param(WITH_PROMPT, "CANVAS_TOOL_INSTRUCTION", SYSTEM_PROMPT, id = "render_html"),
        # No tool support means no render_html and the fenced-HTML fallback instead. With
        # no system prompt to append to, it becomes the leading system turn -- exactly what
        # the adapter's addSystemInstruction does.
        pytest.param(
            "{ artifactsEnabled: true, supportsTools: false }",
            "CANVAS_FALLBACK_INSTRUCTION",
            "",
            id = "fenced_html_fallback",
        ),
        # The pill is off by default; the count must not invent a prompt the request has
        # no reason to send.
        pytest.param("{ artifactsEnabled: false, supportsTools: true }", None, "", id = "canvas_off"),
    ],
)
def test_the_recount_prices_the_canvas_instruction(seed_patch, constant, prompt):
    """#7450's bar answers "does this chat still fit", so it has to price every part of
    the prompt the next completion builds -- including the Canvas instruction, which is
    not a tool schema and so cannot be added back server-side from the tool flags."""
    instruction = _instruction(constant) if constant else ""
    expected_system = "\n\n".join(part for part in (prompt, instruction) if part)
    out = _run(_count_script(seed_patch))
    assert out.get("system") == (expected_system or None)
    assert out.get("inputTokens") == _estimate(
        ([expected_system] if expected_system else []) + [USER_TURN]
    ), "the recount must price the Canvas instruction the completion sends"


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


@pytest.mark.parametrize(
    ("seed_patch", "expected"),
    [
        # No reasoning support: send nothing, and llama-server keeps its own defaults.
        pytest.param("{ supportsReasoning: false }", {}, id = "no_reasoning_support"),
        # Qwen3-style gate turned off. The completion sends this, the template prefills an
        # empty thinking block for it, and a count without it prices the loaded default.
        pytest.param(
            '{ supportsReasoning: true, reasoningStyle: "enable_thinking", reasoningEnabled: false }',
            {"enable_thinking": False},
            id = "thinking_turned_off",
        ),
        # gpt-oss-style: the effort level is rendered into the prompt.
        pytest.param(
            '{ supportsReasoning: true, reasoningStyle: "reasoning_effort", reasoningEnabled: true,'
            ' reasoningEffort: "low" }',
            {"reasoning_effort": "low"},
            id = "effort_level",
        ),
        # GLM-style: the on/off gate plus a level, clamped to the levels this template
        # offers, exactly as the request build clamps it. "high" is not one of them here.
        pytest.param(
            '{ supportsReasoning: true, reasoningStyle: "enable_thinking_effort",'
            ' reasoningEnabled: true, reasoningEffort: "high", reasoningEffortLevels: ["max"] }',
            {"enable_thinking": True, "reasoning_effort": "max"},
            id = "effort_clamped_to_the_template_levels",
        ),
        # Independent of the gate: decides whether past <think> blocks stay in the prompt.
        pytest.param(
            "{ supportsPreserveThinking: true, preserveThinking: true }",
            {"preserve_thinking": True},
            id = "preserve_thinking",
        ),
    ],
)
def test_the_recount_sends_the_reasoning_mode_the_completion_would(seed_patch, expected):
    """llama-server layers a request's chat_template_kwargs over the load-time
    --chat-template-kwargs, so a count that omits them renders the template in whatever
    mode the model was LOADED in and reports a prompt size the next completion will not
    match."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ buildLocalTokenCountReasoning, seed }} from "./harness.ts";
            seed({seed_patch});
            console.log(JSON.stringify({{ reasoning: buildLocalTokenCountReasoning() }}));
            """
        )
    )
    assert out.get("reasoning") == expected


def test_the_request_path_clamps_the_effort_the_same_way():
    """Both payloads have to clamp against the loaded template's levels, or the count
    sends a level the backend drops and prices the template default instead."""
    src = " ".join(read(ADAPTER).split())
    assert (
        src.count("clampReasoningEffortToLevels( reasoningEffort, reasoningEffortLevels, )") == 2
    ), "the request build and the token recount must clamp from the same store fields"
