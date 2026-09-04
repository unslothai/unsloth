# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The token recount must price the same system prompt the completion would send.

``createOpenAIStreamAdapter`` appends a Canvas instruction to the outbound system prompt whenever
the Canvas pill is on -- render_html wording when the model can call the tool, the fenced-HTML
fallback otherwise. Neither is a tool schema, so the server cannot add it back from the flags
``buildLocalTokenCountExtras`` sends. Same for reasoning: llama-server layers a request's
``chat_template_kwargs`` over the load-time ``--chat-template-kwargs``, so a count sending none
renders the template in whatever mode the model was LOADED in. Either way the count reads low.

The builders, both instruction constants and the shared effort clamp are sliced verbatim out of
the studio sources and run under ``node`` (see ``_node_harness``).
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
MODEL_SIZE = source_path("studio/frontend/src/lib/model-size.ts")

TEMP = WORKDIR / "temp" / "token_count_prompt_parity"

SOURCES = (ADAPTER, CAPABILITIES, MODEL_SIZE)


def _canvas_constants() -> str:
    return slice_between(
        read(ADAPTER),
        "export const CANVAS_TOOL_INSTRUCTION =",
        "export async function buildLocalTokenCountHistory(",
    )


def _prune_helpers() -> str:
    """isAbandonedAssistantTurn + pruneOutboundHistory, which the outbound builder calls."""
    return slice_between(
        read(ADAPTER),
        "/** Payload the turn carries in its own parts",
        "function extractImageBase64(",
    )


def _outbound_builder() -> str:
    return slice_between(
        read(ADAPTER),
        "export async function buildLocalTokenCountHistory(",
        "/**\n * The reasoning fields a completion would send",
    )


def _extras_builder() -> str:
    """buildLocalTokenCountExtras, the tool flags the count sends, and the Auto-inject
    resolution it shares with the request build.

    Joined on blank lines, not concatenated: a slice that starts on the previous slice's
    closing brace is not a declaration _harness_bindings can see, so resolve_dependencies
    pulls its own copy and node refuses the duplicate.
    """
    parts = [
        read(MODEL_SIZE).split("\n", 2)[2],
        slice_between(
            read(ADAPTER),
            "const AUTOINJECT_AUTO_MAX_SIZE_B =",
            "\n\ntype ThreadRecordReader",
        ),
        slice_between(
            read(ADAPTER),
            "function resolveAutoInject(",
            "\n\n/** Server-side usage data",
        ),
        slice_between(
            read(ADAPTER),
            "export async function buildLocalTokenCountExtras(",
            "\n\nasync function resolveUseAdapter(",
        ),
    ]
    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


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

function sanitizeAssistantReplayText(text: string): string {
  return text;
}

function readIncompleteInfo(_metadata: any): any {
  return null;
}

function collectImageParts(_message: any): any[] {
  return [];
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

// The extras builder resolves a project from the thread; no project is configured here, so
// the RAG scope depends on the Docs pill and the thread id alone.
async function resolveProjectId(_threadId: any): Promise<string | null> {
  return null;
}

async function projectHasSources(_projectId: any): Promise<boolean> {
  return false;
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
    return (
        HARNESS
        + _canvas_constants()
        + _prune_helpers()
        + _outbound_builder()
        + _reasoning_builder()
        + _extras_builder()
    )


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script, sources = SOURCES)


def _count_script(seed_patch: str) -> str:
    return textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{
          buildLocalTokenCountHistory,
          estimateTokens,
          seed,
        }} from "./harness.ts";
        seed({seed_patch});
        const {{ messages: outbound }} = await buildLocalTokenCountHistory(
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
        # Canvas on, tool-capable: the request appends the render_html wording to the prompt.
        pytest.param(WITH_PROMPT, "CANVAS_TOOL_INSTRUCTION", SYSTEM_PROMPT, id = "render_html"),
        # No tool support: the fenced-HTML fallback, and with no prompt to append to it leads.
        pytest.param(
            "{ artifactsEnabled: true, supportsTools: false }",
            "CANVAS_FALLBACK_INSTRUCTION",
            "",
            id = "fenced_html_fallback",
        ),
        # The pill is off by default; the count must not invent a prompt.
        pytest.param("{ artifactsEnabled: false, supportsTools: true }", None, "", id = "canvas_off"),
    ],
)
def test_the_recount_prices_the_canvas_instruction(seed_patch, constant, prompt):
    """#7450's bar answers "does this chat still fit", so it must price every part of the next
    prompt -- including the Canvas instruction, which no tool flag can add back server-side."""
    instruction = _instruction(constant) if constant else ""
    expected_system = "\n\n".join(part for part in (prompt, instruction) if part)
    out = _run(_count_script(seed_patch))
    assert out.get("system") == (expected_system or None)
    assert out.get("inputTokens") == _estimate(
        ([expected_system] if expected_system else []) + [USER_TURN]
    ), "the recount must price the Canvas instruction the completion sends"


def test_the_request_path_sends_the_same_constants():
    """The adapter and the recount must read one source of truth, or the count drifts on edit."""
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
        # Qwen3-style gate off: the template prefills an empty thinking block for this flag.
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
        # GLM-style: gate plus a level, clamped to this template's levels as the request build is.
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
    --chat-template-kwargs, so a count omitting them prices the mode the model was LOADED in."""
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


RAG_ON = (
    "{ supportsTools: true, toolsEnabled: false, codeToolsEnabled: false, "
    "artifactsEnabled: false, mcpEnabledForChat: false, ragEnabled: true, "
    'ragSource: { type: "thread" }, ragMode: "hybrid", ragTopK: 5, '
    "autoHealToolCalls: true }"
)


@pytest.mark.parametrize(
    ("thread_id", "expected_thread_id"),
    [("undefined", None), ('"thread-a"', "thread-a")],
    ids = ["unpersisted_new_chat", "persisted_thread"],
)
def test_the_rag_scope_a_count_sends_is_never_empty(thread_id, expected_thread_id):
    """The backend keeps search_knowledge_base and its grounding nudge only while rag_scope
    is truthy, and ``{}`` is falsy in Python. A New Chat has no thread and no project, so an
    id-only scope would drop from the count a tool schema and a nudge the send still pays."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ buildLocalTokenCountExtras, seed }} from "./harness.ts";
            seed({RAG_ON});
            const extras = await buildLocalTokenCountExtras({thread_id});
            console.log(JSON.stringify({{
              scope: extras.rag_scope,
              keys: Object.keys(extras.rag_scope ?? {{}}),
              enabledTools: extras.enabled_tools,
            }}));
            """
        )
    )
    assert "search_knowledge_base" in (
        out.get("enabledTools") or []
    ), "the Docs pill must still ask for the tool"
    assert out.get(
        "keys"
    ), "an empty rag_scope is falsy server-side and drops the tool and the nudge"
    assert (out.get("scope") or {}).get("thread_id") == expected_thread_id


def test_the_count_sends_every_setting_that_changes_the_rendered_prompt():
    """The backend prices the tool loop the settings describe: its gate, its call budget,
    and whether it retrieves. Omitting one priced the server's defaults."""
    settings = RAG_ON.rstrip(" }") + ', permissionMode: "ask", maxToolCallsPerMessage: 0 }'
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ buildLocalTokenCountExtras, seed }} from "./harness.ts";
            seed({settings});
            const on = await buildLocalTokenCountExtras("thread-a");
            seed({{ residentCheckpoint: "org/Model-70B" }});
            const large = await buildLocalTokenCountExtras("thread-a");
            seed({{ ragAutoInject: "off", residentCheckpoint: "org/Model-4B" }});
            const injectOff = await buildLocalTokenCountExtras("thread-a");
            seed({{ supportsTools: false }});
            const off = await buildLocalTokenCountExtras("thread-a");
            console.log(JSON.stringify({{ on, large, injectOff, off }}));
            """
        )
    )
    on, off = out["on"], out["off"]
    assert on.get("permission_mode") == "ask", "the gate that holds the loop's retrieval"
    assert on.get("max_tool_calls_per_message") == 0, "Off suppresses the loop entirely"
    assert (on.get("rag_scope") or {}).get("autoinject") is True
    # The values, not the keys: unknown size on, Auto off above the threshold.
    assert (out["large"].get("rag_scope") or {}).get("autoinject") is False
    off_scope = out["injectOff"].get("rag_scope") or {}
    assert (off_scope.get("autoinject"), off_scope.get("whole_doc")) == (False, False)
    # Explicit, and with no budget beside it, as the send is: an omitted flag would let
    # `unsloth studio run --enable-tools` answer for the count.
    assert off.get("enable_tools") is False
    assert "max_tool_calls_per_message" not in off


# Tools on, RAG deliberately off: the archive tool is gated on the thread id alone, so a
# scope-only id would leave it unpriced exactly when RAG is not in play.
TOOLS_ON_RAG_OFF = (
    "{ supportsTools: true, toolsEnabled: true, codeToolsEnabled: false, "
    "artifactsEnabled: false, mcpEnabledForChat: false, ragEnabled: false, "
    'ragSource: { type: "thread" }, ragMode: "hybrid", ragTopK: 5, '
    "autoHealToolCalls: true }"
)


@pytest.mark.parametrize(
    ("thread_id", "expected"),
    [("undefined", None), ('"thread-a"', "thread-a")],
    ids = ["unpersisted_new_chat", "persisted_thread"],
)
def test_the_count_sends_the_thread_id_at_top_level_even_with_rag_off(thread_id, expected):
    """`_select_request_tools` reads `payload.thread_id`, not the one inside `rag_scope`.

    An archived thread puts `search_conversation` and its compaction nudge in the prompt,
    so a count that only ever nests the id under a RAG scope under-reports every archived
    conversation whose Docs pill is off, and the bar claims room the completion lacks.
    """
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ buildLocalTokenCountExtras, seed }} from "./harness.ts";
            seed({TOOLS_ON_RAG_OFF});
            const extras = await buildLocalTokenCountExtras({thread_id});
            console.log(JSON.stringify({{
              threadId: extras.thread_id ?? null,
              ragScope: extras.rag_scope ?? null,
            }}));
            """
        )
    )
    assert out.get("ragScope") is None, "RAG is off, so there is no scope to hide the id in"
    assert (
        out.get("threadId") == expected
    ), "the archive tool and its nudge are gated on the top-level thread id"
