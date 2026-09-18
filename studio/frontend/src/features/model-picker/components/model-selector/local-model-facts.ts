// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The half of "Model info" read from the local GGUF header, so the panel answers with no
// network. Pure, same contract as model-info-facts.ts.

export const LOCAL_MODEL_INFO_FIELDS = [
  "contextLength",
  "layers",
  "moeLayers",
  "reasoning",
  "chatTemplate",
] as const;

export type LocalModelInfoField = (typeof LOCAL_MODEL_INFO_FIELDS)[number];

export interface LocalModelInfoFact {
  key: LocalModelInfoField;
  label: string;
  value: string;
  /** Longer explanation for a tooltip, when the bare value understates the row. */
  detail?: string;
}

/** What /api/inference/validate reports for a local file. Null means not read. */
export interface LocalModelMeta {
  contextLength?: number | null;
  layerCount?: number | null;
  moeLayerCount?: number | null;
  chatTemplate?: string | null;
}

// Jinja comments are documentation, not behaviour: a `{# ... #}` note mentioning `/no_think`
// made a non-hybrid model advertise a toggle it does not have.
const JINJA_COMMENT = /\{#[\s\S]*?#\}/g;

// A marker being REMOVED, not emitted: `{{ content.split('</think>')[-1] }}` strips a previous
// turn's reasoning, which is what the NON-thinking variant of a pair does, so counting it
// inverts the answer. Blanking the whole call leaves a genuinely emitted marker standing.
const MARKER_CONSUMING_CALL =
  /\.\s*(?:split|rsplit|replace|partition|rpartition|find|rfind|index|startswith|endswith|strip|lstrip|rstrip)\s*\(\s*(['"])[\s\S]*?\1[^)]*\)/g;

// A switch means thinking can be turned OFF, so a variable counts only where the template
// branches on it: a bare `{%- set enable_thinking = true %}` names it without honouring it.
// Read from the template, not the repo name, so a renamed quant still answers correctly.
const THINKING_VAR_IN_CONDITION =
  /\{[%{][^%}]*?\b(?:if|elif)\b[^%}]*?(?:\benable_thinking\b|\bthinking_?budget\b)/;

// A sentinel rather than a word, so emitting it at all implies a model that honours it. The
// mention that must not count lives in a `{# ... #}` note, already stripped above.
const NO_THINK_SENTINEL = /\/no_?think\b/;

// Markers alone do not establish that thinking can be DISABLED. Beyond `<think>`: `[THINK]` is
// Magistral, `<seed:think>` ByteDance Seed. Omitting them called Magistral "Not detected".
const REASONING_MARKERS =
  /\breasoning_effort\b|<\/?think>|<\/?thinking>|<\|\/?think\|>|<\/?seed:think>|\[\/?THINK\]|<\|\/?start_of_thought\|>|<\|channel\|>analysis|\breasoning_content\b/;

/** Only what the template would actually emit or branch on. */
function executableTemplate(template: string): string {
  return template.replace(JINJA_COMMENT, " ").replace(MARKER_CONSUMING_CALL, " ");
}

export type ReasoningSupport = "hybrid" | "detected" | "unknown";

/** Detect template switches and markers without inferring an always-on capability. */
export function reasoningSupport(
  template: string | null | undefined,
): ReasoningSupport {
  if (!template || !template.trim()) return "unknown";
  const executable = executableTemplate(template);
  if (
    THINKING_VAR_IN_CONDITION.test(executable) ||
    NO_THINK_SENTINEL.test(executable)
  )
    return "hybrid";
  return REASONING_MARKERS.test(executable) ? "detected" : "unknown";
}

const REASONING_VALUE: Record<ReasoningSupport, string> = {
  hybrid: "Hybrid",
  detected: "Detected",
  unknown: "Not detected",
};

const REASONING_DETAIL: Record<ReasoningSupport, string> = {
  hybrid:
    "Reasoning can be turned on or off per request, so this model can answer with or without thinking first.",
  detected:
    "Reasoning markers are present in the chat template. They do not establish whether thinking can be turned off.",
  unknown:
    "No thinking markers in this model's chat template. It may still reason, so this is not a verdict that it does not.",
};

function formatTokens(tokens: number): string {
  // A K-only unit renders Llama 4's 10,485,760 as "10240K tokens".
  const MEGA = 1024 * 1024;
  if (tokens >= MEGA && tokens % MEGA === 0) return `${tokens / MEGA}M tokens`;
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K tokens`;
  return `${tokens.toLocaleString("en-US")} tokens`;
}

/** A count the header actually reported, including a real 0 for a dense model's MoE layers. */
function isReportedCount(n: number | null | undefined): n is number {
  return typeof n === "number" && Number.isFinite(n) && n >= 0;
}

/** Whether the probe read the file at all: every GGUF header carries a context length or a
 *  block count, so neither means nothing was read, and no rows beat "none embedded".
 *
 *  POSITIVE, not merely reported. 0 is a real `moeLayerCount` but not a real context length,
 *  and admitting it cleared this gate while failing every row guard, leaving the panel on
 *  "Chat template: Not available" alone — the claim the line above forbids. */
function headerWasRead(meta: LocalModelMeta): boolean {
  return (
    (isReportedCount(meta.contextLength) && meta.contextLength > 0) ||
    (isReportedCount(meta.layerCount) && meta.layerCount > 0)
  );
}

export function localModelInfoFacts(
  meta: LocalModelMeta,
): LocalModelInfoFact[] {
  const facts: LocalModelInfoFact[] = [];
  if (!headerWasRead(meta)) return facts;

  if (isReportedCount(meta.contextLength) && meta.contextLength > 0) {
    facts.push({
      key: "contextLength",
      label: "Context length",
      value: formatTokens(meta.contextLength),
      detail:
        "The model's native context window, read from the local file. Run settings may load it with a smaller one.",
    });
  }

  if (isReportedCount(meta.layerCount) && meta.layerCount > 0) {
    facts.push({
      key: "layers",
      label: "Layers",
      value: String(meta.layerCount),
    });
  }

  if (isReportedCount(meta.moeLayerCount)) {
    facts.push({
      key: "moeLayers",
      label: "MoE expert layers",
      value:
        meta.moeLayerCount > 0 ? String(meta.moeLayerCount) : "None (dense)",
    });
  }

  const hasTemplate = Boolean(meta.chatTemplate?.trim());

  if (hasTemplate) {
    const reasoning = reasoningSupport(meta.chatTemplate);
    facts.push({
      key: "reasoning",
      label: "Reasoning",
      value: REASONING_VALUE[reasoning],
      detail: REASONING_DETAIL[reasoning],
    });
  }

  // Null is not absence: the probe drops templates over 64KB, and read_gguf_chat_template
  // returns null for absent, unreadable and not-a-GGUF alike. Only presence is stated.
  facts.push({
    key: "chatTemplate",
    label: "Chat template",
    value: hasTemplate ? "Embedded" : "Not available",
    detail: hasTemplate
      ? "The file carries its own chat template, so it formats conversations without one being supplied. Click to read it."
      : "No template was returned for this file. It may carry none, or one too large to read here.",
  });

  return facts;
}
