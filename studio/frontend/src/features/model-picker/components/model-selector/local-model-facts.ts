// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The half of "Model info" read from the local GGUF header, so the panel still answers with
// no network. Pure, same contract as model-info-facts.ts.

/** Panel reading order for the local section. */
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

// Jinja comments are documentation, not behaviour. A template that merely MENTIONS `/no_think`
// in a `{# ... #}` note describing the model is not a template that honours it, and reading the
// note as a switch is how a non-hybrid model came to advertise a toggle it does not have.
const JINJA_COMMENT = /\{#[\s\S]*?#\}/g;

// String-consuming calls whose literal argument is a marker being REMOVED, not emitted. The
// giveaway idiom is `{{ content.split('</think>')[-1] }}`, which strips a previous turn's
// reasoning out of the history — something the NON-thinking variant of a pair does, so reading
// it as evidence of reasoning inverts the answer. Blanking the whole call, argument included,
// leaves any genuinely emitted marker still standing: a template that both strips history and
// writes `<think>` itself keeps the second one and still reads as detected.
const MARKER_CONSUMING_CALL =
  /\.\s*(?:split|rsplit|replace|partition|rpartition|find|rfind|index|startswith|endswith|strip|lstrip|rstrip)\s*\(\s*(['"])[\s\S]*?\1[^)]*\)/g;

// A thinking switch means thinking can be turned off. Read off the template, not the repo name,
// so a renamed quant still answers correctly — and, for a variable, only where the template
// BRANCHES on it. A bare `{%- set enable_thinking = true %}` names the variable without
// honouring it, and calling that "Hybrid" asserts, in the one tooltip here that makes a hard
// claim, that reasoning can be turned off per request when nothing in the template can turn it
// off.
const THINKING_VAR_IN_CONDITION =
  /\{[%{][^%}]*?\b(?:if|elif)\b[^%}]*?(?:\benable_thinking\b|\bthinking_?budget\b)/;

// `/no_think` needs no branch around it: it is a model-specific sentinel rather than a word, so
// a template that emits it at all is a template built for a model that honours it. The mention
// this must NOT count is the one in a `{# ... #}` note, and comments are already gone by here.
const NO_THINK_SENTINEL = /\/no_?think\b/;

// Markers alone do not establish whether thinking can be disabled. Beyond the `<think>` family:
// `[THINK]` is Mistral's Magistral, `<seed:think>` ByteDance Seed, and `<|start_of_thought|>`
// and `<thinking>` are used by several others. Omitting them reported a reasoning model this
// very PR links a guide for — Magistral — as "Not detected".
const REASONING_MARKERS =
  /\breasoning_effort\b|<\/?think>|<\/?thinking>|<\|\/?think\|>|<\/?seed:think>|\[\/?THINK\]|<\|\/?start_of_thought\|>|<\|channel\|>analysis|\breasoning_content\b/;

/** Template text with documentation and marker-stripping expressions removed, so both tests
 *  below see only what the template would actually emit or branch on. */
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
  // Million-token windows ship now, and a K-only unit renders Llama 4's 10,485,760 as
  // "10240K tokens". Step up to M first, so 1M reads as 1M rather than 1024K.
  const MEGA = 1024 * 1024;
  if (tokens >= MEGA && tokens % MEGA === 0) return `${tokens / MEGA}M tokens`;
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K tokens`;
  return `${tokens.toLocaleString("en-US")} tokens`;
}

/** A count the header actually reported, including a real 0 for a dense model's MoE layers. */
function isReportedCount(n: number | null | undefined): n is number {
  return typeof n === "number" && Number.isFinite(n) && n >= 0;
}

/** Whether the probe read the file at all. Every GGUF header carries a context length or a
 *  block count, so neither being reported means nothing was read. Absence of a reading is not
 *  a finding, so no local rows beat turning "not read" into "none embedded".
 *
 *  Both counts must be POSITIVE, not merely reported. `isReportedCount` accepts 0 so a dense
 *  model's `moeLayerCount: 0` can render as "None (dense)", but a context length or block count
 *  of 0 is not a reading — no real GGUF has either — and letting it through cleared this gate
 *  while failing every row guard below, leaving the panel showing "Chat template: Not available"
 *  on its own. That is the exact claim the paragraph above says must never be made. */
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

  // 0 is meaningful here (dense, not unknown), so it is reported rather than dropped.
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

  // A null template is not evidence of absence. The probe drops one over 64KB, a read failure
  // after the dims succeeded leaves the same shape, and read_gguf_chat_template returns null
  // for absent, unreadable and not-a-GGUF alike. So only presence is ever stated.
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
