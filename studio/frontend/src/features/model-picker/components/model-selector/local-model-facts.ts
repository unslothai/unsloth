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

// A thinking switch means thinking can be turned off. Read off the template, not the repo
// name, so a renamed quant still answers correctly.
const THINKING_TOGGLE =
  /\benable_thinking\b|\/no_?think\b|\bthinking_?budget\b/;

// Markers alone do not establish whether thinking can be disabled.
const REASONING_MARKERS =
  /\breasoning_effort\b|<\/?think>|<\|channel\|>analysis|\breasoning_content\b/;

export type ReasoningSupport = "hybrid" | "detected" | "unknown";

/** Detect template switches and markers without inferring an always-on capability. */
export function reasoningSupport(
  template: string | null | undefined,
): ReasoningSupport {
  if (!template || !template.trim()) return "unknown";
  if (THINKING_TOGGLE.test(template)) return "hybrid";
  return REASONING_MARKERS.test(template) ? "detected" : "unknown";
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
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K tokens`;
  return `${tokens.toLocaleString("en-US")} tokens`;
}

/** A count the header actually reported, including a real 0 for a dense model's MoE layers. */
function isReportedCount(n: number | null | undefined): n is number {
  return typeof n === "number" && Number.isFinite(n) && n >= 0;
}

/** Whether the probe read the file at all. Every GGUF header carries a context length or a
 *  block count, so neither being reported means nothing was read. Absence of a reading is not
 *  a finding, so no local rows beat turning "not read" into "none embedded". */
function headerWasRead(meta: LocalModelMeta): boolean {
  return (
    isReportedCount(meta.contextLength) || isReportedCount(meta.layerCount)
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
