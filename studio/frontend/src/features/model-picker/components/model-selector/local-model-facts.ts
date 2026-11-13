// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const LOCAL_MODEL_INFO_FIELDS = [
  "contextLength",
  "layers",
  "moeLayers",
  "chatTemplate",
] as const;

export type LocalModelInfoField = (typeof LOCAL_MODEL_INFO_FIELDS)[number];

export interface LocalModelInfoFact {
  key: LocalModelInfoField;
  label: string;
  value: string;
  detail?: string;
}

/** What /api/inference/validate reports for a local file. Null means not read. */
export interface LocalModelMeta {
  contextLength?: number | null;
  layerCount?: number | null;
  moeLayerCount?: number | null;
  chatTemplate?: string | null;
}

function formatTokens(tokens: number): string {
  // A K-only unit renders Llama 4's 10,485,760 as "10240K tokens".
  const MEGA = 1024 * 1024;
  if (tokens >= MEGA && tokens % MEGA === 0) return `${tokens / MEGA}M tokens`;
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K tokens`;
  return `${tokens.toLocaleString("en-US")} tokens`;
}

function isReportedCount(n: number | null | undefined): n is number {
  return typeof n === "number" && Number.isFinite(n) && n >= 0;
}

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

  // Null is not absence: the probe drops templates over 64KB and returns null when unreadable.
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
