// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Model-only feedback (the tool budget notice) on a finished card, replayed with its result. */

type NotedPart = {
  toolCallId: string;
  backendToolCallId?: string;
  provenance?: Record<string, unknown>;
};

/** The LAST match: generated ids restart at call_0 every round. */
export function toolModelNoteTarget(
  parts: readonly NotedPart[],
  backendToolCallId: string,
): number {
  if (!backendToolCallId) return -1;
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    const part = parts[i];
    if (
      part.backendToolCallId === backendToolCallId ||
      part.toolCallId === backendToolCallId
    ) {
      return i;
    }
  }
  return -1;
}

/** A note delivered twice (a reconnect replay) is kept once. */
export function withToolModelSuffix(
  provenance: Record<string, unknown> | undefined,
  suffix: string,
): Record<string, unknown> {
  const previous =
    typeof provenance?.model_suffix === "string" ? provenance.model_suffix : "";
  if (suffix && previous.endsWith(suffix)) return { ...(provenance ?? {}) };
  return {
    ...(provenance ?? {}),
    model_suffix: previous + suffix,
  };
}

export function toolModelSuffixExtraContent(
  provenance: Record<string, unknown> | null | undefined,
): { extra_content: { studio_model_suffix: string } } | Record<string, never> {
  const suffix = provenance?.model_suffix;
  return typeof suffix === "string" && suffix
    ? { extra_content: { studio_model_suffix: suffix } }
    : {};
}
