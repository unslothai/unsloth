// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Feedback the backend folded into a finished tool result for the model only (the tool budget
 *  notice). It is never shown; it rides the card's provenance and goes back with the result on the
 *  next request, so the replayed prompt matches the prefix llama-server already cached. */

type NotedPart = {
  toolCallId: string;
  backendToolCallId?: string;
  provenance?: Record<string, unknown>;
};

/** Index of the card a `tool_model_note` targets: the LAST match, since generated ids restart at
 *  call_0 every round. */
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

/** Provenance with `suffix` appended to any suffix already noted. A note delivered twice (a
 *  reconnect replaying the stream) is kept once. */
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

/** `extra_content` for the replayed role="tool" message, or nothing. */
export function toolModelSuffixExtraContent(
  provenance: Record<string, unknown> | null | undefined,
): { extra_content: { studio_model_suffix: string } } | Record<string, never> {
  const suffix = provenance?.model_suffix;
  return typeof suffix === "string" && suffix
    ? { extra_content: { studio_model_suffix: suffix } }
    : {};
}
