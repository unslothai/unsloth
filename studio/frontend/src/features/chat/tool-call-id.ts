// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function resolveToolCallPartId(
  ids: Map<string, string>,
  backendId: string,
  confirmationId: string | undefined,
  lastPartId: string,
  createId: () => string,
): string {
  if (!backendId) return lastPartId;
  if (confirmationId) return confirmationId;
  const existing = ids.get(backendId);
  if (existing) return existing;
  const partId = createId();
  ids.set(backendId, partId);
  return partId;
}

export interface StreamedToolCallPart {
  toolCallId: string;
  _delta_index?: number;
  _has_stable_id?: boolean;
}

/**
 * Newest part holding `deltaIndex`, or -1. `unownedOnly` restricts the match to
 * a slot no provider id has claimed yet.
 */
function findDeltaIndexSlot(
  parts: readonly StreamedToolCallPart[],
  deltaIndex: number | undefined,
  unownedOnly: boolean,
): number {
  if (deltaIndex === undefined) {
    return -1;
  }
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    const part = parts[i];
    if (part._delta_index !== deltaIndex) {
      continue;
    }
    return unownedOnly && part._has_stable_id ? -1 : i;
  }
  return -1;
}

/**
 * Index of the tool-call part a `delta.tool_calls[]` fragment continues, or -1
 * when the fragment starts a new call.
 *
 * Providers restart `tool_calls[].index` at 0 for every tool round inside one
 * assistant response, so the index slot alone cannot separate a continuation
 * from the next round's opening fragment. A fragment carrying a stable id
 * therefore matches on that id, and falls back to the index slot only while no
 * other id owns it, which covers servers that stamp the real id on a later
 * fragment. Id-less fragments continue the newest part in their slot.
 */
export function findStreamedToolCallPartIndex(
  parts: readonly StreamedToolCallPart[],
  partId: string | undefined,
  deltaIndex: number | undefined,
): number {
  if (!partId) {
    return findDeltaIndexSlot(parts, deltaIndex, false);
  }
  const byId = parts.findIndex((part) => part.toolCallId === partId);
  return byId === -1 ? findDeltaIndexSlot(parts, deltaIndex, true) : byId;
}
