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
 * The id for a card drawn from a delta the provider gave no id to.
 *
 * The slot's own `tool_call_<index>` when that is free, then the lowest
 * `tool_call_<n>` that is not. `_mint_streamed_card_id` in the backend loop
 * walks the same deltas in the same order under the same rule, so it lands on
 * the same spelling and its `tool_start` and `tool_end` reach the card this
 * stream already drew rather than opening a second one beside it.
 *
 * `reserved` holds the ids the provider itself sent, so a provider that spells
 * one `tool_call_0` keeps it. No colon, because a replayed id has to satisfy
 * `^[a-zA-Z0-9_-]+$`.
 */
export function mintStreamedToolCallId(
  parts: StreamedToolCallPart[],
  deltaIndex: number | undefined,
  reserved: Set<string>,
): string {
  const isTaken = (candidate: string) =>
    reserved.has(candidate) || parts.some((part) => part.toolCallId === candidate);
  const preferred = deltaIndex === undefined ? "" : `tool_call_${deltaIndex}`;
  if (preferred && !isTaken(preferred)) return preferred;
  let position = 0;
  while (isTaken(`tool_call_${position}`)) position += 1;
  return `tool_call_${position}`;
}

/**
 * Let a card drawn by the deltas answer to its own id.
 *
 * Without this `resolveToolCallPartId` mints `<backend id>:<uuid>` the first
 * time the backend names an id-less call, `tool_start` finds no card under
 * that and pushes one, and the turn ends holding two cards and persisting two
 * tool-call parts for every call the stream drew.
 */
export function bindStreamedToolCallCard(
  ids: Map<string, string>,
  partId: string,
): void {
  if (!ids.has(partId)) ids.set(partId, partId);
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
