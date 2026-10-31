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

/** Must match backend `_mint_streamed_card_id`. No colon: replayed ids need ^[a-zA-Z0-9_-]+$. */
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

/** Without this binding the turn persists two parts per id-less call. */
export function bindStreamedToolCallCard(
  ids: Map<string, string>,
  partId: string,
): void {
  if (!ids.has(partId)) ids.set(partId, partId);
}

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

/** Providers restart tool_calls[].index per round, so match on id first, index slot only if unowned. */
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
