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
  toolName?: string;
  argsText?: string;
  _delta_index?: number;
  _has_stable_id?: boolean;
}

export interface JsonDocumentSplit {
  complete: string[];
  tail: string;
}

/** split a stream slot into complete top-level JSON documents and its open tail. */
export function splitTopLevelJsonDocuments(text: string): JsonDocumentSplit {
  const unsplit = { complete: [], tail: text };
  const complete: string[] = [];
  const closing: string[] = [];
  let start = -1;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (closing.length === 0) {
      if (character === "{" || character === "[") {
        start = index;
        closing.push(character === "{" ? "}" : "]");
        continue;
      }
      if (/\s/u.test(character)) continue;
      return unsplit;
    }
    if (character === '"') {
      inString = true;
      continue;
    }
    if (character === "{" || character === "[") {
      closing.push(character === "{" ? "}" : "]");
      continue;
    }
    if (character !== "}" && character !== "]") continue;
    if (closing.pop() !== character) return unsplit;
    if (closing.length !== 0) continue;

    const document = text.slice(start, index + 1);
    try {
      JSON.parse(document);
    } catch {
      return unsplit;
    }
    complete.push(document);
    start = -1;
  }

  return {
    complete,
    tail: start === -1 ? "" : text.slice(start),
  };
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
    if (unownedOnly && part._has_stable_id) {
      continue;
    }
    return i;
  }
  return -1;
}

/** oldest id-less part holding `deltaIndex`, or -1. */
export function findOldestUnownedStreamedToolCallPartIndex(
  parts: readonly StreamedToolCallPart[],
  deltaIndex: number | undefined,
): number {
  if (deltaIndex === undefined) {
    return -1;
  }
  return parts.findIndex(
    (part) =>
      part._delta_index === deltaIndex && part._has_stable_id !== true,
  );
}

export function findDelayedStableToolCallPartIndex(
  parts: readonly StreamedToolCallPart[],
  deltaIndex: number | undefined,
  name: string,
  argumentsText: string,
): number {
  if (deltaIndex === undefined) {
    return -1;
  }
  return parts.findIndex((part) => {
    if (
      part._delta_index !== deltaIndex ||
      part._has_stable_id === true ||
      (name && part.toolName && part.toolName !== name)
    ) {
      return false;
    }
    return !argumentsText || part.argsText === argumentsText;
  });
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

/**
 * True when an id-less fragment cannot be a continuation of the arguments a
 * slot has already accumulated: the accumulated text forms a complete JSON
 * document and the fragment opens another one.
 *
 * Some proxies strip tool-call ids AND renumber every parallel call's index
 * to 0 (LiteLLM), so slot matching alone lands each call's opening fragment
 * on the previous call's part and the argument JSONs concatenate into one
 * malformed string, which then poisons the thread on replay (#9807). A
 * complete JSON document never continues into another `{`/`[`, so this
 * boundary is safe for genuinely chunked arguments.
 */
export function fragmentStartsNewToolCall(
  existingArgsText: string | undefined,
  fragmentArguments: string,
): boolean {
  const existing = existingArgsText ?? "";
  if (!existing.trim() || !fragmentArguments.trim()) return false;
  const split = splitTopLevelJsonDocuments(existing + fragmentArguments);
  return (
    split.complete.length > 1 ||
    (split.complete.length === 1 && Boolean(split.tail))
  );
}

/** a `tool_call_<n>` id no existing part already carries. */
export function mintStreamedToolCallId(
  parts: readonly StreamedToolCallPart[],
  _deltaIndex: number | undefined,
  reservedIds: Iterable<string> = [],
): string {
  const taken = new Set(reservedIds);
  for (const part of parts) {
    taken.add(part.toolCallId);
  }
  let next = 0;
  while (taken.has(`tool_call_${next}`)) {
    next += 1;
  }
  return `tool_call_${next}`;
}
