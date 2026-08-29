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

export function bindStreamedToolCallBackendIds(
  ids: Map<string, string>,
  providerId: string,
  streamId: string,
): void {
  if (!ids.has(providerId)) ids.set(providerId, streamId);
  ids.set(streamId, streamId);
}

export interface StreamedToolCallPart {
  toolCallId: string;
  toolName?: string;
  argsText?: string;
  result?: unknown;
  _delta_index?: number;
  _has_stable_id?: boolean;
}

export interface JsonDocumentSplit {
  complete: string[];
  tail: string;
}

function sameJsonValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => sameJsonValue(value, right[index]))
    );
  }
  if (
    left === null ||
    right === null ||
    typeof left !== "object" ||
    typeof right !== "object"
  ) {
    return false;
  }
  const leftObject = left as Record<string, unknown>;
  const rightObject = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftObject);
  return (
    leftKeys.length === Object.keys(rightObject).length &&
    leftKeys.every(
      (key) =>
        Object.hasOwn(rightObject, key) &&
        sameJsonValue(leftObject[key], rightObject[key]),
    )
  );
}

function containsUnsafeJsonInteger(text: string): boolean {
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
    if (character === '"') {
      inString = true;
      continue;
    }
    if (character !== "-" && (character < "0" || character > "9")) {
      continue;
    }
    const match = text
      .slice(index)
      .match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/u);
    if (!match) continue;
    const value = Number(match[0]);
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) {
      return true;
    }
    index += match[0].length - 1;
  }
  return false;
}

export function sameJsonDocument(left: string, right: string): boolean {
  if (left.trim() === right.trim()) {
    try {
      JSON.parse(left);
      return true;
    } catch {
      return false;
    }
  }
  if (containsUnsafeJsonInteger(left) || containsUnsafeJsonInteger(right)) {
    return false;
  }
  try {
    return sameJsonValue(JSON.parse(left), JSON.parse(right));
  } catch {
    return false;
  }
}

export function mergeStreamedToolCallName(
  current: string,
  fragment: string,
): string {
  if (!fragment) return current;
  return fragment.startsWith(current) ? fragment : current + fragment;
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

export function isRepeatedJsonSnapshot(
  existing: string,
  fragment: string,
): boolean {
  if (!fragment.trim()) return false;
  if (!fragment.includes("{") && !fragment.includes("[")) return false;
  const documents = splitTopLevelJsonDocuments(fragment);
  return (
    documents.complete.length === 1 &&
    !documents.tail &&
    (existing === fragment || sameJsonDocument(existing, fragment))
  );
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
      part.result !== undefined ||
      (name && part.toolName && part.toolName !== name)
    ) {
      return false;
    }
    return (
      !argumentsText.trim() ||
      sameJsonDocument(part.argsText ?? "", argumentsText)
    );
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
  if (!fragmentArguments.includes("{") && !fragmentArguments.includes("[")) {
    return false;
  }
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
