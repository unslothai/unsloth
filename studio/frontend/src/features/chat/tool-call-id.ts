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

/** Bind an id-less streamed card to the backend's matching minted id. */
export function bindStreamedToolCallBackendId(
  ids: Map<string, string>,
  partId: string,
): void {
  if (!ids.has(partId)) ids.set(partId, partId);
}

export interface StreamedToolCallPart {
  toolCallId: string;
  toolName?: string;
  argsText?: string;
  _delta_index?: number;
  _has_stable_id?: boolean;
}

/** Find adjacent JSON documents incrementally, leaving malformed input whole. */
export class ToolCallArgumentBoundaries {
  private depth = 0;
  private inString = false;
  private escaped = false;
  private closed = false;
  private invalid = false;
  /** Provider text used as the coordinate space for boundary offsets. */
  private raw = "";
  /** Current document text, retained only until it parses. */
  private open: string[] = [];
  /** Exposed for deterministic complexity tests. */
  scanned = 0;

  /** Offsets into `text()` where `fragment` opens another call. */
  feed(fragment: string): number[] {
    if (this.invalid) return [];
    const base = this.raw.length;
    this.raw += fragment;
    const boundaries: number[] = [];
    // Start of this fragment's contribution to the open document.
    let from = this.depth > 0 ? 0 : -1;

    for (let i = 0; i < fragment.length; i += 1) {
      this.scanned += 1;
      const character = fragment[i];
      if (this.inString) {
        if (this.escaped) this.escaped = false;
        else if (character === "\\") this.escaped = true;
        else if (character === '"') this.inString = false;
        continue;
      }
      if (this.depth === 0) {
        if (/\s/u.test(character)) continue;
        if (character !== "{" && character !== "[") {
          this.invalid = true;
          return [];
        }
        if (this.closed) {
          boundaries.push(base + i);
          this.closed = false;
        }
        from = i;
        this.depth = 1;
        continue;
      }
      if (character === '"') {
        this.inString = true;
      } else if (character === "{" || character === "[") {
        this.depth += 1;
      } else if (character === "}" || character === "]") {
        this.depth -= 1;
        if (this.depth === 0) {
          const document =
            this.open.join("") + fragment.slice(from < 0 ? 0 : from, i + 1);
          this.open = [];
          from = -1;
          try {
            // Parsing also rejects mismatched delimiters.
            JSON.parse(document);
          } catch {
            this.invalid = true;
            return [];
          }
          this.closed = true;
        }
      }
    }

    if (this.depth > 0) this.open.push(fragment.slice(from < 0 ? 0 : from));
    return boundaries;
  }

  /** The arguments as the provider streamed them, which `feed` indexes into. */
  text(): string {
    return this.raw;
  }

  holdsOneCompleteDocument(): boolean {
    return this.closed && this.depth === 0 && !this.invalid;
  }

  /** Whether a document has begun here and has not finished. */
  isOpen(): boolean {
    return this.depth > 0;
  }

  /** Whether this scan never turned into a valid document. A boundary is
   * recorded when the next `{` arrives, before that document has parsed. */
  isUnfinished(): boolean {
    return this.depth > 0 || this.invalid;
  }

  /** Reset offsets after moving the scan to a split call. */
  rebase(text: string): void {
    this.raw = text;
    this.open = this.depth > 0 ? [text] : [];
  }
}

/** Cut `text` at `boundaries` into one string per call. */
export function toolCallArgumentSegments(
  text: string,
  boundaries: readonly number[],
): string[] {
  const edges = [0, ...boundaries, text.length];
  return edges.slice(0, -1).map((from, i) => text.slice(from, edges[i + 1]));
}

/** Return `preferred` or the lowest unreserved `tool_call_<n>`. */
export function mintStreamedToolCallId(
  parts: readonly StreamedToolCallPart[],
  preferred: string,
  reserved: Iterable<string> = [],
): string {
  const taken = new Set(parts.map((part) => part.toolCallId));
  for (const id of reserved) taken.add(id);
  if (!taken.has(preferred)) return preferred;
  let next = 0;
  while (taken.has(`tool_call_${next}`)) next += 1;
  return `tool_call_${next}`;
}

/** Return the first part in `deltaIndex`'s slot no provider id has claimed, or
 * -1. Not "and unnamed": a split hands its name down to every segment, so a
 * named one can still be waiting for the id that belongs to it. */
export function findUnclaimedToolCallPartIndex(
  parts: readonly StreamedToolCallPart[],
  deltaIndex: number | undefined,
): number {
  if (deltaIndex === undefined) return -1;
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    if (part._delta_index === deltaIndex && !part._has_stable_id) return i;
  }
  return -1;
}

/** Return the first unnamed part in `deltaIndex`'s slot, or -1. */
export function findUnnamedToolCallPartIndex(
  parts: readonly StreamedToolCallPart[],
  deltaIndex: number | undefined,
): number {
  if (deltaIndex === undefined) return -1;
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    if (part._delta_index === deltaIndex && !part.toolName) return i;
  }
  return -1;
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
