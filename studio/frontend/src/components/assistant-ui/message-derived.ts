// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Computed once per revision (#12552): assistant-ui rebuilds `parts` / `content` / `messages` on
// any change and keeps the same array otherwise, so the array identity is the revision key.

import { isRenderableRenderHtmlToolPart } from "../../features/chat/artifacts/html-fences.ts";
import { searchImagesSignature } from "../../features/chat/search-images/search-images.ts";
import {
  type CompactionMarker,
  type ContextTruncation,
  compactionBoundary,
  shouldShowCompactionNotice,
} from "../../features/chat/utils/context-truncation.ts";

export function memoOnArray<A extends readonly unknown[], T>(
  compute: (array: A) => T,
): (array: A) => T {
  const cache = new WeakMap<A, T>();
  return (array) => {
    if (cache.has(array)) return cache.get(array) as T;
    const value = compute(array);
    cache.set(array, value);
    return value;
  };
}

const memoByArrayAndKey = new WeakMap<object, Map<unknown, unknown>>();

export function memoOnArrayByKey<K, V>(
  array: readonly unknown[],
  key: K,
  compute: () => V,
): V {
  let byKey = memoByArrayAndKey.get(array);
  if (byKey === undefined) {
    byKey = new Map();
    memoByArrayAndKey.set(array, byKey);
  }
  if (byKey.has(key)) return byKey.get(key) as V;
  const value = compute();
  byKey.set(key, value);
  return value;
}

type PartLike = { readonly type: string; readonly text?: unknown };

export const partsHaveRenderableRenderHtmlTool = memoOnArray(
  (parts: readonly unknown[]) => parts.some(isRenderableRenderHtmlToolPart),
);

export const partsTextKey = memoOnArray((parts: readonly PartLike[]) =>
  JSON.stringify(
    parts.filter((part) => part.type === "text").map((part) => part.text),
  ),
);

export const partsHaveNonEmptyText = memoOnArray((parts: readonly PartLike[]) =>
  parts.some(
    (part) =>
      part.type === "text" &&
      "text" in part &&
      (part as { text: string }).text.length > 0,
  ),
);

export const partsSearchImagesSignature = memoOnArray(
  (
    parts: ReadonlyArray<{ type: string; toolName?: string; result?: unknown }>,
  ) => searchImagesSignature(parts),
);

// Slices of one joined string (as answerTextFromParts joins); a string per part was quadratic.
const precedingTextLayout = memoOnArray(
  (parts: ReadonlyArray<{ type: string; text?: unknown }>) => {
    const texts: string[] = [];
    const ends: number[] = [];
    let end = 0;
    for (const part of parts) {
      ends.push(end);
      if (part.type === "text" && typeof part.text === "string") {
        end += (texts.length > 0 ? 2 : 0) + part.text.length;
        texts.push(part.text);
      }
    }
    ends.push(end);
    const joined = texts.join("\n\n");
    return { joined, ends, slices: new Map<number, string>() };
  },
);

export function partsPrecedingText(
  parts: ReadonlyArray<{ type: string; text?: unknown }>,
  partIndex: number,
): string {
  const layout = precedingTextLayout(parts);
  const index = Math.max(0, Math.min(partIndex, parts.length));
  let text = layout.slices.get(index);
  if (text === undefined) {
    text = layout.joined.slice(0, layout.ends[index]);
    layout.slices.set(index, text);
  }
  return text;
}

type MessageLike = {
  readonly id: string;
  readonly role: string;
  readonly metadata?: unknown;
};

export const compactionNoticeMessageIds = memoOnArray(
  (messages: readonly MessageLike[]): ReadonlySet<string> => {
    const ids = new Set<string>();
    let previousDropped = 0;
    for (const message of messages) {
      if (message.role !== "assistant") continue;
      const value = (
        message.metadata as
          | { custom?: { contextTruncation?: unknown } }
          | undefined
      )?.custom?.contextTruncation as ContextTruncation | undefined;
      if (shouldShowCompactionNotice(value, previousDropped)) {
        ids.add(message.id);
        previousDropped = Math.max(previousDropped, compactionBoundary(value));
      }
    }
    return ids;
  },
);

type CompactionDividers = {
  /** Row id -> the compaction drawn above that row: the prompt whose request compacted, or the reply. */
  rows: ReadonlyMap<string, ContextTruncation>;
  /** Tool call id -> the compaction a tool loop ran right after that call returned. */
  afterToolCalls: ReadonlyMap<string, ContextTruncation>;
};

const NO_COMPACTION_DIVIDERS: CompactionDividers = {
  rows: new Map(),
  afterToolCalls: new Map(),
};

function compactionMarkersOf(
  message: MessageLike,
): readonly CompactionMarker[] | null {
  const markers = (
    message.metadata as { custom?: { compactionMarkers?: unknown } } | undefined
  )?.custom?.compactionMarkers;
  return Array.isArray(markers) && markers.length > 0
    ? (markers as CompactionMarker[])
    : null;
}

function hasToolCall(message: MessageLike, toolCallId: string): boolean {
  const { parts, content } = message as { parts?: unknown; content?: unknown };
  const list = Array.isArray(parts) ? parts : content;
  return (
    Array.isArray(list) &&
    list.some(
      (part) =>
        (part as { type?: string }).type === "tool-call" &&
        (part as { toolCallId?: string }).toolCallId === toolCallId,
    )
  );
}

/** Places each compaction where it cut: above the prompt whose request compacted (or the reply when no
 *  prompt precedes it), or, for one a tool loop ran mid-reply, right after the tool call it followed.
 *  Replies saved before markers existed keep one divider above the prompt. */
const compactionDividers = memoOnArray(
  (messages: readonly MessageLike[]): CompactionDividers => {
    const notices = compactionNoticeMessageIds(messages);
    // streamed tokens create new arrays, so reuse this result until a thread compacts.
    if (notices.size === 0) return NO_COMPACTION_DIVIDERS;
    const rows = new Map<string, ContextTruncation>();
    const afterToolCalls = new Map<string, ContextTruncation>();
    let previous: MessageLike | undefined;
    let previousBoundary = 0;
    for (const message of messages) {
      const row = previous?.role === "user" ? previous.id : message.id;
      const markers =
        message.role === "assistant" ? compactionMarkersOf(message) : null;
      if (markers) {
        for (const marker of markers) {
          if (!shouldShowCompactionNotice(marker.truncation, previousBoundary)) {
            continue;
          }
          previousBoundary = Math.max(
            previousBoundary,
            compactionBoundary(marker.truncation),
          );
          if (
            marker.afterToolCallId !== null &&
            hasToolCall(message, marker.afterToolCallId)
          ) {
            afterToolCalls.set(marker.afterToolCallId, marker.truncation);
          } else if (!rows.has(row)) {
            rows.set(row, marker.truncation);
          }
        }
      } else if (notices.has(message.id)) {
        const truncation = (
          message.metadata as {
            custom: { contextTruncation: ContextTruncation };
          }
        ).custom.contextTruncation;
        rows.set(row, truncation);
        previousBoundary = Math.max(
          previousBoundary,
          compactionBoundary(truncation),
        );
      }
      previous = message;
    }
    return rows.size === 0 && afterToolCalls.size === 0
      ? NO_COMPACTION_DIVIDERS
      : { rows, afterToolCalls };
  },
);

export const compactionDividerRows = (
  messages: readonly MessageLike[],
): ReadonlyMap<string, ContextTruncation> => compactionDividers(messages).rows;

export const compactionToolCallDividers = (
  messages: readonly MessageLike[],
): ReadonlyMap<string, ContextTruncation> =>
  compactionDividers(messages).afterToolCalls;

/** What the context meter counts: one per divider on screen. */
export const compactionDividerCount = (
  messages: readonly MessageLike[],
): number => {
  const dividers = compactionDividers(messages);
  return dividers.rows.size + dividers.afterToolCalls.size;
};
