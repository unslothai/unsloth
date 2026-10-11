// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Computed once per revision (#12552): assistant-ui rebuilds `parts` / `content` / `messages` on
// any change and keeps the same array otherwise, so the array identity is the revision key.

import { isRenderableRenderHtmlToolPart } from "../../features/chat/artifacts/html-fences.ts";
import { searchImagesSignature } from "../../features/chat/search-images/search-images.ts";
import {
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
