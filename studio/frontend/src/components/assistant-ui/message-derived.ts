// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Values derived from a whole message or thread, computed once per revision (#12552).
//
// One store write re-runs every useAuiState selector in the tree (#9054): every keystroke in the
// composer and every streamed delta. A selector that scans or joins a message's parts therefore
// pays for the whole message once per subscriber per write, and the subscribers that ask are per
// part, per group or per message, so the cost was quadratic in a long agent turn and quadratic in
// the thread for the compaction notice. Typing into a long Rolling Context chat stalled the page.
//
// assistant-ui rebuilds `message.parts`, `message.content` and `thread.messages` whenever anything
// in them changes (each is a `tapClientLookup` state, or the runtime's immutable content array),
// and hands back the same array on a write that changes nothing in them. The array is therefore
// the revision: caching on its identity is exact, and a WeakMap lets an old revision go with it.

import { isRenderableRenderHtmlToolPart } from "../../features/chat/artifacts/html-fences.ts";
import {
  precedingTextForMessagePart,
  searchImagesSignature,
} from "../../features/chat/search-images/search-images.ts";
import {
  type ContextTruncation,
  compactionBoundary,
  shouldShowCompactionNotice,
} from "../../features/chat/utils/context-truncation.ts";

/** `compute(array)`, computed once per array object. */
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

/** Whether any part is a render_html call that renders. */
export const partsHaveRenderableRenderHtmlTool = memoOnArray(
  (parts: readonly unknown[]) => parts.some(isRenderableRenderHtmlToolPart),
);

/** Every text part's text, as one string a selector can compare by value. */
export const partsTextKey = memoOnArray((parts: readonly PartLike[]) =>
  JSON.stringify(
    parts.filter((part) => part.type === "text").map((part) => part.text),
  ),
);

/** Whether any text part has something in it. */
export const partsHaveNonEmptyText = memoOnArray((parts: readonly PartLike[]) =>
  parts.some(
    (part) =>
      part.type === "text" &&
      "text" in part &&
      (part as { text: string }).text.length > 0,
  ),
);

/** The search-image signature of the parts (see searchImagesSignature). */
export const partsSearchImagesSignature = memoOnArray(
  (parts: ReadonlyArray<{ type: string; toolName?: string; result?: unknown }>) =>
    searchImagesSignature(parts),
);

const precedingTextByIndex = memoOnArray(
  (_parts: ReadonlyArray<{ type: string; text?: unknown }>) =>
    new Map<number, string>(),
);

/** precedingTextForMessagePart, computed once per parts array and index. */
export function partsPrecedingText(
  parts: ReadonlyArray<{ type: string; text?: unknown }>,
  partIndex: number,
): string {
  const byIndex = precedingTextByIndex(parts);
  let text = byIndex.get(partIndex);
  if (text === undefined) {
    text = precedingTextForMessagePart(parts, partIndex);
    byIndex.set(partIndex, text);
  }
  return text;
}

type MessageLike = {
  readonly id: string;
  readonly role: string;
  readonly metadata?: unknown;
};

/**
 * Ids of the assistant messages that show a compaction notice: where the eviction boundary rose
 * above every earlier notice, or a checkpoint started. One pass over the thread, where asking
 * per message walked the thread up to that message for each of them.
 */
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
