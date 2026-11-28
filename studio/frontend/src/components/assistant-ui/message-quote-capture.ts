// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// DOM capture for multi-section quoting. The resulting QuoteAnchor contains no
// Range or element references, so it survives message windowing and remounts.

import {
  type FindElementLike,
  type FindTextIndex,
  buildTextIndex,
} from "../../features/find-in-page/lib/find-text-index.ts";
import {
  type QuoteAnchor,
  canonicalQuoteText,
  createQuoteAnchor,
  normalizeQuoteText,
  quoteSearchIndex,
} from "./message-quote-model";
import {
  type QuotePreviewNode,
  tableQuotePreview,
} from "./message-quote-preview";
import { QUOTE_PRUNE_SELECTOR } from "./message-selection-model";

export type QuoteCaptureInput = {
  readonly messageId: string;
  readonly turn: number;
  readonly markdown: string;
  readonly range: Range;
  readonly content: HTMLElement;
};

/** One clone per capture: chrome is removed before both text and preview use it. */
function quoteHolderFromRange(range: Range): HTMLElement {
  const holder = document.createElement("div");
  holder.append(range.cloneContents());
  for (const element of Array.from(
    holder.querySelectorAll(QUOTE_PRUNE_SELECTOR),
  )) {
    element.remove();
  }
  return holder;
}

/** Preserve the captured-text fallback when the clone carries no usable text. */
function quoteTextFromHolder(holder: HTMLElement, range: Range): string {
  const pruned = normalizeQuoteText(holder.textContent ?? "");
  return pruned.length > 0 ? pruned : normalizeQuoteText(range.toString());
}

function indexOffsetAtBoundary(
  index: FindTextIndex,
  container: Node,
  offset: number,
  edge: "start" | "end",
): number | null {
  if (container.nodeType === 3) {
    const segment = index.segments.find(
      (candidate) =>
        (candidate.node as unknown as Node) === container,
    );
    if (segment === undefined) {
      return null;
    }
    return (
      segment.start +
      Math.min(Math.max(0, offset), segment.length)
    );
  }

  try {
    const probe = document.createRange();
    probe.setStart(container, offset);
    probe.collapse(true);

    if (edge === "start") {
      for (const segment of index.segments) {
        if (
          probe.comparePoint(
            segment.node as unknown as Node,
            0,
          ) >= 0
        ) {
          return segment.start;
        }
      }
      return index.text.length;
    }

    let end = 0;
    for (const segment of index.segments) {
      if (
        probe.comparePoint(
          segment.node as unknown as Node,
          segment.length,
        ) <= 0
      ) {
        end = segment.start + segment.length;
      }
    }
    return end;
  } catch {
    return null;
  }
}

/**
 * Captures a serializable anchor from a live selection Range. The caller must
 * capture while the source message is mounted; resolution after remounting is
 * handled by the pure model.
 */
export function captureQuoteAnchor(
  input: QuoteCaptureInput,
): QuoteAnchor | null {
  if (!input.content.isConnected || input.range.collapsed) {
    return null;
  }
  const index = buildTextIndex(
    input.content as unknown as FindElementLike,
    [],
    QUOTE_PRUNE_SELECTOR,
  );
  const start = indexOffsetAtBoundary(
    index,
    input.range.startContainer,
    input.range.startOffset,
    "start",
  );
  const end = indexOffsetAtBoundary(
    index,
    input.range.endContainer,
    input.range.endOffset,
    "end",
  );
  if (start === null || end === null) {
    return null;
  }
  const holder = quoteHolderFromRange(input.range);
  const captured = quoteTextFromHolder(holder, input.range);
  const spanText = quoteSearchIndex(index).text.slice(start, end);
  const text = canonicalQuoteText(captured, spanText);
  return createQuoteAnchor({
    messageId: input.messageId,
    turn: input.turn,
    text,
    previewText:
      tableQuotePreview(holder as unknown as QuotePreviewNode, text) ??
      undefined,
    markdown: input.markdown,
    index,
    start,
    end,
  });
}

/** The assistant content root belonging to a mounted message id. */
export function assistantContentForMessage(
  viewport: HTMLElement,
  messageId: string,
): HTMLElement | null {
  const row = viewport.querySelector<HTMLElement>(
    `[data-role="assistant"][data-message-id="${CSS.escape(messageId)}"]`,
  );
  return (
    row?.querySelector<HTMLElement>(".aui-assistant-message-content") ??
    null
  );
}
