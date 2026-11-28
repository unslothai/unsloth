// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Native selection stays owned by the browser. This module only snapshots it:
// identity, Markdown payload, and the live cloned Range used for toolbar placement.

import {
  htmlNodeToMarkdown,
  type PasteNode,
} from "../../features/chat/utils/rich-paste.ts";
import { faithfulSelectionText } from "./thread-fast-copy.ts";
import { turnNumberAt } from "./thread-turns.ts";

const INTERACTIVE_SELECTOR = [
  "input",
  "textarea",
  "select",
  "button",
  "[role='button']",
  "[contenteditable='']",
  "[contenteditable='true']",
  "[contenteditable='plaintext-only']",
].join(",");

// Rendered Markdown chrome that must never become part of a quote.
const CHROME_SELECTOR = [
  "[data-streamdown='code-block-header']",
  "[data-streamdown='code-block-actions']",
  "[data-streamdown='mermaid-block-actions']",
  "[data-streamdown='table-fullscreen']",
  ".aui-md-table-actions",
].join(",");

export const QUOTE_PRUNE_SELECTOR = [
  "button",
  "svg",
  "[data-streamdown='code-block-header']",
  "[data-streamdown='code-block-actions']",
  "[data-streamdown='mermaid-block-actions']",
  "[data-streamdown='table-fullscreen']",
  ".aui-md-table-actions",
  "[aria-hidden='true']",
].join(",");

export type MessageIdentity = {
  readonly id: string;
  readonly role: string;
};

export type SelectionRect = {
  readonly top: number;
  readonly left: number;
  readonly right: number;
  readonly bottom: number;
  readonly width: number;
  readonly height: number;
};

export type SelectionSnapshot = {
  readonly messageId: string;
  readonly turn: number;
  readonly markdown: string;
  /** Live clone: tracks DOM mutations but is independent of the browser's selection. */
  readonly range: Range;
  readonly rect: SelectionRect;
};

export type SelectionPosition = {
  readonly startContainer: Node;
  readonly startOffset: number;
  readonly endContainer: Node;
  readonly endOffset: number;
};

export type ToolbarSize = {
  readonly width: number;
  readonly height: number;
};

export type ViewportRect = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

export type SelectionRectLike = {
  readonly top: number;
  readonly left: number;
  readonly right: number;
  readonly bottom: number;
};

export type ToolbarPlacement = {
  readonly left: number;
  readonly top: number;
};

const KEYBOARD_SELECTION_EXTEND_KEYS = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  "Home",
  "End",
  "PageUp",
  "PageDown",
]);

export type KeyboardSelectionInput = {
  readonly key: string;
  readonly shiftKey: boolean;
  readonly ctrlOrMetaKey: boolean;
  readonly targetBlocksDocumentSelection: boolean;
  readonly targetInThread: boolean;
  readonly targetIsDocumentBody: boolean;
  readonly selectionInThread: boolean;
};

export function rectOf(rect: DOMRect): SelectionRect {
  return {
    top: rect.top,
    left: rect.left,
    right: rect.right,
    bottom: rect.bottom,
    width: rect.width,
    height: rect.height,
  };
}

export function shortMessageId(messageId: string): string {
  return messageId.slice(0, 8);
}

/** Normalises an event target without depending on the global Element
 * constructor, so the predicate is testable outside a browser. */
export function keyboardSelectionTarget(target: unknown): Element | null {
  if (target === null || typeof target !== "object") {
    return null;
  }
  const closest = (target as { closest?: unknown }).closest;
  return typeof closest === "function" ? (target as Element) : null;
}

/** Null/non-element targets are not controls and must read as false. */
export function keyboardTargetBlocksDocumentSelection(
  target: Element | null,
  selector: string,
): boolean {
  return target !== null && target.closest(selector) !== null;
}

/** Both endpoints must belong to the thread; one endpoint outside means the
 * selection crosses the thread boundary and is not a thread selection. */
export function selectionEndpointsInside<T>(
  anchorNode: T | null,
  focusNode: T | null,
  contains: (node: T) => boolean,
): boolean {
  return (
    anchorNode !== null &&
    focusNode !== null &&
    contains(anchorNode) &&
    contains(focusNode)
  );
}

export function shouldTrackKeyboardSelection({
  key,
  shiftKey,
  ctrlOrMetaKey,
  targetBlocksDocumentSelection,
  targetInThread,
  targetIsDocumentBody,
  selectionInThread,
}: KeyboardSelectionInput): boolean {
  if (targetBlocksDocumentSelection || !selectionInThread) {
    return false;
  }

  if (shiftKey && KEYBOARD_SELECTION_EXTEND_KEYS.has(key)) {
    return targetInThread || targetIsDocumentBody;
  }

  // Ctrl/Cmd+A must be rooted in the thread itself, not merely in a stale
  // document selection after focus moved to the page body.
  return ctrlOrMetaKey && key.toLowerCase() === "a" && targetInThread;
}

/** A selectionchange is actionable only during an eligible pointer gesture, or
 * when the last pointer press did not dismiss the toolbar. */
export function selectionChangeAllowed(
  pointerEligible: boolean | null,
  ignoreAfterOutsidePress: boolean,
): boolean {
  if (pointerEligible === false) {
    return false;
  }
  return pointerEligible === true || !ignoreAfterOutsidePress;
}

/** Node/offset identity, not text identity: the same words at another position
 * are a different selection and must reset transient action state. */
export function sameSelectionPosition(
  a: SelectionPosition,
  b: SelectionPosition,
): boolean {
  return (
    a.startContainer === b.startContainer &&
    a.startOffset === b.startOffset &&
    a.endContainer === b.endContainer &&
    a.endOffset === b.endOffset
  );
}

/** Places the fully measured toolbar above the selection when it fits inside
 * the chat viewport, otherwise below it. Returns null when the toolbar cannot
 * fit inside the viewport at all, or when neither side can hold it without
 * covering the selection (for example, a selection taller than the viewport).
 * The caller keeps the toolbar hidden rather than clamping it over the text. */
export function placeToolbar(
  anchor: SelectionRect,
  size: ToolbarSize,
  viewport: ViewportRect,
  margin = 8,
): ToolbarPlacement | null {
  const availableWidth = viewport.width - margin * 2;
  const availableHeight = viewport.height - margin * 2;
  if (
    availableWidth <= 0 ||
    availableHeight <= 0 ||
    size.width > availableWidth ||
    size.height > availableHeight
  ) {
    return null;
  }

  const viewportRight = viewport.left + viewport.width;
  const viewportBottom = viewport.top + viewport.height;
  const maxLeft = viewportRight - margin - size.width;
  const centeredLeft = anchor.left + anchor.width / 2 - size.width / 2;
  const left = Math.min(
    Math.max(viewport.left + margin, centeredLeft),
    maxLeft,
  );

  const minTop = viewport.top + margin;
  const maxBottom = viewportBottom - margin;
  const aboveTop = anchor.top - margin - size.height;
  if (aboveTop >= minTop && aboveTop + size.height <= maxBottom) {
    return { left, top: aboveTop };
  }

  const belowTop = anchor.bottom + margin;
  if (belowTop >= minTop && belowTop + size.height <= maxBottom) {
    return { left, top: belowTop };
  }

  return null;
}

/** A selection's line boxes unioned into one rectangle. Used for placement so
 * the toolbar anchors to the whole selection, not just its last line. */
export function unionSelectionRects(
  rects: readonly SelectionRectLike[],
): SelectionRect | null {
  let left = Number.POSITIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;
  let found = false;

  for (const rect of rects) {
    if (rect.right <= rect.left && rect.bottom <= rect.top) {
      continue;
    }
    found = true;
    left = Math.min(left, rect.left);
    top = Math.min(top, rect.top);
    right = Math.max(right, rect.right);
    bottom = Math.max(bottom, rect.bottom);
  }

  if (!found) {
    return null;
  }
  return {
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  };
}

/** The union of every selected line box, so "above" means above the whole
 * selection rather than above its last line. Falls back to the range's
 * bounding rect when no client rect is usable. */
export function rangeBoundsRect(range: Range): SelectionRect | null {
  try {
    const bounds = unionSelectionRects(Array.from(range.getClientRects()));
    if (bounds !== null) {
      return bounds;
    }
    const fallback = range.getBoundingClientRect();
    return fallback.width > 0 || fallback.height > 0
      ? rectOf(fallback)
      : null;
  } catch {
    return null;
  }
}

export function formatQuoteMarkdown(
  markdown: string,
  attribution: string,
): string {
  const body = markdown.replace(/\r\n/g, "\n").trim();
  const quoted =
    body.length === 0
      ? ">"
      : body
          .split("\n")
          .map((line) => (line.length > 0 ? `> ${line}` : ">"))
          .join("\n");
  return `${attribution}\n${quoted}`;
}

function elementAt(node: Node | null): HTMLElement | null {
  if (node === null) {
    return null;
  }
  if (node.nodeType === 1) {
    return node as HTMLElement;
  }
  return node.parentElement;
}

function assistantContentFor(node: Node | null): HTMLElement | null {
  return (
    elementAt(node)?.closest<HTMLElement>(".aui-assistant-message-content") ??
    null
  );
}

function isInteractiveBoundary(node: Node | null): boolean {
  const element = elementAt(node);
  if (element === null) {
    return true;
  }
  if (element.closest(INTERACTIVE_SELECTOR) !== null) {
    return true;
  }
  if (element.closest(CHROME_SELECTOR) !== null) {
    return true;
  }
  const style = getComputedStyle(element);
  return (
    style.getPropertyValue("user-select") === "none" ||
    style.getPropertyValue("-webkit-user-select") === "none"
  );
}

/** A pointer press that may start a genuine assistant-text selection. */
export function isAssistantSelectionOrigin(node: Node | null): boolean {
  const content = assistantContentFor(node);
  if (content === null) {
    return false;
  }
  if (
    content.closest<HTMLElement>(
      "[data-role='assistant'][data-message-id]",
    ) === null
  ) {
    return false;
  }
  return !isInteractiveBoundary(node);
}

/**
 * Converts the selected DOM range to Markdown. The rendered Markdown is the
 * source of truth here: it is what the user sees selected. Math and Mermaid
 * have no reliable HTML-to-Markdown shape, so they fall back to plain text.
 */
export function selectionMarkdown(
  selection: Selection,
  content: HTMLElement,
): string {
  const range = selection.getRangeAt(0);
  const holder = document.createElement("div");
  holder.append(range.cloneContents());

  const complexSelection =
    holder.querySelector(
      ".katex, mjx-container, [data-streamdown='mermaid-block']",
    ) !== null;
  holder
    .querySelectorAll(QUOTE_PRUNE_SELECTOR)
    .forEach((element) => element.remove());

  let converted = "";
  try {
    converted = htmlNodeToMarkdown(
      holder as unknown as PasteNode,
    ).trim();
  } catch {
    converted = "";
  }

  if (converted.length > 0 && !complexSelection) {
    return converted;
  }

  let fallback = "";
  try {
    fallback = faithfulSelectionText(selection, content).trim();
  } catch {
    fallback = "";
  }
  return fallback || converted;
}

export function buildSelectionSnapshot(
  selection: Selection | null,
  viewport: HTMLElement,
  messages: readonly MessageIdentity[],
): SelectionSnapshot | null {
  if (
    selection === null ||
    selection.isCollapsed ||
    selection.rangeCount !== 1
  ) {
    return null;
  }

  const range = selection.getRangeAt(0);
  if (range.collapsed || !viewport.contains(range.commonAncestorContainer)) {
    return null;
  }

  const anchorContent = assistantContentFor(selection.anchorNode);
  const focusContent = assistantContentFor(selection.focusNode);
  // One assistant response only. Multi-message and cross-role selections keep
  // their native Ctrl+C behavior but get no quote toolbar.
  if (
    anchorContent === null ||
    anchorContent !== focusContent ||
    isInteractiveBoundary(selection.anchorNode) ||
    isInteractiveBoundary(selection.focusNode)
  ) {
    return null;
  }

  const row = anchorContent.closest<HTMLElement>(
    "[data-role='assistant'][data-message-id]",
  );
  const messageId = row?.getAttribute("data-message-id") ?? "";
  if (row === null || messageId.length === 0) {
    return null;
  }

  // Clone before the fallback path temporarily patches/restores the selection.
  const snapshotRange = range.cloneRange();
  const markdown = selectionMarkdown(selection, anchorContent);
  if (markdown.length === 0) {
    return null;
  }

  const rect = rangeBoundsRect(snapshotRange);
  if (rect === null) {
    return null;
  }

  const messageIndex = messages.findIndex(
    (message) => message.id === messageId,
  );
  const turn =
    messageIndex >= 0 ? turnNumberAt(messages, messageIndex) : 0;

  return {
    messageId,
    turn,
    markdown,
    range: snapshotRange,
    rect,
  };
}

/** Hidden panes (Compare keeps the single chat mounted) must not react. */
export function viewportUsable(viewport: HTMLElement | null): viewport is HTMLElement {
  if (viewport === null || viewport.getClientRects().length === 0) {
    return false;
  }
  if (viewport.closest("[inert], [aria-hidden='true']") !== null) {
    return false;
  }
  const style = window.getComputedStyle(viewport);
  return style.display !== "none" && style.visibility !== "hidden";
}
