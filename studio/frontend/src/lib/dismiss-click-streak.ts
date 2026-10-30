// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The OS counts a menu-dismissing press as click 1; later presses select as if it had not (caret, word, block).

const OPEN_FLOATING =
  '[data-radix-popper-content-wrapper] :is([role="menu"],[role="listbox"],[role="dialog"])';

const TEXT_ENTRY =
  'input,textarea,select,[contenteditable]:not([contenteditable="false"])';

type Doc = Pick<
  Document,
  "addEventListener" | "removeEventListener" | "querySelector"
>;

export type ShiftedSelection = "caret" | "word" | "paragraph";

export function shiftedSelection(detail: number): ShiftedSelection {
  if (detail <= 2) return "caret";
  return detail === 3 ? "word" : "paragraph";
}

function asElement(target: EventTarget | null): Element | null {
  return target && typeof (target as Element).closest === "function"
    ? (target as Element)
    : null;
}

type CaretDoc = Document & {
  caretPositionFromPoint?: (
    x: number,
    y: number,
  ) => { offsetNode: Node; offset: number } | null;
};

function caretRangeAt(x: number, y: number): Range | null {
  const doc = document as CaretDoc;
  if (typeof doc.caretRangeFromPoint === "function") {
    return doc.caretRangeFromPoint(x, y);
  }
  const position = doc.caretPositionFromPoint?.(x, y);
  if (!position) return null;
  const range = doc.createRange();
  range.setStart(position.offsetNode, position.offset);
  range.collapse(true);
  return range;
}

type Extend = (x: number, y: number) => void;

function unitRangeAt(
  x: number,
  y: number,
  detail: number,
  block: Element,
): Range | null {
  const range = caretRangeAt(x, y);
  if (!range) return null;
  const unit = shiftedSelection(detail);
  if (unit === "word") expandToWord(range);
  if (unit === "paragraph") expandToBlock(range, block);
  return range;
}

function selectAt(event: MouseEvent, target: Element | null): Extend | null {
  const selection = window.getSelection();
  if (!selection) return null;
  selection.removeAllRanges();
  const anchor =
    target && unitRangeAt(event.clientX, event.clientY, event.detail, target);
  // Native selection never starts in unselectable text or outside the pressed element.
  if (
    !anchor ||
    !target?.contains(anchor.startContainer) ||
    getComputedStyle(target).userSelect === "none"
  ) {
    return null;
  }
  selection.addRange(anchor);
  return (x, y) => {
    const focus = unitRangeAt(x, y, event.detail, document.body);
    if (!focus) return;
    if (focus.compareBoundaryPoints(Range.START_TO_START, anchor) < 0) {
      selection.setBaseAndExtent(
        anchor.endContainer,
        anchor.endOffset,
        focus.startContainer,
        focus.startOffset,
      );
    } else if (focus.compareBoundaryPoints(Range.END_TO_END, anchor) > 0) {
      selection.setBaseAndExtent(
        anchor.startContainer,
        anchor.startOffset,
        focus.endContainer,
        focus.endOffset,
      );
    } else {
      selection.removeAllRanges();
      selection.addRange(anchor);
    }
  };
}

/** Segmenter, not `Selection.modify`: its word steps differ by platform. */
export function wordBounds(text: string, offset: number): [number, number] {
  if (!text) return [0, 0];
  const at = Math.min(offset, text.length - 1);
  const segment = new Intl.Segmenter(undefined, { granularity: "word" })
    .segment(text)
    .containing(at);
  return segment
    ? [segment.index, segment.index + segment.segment.length]
    : [at, at];
}

function expandToWord(range: Range): void {
  const node = range.startContainer;
  if (node.nodeType !== Node.TEXT_NODE) return;
  const [start, end] = wordBounds(node.textContent ?? "", range.startOffset);
  range.setStart(node, start);
  range.setEnd(node, end);
}

function expandToBlock(range: Range, target: Element): void {
  let block: Element | null =
    range.startContainer instanceof Element
      ? range.startContainer
      : range.startContainer.parentElement;
  while (
    block &&
    block !== target &&
    getComputedStyle(block).display.startsWith("inline")
  ) {
    block = block.parentElement;
  }
  range.selectNodeContents(block ?? target);
}

export function installDismissClickStreakGuard(
  doc: Doc = document,
  select: (
    event: MouseEvent,
    target: Element | null,
  ) => Extend | null = selectAt,
): () => void {
  let pressDismisses = false;
  let streakFromDismiss = false;
  let extend: Extend | null = null;

  const onPointerDown = (event: PointerEvent): void => {
    // Capture phase: runs before Radix closes the menu.
    const target = asElement(event.target);
    pressDismisses =
      event.button === 0 &&
      doc.querySelector(OPEN_FLOATING) !== null &&
      !target?.closest("[data-radix-popper-content-wrapper]");
  };

  const onMouseDown = (event: MouseEvent): void => {
    if (event.button !== 0) return;
    extend = null;
    const target = asElement(event.target);
    if (event.detail <= 1) {
      streakFromDismiss = pressDismisses;
    } else if (streakFromDismiss && !target?.closest(TEXT_ENTRY)) {
      event.preventDefault();
      extend = select(event, target);
    }
    pressDismisses = false;
  };

  const onMouseMove = (event: MouseEvent): void => {
    if (!extend) return;
    if (event.buttons & 1) extend(event.clientX, event.clientY);
    else extend = null;
  };

  const onMouseUp = (): void => {
    extend = null;
  };

  const listeners: [string, EventListener][] = [
    ["pointerdown", onPointerDown as EventListener],
    ["mousedown", onMouseDown as EventListener],
    ["mousemove", onMouseMove as EventListener],
    ["mouseup", onMouseUp as EventListener],
  ];
  for (const [type, fn] of listeners) doc.addEventListener(type, fn, true);
  return () => {
    for (const [type, fn] of listeners) doc.removeEventListener(type, fn, true);
  };
}
