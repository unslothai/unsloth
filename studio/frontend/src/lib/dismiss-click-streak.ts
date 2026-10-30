// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The OS counts a menu-dismissing press as click 1; later presses in that run select one step coarser
// than the user meant, so they select as if it had not counted (2 = caret, 3 = word, 4+ = paragraph).

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

function selectAt(event: MouseEvent, target: Element | null): void {
  const selection = window.getSelection();
  if (!selection) return;
  const range = caretRangeAt(event.clientX, event.clientY);
  selection.removeAllRanges();
  // Native selection never starts in unselectable text or jumps to text outside the pressed element.
  if (
    !range ||
    !target?.contains(range.startContainer) ||
    getComputedStyle(target).userSelect === "none"
  ) {
    return;
  }
  const unit = shiftedSelection(event.detail);
  if (unit === "word") expandToWord(range);
  if (unit === "paragraph") expandToBlock(range, target);
  selection.addRange(range);
}

/** `Selection.modify` word steps differ by platform (macOS stops at a word's end); segment instead. */
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
  select: (event: MouseEvent, target: Element | null) => void = selectAt,
): () => void {
  let pressDismisses = false;
  let streakFromDismiss = false;

  const onPointerDown = (event: PointerEvent): void => {
    // Capture: runs before Radix closes the menu on this press.
    const target = asElement(event.target);
    pressDismisses =
      event.button === 0 &&
      doc.querySelector(OPEN_FLOATING) !== null &&
      !target?.closest("[data-radix-popper-content-wrapper]");
  };

  const onMouseDown = (event: MouseEvent): void => {
    if (event.button !== 0) return;
    const target = asElement(event.target);
    if (event.detail <= 1) {
      streakFromDismiss = pressDismisses;
    } else if (streakFromDismiss && !target?.closest(TEXT_ENTRY)) {
      event.preventDefault();
      select(event, target);
    }
    pressDismisses = false;
  };

  doc.addEventListener("pointerdown", onPointerDown as EventListener, true);
  doc.addEventListener("mousedown", onMouseDown as EventListener, true);
  return () => {
    doc.removeEventListener(
      "pointerdown",
      onPointerDown as EventListener,
      true,
    );
    doc.removeEventListener("mousedown", onMouseDown as EventListener, true);
  };
}
