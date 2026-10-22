// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// In the macOS desktop app, an arrow key that cannot move the caret (left at the start of a field,
// right at its end) is not consumed by the page, and the native fallback types the key's
// private-use character (U+F702, U+F703, ...) into the field (#14764). Marking that one keypress as
// handled leaves the field as it was, which is what Windows and Safari already do.

export type CaretField = {
  value: string;
  selectionStart: number | null;
  selectionEnd: number | null;
};

export type ArrowPress = {
  key: string;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  isComposing?: boolean;
};

/** Whether this arrow would leave the caret where it is. `rtl` swaps left and right, since in a
 * right-to-left field ArrowLeft moves toward the end. Selections and modified arrows always do
 * something (collapse, extend, jump), so they are left to the browser. */
export function arrowCannotMoveCaret(
  press: ArrowPress,
  field: CaretField,
  rtl = false,
): boolean {
  if (press.altKey || press.ctrlKey || press.metaKey || press.shiftKey || press.isComposing) {
    return false;
  }
  const { value, selectionStart, selectionEnd } = field;
  if (selectionStart === null || selectionEnd === null || selectionStart !== selectionEnd) {
    return false;
  }
  const atStart = selectionStart === 0;
  const atEnd = selectionEnd === value.length;
  switch (press.key) {
    case "ArrowLeft":
      return rtl ? atEnd : atStart;
    case "ArrowRight":
      return rtl ? atStart : atEnd;
    case "ArrowUp":
      return atStart;
    case "ArrowDown":
      return atEnd;
    default:
      return false;
  }
}

function caretField(target: EventTarget | null): (CaretField & Element) | null {
  if (!target || typeof (target as Element).tagName !== "string") return null;
  const element = target as Element & Partial<CaretField>;
  const tag = element.tagName.toUpperCase();
  if (tag !== "TEXTAREA" && tag !== "INPUT") return null;
  try {
    // Inputs without a caret (checkbox, number, ...) report null here.
    if (typeof element.selectionStart !== "number") return null;
  } catch {
    return null;
  }
  return element as CaretField & Element;
}

/** Bubble phase on the window, so the app's own arrow handlers run first and keep their say. */
export function watchMacArrowBoundary(win: Window): void {
  win.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) return;
    const field = caretField(event.target);
    if (!field) return;
    const rtl = win.getComputedStyle(field).direction === "rtl";
    if (arrowCannotMoveCaret(event, field, rtl)) {
      event.preventDefault();
    }
  });
}
