// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The OS counts a menu-dismissing press as click 1, so quick follow-ups double/triple-click select text.

const OPEN_FLOATING =
  '[data-radix-popper-content-wrapper] :is([role="menu"],[role="listbox"],[role="dialog"])';

const TEXT_ENTRY =
  'input,textarea,select,[contenteditable]:not([contenteditable="false"])';

type Doc = Pick<
  Document,
  "addEventListener" | "removeEventListener" | "querySelector"
>;

function asElement(target: EventTarget | null): Element | null {
  return target && typeof (target as Element).closest === "function"
    ? (target as Element)
    : null;
}

export function installDismissClickStreakGuard(
  doc: Doc = document,
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
    if (event.detail <= 1) {
      streakFromDismiss = pressDismisses;
    } else if (
      streakFromDismiss &&
      !asElement(event.target)?.closest(TEXT_ENTRY)
    ) {
      event.preventDefault();
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
