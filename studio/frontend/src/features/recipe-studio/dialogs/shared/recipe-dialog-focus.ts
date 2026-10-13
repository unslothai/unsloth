// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type AutoFocusEvent = {
  defaultPrevented: boolean;
  preventDefault(): void;
  currentTarget: EventTarget | null;
};

/** Radix focuses a dialog's first tabbable element on open. In recipe dialogs that is a FieldLabel
 *  hint button, and its tooltip opens on focus, so a hint pops up unasked. Focus the dialog itself
 *  instead: Tab still reaches every hint, and nothing opens until the user moves. A caller's own
 *  handler runs first and wins if it picks a target. */
export function focusRecipeDialogOnOpen<E extends AutoFocusEvent>(
  event: E,
  onOpenAutoFocus?: (event: E) => void,
): void {
  onOpenAutoFocus?.(event);
  if (event.defaultPrevented) {
    return;
  }
  event.preventDefault();
  (event.currentTarget as HTMLElement | null)?.focus({ preventScroll: true });
}
