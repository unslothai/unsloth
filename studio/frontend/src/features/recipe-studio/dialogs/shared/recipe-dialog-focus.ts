// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type AutoFocusEvent = {
  defaultPrevented: boolean;
  preventDefault(): void;
  currentTarget: EventTarget | null;
};

/** Radix focuses the first tabbable on open, a FieldLabel hint whose tooltip opens on focus, so
 *  focus the dialog itself; a caller's handler runs first and wins if it prevents the default. */
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
