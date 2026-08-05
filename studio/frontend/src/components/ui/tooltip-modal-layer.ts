// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Radix's DismissableLayer sets pointer-events:none on the body while a modal
 * is up, and pointer-events:auto on the active layer. A trigger inside that
 * layer therefore computes "auto" and one outside computes "none", which is
 * what says whether a tooltip belongs to the modal.
 *
 * Read the trigger, never the tooltip content: layers rank by mount order, so
 * content opened after the modal reads "auto" wherever its trigger sits.
 */
export function isBlockedByActiveModal(element: HTMLElement): boolean {
  return (
    element.ownerDocument.defaultView?.getComputedStyle(element)
      .pointerEvents === "none"
  );
}
