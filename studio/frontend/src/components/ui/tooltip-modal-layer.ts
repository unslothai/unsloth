// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Radix's DismissableLayer sets pointer-events:none on layers below the active
 * modal and pointer-events:auto on layers that belong to it. Reading the
 * tooltip layer itself avoids guessing which dialog, sheet, or menu owns its
 * trigger.
 */
export function isTooltipLayerBlocked(element: HTMLElement): boolean {
  return (
    element.ownerDocument.defaultView?.getComputedStyle(element).pointerEvents ===
    "none"
  );
}
