// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isSurfaceBackgrounded } from "@/features/settings/hooks/use-shortcut";
import { hasTrackedSurface, MODAL_SURFACE } from "@/lib/surface-registry";
import { FIND_SCOPE_ATTRIBUTE } from "./find-attributes.ts";

export const MODAL_BACKDROP_SELECTOR = [
  ...["dialog-overlay", "alert-dialog-overlay", "sheet-overlay"].map(
    (slot) => `[data-slot="${slot}"]:not([data-state="closed"])`,
  ),
  '[aria-modal="true"]',
  "[data-blocking-screen]",
].join(", ");

/** Radix never aria-hides ancestors of an aria-live region, so check the backdrop too. */
export function isFindScopeBackgrounded(): boolean {
  if (typeof document === "undefined") return false;
  return (
    hasTrackedSurface(MODAL_SURFACE, MODAL_BACKDROP_SELECTOR) ||
    isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`)
  );
}
