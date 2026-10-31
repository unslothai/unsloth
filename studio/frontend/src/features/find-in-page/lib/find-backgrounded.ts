// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isSurfaceBackgrounded } from "@/features/settings";
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
    document.querySelector(MODAL_BACKDROP_SELECTOR) !== null ||
    isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`)
  );
}
