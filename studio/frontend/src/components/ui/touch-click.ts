// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";

/** The click's own pointerType, not the media query, which mislabels hybrids. Keyboard gives "". */
export function isTouchClick(event: React.MouseEvent): boolean {
  const pointerType = (event.nativeEvent as Partial<PointerEvent>).pointerType;
  if (typeof pointerType === "string") return pointerType === "touch";
  // No PointerEvent (older WebViews): fall back to the device class.
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}
