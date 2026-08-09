// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tauri types every drop position as physical, but wry only produces one on
// WebView2 (ScreenToClient device pixels). macOS reports NSView points and GTK
// reports widget coordinates, both of which are already CSS pixels.
function isPhysicalDropPosition(): boolean {
  return (
    typeof navigator !== "undefined" && navigator.userAgent.includes("Windows")
  );
}

/** A drop position in the CSS pixels the DOM is measured in. */
export function nativeDropPointToCss(
  position: { x: number; y: number },
  windowScaleFactor: number,
): { x: number; y: number } {
  if (!isPhysicalDropPosition()) return position;
  const scale =
    Number.isFinite(windowScaleFactor) && windowScaleFactor > 0
      ? windowScaleFactor
      : 1;
  return { x: position.x / scale, y: position.y / scale };
}
