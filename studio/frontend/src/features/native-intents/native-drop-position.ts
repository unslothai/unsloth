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

// The DOM is measured in CSS pixels, not the logical window ones the monitor
// scale gives, and webview zoom moves the two apart either way. Take
// devicePixelRatio, which is physical per CSS by definition.
function physicalPerCssPx(windowScaleFactor: number): number {
  const ratio = typeof window === "undefined" ? NaN : window.devicePixelRatio;
  if (Number.isFinite(ratio) && ratio > 0) return ratio;
  return Number.isFinite(windowScaleFactor) && windowScaleFactor > 0
    ? windowScaleFactor
    : 1;
}

/** A drop position in the CSS pixels the DOM is measured in. */
export function nativeDropPointToCss(
  position: { x: number; y: number },
  windowScaleFactor: number,
): { x: number; y: number } {
  if (!isPhysicalDropPosition()) return position;
  const scale = physicalPerCssPx(windowScaleFactor);
  return { x: position.x / scale, y: position.y / scale };
}
