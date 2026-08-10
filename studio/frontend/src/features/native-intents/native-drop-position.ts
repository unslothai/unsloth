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

// Windows text scaling and page zoom put devicePixelRatio above the monitor
// scale, and the DOM is measured in CSS pixels, not logical window ones. Same
// distinction app/provider.tsx draws with `devicePixelRatio / monitorScale`.
function physicalPerCssPx(windowScaleFactor: number): number {
  const scale =
    Number.isFinite(windowScaleFactor) && windowScaleFactor > 0
      ? windowScaleFactor
      : 1;
  if (typeof window === "undefined") return scale;
  const ratio = window.devicePixelRatio;
  return Number.isFinite(ratio) && ratio > scale ? ratio : scale;
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
