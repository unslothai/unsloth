// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAppliedInterfaceZoom } from "../settings/lib/interface-scale-runtime.ts";

// Tauri types every drop position as physical, but wry only produces one on
// WebView2 (ScreenToClient device pixels). macOS reports NSView points and GTK
// reports widget coordinates, and neither of those moves with page zoom, so they
// are CSS pixels only at 100%.
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

/**
 * A drop position in the CSS pixels the DOM is measured in.
 *
 * The two branches read zoom from different places and that asymmetry is forced, not an
 * oversight. **Do not collapse them onto `devicePixelRatio`.** Page zoom is not readable
 * from JS, so macOS and GTK have only the value we last handed the webview; and on macOS
 * `devicePixelRatio` carries a backing-scale factor that NSView points do not have, so
 * using it there would divide by the Retina factor twice.
 */
export function nativeDropPointToCss(
  position: { x: number; y: number },
  windowScaleFactor: number,
  webviewZoom = getAppliedInterfaceZoom(),
): { x: number; y: number } {
  if (!isPhysicalDropPosition()) {
    const scale =
      Number.isFinite(webviewZoom) && webviewZoom > 0 ? webviewZoom : 1;
    return { x: position.x / scale, y: position.y / scale };
  }
  const scale = physicalPerCssPx(windowScaleFactor);
  return { x: position.x / scale, y: position.y / scale };
}
