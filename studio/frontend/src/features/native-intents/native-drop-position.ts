// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAppliedInterfaceZoom } from "../settings/lib/interface-scale-runtime.ts";

// Only WebView2 produces physical positions; macOS and GTK report zoom-independent coordinates.
function isPhysicalDropPosition(): boolean {
  return (
    typeof navigator !== "undefined" && navigator.userAgent.includes("Windows")
  );
}

// devicePixelRatio is physical per CSS pixel, which the DOM uses; monitor scale ignores zoom.
function physicalPerCssPx(windowScaleFactor: number): number {
  const ratio = typeof window === "undefined" ? NaN : window.devicePixelRatio;
  if (Number.isFinite(ratio) && ratio > 0) return ratio;
  return Number.isFinite(windowScaleFactor) && windowScaleFactor > 0
    ? windowScaleFactor
    : 1;
}

/**
 * Drop position in CSS pixels. Do not collapse both branches onto `devicePixelRatio`: on macOS it
 * includes the Retina backing scale that NSView points lack, so it would divide twice.
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
