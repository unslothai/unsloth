// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const FLOATING_ATTRIBUTE = "data-browser-floating";
/** Per frame of a move, so a native page follows (native-view.ts). */
export const FLOATING_MOVE_EVENT = "unsloth-browser-floating-move";
export const FLOATING_SCALE_VAR = "--browser-floating-scale";
/** `browser_view_zoom`'s floor (browser_webview.rs): below it the call fails and the tab shows an error. */
export const NATIVE_MIN_ZOOM = 0.25;

export function floatingNativeZoom(zoom: number, scale: number): number {
  return scale < 1 ? Math.max(NATIVE_MIN_ZOOM, zoom * scale) : zoom;
}
