// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** On the floating preview's surface, as chat-page.tsx renders it. */
export const FLOATING_ATTRIBUTE = "data-browser-floating";
/** Fired on window per frame while the preview moves: a native page under it follows (native-view.ts). */
export const FLOATING_MOVE_EVENT = "unsloth-browser-floating-move";
/** On that surface: the page's scale, its side-pane layout width shown in the preview's width. */
export const FLOATING_SCALE_VAR = "--browser-floating-scale";
/** `browser_view_zoom`'s floor (browser_webview.rs): below it the call fails and the tab shows an error. */
export const NATIVE_MIN_ZOOM = 0.25;

/** A native page's zoom in the floating window: its own zoom scaled down with the window, never
 *  under the native floor, so a page zoomed out before floating comes out a little larger instead. */
export function floatingNativeZoom(zoom: number, scale: number): number {
  return scale < 1 ? Math.max(NATIVE_MIN_ZOOM, zoom * scale) : zoom;
}
