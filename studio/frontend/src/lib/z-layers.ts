// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The stacking order of Studio's full-viewport surfaces, in one place.
 *
 * Everything here is a `position: fixed` surface that covers whatever is under
 * it, so the only thing deciding who wins is the number. Written down together
 * because these numbers are meaningless apart: raising one file's z-index to
 * fix one screenshot is what produced the 9998/9999 pair this replaces.
 *
 * Ordered bottom to top:
 *
 *   OVERLAY_STACK        passive corner status, nothing to click through to
 *   FLOATING_PANEL       windows the user drags, resizes and closes
 *   FLOATING_PANEL_TOP   the one of those the user touched last
 *   STARTUP_SCREEN       blocks the app while the backend comes up or quits
 *   TOOLTIP              transient, must be readable above whatever spawned it
 *
 * In-page surfaces -- dropdowns, popovers, dialogs, sheets, the sidebar, the
 * Tauri titlebar -- all sit in the 1..120 band on Tailwind's own `z-*` scale
 * and are deliberately not renumbered here: nothing in this file competes with
 * them for a number, so folding them in would change what paints over what
 * without fixing anything. See tests/studio/test_overlay_layering.py.
 */
export const Z_LAYER = {
  /**
   * The bottom-right notification stack: update banners, the download panel,
   * the loaded models card. Passive; it already keeps clear of the panels
   * below it geometrically (see monitor-frame-store), and where it cannot, it
   * loses.
   */
  OVERLAY_STACK: 9000,
  /**
   * Floating panels: the Live resource monitor and the API monitor overlay.
   * Above the stack because a window the user is dragging, resizing and
   * closing outranks a status card that happened to land on top of it.
   */
  FLOATING_PANEL: 9100,
  /**
   * The floating panel the user touched last. Only ever one panel at a time,
   * so a single step above FLOATING_PANEL is enough; see floating-panel-order.
   */
  FLOATING_PANEL_TOP: 9101,
  /**
   * The startup and closing screens, which stand in for the whole app while
   * the backend comes up or shuts down. Nothing below may show through.
   */
  STARTUP_SCREEN: 9999,
  /** Tooltips, which have to be legible above whatever spawned them. */
  TOOLTIP: 999999,
} as const;

export type ZLayer = (typeof Z_LAYER)[keyof typeof Z_LAYER];
