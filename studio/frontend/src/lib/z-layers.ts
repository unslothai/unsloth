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
 *   WINDOW_RESIZE_EDGE   the window's own bottom resize grips, under that stack
 *   FLOATING_PANEL       windows the user drags, resizes and closes
 *   FLOATING_PANEL_TOP   the one of those the user touched last
 *   STARTUP_SCREEN       blocks the app while the backend comes up or quits
 *   TOOLTIP              transient, must be readable above whatever spawned it
 *   DRAG_CURSOR_OVERLAY  owns the cursor and the hit test during a panel drag
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
   * The custom titlebar's window-resize targets along the bottom edge, which
   * the notification stack reaches: `right-4` holds its margin edge 16px in,
   * but `-mx-3` puts its border box 4px from the window edge, and the shadow
   * gutter drops that box the last 16px to the floor. A scrolling stack is
   * pointer-active over the whole box, so without a number here it takes two
   * thirds of the corner grip and the strip along the bottom.
   *
   * Only the bottom two. The other six cannot be reached by a stack anchored
   * to that corner, and raising them is not free: the north-east grip sits on
   * the window controls, which are `z-[80]` and win today.
   */
  WINDOW_RESIZE_EDGE: 9050,
  /**
   * The custom titlebar's window controls, which the north-east grip's 12x12 corner lands
   * on: `right-1` plus `px-1` leaves Close's right edge 8px in, and a 30px button centred
   * in the band puts its top a couple of px down. They were `z-[80]`, which beat the grips
   * at `z-[70]` and lost to the stack at 9000; both of those were the wrong way round, so
   * they move up with the grips rather than being left behind by them.
   */
  WINDOW_CONTROLS: 9060,
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
  /**
   * The transparent sheet PanelResizeHandle mounts for the life of a panel
   * drag. Top of the scale because it owns two things for the whole viewport
   * while it exists: the resize cursor, and the hit test. It replaces
   * `html[data-panel-resizing] *`, which was `!important` and so beat every
   * surface here, and the `pointer-events: none` rule that used to blank the
   * sidebar and the main content; being above them is what makes it their
   * equal rather than a partial stand-in. Transparent, and it exists only
   * between pointerdown and pointerup, so nothing it covers is hidden.
   */
  DRAG_CURSOR_OVERLAY: 1000000,
} as const;

export type ZLayer = (typeof Z_LAYER)[keyof typeof Z_LAYER];
