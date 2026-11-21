// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Stacking order of Unsloth's full-viewport fixed surfaces, bottom to top. These numbers only
 * mean something relative to each other, so change them here, together. In-page surfaces stay on
 * Tailwind's 1..120 z-scale. See tests/studio/test_overlay_layering.py.
 */
export const Z_LAYER = {
  /** Bottom-right notification stack; passive, so it loses to the panels above it. */
  OVERLAY_STACK: 9000,
  /**
   * Window-resize grips (all eight), above the notification stack, which can reach every edge
   * in a narrow window and is pointer-active over its whole box.
   */
  WINDOW_RESIZE_EDGE: 9050,
  /** Draggable floating panels (resource monitor, API monitor). */
  FLOATING_PANEL: 9100,
  /** The most recently touched floating panel; only one at a time, so +1 suffices. */
  FLOATING_PANEL_TOP: 9101,
  /** Startup/closing screens; nothing below may show through. */
  STARTUP_SCREEN: 9999,
  TOOLTIP: 999999,
  /**
   * Transparent sheet PanelResizeHandle mounts during a drag; it must own the cursor and hit test
   * for the whole viewport, so it sits above everything it covers.
   */
  DRAG_CURSOR_OVERLAY: 1000000,
  /** Find bar. Above toasts (999999999), below only the reload snapshot. */
  WINDOW_BARS: 2147483000,
  ZOOM_POPUP: 2147483001,
} as const;

export type ZLayer = (typeof Z_LAYER)[keyof typeof Z_LAYER];
