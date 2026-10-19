// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The floating orb window: one always-on-top ball that stands in for the hidden main window. */

export const ORB_WIDGET_LABEL = "orb-widget";
export const ORB_WIDGET_QUERY = "orb-widget";
export const ORB_WIDGET_WIDTH = 132;
export const ORB_WIDGET_HEIGHT = 168;
export const ORB_WIDGET_LONG_PRESS_MS = 500;

/** main -> widget: the orb's live state. */
export const ORB_WIDGET_STATE_EVENT = "orb-widget:state";
/** widget -> main: asks for a state push once its listener is up. */
export const ORB_WIDGET_READY_EVENT = "orb-widget:ready";
/** widget -> main: a control-bar press. */
export const ORB_WIDGET_ACTION_EVENT = "orb-widget:action";

export const ORB_WIDGET_POSITION_KEY = "unsloth.orb-widget.position.v1";

export type OrbWidgetAction = "toggle-pause" | "stop" | "back";

export interface OrbWidgetSnapshot {
  orbState: string | null;
  paused: boolean;
  muted: boolean;
  active: boolean;
}

export interface Point {
  x: number;
  y: number;
}

/** A monitor's work area in logical pixels. */
export interface Area {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function isOrbWidgetAction(value: unknown): value is OrbWidgetAction {
  return value === "toggle-pause" || value === "stop" || value === "back";
}

export function parseSavedPosition(raw: string | null): Point | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<Point>;
    if (Number.isFinite(value.x) && Number.isFinite(value.y)) {
      return { x: Math.round(value.x as number), y: Math.round(value.y as number) };
    }
  } catch {
    // Corrupt entry: fall back to the default corner.
  }
  return null;
}

/**
 * Where to open the widget: the saved spot while most of the ball is still on a
 * connected monitor (a display can be unplugged between uses), else the bottom
 * right of the fallback monitor.
 */
export function placeOrbWidget(
  saved: Point | null,
  monitors: readonly Area[],
  fallback: Area | null,
  size: { width: number; height: number } = { width: ORB_WIDGET_WIDTH, height: ORB_WIDGET_HEIGHT },
): Point | null {
  if (saved) {
    const cx = saved.x + size.width / 2;
    const cy = saved.y + size.height / 2;
    const visible = monitors.some(
      (m) => cx >= m.x && cx < m.x + m.width && cy >= m.y && cy < m.y + m.height,
    );
    if (visible) return saved;
  }
  if (!fallback) return null;
  return {
    x: Math.round(fallback.x + fallback.width - size.width - 24),
    y: Math.round(fallback.y + fallback.height - size.height - 72),
  };
}
