// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The floating orb window: one always-on-top ball that stands in for the hidden main window. */

export const ORB_WIDGET_LABEL = "orb-widget";
export const ORB_WIDGET_QUERY = "orb-widget";
export const ORB_WIDGET_BALL = 72;
// voice-orb.tsx, at its own 140 px: the spectrum edge reaches 70 + 3 + 16; the glow is a
// 70 px disc with an 8 px spread and a 32 px blur, whose Gaussian tail is still faintly
// visible at 1.5x the blur; the biggest pulse (speaking) scales both by 1.1.
const K = ORB_WIDGET_BALL / 140;
export const ORB_WIDGET_PEAK_SCALE = 1.1;
export const ORB_WIDGET_EDGE_REACH = (70 + 3 + 16) * K * ORB_WIDGET_PEAK_SCALE;
export const ORB_WIDGET_GLOW_REACH = (70 + 8 + 1.5 * 32) * K * ORB_WIDGET_PEAK_SCALE;
/** Clear space past the furthest pixel, so nothing is ever cut at the window's edge. */
export const ORB_WIDGET_MARGIN = 24;
const HALF = Math.ceil(Math.max(ORB_WIDGET_EDGE_REACH, ORB_WIDGET_GLOW_REACH) + ORB_WIDGET_MARGIN);
/** The control bar's height: 28 px buttons, 2 px padding and a 1 px border each way. */
export const ORB_WIDGET_BAR_HEIGHT = 34;
/** The bar's top: low enough that only the ball's pulse and wave cross it, drawn in front. */
export const ORB_WIDGET_BAR_TOP = HALF + 43;
export const ORB_WIDGET_WIDTH = 2 * HALF;
export const ORB_WIDGET_HEIGHT = Math.max(2 * HALF, ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT + 8);
/** The ball's centre inside the window. */
export const ORB_WIDGET_CENTER = { x: HALF, y: HALF };
export const ORB_WIDGET_LONG_PRESS_MS = 500;

/** main -> widget: the orb's live state. */
export const ORB_WIDGET_STATE_EVENT = "orb-widget:state";
/** widget -> main: asks for a state push once its listener is up. */
export const ORB_WIDGET_READY_EVENT = "orb-widget:ready";
/** main -> widget: live spectrum bands for the ball's edge, while one side is talking. */
export const ORB_WIDGET_SPECTRUM_EVENT = "orb-widget:spectrum";
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

/** A monitor rectangle (whole screen or work area) in logical pixels. */
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

function centreOn(p: Point, m: Area, size: { width: number; height: number }): boolean {
  const cx = p.x + size.width / 2;
  const cy = p.y + size.height / 2;
  return cx >= m.x && cx < m.x + m.width && cy >= m.y && cy < m.y + m.height;
}

const WIDGET_SIZE = { width: ORB_WIDGET_WIDTH, height: ORB_WIDGET_HEIGHT };

/**
 * Where to open the widget: the saved spot while its centre is still on a
 * connected screen (a display can be unplugged between uses), else the bottom
 * right of the fallback work area.
 */
export function placeOrbWidget(
  saved: Point | null,
  screens: readonly Area[],
  fallback: Area | null,
  size: { width: number; height: number } = WIDGET_SIZE,
): Point | null {
  if (saved && screens.some((m) => centreOn(saved, m, size))) return saved;
  if (!fallback) return null;
  return {
    x: Math.round(fallback.x + fallback.width - size.width - 24),
    y: Math.round(fallback.y + fallback.height - size.height - 72),
  };
}

/**
 * A dragged widget may go anywhere, menu bar included, so long as the ball's centre
 * stays on some screen; past every edge it is pulled back onto the nearest one.
 */
export function clampToScreens(
  p: Point,
  screens: readonly Area[],
  center: Point = ORB_WIDGET_CENTER,
): Point {
  if (!screens.length) return p;
  const cx = p.x + center.x;
  const cy = p.y + center.y;
  let best: Point = p;
  let bestDist = Infinity;
  for (const m of screens) {
    const x = Math.min(Math.max(cx, m.x), m.x + m.width - 1);
    const y = Math.min(Math.max(cy, m.y), m.y + m.height - 1);
    const d = (x - cx) ** 2 + (y - cy) ** 2;
    if (d < bestDist) {
      bestDist = d;
      best = { x: x - center.x, y: y - center.y };
    }
    if (d === 0) break;
  }
  return best;
}

export interface SpectrumFrame {
  side: "input" | "output";
  level: number;
  bands: number[];
}

/** Two decimals is finer than the edge can show, and keeps 30 frames a second small. */
export function packSpectrum(side: "input" | "output", level: number, bands: ArrayLike<number>): SpectrumFrame {
  const out: number[] = new Array(bands.length);
  for (let i = 0; i < bands.length; i++) out[i] = Math.round(bands[i] * 100) / 100;
  return { side, level: Math.round(level * 1000) / 1000, bands: out };
}

/** Copies a received frame into `out`; anything malformed reads as silence. */
export function unpackSpectrum(frame: unknown, out: Float32Array): number {
  const f = frame as Partial<SpectrumFrame> | null;
  if (!f || !Array.isArray(f.bands) || typeof f.level !== "number" || f.bands.length !== out.length) {
    out.fill(0);
    return 0;
  }
  for (let i = 0; i < out.length; i++) {
    const v = Number(f.bands[i]);
    out[i] = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
  }
  return Math.min(1, Math.max(0, f.level));
}

/**
 * Opens the widget with the ball centred on the cursor, pulled in so the whole ball
 * stays on the cursor's screen. Null when the cursor is on no known screen.
 */
export function placeAtCursor(cursor: Point, screens: readonly Area[]): Point | null {
  const m = screens.find(
    (a) => cursor.x >= a.x && cursor.x < a.x + a.width && cursor.y >= a.y && cursor.y < a.y + a.height,
  );
  if (!m) return null;
  const r = ORB_WIDGET_BALL / 2;
  const cx = Math.min(Math.max(cursor.x, m.x + r), m.x + m.width - r);
  const cy = Math.min(Math.max(cursor.y, m.y + r), m.y + m.height - r);
  return { x: Math.round(cx - ORB_WIDGET_CENTER.x), y: Math.round(cy - ORB_WIDGET_CENTER.y) };
}

/** A press has to travel this far before it moves the window, so a click never does. */
export const ORB_WIDGET_DRAG_THRESHOLD = 4;

export interface DragStart {
  /** The pointer at press, in screen points. */
  screen: Point;
  /** Where in the window it was grabbed: the pointer's offset from the window's top left. */
  grab: Point;
}

/**
 * The window's new top left for a pointer at `now` (screen points, the same logical
 * units setPosition takes), or null while the press is still inside the click slop.
 */
export function dragTarget(start: DragStart, now: Point, moving: boolean): Point | null {
  if (!moving && Math.hypot(now.x - start.screen.x, now.y - start.screen.y) <= ORB_WIDGET_DRAG_THRESHOLD) {
    return null;
  }
  return { x: Math.round(now.x - start.grab.x), y: Math.round(now.y - start.grab.y) };
}
