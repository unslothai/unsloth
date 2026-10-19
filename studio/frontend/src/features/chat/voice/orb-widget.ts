// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The floating orb window: one always-on-top ball that stands in for the hidden main window. */

export const ORB_WIDGET_LABEL = "orb-widget";
export const ORB_WIDGET_QUERY = "orb-widget";
export const ORB_WIDGET_BALL = 72;
// voice-orb.tsx, at its own 140 px: the spectrum edge reaches 70 + 3 + 22.4 (BLOB_AMP_PX); the glow is a
// 70 px disc with an 8 px spread and a 32 px blur, whose Gaussian tail is still faintly
// visible at 1.5x the blur; the biggest pulse (speaking) scales both by 1.1.
const K = ORB_WIDGET_BALL / 140;
export const ORB_WIDGET_PEAK_SCALE = 1.1;
export const ORB_WIDGET_EDGE_REACH = (70 + 3 + 22.4) * K * ORB_WIDGET_PEAK_SCALE;
export const ORB_WIDGET_GLOW_REACH = (70 + 8 + 1.5 * 32) * K * ORB_WIDGET_PEAK_SCALE;
/** Clear space past the furthest pixel, so nothing is ever cut at the window's edge. */
export const ORB_WIDGET_MARGIN = 24;
const HALF = Math.ceil(Math.max(ORB_WIDGET_EDGE_REACH, ORB_WIDGET_GLOW_REACH) + ORB_WIDGET_MARGIN);
/** The control bar's height: 28 px buttons, 2 px padding and a 1 px border each way. */
export const ORB_WIDGET_BAR_HEIGHT = 34;
/** Between the ball's bottom and the bar's top. */
export const ORB_WIDGET_BAR_GAP = 7;
/** Just under the ball. The edge is the in-app orb's, unflattened, and may cross the bar, drawn in front. */
export const ORB_WIDGET_BAR_TOP = HALF + ORB_WIDGET_BALL / 2 + ORB_WIDGET_BAR_GAP;
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
/** widget -> main: Allow or Deny pressed on the approval popup. */
export const ORB_WIDGET_APPROVAL_EVENT = "orb-widget:approval";

export const ORB_WIDGET_POSITION_KEY = "unsloth.orb-widget.position.v1";

export type OrbWidgetAction = "toggle-pause" | "stop" | "back";

/** A tool call waiting for Allow / Deny, worded by the main window (the widget has no locale). */
export interface OrbWidgetApproval {
  id: string;
  /** The full "Using tool: ..." line, for the tooltip and screen readers. */
  line: string;
  /** What the card shows: the tool's name and a short query, e.g. web_search · "who is ...". */
  brief: string;
  /** "+N more" when others are queued behind it. */
  more: string | null;
  allow: string;
  deny: string;
}

export interface OrbWidgetSnapshot {
  orbState: string | null;
  paused: boolean;
  muted: boolean;
  active: boolean;
  approval: OrbWidgetApproval | null;
}

export interface OrbWidgetDecision {
  id: string;
  decision: "allow" | "deny";
}

export function isOrbWidgetDecision(value: unknown): value is OrbWidgetDecision {
  const v = value as Partial<OrbWidgetDecision> | null;
  return (
    typeof v === "object" &&
    v !== null &&
    typeof v.id === "string" &&
    v.id.length > 0 &&
    (v.decision === "allow" || v.decision === "deny")
  );
}

/**
 * The popup's subject: the first pending call (insertion order, the order they were asked)
 * and how many wait behind it. Null when nothing is waiting.
 */
export function pendingApproval(confirmations: Readonly<Record<string, unknown>>): { id: string; more: number } | null {
  const ids = Object.keys(confirmations);
  return ids.length ? { id: ids[0], more: ids.length - 1 } : null;
}

export function sameApproval(a: OrbWidgetApproval | null, b: OrbWidgetApproval | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.id === b.id &&
    a.line === b.line &&
    a.brief === b.brief &&
    a.more === b.more &&
    a.allow === b.allow &&
    a.deny === b.deny
  );
}

/** The approval card: one line of text over a row of small pills. */
export const ORB_WIDGET_POPUP_WIDTH = 156;
export const ORB_WIDGET_POPUP_CARD_HEIGHT = 54;
const POPUP_GAP = 6;
/** The card's top inside the window when it sits under the bar. */
export const ORB_WIDGET_POPUP_BELOW_TOP = ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT + POPUP_GAP;
/** The window's height while the card shows, and how much that adds. */
export const ORB_WIDGET_GROWN_HEIGHT = Math.max(
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_POPUP_BELOW_TOP + ORB_WIDGET_POPUP_CARD_HEIGHT + POPUP_GAP,
);
export const ORB_WIDGET_POPUP_HEIGHT = ORB_WIDGET_GROWN_HEIGHT - ORB_WIDGET_HEIGHT;
/** Room kept between the card and the edge of the screen's visible frame. */
const SCREEN_MARGIN = 8;

export interface PopupPlacement {
  /** The window's new top left. */
  at: Point;
  /** The card sits above the ball: the page is pushed down by `pad` (logical px). */
  above: boolean;
  pad: number;
  /** The card's left edge inside the window (logical px). */
  popupX: number;
  /** The card's top inside the window (logical px). */
  popupY: number;
}

/**
 * Where the window and the card go while an approval waits. Works in whatever units the
 * position and the screens come in (physical pixels in the app), with `scale` the factor
 * from logical pixels to those. `screens` are the monitors' visible frames.
 *
 * The card goes under the bar, or above the ball when that would leave the bottom of
 * the frame, and shifts sideways (off centre if it must) to stay inside it. The window
 * moves only as far as needed to hold the card, so the ball mostly keeps its place.
 */
export function placeApprovalPopup(pos: Point, screens: readonly Area[], scale = 1): PopupPlacement {
  const s = scale || 1;
  const width = ORB_WIDGET_WIDTH * s;
  const cardW = ORB_WIDGET_POPUP_WIDTH * s;
  const cardH = ORB_WIDGET_POPUP_CARD_HEIGHT * s;
  const margin = SCREEN_MARGIN * s;
  const cx = pos.x + ORB_WIDGET_CENTER.x * s;
  const cy = pos.y + ORB_WIDGET_CENTER.y * s;
  const fallback = {
    at: pos,
    above: false,
    pad: 0,
    popupX: (ORB_WIDGET_WIDTH - ORB_WIDGET_POPUP_WIDTH) / 2,
    popupY: ORB_WIDGET_POPUP_BELOW_TOP,
  };
  if (!screens.length) return fallback;
  let m = screens[0];
  let best = Number.POSITIVE_INFINITY;
  for (const a of screens) {
    const dx = Math.max(a.x - cx, 0, cx - (a.x + a.width - 1));
    const dy = Math.max(a.y - cy, 0, cy - (a.y + a.height - 1));
    const d = dx * dx + dy * dy;
    if (d < best) {
      best = d;
      m = a;
    }
  }
  const left = m.x + margin;
  const right = m.x + m.width - margin;
  const top = m.y + margin;
  const bottom = m.y + m.height - margin;

  // Horizontal: centred under the ball when it fits, else pushed inside the frame.
  const cardLeft = Math.min(Math.max(cx - cardW / 2, left), right - cardW);
  const winX = Math.min(Math.max(pos.x, cardLeft + cardW - width), cardLeft);

  // Vertical: under the bar, else above the ball, else lifted until it fits under.
  const padPx = ORB_WIDGET_POPUP_HEIGHT * s;
  const belowTop = ORB_WIDGET_POPUP_BELOW_TOP * s;
  let winY = pos.y;
  let above = false;
  if (pos.y + belowTop + cardH > bottom) {
    if (pos.y - padPx + POPUP_GAP * s >= top) {
      above = true;
      winY = pos.y - padPx;
    } else {
      winY = bottom - cardH - belowTop;
    }
  }
  return {
    at: { x: Math.round(winX), y: Math.round(winY) },
    above,
    pad: above ? ORB_WIDGET_POPUP_HEIGHT : 0,
    popupX: (cardLeft - Math.round(winX)) / s,
    popupY: above ? POPUP_GAP : ORB_WIDGET_POPUP_BELOW_TOP,
  };
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
  /** The bottom of the visible frame (above a dock or taskbar), when it is known. */
  visibleBottom?: number;
}

/** From the ball's centre to the control bar's bottom edge. */
export const ORB_WIDGET_BAR_BELOW_CENTER = ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT - ORB_WIDGET_CENTER.y;

/** The lowest the ball's centre may go on `m`: the bar's bottom on the visible frame's bottom. */
function lowestCenter(m: Area): number {
  return Math.min(m.y + m.height - 1, (m.visibleBottom ?? m.y + m.height) - ORB_WIDGET_BAR_BELOW_CENTER);
}

export function isOrbWidgetAction(value: unknown): value is OrbWidgetAction {
  return value === "toggle-pause" || value === "stop" || value === "back";
}

export function parseSavedPosition(raw: string | null): Point | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<Point>;
    if (Number.isFinite(value.x) && Number.isFinite(value.y)) {
      return {
        x: Math.round(value.x as number),
        y: Math.round(value.y as number),
      };
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
export function clampToScreens(p: Point, screens: readonly Area[], center: Point = ORB_WIDGET_CENTER): Point {
  if (!screens.length) return p;
  const cx = p.x + center.x;
  const cy = p.y + center.y;
  let best: Point = p;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const m of screens) {
    const x = Math.min(Math.max(cx, m.x), m.x + m.width - 1);
    // Any side may take the ball to the edge, except the bottom: the bar hangs below it.
    const y = Math.min(Math.max(cy, m.y), lowestCenter(m));
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
  // Three places: two made the widget's edge step where the in-app one glides.
  for (let i = 0; i < bands.length; i++) out[i] = Math.round(bands[i] * 1000) / 1000;
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
  const cy = Math.min(Math.max(cursor.y, m.y + r), Math.min(m.y + m.height - r, lowestCenter(m)));
  return {
    x: Math.round(cx - ORB_WIDGET_CENTER.x),
    y: Math.round(cy - ORB_WIDGET_CENTER.y),
  };
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
  return {
    x: Math.round(now.x - start.grab.x),
    y: Math.round(now.y - start.grab.y),
  };
}
