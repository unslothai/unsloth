// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export const ORB_WIDGET_LABEL = "orb-widget";
export const ORB_WIDGET_QUERY = "orb-widget";
export const ORB_WIDGET_BALL = 72;
const K = ORB_WIDGET_BALL / 140;
export const ORB_WIDGET_PEAK_SCALE = 1.1;
export const ORB_WIDGET_EDGE_REACH = (70 + 3 + 22.4) * K * ORB_WIDGET_PEAK_SCALE;
export const ORB_WIDGET_GLOW_REACH = (70 + 8 + 1.5 * 32) * K * ORB_WIDGET_PEAK_SCALE;
export const ORB_WIDGET_MARGIN = 24;
const HALF = Math.ceil(Math.max(ORB_WIDGET_EDGE_REACH, ORB_WIDGET_GLOW_REACH) + ORB_WIDGET_MARGIN);
export const ORB_WIDGET_BAR_HEIGHT = 34;
export const ORB_WIDGET_BAR_GAP = 7;
export const ORB_WIDGET_BAR_TOP = HALF + ORB_WIDGET_BALL / 2 + ORB_WIDGET_BAR_GAP;
export const ORB_WIDGET_WIDTH = 2 * HALF;
export const ORB_WIDGET_HEIGHT = Math.max(2 * HALF, ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT + 8);
export const ORB_WIDGET_CENTER = { x: HALF, y: HALF };
export const ORB_WIDGET_LONG_PRESS_MS = 500;

export const ORB_WIDGET_STATE_EVENT = "orb-widget:state";
export const ORB_WIDGET_READY_EVENT = "orb-widget:ready";
export const ORB_WIDGET_SPECTRUM_EVENT = "orb-widget:spectrum";
export const ORB_WIDGET_ACTION_EVENT = "orb-widget:action";
export const ORB_WIDGET_APPROVAL_EVENT = "orb-widget:approval";

export const ORB_WIDGET_POSITION_KEY = "unsloth.orb-widget.position.v1";

export type OrbWidgetAction = "toggle-pause" | "stop" | "back";

export interface OrbWidgetApproval {
  id: string;
  line: string;
  brief: string;
  more: string | null;
  allow: string;
  deny: string;
}

export interface OrbWidgetLabels {
  pause: string;
  resume: string;
  stop: string;
  back: string;
}

export interface OrbWidgetTheme {
  dark: boolean;
  palette: string | null;
}

export interface OrbWidgetSnapshot {
  orbState: string | null;
  paused: boolean;
  muted: boolean;
  active: boolean;
  approval: OrbWidgetApproval | null;
  labels: OrbWidgetLabels;
  theme: OrbWidgetTheme;
}

type ThemeRoot = {
  classList: { toggle(name: string, force: boolean): unknown };
  style: { colorScheme: string };
  setAttribute(name: string, value: string): void;
  removeAttribute(name: string): void;
};

export function applyOrbWidgetTheme(root: ThemeRoot, theme: OrbWidgetTheme): void {
  root.classList.toggle("dark", theme.dark);
  root.classList.toggle("light", !theme.dark);
  root.style.colorScheme = theme.dark ? "dark" : "light";
  if (theme.palette) root.setAttribute("data-palette", theme.palette);
  else root.removeAttribute("data-palette");
}

export function sameTheme(a: OrbWidgetTheme | undefined, b: OrbWidgetTheme | undefined): boolean {
  return a?.dark === b?.dark && (a?.palette ?? null) === (b?.palette ?? null);
}

export function sameLabels(a: OrbWidgetLabels | undefined, b: OrbWidgetLabels | undefined): boolean {
  return a?.pause === b?.pause && a?.resume === b?.resume && a?.stop === b?.stop && a?.back === b?.back;
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

export function pendingApproval(
  confirmations: Readonly<Record<string, { autoAllowKey?: string }>>,
  voiceThreadId: string | null,
  inVoiceThread: (toolCallId: string) => boolean,
): { id: string; more: number } | null {
  const suffix = voiceThreadId ? `:${voiceThreadId}` : null;
  const ids = Object.keys(confirmations).filter(
    (id) => (suffix !== null && confirmations[id]?.autoAllowKey?.endsWith(suffix)) || inVoiceThread(id),
  );
  return ids.length ? { id: ids[0], more: ids.length - 1 } : null;
}

/** The name Always allow would grant, or null to not offer it: never a guessed "tool". */
export function alwaysAllowToolName(call: { toolName?: string } | null | undefined): string | null {
  const name = call?.toolName?.trim();
  return name ? name : null;
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

export const ORB_WIDGET_POPUP_WIDTH = 156;
export const ORB_WIDGET_POPUP_CARD_HEIGHT = 54;
const POPUP_GAP = 6;
export const ORB_WIDGET_POPUP_BELOW_TOP = ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT + POPUP_GAP;
export const ORB_WIDGET_GROWN_HEIGHT = Math.max(
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_POPUP_BELOW_TOP + ORB_WIDGET_POPUP_CARD_HEIGHT + POPUP_GAP,
);
export const ORB_WIDGET_POPUP_HEIGHT = ORB_WIDGET_GROWN_HEIGHT - ORB_WIDGET_HEIGHT;
const SCREEN_MARGIN = 8;

export interface PopupPlacement {
  at: Point;
  above: boolean;
  pad: number;
  popupX: number;
  popupY: number;
  shift: Point;
}

export function restoreAfterPopup(current: Point, shift: Point): Point {
  return { x: current.x - shift.x, y: current.y - shift.y };
}

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
    shift: { x: 0, y: 0 },
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

  const cardLeft = Math.min(Math.max(cx - cardW / 2, left), right - cardW);
  const winX = Math.min(Math.max(pos.x, cardLeft + cardW - width), cardLeft);

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
  const at = { x: Math.round(winX), y: Math.round(winY) };
  return {
    at,
    above,
    pad: above ? ORB_WIDGET_POPUP_HEIGHT : 0,
    popupX: (cardLeft - at.x) / s,
    popupY: above ? POPUP_GAP : ORB_WIDGET_POPUP_BELOW_TOP,
    shift: { x: at.x - pos.x, y: at.y - pos.y },
  };
}

export interface Point {
  x: number;
  y: number;
}

export interface Area {
  x: number;
  y: number;
  width: number;
  height: number;
  visibleBottom?: number;
}

export const ORB_WIDGET_BAR_BELOW_CENTER = ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT - ORB_WIDGET_CENTER.y;

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
  }
  return null;
}

function centreOn(p: Point, m: Area, size: { width: number; height: number }): boolean {
  const cx = p.x + size.width / 2;
  const cy = p.y + size.height / 2;
  return cx >= m.x && cx < m.x + m.width && cy >= m.y && cy < m.y + m.height;
}

const WIDGET_SIZE = { width: ORB_WIDGET_WIDTH, height: ORB_WIDGET_HEIGHT };

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

export function clampToScreens(p: Point, screens: readonly Area[], center: Point = ORB_WIDGET_CENTER): Point {
  if (!screens.length) return p;
  const cx = p.x + center.x;
  const cy = p.y + center.y;
  let best: Point = p;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const m of screens) {
    const x = Math.min(Math.max(cx, m.x), m.x + m.width - 1);
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

export function packSpectrum(side: "input" | "output", level: number, bands: ArrayLike<number>): SpectrumFrame {
  const out: number[] = new Array(bands.length);
  for (let i = 0; i < bands.length; i++) out[i] = Math.round(bands[i] * 1000) / 1000;
  return { side, level: Math.round(level * 1000) / 1000, bands: out };
}

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
  screen: Point;
  grab: Point;
}

export function dragTarget(start: DragStart, now: Point, moving: boolean): Point | null {
  if (!moving && Math.hypot(now.x - start.screen.x, now.y - start.screen.y) <= ORB_WIDGET_DRAG_THRESHOLD) {
    return null;
  }
  return {
    x: Math.round(now.x - start.grab.x),
    y: Math.round(now.y - start.grab.y),
  };
}
