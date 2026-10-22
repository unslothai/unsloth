// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const FLOATING_PANEL_POSITION_KEYS = {
  liveMonitor: "unsloth_live_monitor_position",
  apiMonitor: "unsloth_api_monitor_position",
} as const;

/** User's drop spot in its constraints box; a null axis stays anchored to the far edge. */
export type PanelIntent = { left: number | null; top: number | null };

function axis(value: unknown): number | null | undefined {
  if (value === null) return null;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function readPanelIntent(key: string | undefined): PanelIntent | null {
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Record<string, unknown> | null;
    const left = axis(parsed?.left);
    const top = axis(parsed?.top);
    return left === undefined || top === undefined ? null : { left, top };
  } catch {
    return null;
  }
}

export function storePanelIntent(key: string | undefined, intent: PanelIntent): void {
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(intent));
  } catch {
    // storage unavailable
  }
}

export function placeAxis(intent: number | null, max: number): number {
  return intent === null ? max : Math.max(0, Math.min(max, intent));
}

/** Minimized Windows Desktop resizes the page to 0x0 (tao); laying out against it parks panels top-left. */
export function isDegenerateBox(box: { width: number; height: number }): boolean {
  return !(box.width > 0 && box.height > 0);
}

export function clearPanelIntent(key: string | undefined): void {
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch {
    // storage unavailable
  }
}

export function nudgeDelta(
  key: string,
  shift: boolean,
): { dx: number; dy: number } | null {
  const step = shift ? 32 : 8;
  switch (key) {
    case "ArrowLeft":
      return { dx: -step, dy: 0 };
    case "ArrowRight":
      return { dx: step, dy: 0 };
    case "ArrowUp":
      return { dx: 0, dy: -step };
    case "ArrowDown":
      return { dx: 0, dy: step };
    default:
      return null;
  }
}

export type FrameBox = { left: number; top: number; right: number; bottom: number };

export function liftAboveObstacle(
  panel: { left: number; top: number; width: number; height: number },
  obstacle: FrameBox | null | undefined,
  gap: number,
): number | null {
  if (!obstacle) return null;
  const overlapsX =
    panel.left < obstacle.right && panel.left + panel.width > obstacle.left;
  // Not only on collision: a card the composer lifted would leave the panel below it, over the composer.
  const reaches = panel.top + panel.height > obstacle.top - gap;
  return overlapsX && reaches ? obstacle.top - gap - panel.height : null;
}

export const ANCHOR_SNAP_PX = 16;

export function snapsToAnchor(
  drop: { left: number; top: number },
  rest: { left: number; top: number },
  max: { left: number; top: number },
  threshold: number = ANCHOR_SNAP_PX,
): boolean {
  const near = (a: { left: number; top: number }) =>
    Math.abs(drop.left - a.left) <= threshold &&
    Math.abs(drop.top - a.top) <= threshold;
  return near(rest) || near(max);
}
