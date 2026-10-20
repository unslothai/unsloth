// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const FLOATING_PANEL_POSITION_KEYS = {
  liveMonitor: "unsloth_live_monitor_position",
  apiMonitor: "unsloth_api_monitor_position",
} as const;

/** Where the user dropped a panel, relative to its constraints box. A null axis is still anchored
 *  to the far edge, as an undragged panel is. */
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

/** Display position for one axis: the user's spot clamped to the room there is now, or the far
 *  edge when they never moved it. The intent itself is never rewritten by this. */
export function placeAxis(intent: number | null, max: number): number {
  return intent === null ? max : Math.max(0, Math.min(max, intent));
}

/** A minimized Desktop window on Windows resizes the page to 0x0 (tao reports the minimized
 *  client size and the webview is auto-resized to it); laying out against that parks every panel
 *  in the top-left corner. */
export function isDegenerateBox(box: { width: number; height: number }): boolean {
  return !(box.width > 0 && box.height > 0);
}
