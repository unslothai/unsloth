// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Buffer and polling maths for the Settings > Debugging log viewer.
 *
 * Kept free of React so it can be tested: the tab component pulls in the
 * router, motion and hugeicons and cannot be imported under node:test.
 */

export type RefreshMode = "live" | "3s" | "manual";

export const DEFAULT_REFRESH_MODE: RefreshMode = "3s";
export const REFRESH_MODE_STORAGE_KEY = "unsloth_debug_log_refresh_mode";

/** Bounds on what the viewer holds. The server already caps a response; this
 * caps the accumulation across a long session. */
export const MAX_CLIENT_LINES = 2000;
export const MAX_CLIENT_CHARS = 400_000;

export interface DebugLogChunk {
  lines: string[];
  cursor: string | null;
  reset: boolean;
}

export interface LogBufferState {
  lines: string[];
  cursor: string | null;
}

export const EMPTY_BUFFER: LogBufferState = { lines: [], cursor: null };

/** Poll delay for a mode, or null when the user drives it by hand.
 *
 * "live" is a one second poll rather than a socket: this endpoint is the same
 * shape the export log had to fall back to because Cloudflare quick tunnels
 * buffer text/event-stream and only flush at close.
 */
export function pollDelayMs(mode: RefreshMode): number | null {
  switch (mode) {
    case "live":
      return 1000;
    case "3s":
      return 3000;
    case "manual":
      return null;
  }
}

export function parseRefreshMode(value: unknown): RefreshMode {
  return value === "live" || value === "3s" || value === "manual"
    ? value
    : DEFAULT_REFRESH_MODE;
}

/** Drop from the front until the buffer is within both caps. */
export function trimBuffer(lines: string[]): string[] {
  let trimmed =
    lines.length > MAX_CLIENT_LINES ? lines.slice(-MAX_CLIENT_LINES) : lines;
  let chars = 0;
  for (const line of trimmed) chars += line.length + 1;
  if (chars <= MAX_CLIENT_CHARS) return trimmed;
  let start = 0;
  while (start < trimmed.length && chars > MAX_CLIENT_CHARS) {
    chars -= trimmed[start].length + 1;
    start += 1;
  }
  trimmed = trimmed.slice(start);
  return trimmed;
}

/** Fold one response into the buffer.
 *
 * Returns the SAME object when the chunk carries nothing new, so the caller can
 * skip a re-render: at a one second poll on a quiet server, almost every tick
 * is empty.
 */
export function applyLogChunk(
  previous: LogBufferState,
  chunk: DebugLogChunk,
): LogBufferState {
  if (chunk.reset) {
    return { lines: trimBuffer(chunk.lines.slice()), cursor: chunk.cursor };
  }
  if (chunk.lines.length === 0) {
    if (chunk.cursor === previous.cursor) return previous;
    return { lines: previous.lines, cursor: chunk.cursor };
  }
  return {
    lines: trimBuffer(previous.lines.concat(chunk.lines)),
    cursor: chunk.cursor,
  };
}
