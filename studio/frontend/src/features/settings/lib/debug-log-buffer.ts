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

/** Generous next to a 1s poll: this is the "this request will never answer"
 * backstop, not a latency budget. A remote tunnel can legitimately be slow. */
export const REQUEST_TIMEOUT_MS = 20_000;

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

/** Run a request under the "this will never answer" backstop, linked to the
 * caller's signal.
 *
 * EVERY request the viewer awaits needs this, not just the tail read: the auth
 * client passes `init` straight to `fetch` and adds no timeout of its own, so a
 * request that opens and never settles hangs whatever is awaiting it for good.
 * The poll loop awaits the source rescan before the tail read, so an unanswered
 * /sources froze the pane exactly the way the tail read's own backstop was
 * added to prevent. `AbortSignal.any` would fold the two signals together, but
 * it is Safari 17.4 against this project's 16.4 floor, so they are linked by
 * hand.
 */
export async function withRequestTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<T> {
  const local = new AbortController();
  if (signal?.aborted) local.abort();
  const relay = () => local.abort();
  signal?.addEventListener("abort", relay);
  const timer = setTimeout(() => local.abort(), timeoutMs);
  try {
    return await task(local.signal);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", relay);
  }
}

/** Whether a response that has just landed still belongs on screen.
 *
 * `selection` counts source changes. A manual refresh carries no abort signal,
 * so a slow read of source A can answer after the user has picked B, and
 * comparing the page's own source id does not catch it: the answer really is
 * A's, it is the view underneath that moved. Counting also covers A -> B -> A,
 * where the id matches again but the cursor and the buffer have been reset in
 * between, so the old page's cursor would rewind the new read.
 */
export function isPageStale(page: {
  requestSelection: number;
  currentSelection: number;
  requestSourceId: string | null;
  pageSourceId: string | null;
}): boolean {
  if (page.requestSelection !== page.currentSelection) return true;
  // The server answers an unset source with its default, so only disagree when
  // both ends named one.
  return Boolean(
    page.requestSourceId &&
      page.pageSourceId &&
      page.requestSourceId !== page.pageSourceId,
  );
}

/** Whether the "some lines were skipped" warning shows after this response.
 *
 * Sticky, because the gap it reports stays in the buffer: recomputing it from
 * the newest response alone cleared the warning on the next poll (one second in
 * live mode) while the pane was still missing those lines. It clears when the
 * buffer does, and a reset replaces everything on screen with a fresh tail.
 */
export function nextDroppedState(
  previous: boolean,
  page: { droppedBytes: number; reset: boolean },
): boolean {
  if (page.droppedBytes > 0) return true;
  return page.reset ? false : previous;
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
