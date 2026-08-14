// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Buffer and polling maths for the Settings > Logs log viewer. Kept free
 * of React so it can be tested: the tab component pulls in the router, motion
 * and hugeicons and cannot be imported under node:test.
 */

export type RefreshMode = "live" | "3s" | "manual";

export const DEFAULT_REFRESH_MODE: RefreshMode = "3s";
export const REFRESH_MODE_STORAGE_KEY = "unsloth_debug_log_refresh_mode";

/** The request backstop fired. Distinct from an AbortError because the two
 * arrive as the same rejection: the timer aborts the very controller an unmount
 * or a source switch uses, and those are silent by design. Sharing the name
 * made a hung tunnel silent too, so the pane stopped updating and said nothing,
 * which is the failure this viewer exists to make visible.
 *
 * Here rather than beside the other error types because this module is imported
 * by the node:test runner, which resolves no extensionless src import.
 */
export class DebugLogTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    this.name = "DebugLogTimeoutError";
  }
}

export function isRequestTimeout(error: unknown): boolean {
  return error instanceof DebugLogTimeoutError;
}

/** What `fetch` rejects with on an abort. Built by hand rather than taken from
 * a cancelled request, for the tie where the deadline won the race but the
 * caller had already asked to stop. */
function abortError(): Error {
  const error = new Error("Aborted.");
  error.name = "AbortError";
  return error;
}

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

/** Poll delay for a mode, or null when the user drives it by hand. "live" polls
 * rather than opening a socket: Cloudflare quick tunnels buffer
 * text/event-stream and only flush at close, which is why the export log polls too.
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
 * client passes `init` straight to `fetch` and adds no timeout, so a request
 * that opens and never settles hangs its awaiter for good. The poll loop awaits
 * the source rescan first, so an unanswered /sources froze the pane. Signals are
 * linked by hand because `AbortSignal.any` is Safari 17.4 against a 16.4 floor.
 *
 * The deadline is RACED rather than awaited behind the abort, because aborting
 * only settles work that watches the signal. `authFetch` awaits
 * `refreshSession()` on a 401 and hands it no signal, so an expired token plus a
 * hung refresh left this promise pending for good: the caller's in-flight guard
 * stayed pinned and the pane froze, which is precisely what the backstop is for.
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
  let timer: ReturnType<typeof setTimeout> | undefined;
  const running = task(local.signal);
  // The loser of the race can still reject afterwards, and an unobserved
  // rejection is a console error (a process-level crash under node:test).
  running.catch(() => {});
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      // Reject BEFORE aborting. `local.abort()` settles an abort-aware task
      // synchronously, and that rejection would then win the race and arrive
      // as an ordinary AbortError, which the poll loop drops in silence: the
      // deadline would announce itself only for work that ignores the signal.
      reject(new DebugLogTimeoutError(timeoutMs));
      local.abort();
    }, timeoutMs);
  });
  try {
    return await Promise.race([running, deadline]);
  } catch (error) {
    // The caller's own abort wins the tie: an unmount or a source switch is
    // silent by design, and it can race the timer on a request that was
    // already doomed. Only a backstop with no caller abort behind it is a
    // fault the user needs told about.
    if (signal?.aborted && isRequestTimeout(error)) throw abortError();
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", relay);
  }
}

/** Whether a response that has just landed still belongs on screen.
 *
 * `selection` counts source changes. A manual refresh carries no abort signal,
 * so a slow read of source A can answer after the user picked B; comparing
 * source ids does not catch it, since the answer really is A's and it is the
 * view underneath that moved. Counting also covers A -> B -> A, where the id
 * matches again but the buffer was reset and A's old cursor would rewind it.
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
 * Sticky, because the gap it reports stays in the buffer: recomputing from the
 * newest response alone cleared the warning on the next poll (1s in live mode)
 * while the pane was still missing those lines. Only a reset clears it, since
 * that replaces everything on screen with a fresh tail.
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

/** Fold one response into the buffer. Returns the SAME object when the chunk
 * carries nothing new, so the caller can skip a re-render: at a 1s poll on a
 * quiet server almost every tick is empty.
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
