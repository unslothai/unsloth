// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Kept free of React so node:test can import it. */

export type RefreshMode = "live" | "3s" | "manual";

export const DEFAULT_REFRESH_MODE: RefreshMode = "3s";
export const REFRESH_MODE_STORAGE_KEY = "unsloth_debug_log_refresh_mode";

/** Distinct from AbortError, which unmounts and source switches use silently, so a hung request
 * is reported. Lives here because node:test resolves no extensionless src import. */
export class DebugLogTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    this.name = "DebugLogTimeoutError";
  }
}

export function isRequestTimeout(error: unknown): boolean {
  return error instanceof DebugLogTimeoutError;
}

/** Built by hand for the tie where the deadline won but the caller had already aborted. */
function abortError(): Error {
  const error = new Error("Aborted.");
  error.name = "AbortError";
  return error;
}

/** A never-answers backstop, not a latency budget; tunnels can be slow. */
export const REQUEST_TIMEOUT_MS = 20_000;

/** The server caps a response; this caps accumulation over a long session. */
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

/** null for manual. "live" polls short requests rather than streaming, so it survives proxies. */
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

/**
 * Every awaited request needs this: authFetch adds no timeout and awaits refreshSession() on a 401
 * without a signal, so the deadline is raced, not awaited. Signals are linked by hand because
 * `AbortSignal.any` needs Safari 17.4 and the floor is 16.4.
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
  // An unobserved rejection is a process-level crash under node:test.
  running.catch(() => {});
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      // Reject before aborting, or the synchronous AbortError would win the race and be dropped.
      reject(new DebugLogTimeoutError(timeoutMs));
      local.abort();
    }, timeoutMs);
  });
  try {
    return await Promise.race([running, deadline]);
  } catch (error) {
    // The caller's own abort wins the tie; it is silent by design.
    if (signal?.aborted && isRequestTimeout(error)) throw abortError();
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", relay);
  }
}

/** `selection` counts source changes: a manual refresh has no abort signal, and A -> B -> A
 * would match ids while A's old cursor rewinds the reset buffer. */
export function isPageStale(page: {
  requestSelection: number;
  currentSelection: number;
  requestSourceId: string | null;
  pageSourceId: string | null;
}): boolean {
  if (page.requestSelection !== page.currentSelection) return true;
  // The server answers an unset source with its default, so only compare when both name one.
  return Boolean(
    page.requestSourceId &&
      page.pageSourceId &&
      page.requestSourceId !== page.pageSourceId,
  );
}

/** Sticky until a reset: the skipped lines stay missing from the buffer. */
export function nextDroppedState(
  previous: boolean,
  page: { droppedBytes: number; reset: boolean },
): boolean {
  if (page.droppedBytes > 0) return true;
  return page.reset ? false : previous;
}

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

/** Returns the SAME object when nothing is new so the caller can skip a re-render. */
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
