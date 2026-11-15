// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RefreshMode = "live" | "3s" | "manual";

export const DEFAULT_REFRESH_MODE: RefreshMode = "3s";
export const REFRESH_MODE_STORAGE_KEY = "unsloth_debug_log_refresh_mode";

export class DebugLogTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    this.name = "DebugLogTimeoutError";
  }
}

export function isRequestTimeout(error: unknown): boolean {
  return error instanceof DebugLogTimeoutError;
}

function abortError(): Error {
  const error = new Error("Aborted.");
  error.name = "AbortError";
  return error;
}

export const REQUEST_TIMEOUT_MS = 20_000;

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

/** Race refreshSession (no abort signal); link signals for Safari 16.4 without AbortSignal.any. */
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
  // Observe late rejections after the timeout wins.
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

/** The source-change counter rejects old A responses after A -> B -> A, including manual polls. */
export function isPageStale(page: {
  requestSelection: number;
  currentSelection: number;
  requestSourceId: string | null;
  pageSourceId: string | null;
}): boolean {
  if (page.requestSelection !== page.currentSelection) return true;
  return Boolean(
    page.requestSourceId &&
      page.pageSourceId &&
      page.requestSourceId !== page.pageSourceId,
  );
}

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

export function selectionInside<N>(
  selection: {
    isCollapsed: boolean;
    anchorNode: N | null;
    focusNode: N | null;
  } | null,
  pane: { contains(node: N | null): boolean } | null,
): boolean {
  if (!selection || !pane || selection.isCollapsed) return false;
  return pane.contains(selection.anchorNode) || pane.contains(selection.focusNode);
}

export function clearSelectionInside<N>(
  selection: {
    isCollapsed: boolean;
    anchorNode: N | null;
    focusNode: N | null;
    removeAllRanges(): void;
  } | null,
  pane: { contains(node: N | null): boolean } | null,
): void {
  if (selectionInside(selection, pane)) selection?.removeAllRanges();
}
