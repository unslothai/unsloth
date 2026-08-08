// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ownership of a diffusion page's model pick across an await. Resolving a repo-level GGUF pick is a listing request that can
// take seconds, and neither page sets `busy` for it, so the user can pick again or switch pages meanwhile. No app deps, so
// both pages share one copy and the ordering is testable without a React harness.

/** Which pick owns the page. A claim invalidates every token handed out before it, so the newest pick always wins. */
export interface PickGuard {
  /** Take the page for a new pick. */
  claim(): number;
  /** Give the page up, so no outstanding token owns it: a page switch, an unload, an unmount. */
  release(): void;
  /** Does this token still own the page? */
  holds(token: number): boolean;
}

export function createPickGuard(): PickGuard {
  // Starts at 1 so 0 is never a live token; release() lands on a value nobody was handed.
  let current = 0;
  return {
    claim: () => {
      current += 1;
      return current;
    },
    release: () => {
      current += 1;
    },
    holds: (token) => token === current,
  };
}

/** The page-specific halves of a repo-level GGUF pick. Only `isCurrent` is about staleness; the rest is the page's own state. */
export interface GgufRepoPickHandlers {
  /** The repo's listing, resolved to the one .gguf this pick means, or null when the repo cannot name it. */
  resolve(): Promise<string | null>;
  /** False once a newer pick, a page switch or an unmount has taken the page. */
  isCurrent(): boolean;
  /** Nothing to load: several quants on disk, a stale label, or an unreadable listing. */
  onAmbiguous(): void;
  /** Optimistic quant label plus per-model defaults, applied before the load starts. */
  onResolved(filename: string): void;
  /** Undo `onResolved`: the load never started. */
  onNotStarted(): void;
  /** Start (or stage) the load. True when the pick was accepted. */
  load(filename: string): Promise<boolean>;
}

/** Resolve a repo-level GGUF pick to a filename and load it, doing nothing at all once the pick no longer owns the page. */
export async function runGgufRepoPick(
  handlers: GgufRepoPickHandlers,
): Promise<boolean> {
  const filename = await handlers.resolve();
  // Silent, not just load-free: a toast here would blame a model the user has already moved on from, and touching the quant
  // label or the defaults would overwrite the pick that replaced this one.
  if (!handlers.isCurrent()) return false;
  if (!filename) {
    handlers.onAmbiguous();
    return false;
  }
  handlers.onResolved(filename);
  const started = await handlers.load(filename);
  // Only the pick that applied the optimistic label may take it back: `quantRevert` is one slot, so a stale revert would
  // restore this pick's old label over the newer one.
  if (!started && handlers.isCurrent()) handlers.onNotStarted();
  return started;
}
