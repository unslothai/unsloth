// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ownership of a diffusion page's model pick across an await: resolving one is a listing request that can take seconds, and
// neither page sets `busy` for it. No app deps, so both pages share a copy and the ordering is testable.

/** Which pick owns the page. A claim invalidates every token handed out before it, so the newest pick wins. */
export interface PickGuard {
  claim(): number;
  /** Leave the page unowned: a page switch, an unload, an unmount. */
  release(): void;
  holds(token: number): boolean;
}

export function createPickGuard(): PickGuard {
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

/** The page's own halves of a repo-level GGUF pick. Only `isCurrent` is about staleness. */
export interface GgufRepoPickHandlers {
  /** The listing, resolved to the one .gguf this pick means, or null when the repo cannot name it. */
  resolve(): Promise<string | null>;
  isCurrent(): boolean;
  /** Nothing to load: several quants on disk, a stale label, or an unreadable listing. */
  onAmbiguous(): void;
  /** Optimistic quant label plus per-model defaults, applied before the load starts. */
  onResolved(filename: string): void;
  /** Undo `onResolved`: the load never started. */
  onNotStarted(): void;
  load(filename: string): Promise<boolean>;
}

/** Resolve a repo-level GGUF pick and load it, doing nothing at all once the pick no longer owns the page. */
export async function runGgufRepoPick(
  handlers: GgufRepoPickHandlers,
): Promise<boolean> {
  const filename = await handlers.resolve();
  // Silent, not just load-free: a toast would blame a model the user has moved on from, and the label and defaults now
  // belong to the pick that replaced this one.
  if (!handlers.isCurrent()) return false;
  if (!filename) {
    handlers.onAmbiguous();
    return false;
  }
  handlers.onResolved(filename);
  const started = await handlers.load(filename);
  // `quantRevert` is one slot, so only the pick that set the label may take it back.
  if (!started && handlers.isCurrent()) handlers.onNotStarted();
  return started;
}
