// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ownership of a diffusion page's model pick across an await: resolving is a slow listing request and neither page sets
// `busy` for it. No app deps, so both pages share this and the ordering is testable.

/** Which pick owns the page. A claim invalidates every token handed out before it, so the newest pick wins. */
export interface PickGuard {
  claim(): number;
  /** Leave the page unowned without ending the pick: a page switch, an unmount. */
  release(): void;
  /** End the pick outright: an eject or a deploy, which the staged download must not undo. */
  cancel(): void;
  /** Is this token still the owner? False after a release, so nothing lands on a page nobody is looking at. */
  holds(token: number): boolean;
  /** Is this still the last pick made? Survives a release, so a staged download resumes on the way back. */
  isLatest(token: number): boolean;
}

export function createPickGuard(): PickGuard {
  let latest = 0;
  let owner = 0;
  return {
    claim: () => {
      latest += 1;
      owner = latest;
      return latest;
    },
    release: () => {
      owner = 0;
    },
    cancel: () => {
      // Past every token handed out, so nothing outstanding is the latest pick either.
      latest += 1;
      owner = 0;
    },
    holds: (token) => token !== 0 && token === owner,
    isLatest: (token) => token !== 0 && token === latest,
  };
}

/** The page's own halves of a repo-level GGUF pick. Only `isCurrent` is about staleness. */
export interface GgufRepoPickHandlers {
  /** The one .gguf this pick means, or null when the repo cannot name it. */
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

/** Resolve a repo-level GGUF pick and load it; does nothing once the pick no longer owns the page. */
export async function runGgufRepoPick(
  handlers: GgufRepoPickHandlers,
): Promise<boolean> {
  const filename = await handlers.resolve();
  // Silent, not just load-free: a toast would blame a model the user has moved on from.
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
