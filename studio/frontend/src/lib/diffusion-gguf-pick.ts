// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Neither page sets `busy` while resolving; no app deps so both pages share it and it is testable.

/** A claim invalidates every earlier token, so the newest pick wins. */
export interface PickGuard {
  claim(): number;
  /** Unowned without ending the pick: a page switch, an unmount. */
  release(): void;
  /** An eject or deploy, which a staged download must not undo. */
  cancel(): void;
  /** False after a release. */
  holds(token: number): boolean;
  /** Survives a release, so a staged download resumes on the way back. */
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
      latest += 1;
      owner = 0;
    },
    holds: (token) => token !== 0 && token === owner,
    isLatest: (token) => token !== 0 && token === latest,
  };
}

/** Only `isCurrent` is about staleness. */
export interface GgufRepoPickHandlers {
  resolve(): Promise<string | null>;
  isCurrent(): boolean;
  /** Several quants on disk, a stale label, or an unreadable listing. */
  onAmbiguous(): void;
  onResolved(filename: string): void;
  onNotStarted(): void;
  load(filename: string): Promise<boolean>;
}

export async function runGgufRepoPick(
  handlers: GgufRepoPickHandlers,
): Promise<boolean> {
  const filename = await handlers.resolve();
  // Silent: a toast would blame a model the user moved on from.
  if (!handlers.isCurrent()) return false;
  if (!filename) {
    handlers.onAmbiguous();
    handlers.onNotStarted();
    return false;
  }
  handlers.onResolved(filename);
  const started = await handlers.load(filename);
  // `quantRevert` is one slot, so only the pick that set the label may take it back.
  if (!started && handlers.isCurrent()) handlers.onNotStarted();
  return started;
}
