// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

/**
 * Coalesces sidebar clicks: while the last pushed entry has not rendered, the next goes out as a
 * replace so a burst leaves one history entry.
 */
export function createNavigationCoalescer<T>({
  navigate,
  currentHref,
  hrefOf,
  currentEntry,
  asReplace,
}: {
  navigate: (options: T) => Promise<unknown>;
  currentHref: () => string;
  hrefOf: (options: T) => string;
  currentEntry: () => string | undefined;
  asReplace: (options: T) => T;
}): { go: (options: T) => void; resolved: () => void } {
  let unshown: string | undefined;

  return {
    go(options: T) {
      if (hrefOf(options) === currentHref()) return;
      const replace = unshown !== undefined && currentEntry() === unshown;
      void navigate(replace ? asReplace(options) : options).catch(() => {});
      unshown = currentHref() === hrefOf(options) ? currentEntry() : undefined;
    },
    // The router resolves only when nothing is pending: the entry rendered or history left it.
    resolved() {
      unshown = undefined;
    },
  };
}
