// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

/**
 * Sidebar clicks: drop one on where the router is or is heading; while the entry the last
 * click pushed has not rendered, send the next at once as a replace, so the router cancels
 * the stale load and a burst leaves one history entry. Only that entry: Back/Forward onto a
 * slow page, or a navigation still held by a blocker (unsaved Library note), keeps its entry.
 */
export function createNavigationCoalescer<T>({
  navigate,
  currentHref,
  hrefOf,
  currentEntry,
  asReplace,
}: {
  navigate: (options: T) => Promise<unknown>;
  /** Where the router is, or is already heading. */
  currentHref: () => string;
  hrefOf: (options: T) => string;
  /** Key of the router's latest history entry. */
  currentEntry: () => string | undefined;
  asReplace: (options: T) => T;
}): {
  go: (options: T) => void;
  rendered: (entry: string | undefined) => void;
} {
  let unshown: string | undefined;

  return {
    go(options: T) {
      if (hrefOf(options) === currentHref()) return;
      const replace = unshown !== undefined && currentEntry() === unshown;
      void navigate(replace ? asReplace(options) : options).catch(() => {});
      unshown = currentHref() === hrefOf(options) ? currentEntry() : undefined;
    },
    rendered(entry: string | undefined) {
      if (entry === unshown) unshown = undefined;
    },
  };
}
