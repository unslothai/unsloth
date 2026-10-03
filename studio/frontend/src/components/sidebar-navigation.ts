// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

/**
 * Sidebar clicks: drop one on where the router is or is heading; while the current entry is
 * still loading (never shown), send the click at once as a replace, so the router cancels the
 * stale load and a burst leaves one history entry. Asks the router instead of tracking our own
 * promise: a blocker (unsaved Library note) holds a navigation before it touches history, and
 * a blocked one never settles, so the shown entry must still get a push.
 */
export function createNavigationCoalescer<T>({
  navigate,
  currentHref,
  hrefOf,
  entryShown,
  asReplace,
}: {
  navigate: (options: T) => Promise<unknown>;
  /** Where the router is, or is already heading. */
  currentHref: () => string;
  hrefOf: (options: T) => string;
  /** Whether the current history entry has rendered. */
  entryShown: () => boolean;
  asReplace: (options: T) => T;
}): (options: T) => void {
  return (options: T) => {
    if (hrefOf(options) === currentHref()) return;
    void navigate(entryShown() ? options : asReplace(options)).catch(() => {});
  };
}
