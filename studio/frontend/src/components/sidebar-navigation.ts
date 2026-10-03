// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

/**
 * Sidebar clicks: drop one on where the router is or is heading; send one made during a
 * sidebar load at once as a replace, so the router cancels the stale load and a burst leaves
 * one history entry. Never queue behind the in-flight load: a slow chunk would stall the click.
 */
export function createNavigationCoalescer<T>({
  navigate,
  currentHref,
  hrefOf,
  asReplace,
}: {
  navigate: (options: T) => Promise<unknown>;
  /** Where the router is, or is already heading. */
  currentHref: () => string;
  hrefOf: (options: T) => string;
  asReplace: (options: T) => T;
}): (options: T) => void {
  let latest = 0;
  let loading = false;

  return (options: T) => {
    if (hrefOf(options) === currentHref()) return;
    const id = ++latest;
    const replace = loading;
    loading = true;
    void navigate(replace ? asReplace(options) : options)
      .catch(() => {})
      .finally(() => {
        if (id === latest) loading = false;
      });
  };
}
