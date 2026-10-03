// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

/**
 * Sidebar clicks, last click wins at once, one history entry per burst.
 *
 * A click on where the router already is, or is already heading, is dropped. A click while
 * another sidebar navigation is still loading goes out immediately as a replace: the router
 * cancels the stale load, and the entry it pushed (a page never shown) is overwritten rather
 * than stacked. Never wait for the in-flight load: a slow route chunk would hold every later
 * click until it lands.
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
