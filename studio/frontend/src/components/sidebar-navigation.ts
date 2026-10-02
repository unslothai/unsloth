// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

/**
 * Sidebar clicks, one route load at a time, last click wins.
 *
 * Every navigation runs the async auth and device guards and a lazy chunk, so spam-clicking
 * rows used to stack one full load (and one history entry) per click. A click on where the
 * router already is, or is already heading, is dropped; clicks during a load only replace the
 * next target.
 */
export function createNavigationCoalescer<T>({
  navigate,
  currentHref,
  hrefOf,
}: {
  navigate: (options: T) => Promise<unknown>;
  /** Where the router is, or is already heading. */
  currentHref: () => string;
  hrefOf: (options: T) => string;
}): (options: T) => void {
  let inFlight = false;
  let next: T | null = null;

  function go(options: T) {
    if (inFlight) {
      next = options;
      return;
    }
    if (hrefOf(options) === currentHref()) return;
    inFlight = true;
    void navigate(options)
      .catch(() => {})
      .finally(() => {
        inFlight = false;
        const queued = next;
        next = null;
        if (queued !== null) go(queued);
      });
  }

  return go;
}
