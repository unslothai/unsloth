// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

export const COMPOSER_MIN_ROWS = 2;
const COMPOSER_MAX_ROWS = 10;
const COMPOSER_FLOOR_ROWS = 3;
const COMPOSER_VIEWPORT_SHARE = 0.3;

export function composerMaxRows(
  viewportHeight: number,
  rowHeight: number,
): number {
  const fit = Math.floor(
    (viewportHeight * COMPOSER_VIEWPORT_SHARE) / rowHeight,
  );
  return Math.max(COMPOSER_FLOOR_ROWS, Math.min(COMPOSER_MAX_ROWS, fit));
}

function subscribe(callback: () => void): () => void {
  window.addEventListener("resize", callback);
  return () => window.removeEventListener("resize", callback);
}

/** The row count as snapshot, so a window drag re-renders only when it changes. */
export function useComposerMaxRows(rowHeight: number): number {
  return useSyncExternalStore(
    subscribe,
    () => composerMaxRows(window.innerHeight, rowHeight),
    () => COMPOSER_MAX_ROWS,
  );
}
