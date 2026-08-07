// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How a sidebar nav row renders once its pending state is folded in. Its own import-free
// module so it is testable: app-sidebar.tsx pulls in the whole shell.

export type NavRowState = {
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
  pending?: boolean;
};

/**
 * Fold `pending` into the props the row renders with.
 *
 * A guessed gray-out reads exactly like a measured one, so while the capability verdict is out
 * the row stays enabled and spins instead: the user sees "still checking", and the click lands
 * on a page that shows its own loading state. Both render sites (the inline rows and the More
 * flyout) go through here so they cannot drift.
 */
export function resolveNavRowState(row: NavRowState): {
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
} {
  if (row.pending) {
    return { disabled: false, tooltip: undefined, spinner: true };
  }
  return { disabled: row.disabled, tooltip: row.tooltip, spinner: row.spinner };
}
