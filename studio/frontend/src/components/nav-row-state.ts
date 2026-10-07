// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: app-sidebar.tsx pulls in the whole shell.

export type NavRowState = {
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
  pending?: boolean;
  pendingTooltip?: string;
};

/**
 * A pending row stays enabled and spins rather than guessing a gray-out. Both render sites go
 * through here so they cannot drift.
 */
export function resolveNavRowState(row: NavRowState): {
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
  pending: boolean;
} {
  if (row.pending) {
    return {
      disabled: false,
      tooltip: row.pendingTooltip,
      spinner: true,
      pending: true,
    };
  }
  return {
    disabled: row.disabled,
    tooltip: row.tooltip,
    spinner: row.spinner,
    pending: false,
  };
}

/** More needs two or more rows; `shown` leaves More but is counted first. */
export function placeNavRows<Id extends string>(
  rows: readonly { id: Id; pinned: boolean }[],
  shown: Id | null,
): { inline: Id[]; overflow: Id[] } {
  const unpinned = rows.filter((row) => !row.pinned).map((row) => row.id);
  return {
    inline: rows
      .filter((row) => row.pinned || row.id === shown)
      .map((row) => row.id),
    overflow: unpinned.length > 1 ? unpinned.filter((id) => id !== shown) : [],
  };
}
