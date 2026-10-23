// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where a sidebar nav row goes and how it renders once its pending state is folded in. Its own
// import-free module so it is testable: app-sidebar.tsx pulls in the whole shell.

export type NavRowState = {
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
  pending?: boolean;
  /** What the row says while `pending`. Falls back to the row's own label. */
  pendingTooltip?: string;
};

/**
 * Fold `pending` into the props the row renders with.
 *
 * A guessed gray-out reads exactly like a measured one, so while the capability verdict is out
 * the row stays enabled and spins instead: the user sees "still checking", and the click lands
 * on a page that shows its own loading state. Both render sites (the inline rows and the More
 * flyout) go through here so they cannot drift.
 *
 * A pending row carries its own tooltip rather than the disabled hint, which would state a
 * verdict nobody has reached, or nothing, which reads as a hung row.
 */
export function resolveNavRowState(row: NavRowState): {
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
  /** Whether this tooltip belongs to a pending row, which both renderers hide by default. */
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

/**
 * Split the nav rows between the sidebar and its "More" flyout, keeping their order.
 *
 * More needs two or more rows to be worth a click: with exactly one unpinned, the menu and that
 * row are both dropped. `shown` is a row the current page puts on the sidebar without pinning it,
 * in its own slot. It leaves More, but the count is taken first, so surfacing it never hides
 * another row.
 */
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
