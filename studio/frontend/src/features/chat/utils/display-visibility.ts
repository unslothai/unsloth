// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How Thinking blocks and tool activity open. Shared by Settings -> Display and every block.
//   collapsed: never opens on its own.
//   auto:      opens while streaming or running, closes once the answer starts.
//   expanded:  stays open in every phase, streaming included.
// A block toggled by hand wins over the setting for the rest of its round.
export const DISPLAY_VISIBILITIES = ["collapsed", "auto", "expanded"] as const;

export type DisplayVisibility = (typeof DISPLAY_VISIBILITIES)[number];

export const DEFAULT_THINKING_VISIBILITY: DisplayVisibility = "auto";
// There is usually a lot more tool activity than reasoning, so it starts collapsed.
export const DEFAULT_TOOL_VISIBILITY: DisplayVisibility = "collapsed";

function isDisplayVisibility(value: unknown): value is DisplayVisibility {
  return DISPLAY_VISIBILITIES.includes(value as DisplayVisibility);
}

/** A stored value from a newer build, or a hand-edited one, must not leave a select blank. */
export function normaliseDisplayVisibility(
  value: unknown,
  fallback: DisplayVisibility,
): DisplayVisibility {
  return isDisplayVisibility(value) ? value : fallback;
}

/** Falls back to the boolean older builds wrote: `true` meant collapsed, `false` meant auto. */
export function migrateVisibility(
  stored: unknown,
  legacyCollapseByDefault: unknown,
  fallback: DisplayVisibility,
): DisplayVisibility {
  if (isDisplayVisibility(stored)) {
    return stored;
  }
  if (typeof legacyCollapseByDefault === "boolean") {
    return legacyCollapseByDefault ? "collapsed" : "auto";
  }
  return fallback;
}

/** Where a block sits before anyone touches it. `active` is streaming in, or still running. */
export function defaultOpenFor(
  visibility: DisplayVisibility,
  active: boolean,
): boolean {
  switch (visibility) {
    case "expanded":
      return true;
    case "collapsed":
      return false;
    default:
      return active;
  }
}

/** `override` is null until the user clicks the trigger, then holds for the rest of the round. */
export function resolveOpen(
  visibility: DisplayVisibility,
  active: boolean,
  override: boolean | null,
): boolean {
  return override ?? defaultOpenFor(visibility, active);
}

/** Whether the fold preference applies. Folding hides calls behind a Thinking block, so it
 *  cannot coexist with always expanded: the calls would be pinned open somewhere closed.
 *  Always expanded wins, and the fold preference stays stored so switching back restores it. */
export function foldIsActive(
  foldToolsIntoThinking: boolean,
  toolVisibility: DisplayVisibility,
): boolean {
  return foldToolsIntoThinking && toolVisibility !== "expanded";
}
