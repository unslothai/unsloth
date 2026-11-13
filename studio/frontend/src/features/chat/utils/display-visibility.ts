// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// collapsed: never opens; auto: open while streaming; expanded: always open.
// A manual toggle wins for the rest of the round.
export const DISPLAY_VISIBILITIES = ["collapsed", "auto", "expanded"] as const;

export type DisplayVisibility = (typeof DISPLAY_VISIBILITIES)[number];

export const DEFAULT_THINKING_VISIBILITY: DisplayVisibility = "auto";
export const DEFAULT_TOOL_VISIBILITY: DisplayVisibility = "collapsed";

function isDisplayVisibility(value: unknown): value is DisplayVisibility {
  return DISPLAY_VISIBILITIES.includes(value as DisplayVisibility);
}

export function normaliseDisplayVisibility(
  value: unknown,
  fallback: DisplayVisibility,
): DisplayVisibility {
  return isDisplayVisibility(value) ? value : fallback;
}

/** Older builds stored a boolean: `true` meant collapsed, `false` meant auto. */
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

export function resolveOpen(
  visibility: DisplayVisibility,
  active: boolean,
  override: boolean | null,
): boolean {
  return override ?? defaultOpenFor(visibility, active);
}

/** Always expanded wins over folding; the fold preference stays stored. */
export function foldIsActive(
  foldToolsIntoThinking: boolean,
  toolVisibility: DisplayVisibility,
): boolean {
  return foldToolsIntoThinking && toolVisibility !== "expanded";
}
