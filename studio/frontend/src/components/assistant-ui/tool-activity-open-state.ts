// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- this file is in the startup cycle; the chat barrel closes it.
import {
  type DisplayVisibility,
  defaultOpenFor,
} from "@/features/chat/utils/display-visibility";

interface ToolActivityTransition {
  currentOpen: boolean;
  visibility: DisplayVisibility;
  previousVisibility: DisplayVisibility;
  isRunning: boolean;
  hasText: boolean;
  override?: boolean | null;
  /** True when the call starts again; regenerate reuses this card, so hand-set state is stale. */
  startedNewRound?: boolean;
}

/** Changing the setting re-applies it to every card; auto closes a hand-set card once text starts. */
export function resolveToolActivityOpen({
  currentOpen,
  visibility,
  previousVisibility,
  isRunning,
  hasText,
  startedNewRound = false,
  override = null,
}: ToolActivityTransition) {
  if (visibility !== previousVisibility || startedNewRound) {
    // A new round starts where the setting puts it; the old round's hand-set state is dropped.
    return defaultOpenFor(visibility, isRunning);
  }
  if (override !== null) {
    return override;
  }
  // Only auto keeps moving on its own; the other two stay where the user last put them.
  if (visibility !== "auto") {
    return currentOpen;
  }
  if (isRunning) {
    return true;
  }
  if (hasText) {
    return false;
  }
  return currentOpen;
}

export function startsNewToolRound(
  isRunning: boolean,
  wasRunning: boolean,
): boolean {
  return isRunning && !wasRunning;
}

export interface ToolActivityPreferenceState {
  visibility: DisplayVisibility;
  active: boolean;
  override: boolean | null;
}

/** A setting change hands the card back; activity only moves cards the user has not touched. */
export function syncToolActivityPreference(
  current: ToolActivityPreferenceState,
  visibility: DisplayVisibility,
  active: boolean,
): ToolActivityPreferenceState {
  if (current.visibility === visibility && current.active === active) {
    return current;
  }
  const kept =
    current.visibility === visibility && !startsNewToolRound(active, current.active);
  return {
    visibility,
    active,
    override: kept ? current.override : null,
  };
}

export function toolActivityOpen(state: ToolActivityPreferenceState): boolean {
  return state.override ?? defaultOpenFor(state.visibility, state.active);
}
