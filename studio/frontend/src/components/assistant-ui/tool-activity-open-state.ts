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
  /** The answer has started, so the call is finished and no longer worth watching. */
  hasText: boolean;
  /** null until the user clicks this card's trigger. */
  override?: boolean | null;
}

/** Where a card sits after the setting, the run or the answer changed. Changing the setting
 *  re-applies it to every card, so Settings reaches messages already on screen. Between changes
 *  a hand-set card keeps its state, except that auto still closes it once the answer starts. */
export function resolveToolActivityOpen({
  currentOpen,
  visibility,
  previousVisibility,
  isRunning,
  hasText,
  override = null,
}: ToolActivityTransition) {
  if (visibility !== previousVisibility) {
    return defaultOpenFor(visibility, isRunning && !hasText);
  }
  // A click outranks the automatic rules until the setting changes, so a card the user opened
  // stays open when the answer text arrives.
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

export interface ToolActivityPreferenceState {
  visibility: DisplayVisibility;
  /** Still running. A live prop, so it falls to false when the call finishes. */
  active: boolean;
  /** null until the user clicks the trigger. */
  override: boolean | null;
}

/** The state uncontrolled cards and groups keep. A setting change hands the card back to the
 *  setting; activity changing on its own only moves cards the user has not touched, so auto can
 *  close a card when its call ends without discarding a manual open. */
export function syncToolActivityPreference(
  current: ToolActivityPreferenceState,
  visibility: DisplayVisibility,
  active: boolean,
): ToolActivityPreferenceState {
  if (current.visibility === visibility && current.active === active) {
    return current;
  }
  return {
    visibility,
    active,
    override: current.visibility === visibility ? current.override : null,
  };
}

/** Whether such a card is open right now. */
export function toolActivityOpen(state: ToolActivityPreferenceState): boolean {
  return state.override ?? defaultOpenFor(state.visibility, state.active);
}
