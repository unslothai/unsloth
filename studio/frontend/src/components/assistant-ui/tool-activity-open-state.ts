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
}: ToolActivityTransition) {
  if (visibility !== previousVisibility) {
    return defaultOpenFor(visibility, isRunning && !hasText);
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
  open: boolean;
}

/** Same rule for uncontrolled cards and groups: leave them alone until the setting moves. */
export function syncToolActivityPreference(
  current: ToolActivityPreferenceState,
  visibility: DisplayVisibility,
  active: boolean,
) {
  if (current.visibility === visibility) {
    return current;
  }
  return {
    visibility,
    open: defaultOpenFor(visibility, active),
  };
}
