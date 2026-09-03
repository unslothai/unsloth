// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

interface ToolActivityTransition {
  currentOpen: boolean;
  collapseByDefault: boolean;
  previousCollapseByDefault: boolean;
  isRunning: boolean;
  hasText: boolean;
}

export function resolveToolActivityOpen({
  currentOpen,
  collapseByDefault,
  previousCollapseByDefault,
  isRunning,
  hasText,
}: ToolActivityTransition) {
  if (collapseByDefault) {
    return previousCollapseByDefault ? currentOpen : false;
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
  collapseByDefault: boolean;
  open: boolean;
}

export function syncToolActivityPreference(
  current: ToolActivityPreferenceState,
  collapseByDefault: boolean,
  defaultOpen: boolean,
) {
  if (current.collapseByDefault === collapseByDefault) {
    return current;
  }
  return {
    collapseByDefault,
    open: defaultOpen && !collapseByDefault,
  };
}
