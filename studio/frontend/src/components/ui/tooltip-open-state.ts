// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Radix always gets a boolean, so no open state of its own can survive a modal. */
export function resolveTooltipOpen(state: {
  blocked: boolean;
  controlledOpen?: boolean;
  /** A modal blocked it and the owner has not said false since. */
  dismissedUntilOwnerResets: boolean;
  hoverOpen: boolean;
  /** Pinned by a tap, which has no hover to close it. */
  clickOpen: boolean;
}): boolean {
  if (state.blocked) return false;
  if (state.controlledOpen !== undefined) {
    // An owner with no onOpenChange never learns its trigger lost the pointer; wait for false once.
    return state.controlledOpen && !state.dismissedUntilOwnerResets;
  }
  return state.hoverOpen || state.clickOpen;
}
