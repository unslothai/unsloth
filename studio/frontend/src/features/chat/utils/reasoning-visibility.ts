// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Open/closed rules for a reasoning group, kept out of the component so the
// streaming and preference interplay stays testable.
export interface ReasoningOpenStateInput {
  /** Group is receiving reasoning deltas. */
  isStreaming: boolean;
  /** Settings -> Chat: keep thinking collapsed instead of streaming it open. */
  collapseByDefault: boolean;
  /** User closed the auto-opened block mid stream. */
  dismissedWhileStreaming: boolean;
  /** User opened the block by hand. */
  manualOpen: boolean;
}

/**
 * Thinking auto opens while it streams and collapses when it finishes, unless
 * the preference is on: then only an explicit open shows it, in either phase.
 */
export function resolveReasoningOpen({
  isStreaming,
  collapseByDefault,
  dismissedWhileStreaming,
  manualOpen,
}: ReasoningOpenStateInput): boolean {
  if (manualOpen) {
    return true;
  }
  if (collapseByDefault) {
    return false;
  }
  return isStreaming && !dismissedWhileStreaming;
}

/**
 * With the preference on there is no auto open to dismiss, so a mid stream
 * toggle has to land on the sticky manual flag instead.
 */
export function reasoningToggleTargetsManualState(
  isStreaming: boolean,
  collapseByDefault: boolean,
): boolean {
  return collapseByDefault || !isStreaming;
}
