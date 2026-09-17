// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Open/closed rules for a reasoning group, kept out of the component so the streaming and
// preference interplay stays testable.
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

/** Thinking auto opens while it streams and collapses when it finishes, unless the preference is
 *  on: then only an explicit open shows it, in either phase. */
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

/** Whether the block opens on its own for this round, so a toggle dismisses it. */
export function reasoningAutoOpensWhileStreaming(
  isStreaming: boolean,
  collapseByDefault: boolean,
): boolean {
  return isStreaming && !collapseByDefault;
}

/** A new round starts when streaming resumes. Regenerate reuses the component, so last round's
 *  open state has to clear in that same render, not in an effect, or the block paints open
 *  before collapsing. */
export function startsNewReasoningRound(
  isStreaming: boolean,
  wasStreaming: boolean,
): boolean {
  return isStreaming && !wasStreaming;
}

export interface ReasoningToggleResult {
  /** Sticky user open. Cleared on close so a preference flip cannot pin it. */
  manualOpen: boolean;
  /** Set only when the block auto opens, since that is what gets dismissed. */
  dismissedWhileStreaming?: boolean;
  /** Drop the streaming height cap so a hand-opened block shows in full. */
  releaseStreamingHeight: boolean;
}

/** Resolves a trigger click into the next open state. */
export function resolveReasoningToggle(
  open: boolean,
  {
    isStreaming,
    collapseByDefault,
  }: Pick<ReasoningOpenStateInput, "isStreaming" | "collapseByDefault">,
): ReasoningToggleResult {
  const autoOpens = reasoningAutoOpensWhileStreaming(
    isStreaming,
    collapseByDefault,
  );
  return {
    manualOpen: open && !autoOpens,
    ...(autoOpens ? { dismissedWhileStreaming: !open } : {}),
    releaseStreamingHeight: open && !autoOpens,
  };
}
