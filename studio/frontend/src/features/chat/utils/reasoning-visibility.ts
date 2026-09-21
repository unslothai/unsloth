// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type DisplayVisibility,
  defaultOpenFor,
  resolveOpen,
} from "./display-visibility";

// Open/closed rules for a reasoning group, kept out of the component so the streaming and
// preference interplay stays testable.
export interface ReasoningOpenStateInput {
  /** Group is receiving reasoning deltas. */
  isStreaming: boolean;
  /** Settings -> Display: collapsed, auto (open while streaming) or expanded (always open). */
  visibility: DisplayVisibility;
  /** What the user did to this block by hand: null until they click its trigger. */
  override: boolean | null;
}

/** Thinking follows the setting until the user toggles this block. */
export function resolveReasoningOpen({
  isStreaming,
  visibility,
  override,
}: ReasoningOpenStateInput): boolean {
  return resolveOpen(visibility, isStreaming, override);
}

/** Whether the block sits where the setting alone would put it. */
export function reasoningFollowsPreference(
  open: boolean,
  isStreaming: boolean,
  visibility: DisplayVisibility,
): boolean {
  return open === defaultOpenFor(visibility, isStreaming);
}

/** A new round starts when streaming resumes. Regenerate reuses the component, so the previous
 *  round's override has to clear in that same render, not in an effect. */
export function startsNewReasoningRound(
  isStreaming: boolean,
  wasStreaming: boolean,
): boolean {
  return isStreaming && !wasStreaming;
}

export interface ReasoningToggleResult {
  /** The user's choice for this block, kept until the round restarts or the setting changes. */
  override: boolean;
  /** Drop the streaming height cap so a hand-opened block shows in full. */
  releaseStreamingHeight: boolean;
}

/** Resolves a trigger click into the next open state. Every click records an override, which is
 *  what keeps a hand-opened block open once its stream ends. */
export function resolveReasoningToggle(
  open: boolean,
  {
    isStreaming,
    visibility,
  }: Pick<ReasoningOpenStateInput, "isStreaming" | "visibility">,
): ReasoningToggleResult {
  return {
    override: open,
    // A block that opened on its own keeps the streaming cap. One opened against the setting
    // is there to be read, so it gets its full height.
    releaseStreamingHeight:
      open && !defaultOpenFor(visibility, isStreaming),
  };
}
