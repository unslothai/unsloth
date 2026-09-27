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
