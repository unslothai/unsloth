// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type DisplayVisibility,
  defaultOpenFor,
  resolveOpen,
} from "./display-visibility";

export interface ReasoningOpenStateInput {
  isStreaming: boolean;
  visibility: DisplayVisibility;
  override: boolean | null;
}

export function resolveReasoningOpen({
  isStreaming,
  visibility,
  override,
}: ReasoningOpenStateInput): boolean {
  return resolveOpen(visibility, isStreaming, override);
}

/** `auto` opens a block only for its stream, so it shows the newest lines in a bounded box
 *  instead of growing the thread (#11703). Opening it by hand, or `expanded`, shows it all. */
export function reasoningTailCapped({
  visibility,
  override,
}: Omit<ReasoningOpenStateInput, "isStreaming">): boolean {
  return visibility === "auto" && override !== true;
}

/** Whether the block sits where the setting alone would put it. */
export function reasoningFollowsPreference(
  open: boolean,
  isStreaming: boolean,
  visibility: DisplayVisibility,
): boolean {
  return open === defaultOpenFor(visibility, isStreaming);
}

/** Regenerate reuses the component, so clear the override in the same render. */
export function startsNewReasoningRound(
  isStreaming: boolean,
  wasStreaming: boolean,
): boolean {
  return isStreaming && !wasStreaming;
}
