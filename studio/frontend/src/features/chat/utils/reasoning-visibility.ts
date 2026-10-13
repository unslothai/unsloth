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

/** Under `auto` a block shows a bounded box (#11703) until its Expand toggle lifts it. */
export function reasoningTailCapped({
  visibility,
}: Pick<ReasoningOpenStateInput, "visibility">): boolean {
  return visibility === "auto";
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
