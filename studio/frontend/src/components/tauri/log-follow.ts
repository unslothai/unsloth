// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How close to the bottom still counts as "at the bottom". Fractional line heights and
 * browser zoom leave the scroll offset a hair short of the end, so an exact comparison
 * would read as a manual scroll-up on the user's behalf and stop the follow.
 */
export const STICK_THRESHOLD_PX = 8;

/** The three scroll metrics the decision needs, so callers can pass a plain object. */
export interface ScrollMetrics {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
}

/**
 * Whether a scrolling log should keep following its tail. Separate from the component so
 * the thresholds can be exercised without a DOM.
 */
export function isFollowingTail({
  scrollHeight,
  scrollTop,
  clientHeight,
}: ScrollMetrics): boolean {
  // A log shorter than its box never scrolls, so it is always at its own end. Reported
  // dimensions are also 0 while the <details> is closed, which lands here and keeps the
  // follow armed for whenever it opens.
  if (scrollHeight <= clientHeight) {
    return true;
  }
  return scrollHeight - scrollTop - clientHeight <= STICK_THRESHOLD_PX;
}
