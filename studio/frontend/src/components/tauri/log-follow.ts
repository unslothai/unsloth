// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Fractional line heights and zoom leave the offset a hair short of the end. */
export const STICK_THRESHOLD_PX = 8;

export interface ScrollMetrics {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
}

export function isFollowingTail({
  scrollHeight,
  scrollTop,
  clientHeight,
}: ScrollMetrics): boolean {
  // Dimensions are 0 while the <details> is closed, which keeps the follow armed.
  if (scrollHeight <= clientHeight) {
    return true;
  }
  return scrollHeight - scrollTop - clientHeight <= STICK_THRESHOLD_PX;
}
