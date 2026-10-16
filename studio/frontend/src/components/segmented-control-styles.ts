// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SegmentedSize = "compact" | "default";

export const segmentedTrackHeight: Record<SegmentedSize, string> = {
  compact: "h-8",
  default: "h-9",
};

export const segmentedSegmentLabel =
  "whitespace-nowrap px-3 text-ui-12p5" as const;

/** Spring the design system gives segmented controls (design.md, 2.5 Motion). */
export const SEGMENTED_PILL_SPRING = {
  type: "spring",
  stiffness: 500,
  damping: 35,
  mass: 0.5,
} as const;
