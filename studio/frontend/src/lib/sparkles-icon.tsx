// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";

// Hugeicons "AI Security 03" (stroke-rounded). A four-point sparkle inside a
// shield. https://hugeicons.com/icon/ai-security-03
export const SparklesIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M11.6769 8.67348C11.8274 8.43697 12.1726 8.43697 12.3231 8.67348L12.7586 9.35767C13.2401 10.1143 13.8818 10.756 14.6384 11.2375L15.3226 11.6729C15.5591 11.8235 15.5591 12.1687 15.3226 12.3192L14.6384 12.7547C13.8818 13.2362 13.2401 13.8779 12.7586 14.6345L12.3231 15.3187C12.1726 15.5552 11.8274 15.5552 11.6769 15.3187L11.2414 14.6345C10.7599 13.8779 10.1182 13.2362 9.36157 12.7547L8.67738 12.3192C8.44087 12.1687 8.44087 11.8235 8.67738 11.6729L9.36157 11.2375C10.1182 10.756 10.7599 10.1143 11.2414 9.35767L11.6769 8.67348Z",
      stroke: "currentColor",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M3.9068 5.28387C6.87149 5.4984 8.78311 2.49713 12.0262 2.49713C15.2208 2.43341 16.784 5.32395 20.059 5.32395C21.8147 14.2606 18.1622 19.8743 12.053 21.4961C6.38992 20.15 2.13481 14.4788 3.9068 5.28387Z",
      stroke: "currentColor",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
];

/**
 * AI Security shield glyph wrapped as a lucide-compatible component so it can drop
 * into the permission-mode option list alongside lucide icons (same className /
 * strokeWidth props). strokeWidth is a number here (lucide style); Hugeicons
 * accepts it directly.
 */
export function SparklesGlyph({
  className,
  strokeWidth,
}: {
  className?: string;
  strokeWidth?: number;
}) {
  return (
    <HugeiconsIcon
      icon={SparklesIcon}
      className={className}
      strokeWidth={strokeWidth}
    />
  );
}
