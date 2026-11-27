// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ClassValue, clsx } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

// Register text-ui-* as font-size so twMerge does not drop it as a color when a color follows.
const isUiToken = (value: string) => /^ui-\d+(p5)?$/.test(value);

const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      "font-size": [{ text: [isUiToken] }],
      leading: [{ leading: [isUiToken] }],
      // index.css `h-control`, `size-control`, `min-h-menu-row`: so a caller's h-8, size-8 or min-h-0 replaces it instead of both applying.
      h: [{ h: ["control"] }],
      size: [{ size: ["control"] }],
      "min-h": [{ "min-h": ["menu-row"] }],
    },
  },
});

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatCompact(n: number): string {
  if (n >= 1_000_000_000) {
    return `${(n / 1_000_000_000).toFixed(1)}B`;
  }
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M`;
  }
  if (n >= 1_000) {
    return `${(n / 1_000).toFixed(1)}K`;
  }
  return String(n);
}
