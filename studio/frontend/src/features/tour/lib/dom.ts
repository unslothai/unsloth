// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { Rect } from "../types";

export function cssEscape(value: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/"/g, '\\"');
}

export function toRect(domRect: DOMRect): Rect {
  return {
    x: domRect.left,
    y: domRect.top,
    w: domRect.width,
    h: domRect.height,
  };
}

