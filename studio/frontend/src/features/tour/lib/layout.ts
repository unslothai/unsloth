// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { Placement, Rect } from "../types";

export function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

export function padded(r: Rect, pad: number, vw: number, vh: number): Rect {
  const x = clamp(r.x - pad, 8, vw - 8);
  const y = clamp(r.y - pad, 8, vh - 8);
  const w = clamp(r.w + pad * 2, 24, vw - x - 8);
  const h = clamp(r.h + pad * 2, 24, vh - y - 8);
  return { x, y, w, h };
}

export function pickPlacement(
  target: Rect,
  card: { w: number; h: number },
  vw: number,
  vh: number,
  gap: number,
): Placement {
  const canRight = target.x + target.w + gap + card.w <= vw - 12;
  const canLeft = target.x - gap - card.w >= 12;
  const canBottom = target.y + target.h + gap + card.h <= vh - 12;
  const canTop = target.y - gap - card.h >= 12;

  if (canRight) return "right";
  if (canLeft) return "left";
  if (canBottom) return "bottom";
  if (canTop) return "top";
  return "bottom";
}

export function computeCardPos(
  placement: Placement,
  target: Rect,
  card: { w: number; h: number },
  vw: number,
  vh: number,
  gap: number,
): { left: number; top: number } {
  let left = 12;
  let top = 12;

  if (placement === "right") {
    left = target.x + target.w + gap;
    top = target.y + target.h / 2 - card.h / 2;
  }
  if (placement === "left") {
    left = target.x - gap - card.w;
    top = target.y + target.h / 2 - card.h / 2;
  }
  if (placement === "bottom") {
    left = target.x + target.w / 2 - card.w / 2;
    top = target.y + target.h + gap;
  }
  if (placement === "top") {
    left = target.x + target.w / 2 - card.w / 2;
    top = target.y - gap - card.h;
  }

  left = clamp(left, 12, vw - card.w - 12);
  top = clamp(top, 12, vh - card.h - 12);
  return { left, top };
}

