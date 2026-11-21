// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type HoverFlyoutIntent = {
  readonly openDelay: number;
  readonly closeDelay: number;
};

type Point = { readonly x: number; readonly y: number };

type Box = {
  readonly left: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
};

export const NAV_TOOLTIP_INTENT = {
  delayDuration: 400,
  skipDelayDuration: 300,
  disableHoverableContent: true,
} as const;

export const NAV_FLYOUT_INTENT: HoverFlyoutIntent = {
  openDelay: 150,
  closeDelay: 300,
};

const EXIT_BLEED_PX = 5;

function isInside(point: Point, box: Box): boolean {
  return (
    point.x >= box.left &&
    point.x <= box.right &&
    point.y >= box.top &&
    point.y <= box.bottom
  );
}

function sideOf(a: Point, b: Point, point: Point): number {
  return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
}

function isInsideTriangle(point: Point, a: Point, b: Point, c: Point): boolean {
  const ab = sideOf(a, b, point);
  const bc = sideOf(b, c, point);
  const ca = sideOf(c, a, point);
  return (ab >= 0 && bc >= 0 && ca >= 0) || (ab <= 0 && bc <= 0 && ca <= 0);
}

export function isPointerHeadingInto(
  exit: Point,
  pointer: Point,
  target: Box,
): boolean {
  if (isInside(pointer, target)) return true;
  const headingRight = exit.x <= (target.left + target.right) / 2;
  const nearEdge = headingRight ? target.left : target.right;
  const apex = {
    x: exit.x + (headingRight ? -EXIT_BLEED_PX : EXIT_BLEED_PX),
    y: exit.y,
  };
  return isInsideTriangle(
    pointer,
    apex,
    { x: nearEdge, y: target.top },
    { x: nearEdge, y: target.bottom },
  );
}
