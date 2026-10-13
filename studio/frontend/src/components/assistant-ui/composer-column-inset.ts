// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ViewportBox = {
  left: number;
  right: number;
  offsetWidth: number;
  clientWidth: number;
  borderLeft: number;
  borderRight: number;
  paddingLeft: number;
  paddingRight: number;
  /** `scrollbar-gutter` reserves both sides. */
  bothEdges: boolean;
  rtl: boolean;
};

/**
 * Padding that puts the dock's content box exactly on the thread viewport's content box, so the
 * composer and the message column share one width. The scrollbar strip is measured, not assumed:
 * it is 0 for overlay scrollbars and differs by platform.
 */
export function composerColumnInset(
  viewport: ViewportBox,
  dock: { left: number; right: number },
): { left: number; right: number } {
  const scrollbar = Math.max(
    0,
    viewport.offsetWidth -
      viewport.clientWidth -
      viewport.borderLeft -
      viewport.borderRight,
  );
  const [gutterLeft, gutterRight] = viewport.bothEdges
    ? [scrollbar / 2, scrollbar / 2]
    : viewport.rtl
      ? [scrollbar, 0]
      : [0, scrollbar];
  const contentLeft =
    viewport.left + viewport.borderLeft + gutterLeft + viewport.paddingLeft;
  const contentRight =
    viewport.right - viewport.borderRight - gutterRight - viewport.paddingRight;
  return {
    left: Math.max(0, contentLeft - dock.left),
    right: Math.max(0, dock.right - contentRight),
  };
}

export function readViewportBox(el: HTMLElement): ViewportBox {
  const style = getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return {
    left: rect.left,
    right: rect.right,
    offsetWidth: el.offsetWidth,
    clientWidth: el.clientWidth,
    borderLeft: Number.parseFloat(style.borderLeftWidth) || 0,
    borderRight: Number.parseFloat(style.borderRightWidth) || 0,
    paddingLeft: Number.parseFloat(style.paddingLeft) || 0,
    paddingRight: Number.parseFloat(style.paddingRight) || 0,
    bothEdges: style.scrollbarGutter.includes("both-edges"),
    rtl: style.direction === "rtl",
  };
}
