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

/** Dock padding that lands on the viewport's content box; the scrollbar is measured (0 when overlay). */
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
  return composerInsetFromColumn(
    {
      left:
        viewport.left + viewport.borderLeft + gutterLeft + viewport.paddingLeft,
      right:
        viewport.right -
        viewport.borderRight -
        gutterRight -
        viewport.paddingRight,
    },
    dock,
  );
}

export function composerInsetFromColumn(
  column: { left: number; right: number },
  dock: { left: number; right: number },
): { left: number; right: number } {
  return {
    left: Math.max(0, column.left - dock.left),
    right: Math.max(0, dock.right - column.right),
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
    bothEdges: (style.scrollbarGutter ?? "").includes("both-edges"),
    rtl: style.direction === "rtl",
  };
}
