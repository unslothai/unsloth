// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const THREAD_WIDTH_STEPS = [
  { attribute: "data-thread-narrow", rem: 31.5, bound: "max" },
  { attribute: "data-thread-tiny", rem: 18.75, bound: "max" },
  { attribute: "data-thread-wide", rem: 36, bound: "min" },
] as const;

export function applyThreadWidthSteps(
  root: Element,
  width: number,
  remPx: number,
): void {
  for (const { attribute, rem, bound } of THREAD_WIDTH_STEPS) {
    const limit = rem * remPx;
    root.toggleAttribute(
      attribute,
      bound === "max" ? width <= limit : width >= limit,
    );
  }
}

export function observeThreadWidthSteps(root: HTMLElement): () => void {
  if (typeof ResizeObserver === "undefined") return () => {};
  const observer = new ResizeObserver((entries) => {
    const entry = entries[entries.length - 1];
    if (!entry) return;
    const remPx =
      Number.parseFloat(getComputedStyle(document.documentElement).fontSize) ||
      16;
    applyThreadWidthSteps(root, entry.contentRect.width, remPx);
  });
  observer.observe(root);
  return () => observer.disconnect();
}
