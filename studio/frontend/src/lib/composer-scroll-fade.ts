// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Flags the composer edges with text scrolled past, for index.css's fades. Next frame, after autosize;
    bottom needs over half a line past the padding, which typing alone leaves hidden.
    `data-scrollable`: the text outgrows the autosized height (past max rows), so only then may it
    scroll; the frame easing up to a new line clips it briefly and must not flash a scrollbar. */
export function markComposerScrollEdges(el: HTMLTextAreaElement): void {
  requestAnimationFrame(() => {
    const style = getComputedStyle(el);
    const slack =
      (Number.parseFloat(style.paddingBottom) || 0) +
      (Number.parseFloat(style.lineHeight) || 24) / 2;
    el.toggleAttribute("data-fade-top", el.scrollTop > 0);
    el.toggleAttribute(
      "data-fade-bottom",
      el.scrollHeight - (el.scrollTop + el.clientHeight) > slack,
    );
    markComposerScrollable(el);
  });
}

/** The `data-scrollable` half alone, for callers that run after autosize has set the height. */
export function markComposerScrollable(el: HTMLTextAreaElement): void {
  const sized = Number.parseFloat(el.style.height) || el.clientHeight;
  el.toggleAttribute("data-scrollable", el.scrollHeight > sized + 1);
}
