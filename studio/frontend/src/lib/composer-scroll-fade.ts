// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Flags the composer edges with text scrolled past, for index.css's fades. Next frame, after autosize;
    bottom needs over half a line past the padding, which typing alone leaves hidden. */
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
  });
}
