// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Marks which edges of the composer textarea have text scrolled past them, for the fades in index.css.
    Next frame: on input the autosize has not grown the box yet, so measuring now flags a bottom that is not cut.
    Bottom needs over half a line hidden past the padding: typing scrolls the caret line into view but
    leaves the padding and a few pixels below it out of view. */
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
