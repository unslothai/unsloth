// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Marks which edges of the composer textarea have text scrolled past them, for the fades in index.css.
    Next frame: on input the autosize has not grown the box yet, so measuring now flags a bottom that is not cut. */
export function markComposerScrollEdges(el: HTMLTextAreaElement): void {
  requestAnimationFrame(() => {
    el.toggleAttribute("data-fade-top", el.scrollTop > 0);
    el.toggleAttribute(
      "data-fade-bottom",
      el.scrollTop + el.clientHeight < el.scrollHeight - 1,
    );
  });
}
