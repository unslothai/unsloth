// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Bring `el` to the top of the settings pane that scrolls it. scrollIntoView also scrolls every
 * ancestor, overflow-hidden ones included, so a section too near the end to reach the top shifted
 * the whole dialog, nav and all; this scrolls the pane alone and lets it clamp.
 */
export function scrollIntoPane(el: HTMLElement, offset = 24): void {
  let pane = el.parentElement;
  while (pane) {
    const overflowY = getComputedStyle(pane).overflowY;
    if (overflowY === "auto" || overflowY === "scroll") break;
    pane = pane.parentElement;
  }
  if (!pane) {
    el.scrollIntoView({ block: "start", behavior: "smooth" });
    return;
  }
  const top =
    el.getBoundingClientRect().top -
    pane.getBoundingClientRect().top +
    pane.scrollTop -
    offset;
  pane.scrollTo({ top, behavior: "smooth" });
}
