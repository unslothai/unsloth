// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// scrollIntoView also scrolls overflow-hidden ancestors, shifting the whole dialog; this scrolls the pane.
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
