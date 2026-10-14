// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let release: (() => void) | null = null;

/** Hold the page at its size while the split is dragged, so it lays out once on release, not every frame. */
export function pinBrowserPage(handle: Element): () => void {
  // Only the inner box is pinned: the outer one keeps following the panel, so the room a widening
  // drag opens is painted in the page's background rather than the panel's.
  const area = document.querySelector<HTMLElement>("[data-browser-page]");
  const page = area?.querySelector<HTMLElement>(":scope > [data-browser-page]");
  if (release || !area || !page) return () => undefined;
  const saved = page.style.cssText;
  const rect = page.getBoundingClientRect();
  const box = area.getBoundingClientRect();
  // Anchored to the panel's outer edge, which stays put.
  const right = handle.getBoundingClientRect().left < rect.left;
  Object.assign(page.style, {
    position: "absolute",
    // Offsets are from the padding box, inside the area's top border.
    top: `${rect.top - box.top - area.clientTop}px`,
    bottom: "0",
    margin: "0",
    width: `${rect.width}px`,
    [right ? "right" : "left"]: `${right ? box.right - rect.right : rect.left - box.left}px`,
  });
  release = () => {
    page.style.cssText = saved;
    release = null;
  };
  return release;
}
