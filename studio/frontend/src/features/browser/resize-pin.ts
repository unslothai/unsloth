// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let release: (() => void) | null = null;

/** Hold the page at its size while the split is dragged, so it lays out once on release, not every frame. */
export function pinBrowserPage(handle: Element): () => void {
  const page = document.querySelector<HTMLElement>("[data-browser-page]");
  if (release || !page?.parentElement) return () => undefined;
  const saved = page.style.cssText;
  const rect = page.getBoundingClientRect();
  const box = page.parentElement.getBoundingClientRect();
  // Anchored to the panel's outer edge, which stays put.
  const right = handle.getBoundingClientRect().left < rect.left;
  Object.assign(page.style, {
    position: "absolute",
    top: `${rect.top - box.top}px`,
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
