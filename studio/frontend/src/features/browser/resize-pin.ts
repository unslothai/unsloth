// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Hold the panel's page at its width while the split is dragged, so a page, PDF or native view
 * lays out once on release rather than every frame. Returns the release.
 */
export function pinBrowserPage(handle: Element): () => void {
  const page = document.querySelector<HTMLElement>("[data-browser-page]");
  if (!page) return () => undefined;
  const saved = page.style.cssText;
  const rect = page.getBoundingClientRect();
  // Anchored to the panel's outer edge, which stays put.
  const side = handle.getBoundingClientRect().left < rect.left ? "right" : "left";
  Object.assign(page.style, { position: "absolute", top: "0", bottom: "0", [side]: "0", width: `${rect.width}px` });
  return () => {
    page.style.cssText = saved;
  };
}
