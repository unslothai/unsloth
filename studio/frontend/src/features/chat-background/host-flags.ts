// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useLayoutEffect } from "react";

/*
 * What the wallpaper layer shows, mirrored onto the element that holds it, so the wallpaper and
 * glass CSS keys on the host itself. `HOST:has(> .chat-wallpaper[data-ready]) X` puts the style
 * after the :has(): Chromium then re-checks every X-like element under HOST (the whole thread,
 * since the workspace holds it) whenever anything below HOST changes, and WebKit pays the same on
 * focus and input. Each flag is exactly its :has() argument, so the selectors match the same set.
 */
export const WALLPAPER_HOST_FLAGS = [
  ["data-wallpaper-ready", ":scope > .chat-wallpaper[data-ready]"],
  [
    "data-wallpaper-outline-off",
    ':scope > .chat-wallpaper[data-ready][data-outline="false"]',
  ],
  [
    "data-wallpaper-flat",
    ':scope > .chat-wallpaper[data-ready] > .chat-wallpaper-image[data-fade="false"]',
  ],
  [
    "data-wallpaper-frame-flat",
    ':scope > .chat-wallpaper[data-ready] > .chat-wallpaper-frame > .chat-wallpaper-image[data-fade="false"]',
  ],
  [
    "data-wallpaper-image",
    ":scope > .chat-wallpaper[data-ready]:not([data-glass-backdrop])",
  ],
  [
    "data-wallpaper-preview-outline-off",
    ':scope > .chat-wallpaper[data-outline="false"]',
  ],
] as const;

/** Recompute the host flags from the layers it holds now. Idempotent. */
export function syncWallpaperHost(host: Element | null | undefined): void {
  if (!host) return;
  for (const [name, selector] of WALLPAPER_HOST_FLAGS) {
    const on = host.querySelector(selector) !== null;
    // Only write on a change: an attribute write restyles the host even when the value is the same.
    if (on !== host.hasAttribute(name)) host.toggleAttribute(name, on);
  }
}

/**
 * Keep the parent's flags in step with this layer: before paint after every render, and once it is
 * gone. The cleanup re-reads in a microtask, after React has removed or updated the layer and still
 * before the frame paints. Unchanged flags are not rewritten, so a re-render costs a few lookups.
 */
export function useWallpaperHostFlags(
  layer: RefObject<HTMLElement | null>,
): void {
  useLayoutEffect(() => {
    const host = layer.current?.parentElement;
    if (!host) return;
    syncWallpaperHost(host);
    return () => queueMicrotask(() => syncWallpaperHost(host));
  });
}
