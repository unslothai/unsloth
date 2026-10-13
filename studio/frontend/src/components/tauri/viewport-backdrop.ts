// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const BACKDROP_ATTRIBUTES = [
  "data-state",
  "data-viewport-backdrop",
  "data-viewport-backdrop-tone",
];

let backdropOpen = false;
// A popup that dims darker than the standard backdrop names its tone, so the titlebar matches it.
let backdropTone: string | null = null;
const backdropListeners = new Set<() => void>();
let bodyChildObserver: MutationObserver | null = null;
let backdropStateObserver: MutationObserver | null = null;

function readViewportBackdrop(): void {
  let next = false;
  let tone: string | null = null;
  for (const child of document.body.children) {
    if (
      child.getAttribute("data-viewport-backdrop") === "true" &&
      child.getAttribute("data-state") === "open"
    ) {
      next = true;
      tone = child.getAttribute("data-viewport-backdrop-tone");
      break;
    }
  }
  if (next === backdropOpen && tone === backdropTone) return;
  backdropOpen = next;
  backdropTone = tone;
  for (const listener of backdropListeners) listener();
}

function watchBodyChildren(): void {
  if (!backdropStateObserver) return;
  backdropStateObserver.disconnect();
  for (const child of document.body.children) {
    backdropStateObserver.observe(child, {
      attributes: true,
      attributeFilter: BACKDROP_ATTRIBUTES,
    });
  }
  readViewportBackdrop();
}

export function subscribeViewportBackdrop(listener: () => void): () => void {
  const subscription = () => listener();
  backdropListeners.add(subscription);
  if (!bodyChildObserver && typeof MutationObserver !== "undefined") {
    backdropStateObserver = new MutationObserver(readViewportBackdrop);
    bodyChildObserver = new MutationObserver(watchBodyChildren);
    bodyChildObserver.observe(document.body, { childList: true });
    watchBodyChildren();
  }
  return () => {
    backdropListeners.delete(subscription);
    if (backdropListeners.size > 0) return;
    bodyChildObserver?.disconnect();
    backdropStateObserver?.disconnect();
    bodyChildObserver = null;
    backdropStateObserver = null;
    backdropOpen = false;
    backdropTone = null;
  };
}

export function getViewportBackdropOpen(): boolean {
  return backdropOpen;
}

export function getViewportBackdropTone(): string | null {
  return backdropTone;
}
