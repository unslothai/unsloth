// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const BACKDROP_ATTRIBUTES = ["data-state", "data-viewport-backdrop"];

let backdropOpen = false;
const backdropListeners = new Set<() => void>();
let bodyChildObserver: MutationObserver | null = null;
let backdropStateObserver: MutationObserver | null = null;

function readViewportBackdrop(): void {
  let next = false;
  for (const child of document.body.children) {
    if (
      child.getAttribute("data-viewport-backdrop") === "true" &&
      child.getAttribute("data-state") === "open"
    ) {
      next = true;
      break;
    }
  }
  if (next === backdropOpen) return;
  backdropOpen = next;
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
  };
}

export function getViewportBackdropOpen(): boolean {
  return backdropOpen;
}
