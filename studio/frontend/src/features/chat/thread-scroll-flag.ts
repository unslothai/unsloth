// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Header fades show only once their thread scrolls under them: over a flat page every engine rounds
// some gradient pixels a level off the page colour (Firefox in any space, WebKit in sRGB).
export const THREAD_SCROLLED_ATTR = "data-thread-scrolled";
// Static in the JSX, not set by the ref: React attaches child refs first, and they sync on mount.
export const HOST_ATTR = "data-thread-scroll-host";
// Thread viewports, plus the project landing list under the same header.
const SCROLLERS = ".aui-thread-viewport, [data-header-fade-scroller]";

// Also a callback ref on each scroller: a replacement mounts at scrollTop 0 and fires no scroll event.
export function syncThreadScroll(scroller: HTMLElement | null): void {
  // Compare panes are hosts nested in the chat root, which keeps the backgrounded chat mounted.
  const host = scroller?.closest<HTMLElement>(`[${HOST_ATTR}]`);
  if (!host || !scroller) return;
  host.toggleAttribute(THREAD_SCROLLED_ATTR, scroller.scrollTop > 0);
}

function onThreadScroll(this: HTMLElement, event: Event): void {
  const scroller = event.target;
  if (!(scroller instanceof HTMLElement) || !scroller.matches(SCROLLERS)) return;
  if (scroller.closest(`[${HOST_ATTR}]`) === this) syncThreadScroll(scroller);
}

// Stable callback ref: scroll does not bubble, so listen in the capture phase on the fade's parent.
export function trackThreadScroll(host: HTMLElement | null): (() => void) | undefined {
  if (!host) return undefined;
  host.addEventListener("scroll", onThreadScroll, { capture: true, passive: true });
  return () => host.removeEventListener("scroll", onThreadScroll, { capture: true });
}
