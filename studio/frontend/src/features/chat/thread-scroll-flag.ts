// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Header fades show only once their thread scrolls under them: over a flat page every engine rounds
// some gradient pixels a level off the page colour (Firefox in any space, WebKit in sRGB).
export const THREAD_SCROLLED_ATTR = "data-thread-scrolled";
const HOST_ATTR = "data-thread-scroll-host";
// Thread viewports, plus the project landing list under the same header.
const SCROLLERS = ".aui-thread-viewport, [data-header-fade-scroller]";

function onThreadScroll(this: HTMLElement, event: Event): void {
  const viewport = event.target;
  if (!(viewport instanceof HTMLElement) || !viewport.matches(SCROLLERS)) return;
  // Compare panes are hosts nested in the chat root, which keeps the backgrounded chat mounted.
  if (viewport.closest(`[${HOST_ATTR}]`) !== this || viewport.closest(".chat-full-view-dock")) return;
  this.toggleAttribute(THREAD_SCROLLED_ATTR, viewport.scrollTop > 0);
}

// Stable callback ref: scroll does not bubble, so listen in the capture phase on the fade's parent.
export function trackThreadScroll(host: HTMLElement | null): (() => void) | undefined {
  if (!host) return undefined;
  host.setAttribute(HOST_ATTR, "");
  host.addEventListener("scroll", onThreadScroll, { capture: true, passive: true });
  return () => {
    host.removeEventListener("scroll", onThreadScroll, { capture: true });
    host.removeAttribute(HOST_ATTR);
  };
}
