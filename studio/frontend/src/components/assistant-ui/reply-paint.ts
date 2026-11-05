// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keep the normal overflow of replies near the user (focus rings, menus and wide math).
// One observer per scroller also handles side-by-side threads independently.
const observers = new WeakMap<Element, IntersectionObserver>();

export function observeReplyPaint(reply: HTMLElement) {
  const viewport = reply.closest(".aui-thread-viewport");
  if (!viewport || typeof IntersectionObserver === "undefined") return;

  let observer = observers.get(viewport);
  if (!observer) {
    observer = new IntersectionObserver((entries) => {
      // Read before writing any styles. A wide reply can own the thread's horizontal
      // scroll range even while it's well above or below the visible conversation.
      const distant = entries.map(({ target, boundingClientRect, rootBounds }) =>
        rootBounds !== null &&
        (boundingClientRect.bottom < rootBounds.top || boundingClientRect.top > rootBounds.bottom) &&
        target.scrollWidth <= target.clientWidth
      );
      entries.forEach(({ target }, i) => target.toggleAttribute("data-reply-paint", distant[i]));
    }, { root: viewport, rootMargin: "1500px 0px" });
    observers.set(viewport, observer);
  }
  observer.observe(reply);
  return () => {
    observer.unobserve(reply);
    reply.removeAttribute("data-reply-paint");
  };
}
