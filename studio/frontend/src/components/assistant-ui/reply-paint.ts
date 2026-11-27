// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keep reply overflow near the user; one observer per scroller handles side-by-side threads.
const observers = new WeakMap<Element, IntersectionObserver>();
// Width or font changes do not cross the observer margin, so recheck sizes too.
const resizers = new WeakMap<Element, ResizeObserver>();

export function observeReplyPaint(reply: HTMLElement) {
  const viewport = reply.closest(".aui-thread-viewport");
  if (!viewport || typeof IntersectionObserver === "undefined") return;

  let observer = observers.get(viewport);
  if (!observer) {
    observer = new IntersectionObserver((entries) => {
      // Before styling: a wide reply can own the thread's horizontal scroll even off screen.
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

  let resizer = resizers.get(viewport);
  if (!resizer && typeof ResizeObserver !== "undefined") {
    resizer = new ResizeObserver((entries) => {
      // Only bounded replies lose exemptions; read every width before removing any attribute.
      const wide = entries.map(({ target }) =>
        target.hasAttribute("data-reply-paint") && target.scrollWidth > target.clientWidth
      );
      entries.forEach(({ target }, i) => {
        if (wide[i]) target.removeAttribute("data-reply-paint");
      });
    });
    resizers.set(viewport, resizer);
  }
  resizer?.observe(reply);
  return () => {
    observer.unobserve(reply);
    resizer?.unobserve(reply);
    reply.removeAttribute("data-reply-paint");
  };
}
