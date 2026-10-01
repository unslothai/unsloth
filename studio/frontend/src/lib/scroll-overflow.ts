// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";
import { useCallback } from "react";

const SCROLLS = /^(auto|scroll)$/;

/**
 * Only Firefox needs the flag: Chromium and WebKit keep the radius and inset the track instead, so
 * elsewhere the watcher would be two observers per dialog for nothing.
 */
const CLIPS_SCROLLERS =
  typeof CSS !== "undefined" && CSS.supports("-moz-appearance", "none");

/**
 * Marks a scroll container `data-overflowing` while it has something to scroll.
 *
 * Firefox drops a rounded scroller's radius on its scrollbar side, and `.scroll-rounded` clips the
 * box back to it. The clip also cuts an outer shadow, so a surface that carries one (a dialog)
 * takes it only while it overflows: with nothing to scroll there is no scrollbar to square it.
 *
 * Watches the box and each child in it, since content growing inside a capped box leaves the box
 * itself the same size. Writes the DOM only on a change, never state. `data-overflow-watched` says
 * the flag is live, so the clip can wait for it.
 *
 * While flagged it also writes the box's own radius to `--scroll-radius`. The class map in
 * index.css reads the base `rounded-4xl`, which a call site's `rounded-3xl!` or a phone-width
 * `max-sm:rounded-none!` leaves in place, and the clip has to match the radius actually drawn.
 */
export function observeScrollOverflow(element: HTMLElement): () => void {
  element.setAttribute("data-overflow-watched", "");
  let radius = "";
  const sync = () => {
    const style = getComputedStyle(element);
    // Only an axis that scrolls draws a scrollbar: a call site's overflow-hidden clips its content
    // without one, and needs no clip of its own. No tolerance: a single pixel over is enough for
    // overflow:auto to draw a scrollbar. A rounding false positive only clips a shadow; a miss
    // squares the corners.
    const overflowing =
      (SCROLLS.test(style.overflowY) && element.scrollHeight > element.clientHeight) ||
      (SCROLLS.test(style.overflowX) && element.scrollWidth > element.clientWidth);
    if (overflowing !== element.hasAttribute("data-overflowing")) {
      element.toggleAttribute("data-overflowing", overflowing);
    }
    // The scrollbar side's corner; an elliptical radius clips to its horizontal one.
    const next = overflowing ? style.borderTopRightRadius.split(" ")[0] : "";
    if (next !== radius) {
      radius = next;
      if (next) element.style.setProperty("--scroll-radius", next);
      else element.style.removeProperty("--scroll-radius");
    }
  };
  const sizes = new ResizeObserver(sync);
  sizes.observe(element);
  for (const child of element.children) sizes.observe(child);
  const children = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.removedNodes) {
        if (node instanceof Element) sizes.unobserve(node);
      }
      for (const node of record.addedNodes) {
        if (node instanceof Element) sizes.observe(node);
      }
    }
    sync();
  });
  children.observe(element, { childList: true });
  sync();
  return () => {
    sizes.disconnect();
    children.disconnect();
    element.removeAttribute("data-overflow-watched");
    element.removeAttribute("data-overflowing");
    element.style.removeProperty("--scroll-radius");
  };
}

/** A callback ref that forwards to `ref` and runs {@link observeScrollOverflow} while mounted. */
export function useScrollOverflowRef<T extends HTMLElement>(
  ref: React.Ref<T> | undefined,
): (element: T | null) => (() => void) | undefined {
  return useCallback(
    (element: T | null) => {
      if (typeof ref === "function") ref(element);
      else if (ref) ref.current = element;
      if (!element || !CLIPS_SCROLLERS || typeof ResizeObserver === "undefined") {
        return undefined;
      }
      const stop = observeScrollOverflow(element);
      return () => {
        stop();
        if (typeof ref === "function") ref(null);
        else if (ref) ref.current = null;
      };
    },
    [ref],
  );
}
