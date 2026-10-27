// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const THREAD_GUTTER_SHIFT_ATTRIBUTE = "data-gutter-shift";
export const THREAD_GUTTER_SHIFT_PROPERTY = "--thread-gutter-shift";

/** The column moves without resizing, so no ResizeObserver fires. */
export const THREAD_GUTTER_CHANGE_EVENT = "unsloth:thread-gutter-change";

export function threadGutterShift(
  scrollbar: number,
  misplaced: boolean,
  rtl: boolean,
): number {
  return misplaced && !rtl && scrollbar > 0 ? scrollbar : 0;
}

let misplacedCache: boolean | null = null;

/** WebKit reserves a flex scroller's both-edges left gutter on the right. */
export function bothEdgesGutterMisplaced(doc: Document = document): boolean {
  if (misplacedCache !== null) {
    return misplacedCache;
  }
  const probe = doc.createElement("div");
  probe.setAttribute("aria-hidden", "true");
  probe.style.cssText =
    "position:fixed;top:0;left:0;display:flex;flex-direction:column;width:100px;height:10px;overflow-y:scroll;scrollbar-gutter:stable both-edges;visibility:hidden;pointer-events:none";
  const content = doc.createElement("div");
  content.style.height = "20px";
  probe.append(content);
  doc.body.append(probe);
  const scrollbar = probe.offsetWidth - probe.clientWidth;
  const leftGap =
    content.getBoundingClientRect().left - probe.getBoundingClientRect().left;
  probe.remove();
  // Unknown until a classic scroll bar shows.
  if (scrollbar > 0) {
    misplacedCache = leftGap < 0.5;
  }
  return misplacedCache ?? false;
}

export function watchThreadGutter(viewport: HTMLElement): () => void {
  if (typeof ResizeObserver === "undefined") {
    return () => undefined;
  }
  let applied = 0;
  const sync = () => {
    const style = getComputedStyle(viewport);
    const scrollbar = Math.max(
      0,
      viewport.offsetWidth -
        viewport.clientWidth -
        (Number.parseFloat(style.borderLeftWidth) || 0) -
        (Number.parseFloat(style.borderRightWidth) || 0),
    );
    const shift = threadGutterShift(
      scrollbar,
      scrollbar > 0 && bothEdgesGutterMisplaced(viewport.ownerDocument),
      style.direction === "rtl",
    );
    if (shift === applied) {
      return;
    }
    applied = shift;
    viewport.toggleAttribute(THREAD_GUTTER_SHIFT_ATTRIBUTE, shift > 0);
    if (shift > 0) {
      viewport.style.setProperty(THREAD_GUTTER_SHIFT_PROPERTY, `${shift}px`);
    } else {
      viewport.style.removeProperty(THREAD_GUTTER_SHIFT_PROPERTY);
    }
    viewport.dispatchEvent(new Event(THREAD_GUTTER_CHANGE_EVENT));
  };
  const observer = new ResizeObserver(sync);
  observer.observe(viewport);
  sync();
  return () => {
    observer.disconnect();
    viewport.removeAttribute(THREAD_GUTTER_SHIFT_ATTRIBUTE);
    viewport.style.removeProperty(THREAD_GUTTER_SHIFT_PROPERTY);
  };
}
