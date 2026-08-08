// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Publishes the pointer-active strip of an overlay scrollbar for right-edge
 * controls. Classic scrollbars take layout space, so their gutter stays unset.
 */

export const OVERLAY_SCROLLBAR_GUTTER_VAR = "--overlay-scrollbar-gutter";

/** Ignore widths too large to be a scrollbar. */
const MAX_GUTTER_PX = 48;

/** Re-measuring forces layout, so resize bursts are coalesced. */
const RESIZE_SETTLE_MS = 200;

/** `reliable` is false when the sweep could not read the DOM at all. */
type Measurement = { gutter: number; reliable: boolean };

function measure(doc: Document): Measurement {
  const body = doc.body;
  if (!body) {
    return { gutter: 0, reliable: false };
  }

  const probe = doc.createElement("div");
  // Above dialogs; pointer-events:auto beats a Radix modal's body "none".
  probe.style.cssText =
    "position:fixed;top:0;left:0;width:60px;height:60px;margin:0;border:0;padding:0;opacity:0;overflow-y:scroll;pointer-events:auto;z-index:2147483647";
  const content = doc.createElement("div");
  content.style.cssText = "width:100%;height:300px";
  probe.appendChild(content);
  body.appendChild(probe);

  try {
    if (probe.offsetWidth - probe.clientWidth > 0) {
      return { gutter: 0, reliable: true };
    }

    // Reveal scrollbars configured to appear only while scrolling.
    probe.scrollTop = 1;

    const rect = probe.getBoundingClientRect();
    const right = Math.round(rect.right);
    const y = Math.round(rect.top + rect.height / 2);

    let gutter = 0;
    for (let offset = 1; offset <= MAX_GUTTER_PX; offset++) {
      if (doc.elementFromPoint(right - offset, y) === content) {
        return { gutter, reliable: true };
      }
      gutter = offset;
    }
    // Hit testing unusable here: a width is indistinguishable from no scrollbar.
    return { gutter: 0, reliable: false };
  } finally {
    body.removeChild(probe);
  }
}

/** Pointer-active scrollbar width, or 0 when nothing reliable was read. */
export function measureOverlayScrollbarGutter(doc: Document): number {
  return measure(doc).gutter;
}

/** Publishes a positive gutter on the root element. */
export function applyOverlayScrollbarGutter(doc: Document): number {
  const { gutter, reliable } = measure(doc);
  const root = doc.documentElement;
  if (!reliable) {
    // Keep the last good gutter: one unreadable sweep must not shift rows.
    return (
      Number.parseFloat(
        root.style.getPropertyValue(OVERLAY_SCROLLBAR_GUTTER_VAR),
      ) || 0
    );
  }
  if (gutter > 0) {
    root.style.setProperty(OVERLAY_SCROLLBAR_GUTTER_VAR, `${gutter}px`);
  } else {
    root.style.removeProperty(OVERLAY_SCROLLBAR_GUTTER_VAR);
  }
  return gutter;
}

/** Re-measures after resizing, refocusing, or returning to the page. */
export function watchOverlayScrollbarGutter(win: Window): () => void {
  const doc = win.document;
  let resizeTimer: ReturnType<typeof setTimeout> | undefined;

  // Hit testing a hidden page reads nothing, so skip the forced layout.
  const remeasure = () =>
    doc.visibilityState === "hidden" ? 0 : applyOverlayScrollbarGutter(doc);
  const onResize = () => {
    if (resizeTimer !== undefined) {
      clearTimeout(resizeTimer);
    }
    resizeTimer = setTimeout(remeasure, RESIZE_SETTLE_MS);
  };

  remeasure();
  win.addEventListener("resize", onResize);
  win.addEventListener("focus", remeasure);
  doc.addEventListener("visibilitychange", remeasure);

  return () => {
    if (resizeTimer !== undefined) {
      clearTimeout(resizeTimer);
    }
    win.removeEventListener("resize", onResize);
    win.removeEventListener("focus", remeasure);
    doc.removeEventListener("visibilitychange", remeasure);
  };
}
