// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useLayoutEffect } from "react";

const FILL_VAR = "--composer-fill-height";
const EDGE_GAP_PX = 16;
// Re-measures for a few frames while the layout around the composer settles.
const SETTLE_MS = 400;
const SCROLLS = /(auto|scroll)/;

function cssPx(el: Element | null | undefined, prop: "paddingTop"): number {
  return el ? Number.parseFloat(getComputedStyle(el)[prop]) || 0 : 0;
}

function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let node = el.parentElement; node; node = node.parentElement) {
    if (SCROLLS.test(getComputedStyle(node).overflowY)) {
      return node;
    }
  }
  return null;
}

/** Pinned to the bottom (thread footer, compare) it grows up to the panes' top; in flow (welcome, project page) it grows down. */
function availableHeight(root: HTMLElement, compare: boolean): number {
  const rootRect = root.getBoundingClientRect();
  const footer = root.closest(".aui-thread-viewport-footer");
  if (footer || compare) {
    const panes = compare
      ? document.querySelector('[data-tour="chat-compare-view"]')
      : root.closest(".aui-thread-viewport");
    const top =
      (panes?.getBoundingClientRect().top ?? 0) + cssPx(panes, "paddingTop");
    return rootRect.bottom - (top + EDGE_GAP_PX);
  }
  // Unscrolled top, so a page that already overflows does not read as more room. Its bottom padding
  // is ignored: assistant-ui's --aui-scroll-stabilizer adds some for a frame, which made the height bounce.
  const scroller = scrollParent(root);
  const bottom = Math.min(
    scroller?.getBoundingClientRect().bottom ?? window.innerHeight,
    window.innerHeight,
  );
  return bottom - EDGE_GAP_PX - (rootRect.top + (scroller?.scrollTop ?? 0));
}

/** While expanded, sizes the editor so the whole composer (attachments, queue and toolbar included) fills the chat area. */
export function useExpandedComposerFill(
  editorRef: RefObject<HTMLElement | null>,
  expanded: boolean,
  compare = false,
): void {
  useLayoutEffect(() => {
    const editor = editorRef.current;
    const root =
      editor?.closest<HTMLElement>("[data-writing-expanded]") ?? null;
    if (!(expanded && editor && root)) {
      return;
    }
    const measure = () => {
      const chrome =
        root.getBoundingClientRect().height -
        editor.getBoundingClientRect().height;
      const next = `${Math.max(0, Math.floor(availableHeight(root, compare) - chrome))}px`;
      if (root.style.getPropertyValue(FILL_VAR) !== next) {
        root.style.setProperty(FILL_VAR, next);
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    window.addEventListener("resize", measure);
    let frame = 0;
    const settleUntil = performance.now() + SETTLE_MS;
    const settle = () => {
      measure();
      if (performance.now() < settleUntil) {
        frame = window.requestAnimationFrame(settle);
      }
    };
    frame = window.requestAnimationFrame(settle);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      window.cancelAnimationFrame(frame);
      root.style.removeProperty(FILL_VAR);
    };
  }, [editorRef, expanded, compare]);
}
