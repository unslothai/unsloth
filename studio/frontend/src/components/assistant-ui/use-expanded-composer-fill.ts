// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useLayoutEffect } from "react";

const FILL_VAR = "--composer-fill-height";
const EDGE_GAP_PX = 16;
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

function availableHeight(root: HTMLElement, compare: boolean): number {
  const rootRect = root.getBoundingClientRect();
  // A conversation docks the composer beside the viewport, not inside it.
  const dock = root.closest(".aui-thread-composer-dock");
  if (dock || compare) {
    const panes = compare
      ? document.querySelector('[data-tour="chat-compare-view"]')
      : dock?.parentElement?.querySelector(".aui-thread-viewport");
    const top =
      (panes?.getBoundingClientRect().top ?? 0) + cssPx(panes, "paddingTop");
    return rootRect.bottom - (top + EDGE_GAP_PX);
  }
  // Not the scroller's padding: --aui-scroll-stabilizer adds some for a frame.
  const scroller = scrollParent(root);
  const bottom = Math.min(
    scroller?.getBoundingClientRect().bottom ?? window.innerHeight,
    window.innerHeight,
  );
  return bottom - EDGE_GAP_PX - (rootRect.top + (scroller?.scrollTop ?? 0));
}

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
