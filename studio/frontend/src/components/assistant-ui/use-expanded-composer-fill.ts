// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useLayoutEffect } from "react";

const FILL_VAR = "--composer-fill-height";
const EDGE_GAP_PX = 16;
// Covers the welcome screen's 280ms padding-top transition, which moves the composer without resizing it.
const SETTLE_MS = 400;

function cssPx(
  el: HTMLElement | null | undefined,
  prop: "paddingTop" | "paddingBottom",
): number {
  return el ? Number.parseFloat(getComputedStyle(el)[prop]) || 0 : 0;
}

/** Room for the whole composer: the welcome one grows down from its top, the footer one up from its bottom. */
function availableHeight(
  root: HTMLElement,
  area: HTMLElement | null,
  welcome: boolean,
): number {
  const areaRect = area?.getBoundingClientRect() ?? {
    top: 0,
    bottom: window.innerHeight,
  };
  const rootRect = root.getBoundingClientRect();
  if (welcome) {
    // Unscrolled top: a too-tall frame scrolls the welcome page, which would read as more room.
    // Its bottom padding (added with attachments) must fit too, or the page still scrolls.
    const page = root.closest<HTMLElement>(".aui-thread-viewport");
    return (
      Math.min(areaRect.bottom, window.innerHeight) -
      Math.max(EDGE_GAP_PX, cssPx(page, "paddingBottom")) -
      (rootRect.top + (page?.scrollTop ?? 0))
    );
  }
  const headerOffset = cssPx(
    area?.querySelector<HTMLElement>(".aui-thread-viewport"),
    "paddingTop",
  );
  return rootRect.bottom - (areaRect.top + headerOffset + EDGE_GAP_PX);
}

/** While expanded, sizes the editor so the whole composer (attachments, queue and toolbar included) fills the chat area. */
export function useExpandedComposerFill(
  editorRef: RefObject<HTMLElement | null>,
  expanded: boolean,
): void {
  useLayoutEffect(() => {
    const editor = editorRef.current;
    const root =
      editor?.closest<HTMLElement>("[data-writing-expanded]") ?? null;
    if (!(expanded && editor && root)) {
      return;
    }
    const area = root.closest<HTMLElement>("[data-chat-content-root]");
    const welcome = root.closest(".aui-thread-welcome-center") !== null;
    const measure = () => {
      const chrome =
        root.getBoundingClientRect().height -
        editor.getBoundingClientRect().height;
      const next = `${Math.max(0, Math.floor(availableHeight(root, area, welcome) - chrome))}px`;
      if (root.style.getPropertyValue(FILL_VAR) !== next) {
        root.style.setProperty(FILL_VAR, next);
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    if (area) {
      observer.observe(area);
    }
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
  }, [editorRef, expanded]);
}
