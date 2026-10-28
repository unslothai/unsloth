// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useLayoutEffect } from "react";

const FILL_VAR = "--composer-fill-height";
const EDGE_GAP_PX = 16;
// Covers the welcome screen's 280ms padding-top transition, which moves the composer without resizing it.
const SETTLE_MS = 400;

/**
 * While expanded, sizes the editor so the whole composer (attachments, queue and toolbar included)
 * fills the chat area from below the header to the bottom edge. The welcome composer grows down
 * from where it sits; the thread's footer composer grows up from its bottom edge.
 */
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
      const areaRect = area?.getBoundingClientRect() ?? {
        top: 0,
        bottom: window.innerHeight,
      };
      const rootRect = root.getBoundingClientRect();
      const chrome = rootRect.height - editor.getBoundingClientRect().height;
      let available: number;
      if (welcome) {
        // Unscrolled top: a too-tall frame scrolls the welcome page, which would read as more room.
        // Its bottom padding (added with attachments) must fit too, or the page still scrolls.
        const page = root.closest<HTMLElement>(".aui-thread-viewport");
        const padBottom = page
          ? Number.parseFloat(getComputedStyle(page).paddingBottom) || 0
          : 0;
        available =
          Math.min(areaRect.bottom, window.innerHeight) -
          Math.max(EDGE_GAP_PX, padBottom) -
          (rootRect.top + (page?.scrollTop ?? 0));
      } else {
        const viewport = area?.querySelector<HTMLElement>(
          ".aui-thread-viewport",
        );
        const headerOffset = viewport
          ? Number.parseFloat(getComputedStyle(viewport).paddingTop) || 0
          : 0;
        available =
          rootRect.bottom - (areaRect.top + headerOffset + EDGE_GAP_PX);
      }
      const next = Math.max(0, Math.floor(available - chrome));
      if (root.style.getPropertyValue(FILL_VAR) !== `${next}px`) {
        root.style.setProperty(FILL_VAR, `${next}px`);
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
