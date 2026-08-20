// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The DOM half of viewport-gated highlighting: find each gated code block and tell the gate where
// it is. Every decision lives in `code-highlight-gate.ts`; this file only measures.

import type { CodeHighlightGate } from "./code-highlight-gate";
import { CODE_FENCE_ATTRIBUTE } from "./code-plugin";

/** The element whose height a block's rendering actually costs, not the token span carrying the id. */
const BLOCK_SELECTOR = '[data-streamdown="code-block"]';

export type ViewportBinding = { dispose: () => void };

export const bindCodeHighlightViewport = (
  gate: CodeHighlightGate,
  scope: { document: Document; window: Window } = {
    document: globalThis.document,
    window: globalThis.window,
  },
): ViewportBinding => {
  const { document: doc, window: view } = scope;
  const bound = new Map<string, Element>();
  // The id is held beside the element rather than written onto it. Writing an attribute onto a
  // React-owned node works until React re-renders that node and drops it, and a block whose id had
  // been quietly erased would stop being reported and silently stay highlighted forever.
  const idOf = new WeakMap<Element, string>();

  // `rootMargin` expands the observer's box to the gate's own buffer, so a crossing is reported
  // exactly when the gate's answer can change. The gate still decides -- this only says when to
  // ask it -- so the two cannot drift into disagreeing about where the band is.
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const id = idOf.get(entry.target);
        if (id === undefined) continue;
        const box = entry.boundingClientRect;
        gate.place(id, { top: box.top, bottom: box.bottom });
      }
    },
    { root: null, rootMargin: `${gate.bufferPx}px 0px`, threshold: 0 },
  );

  // Announcements arrive from inside the plugin's return path, one frame BEFORE React has
  // committed that result, so the element cannot be queried yet. Batch to the next frame.
  const waiting = new Set<string>();
  let frame: number | null = null;

  const sweep = (): void => {
    frame = null;
    for (const id of waiting) {
      const token = doc.querySelector(
        `[${CODE_FENCE_ATTRIBUTE}="${CSS.escape(id)}"]`,
      );
      const block = token?.closest(BLOCK_SELECTOR);
      if (!block) continue;
      waiting.delete(id);
      const previous = bound.get(id);
      if (previous === block) continue;
      if (previous) observer.unobserve(previous);
      idOf.set(block, id);
      bound.set(id, block);
      observer.observe(block);
    }
  };

  const unannounce = gate.onAnnounce((id) => {
    waiting.add(id);
    if (frame === null) frame = view.requestAnimationFrame(sweep);
  });

  const measureViewport = () => gate.setViewportHeight(view.innerHeight);
  measureViewport();
  view.addEventListener("resize", measureViewport);

  return {
    dispose: () => {
      unannounce();
      observer.disconnect();
      view.removeEventListener("resize", measureViewport);
      if (frame !== null) view.cancelAnimationFrame(frame);
      bound.clear();
      waiting.clear();
    },
  };
};
