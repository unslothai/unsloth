// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { memo, type RefObject, useEffect, useState } from "react";

/*
 * MONOTONIC fence highlighting: a fence is rendered as a plain shell until the
 * first time it comes near the viewport, and from that moment it is highlighted
 * for the rest of its mount. There is no reverse edge.
 *
 * WHY THE DIRECTION MATTERS. The previous attempt at this gated on viewport
 * entry AND on viewport EXIT, so scrolling away from a fence, or collapsing the
 * pane holding it, tore the highlighted subtree down and scrolling back built
 * it again. That makes the gate BIDIRECTIONAL: a user gesture that ought to
 * cost nothing instead schedules a full re-tokenize and a re-mount of every
 * span in the block. It was predicted to save 55% and measured slower.
 *
 * Here the only transition is cheap -> expensive, taken at most once per mount.
 * Scrolling away costs nothing, because there is nothing to undo; the observer
 * has already been disconnected and the component has no effect left running.
 * The worst case is therefore exactly today's cost (every fence looked at), and
 * every fence the reader never reaches is saved outright.
 *
 * WHAT THE SHELL IS. The same text, in the same elements streamdown's own
 * unhighlighted fallback would use, carrying the same `data-streamdown`
 * attributes so every rule in index.css that sizes a code block applies to it
 * unchanged. The text is present in the DOM and selectable, which is the
 * property `content-visibility: auto` does not have: skipped content has to be
 * rendered before it can be selected, and that was measured at +28.8% on
 * `select_all_copy.select_all_ms`. Selection, clipboard and native find-in-page
 * see the same characters here whether the fence has upgraded or not.
 *
 * A STREAMING fence never defers. The block being written is by definition the
 * one the reader is looking at, and deferring it would change what streaming
 * renders rather than what a settled thread costs.
 */

// How far outside the viewport a fence counts as "reached". One viewport of
// slack in each direction, so the upgrade has a frame or two to land before the
// block is actually on screen and the reader never sees the swap.
const REACH_MARGIN = "100% 0px";

/*
 * Three states, not two.
 *
 *   "off"        ship default. Every fence is highlighted at mount, as today.
 *   "defer"      the change: an unreached fence is a plain shell and is never tokenized.
 *   "tokenize"   MEASUREMENT ONLY. An unreached fence is the same plain shell, but the
 *                highlighter is still driven over its source and the result thrown away.
 *
 * `tokenize` exists to answer one question and is not a shipping mode. `defer` removes two
 * things at once: the spans from the document, and the tokenizer work that produces them. An
 * improvement seen under `defer` alone cannot say which of the two paid for it. `tokenize`
 * holds the DOM at `defer`'s size and puts only the tokenizer work back, so the gap between
 * `tokenize` and `defer` is the tokenizer's contribution and the gap between `off` and
 * `tokenize` is the document's.
 */
export type FenceMode = "off" | "defer" | "tokenize";

export const fenceMode = (): FenceMode => {
  const runtime = (globalThis as Record<string, unknown>)
    .__UNSLOTH_DEFER_FENCE_HIGHLIGHT__;
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "defer"
        : runtime === false
          ? "off"
          : readBuildFlag();
  return raw === "1" || raw === "defer"
    ? "defer"
    : raw === "tokenize"
      ? "tokenize"
      : "off";
};

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_DEFER_FENCE_HIGHLIGHT ?? "";
  } catch {
    return "";
  }
};

// Streamdown trims trailing newlines off a fence body before rendering it, so
// the shell has to as well or the two differ by a blank line of height.
export const trimTrailingNewlines = (text: string): string => {
  let end = text.length;
  while (end > 0 && text[end - 1] === "\n") end -= 1;
  return text.slice(0, end);
};

function FenceShell({
  language,
  source,
}: {
  language: string | null;
  source: string;
}) {
  return (
    <div
      className="my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2"
      data-language={language ?? undefined}
      data-streamdown="code-block"
      data-unsloth-fence-deferred="true"
    >
      <div
        className="flex h-8 items-center text-muted-foreground text-xs"
        data-language={language ?? undefined}
        data-streamdown="code-block-header"
      >
        <span className="ml-1 font-mono lowercase">{language}</span>
      </div>
      <div
        className="overflow-x-auto rounded-md border border-border bg-background p-4 text-sm"
        data-language={language ?? undefined}
        data-streamdown="code-block-body"
      >
        <pre>
          <code>{trimTrailingNewlines(source)}</code>
        </pre>
      </div>
    </div>
  );
}

/**
 * Has this fence been reached yet? Latches true and never returns false again.
 *
 * Takes a ref to an element the CALLER already renders rather than mounting a
 * wrapper of its own. An extra div here would sit between a list item and its
 * code block and break `[data-streamdown="list-item"] > [data-streamdown=
 * "code-block"]`, and would push the block one level deeper than the
 * `:last-child` margin chain in index.css walks -- a layout change smuggled in
 * by a performance change, which is the one thing an A/B must not carry.
 */
// No IntersectionObserver, no gate. Read once at module scope so the decision
// is part of the rendered value rather than a state write from inside an
// effect, which would cost a cascading render on every fence in the thread.
const CAN_OBSERVE =
  typeof IntersectionObserver !== "undefined" &&
  typeof globalThis !== "undefined";

export function useFenceReached(
  host: RefObject<HTMLElement | null>,
  immediate: boolean,
): boolean {
  const [latched, setLatched] = useState(false);
  // DERIVED, not stored. `immediate` turns on when a streaming fence completes
  // and its block re-renders as settled; deriving means that transition needs
  // no effect and no extra render, and it can only ever go cheap -> expensive
  // because `latched` never goes back to false.
  const reached = immediate || !CAN_OBSERVE || latched;

  // The one-way edge. Once `reached` is true this effect re-runs, takes the
  // early return, and never observes anything again, so a fence that has been
  // read carries no residual per-scroll cost at all.
  useEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          setLatched(true);
        }
      },
      { rootMargin: REACH_MARGIN },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [reached, host]);

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);
