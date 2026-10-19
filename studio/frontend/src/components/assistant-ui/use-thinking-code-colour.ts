// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";
import { type RefObject, useEffect, useRef, useState } from "react";
import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import {
  cachedSettledColour,
  keepUnchangedLines,
  rememberSettledColour,
} from "./reasoning-code-colour";
import { requestFullHighlight } from "./use-reasoning-highlight";

export const LIVE_COLOUR_MS = 250;

const OBSERVES = typeof IntersectionObserver !== "undefined";

const viewportOf = (element: Element): Element | null =>
  element.closest(".aui-thread-viewport");

// Worker colours for a thinking code block: settled ones once near the viewport, growing ones
// every LIVE_COLOUR_MS while on screen (possibly for a shorter source; callers re-check lines).
export function useThinkingCodeColour(
  surface: RefObject<HTMLElement | null>,
  source: string,
  language: string | null,
  settled: boolean,
): HighlightResult | null {
  const [colour, setColour] = useState<{
    source: string;
    result: HighlightResult;
  } | null>(null);
  const [reached, setReached] = useState(!OBSERVES);
  const [visible, setVisible] = useState(!OBSERVES);
  const latest = useRef(source);
  const flight = useRef<{
    busy: boolean;
    off: boolean;
    startedAt: number;
    cancel: (() => void) | null;
  }>({ busy: false, off: false, startedAt: Number.NEGATIVE_INFINITY, cancel: null });
  useEffect(() => {
    latest.current = source;
  });

  const held = colour !== null && colour.source === source ? colour.result : null;
  const final = settled ? (held ?? cachedSettledColour(language, source)) : null;
  const coloured = final !== null;
  const caughtUp = !settled && colour !== null && colour.source === source;

  useEffect(() => {
    const element = surface.current;
    if (!settled || reached || coloured || !element) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        setReached(true);
      },
      { root: viewportOf(element), rootMargin: "100% 0px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [surface, settled, reached, coloured]);

  useEffect(() => {
    if (!settled || !reached || coloured) return;
    let live = true;
    const cancel = requestFullHighlight(
      source,
      language,
      (late) => {
        if (!live || late === null) return;
        rememberSettledColour(language, source, late);
        setColour((old) => ({
          source,
          result: keepUnchangedLines(old?.result ?? null, late),
        }));
      },
      true,
    );
    return () => {
      live = false;
      cancel?.();
    };
  }, [settled, reached, coloured, source, language]);

  useEffect(() => {
    const element = surface.current;
    if (settled || !OBSERVES || !element) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries.at(-1);
        if (entry) setVisible(entry.isIntersecting);
      },
      { root: viewportOf(element) },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [surface, settled]);

  useEffect(() => {
    const state = flight.current;
    if (settled || !visible || caughtUp || state.busy || state.off) return;
    if (source.length > MAX_HIGHLIGHT_CHARS) return;
    const timer = setTimeout(
      () => {
        const asked = latest.current;
        state.busy = true;
        state.startedAt = performance.now();
        state.cancel = requestFullHighlight(
          asked,
          language,
          (result) => {
            state.busy = false;
            state.cancel = null;
            if (result === null) {
              state.off = true;
              return;
            }
            setColour((old) => ({
              source: asked,
              result: keepUnchangedLines(old?.result ?? null, result),
            }));
          },
          true,
        );
        if (state.cancel === null) {
          state.busy = false;
          state.off = true;
        }
      },
      Math.max(0, state.startedAt + LIVE_COLOUR_MS - performance.now()),
    );
    return () => clearTimeout(timer);
  }, [settled, visible, caughtUp, source, language, colour]);

  useEffect(() => {
    const state = flight.current;
    return () => {
      state.cancel?.();
      state.cancel = null;
      state.busy = false;
    };
  }, []);

  return final ?? colour?.result ?? null;
}
