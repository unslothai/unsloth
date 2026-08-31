// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for the find bar: a vite entry with no backend, driving the real bar, the real index and
// the real CSS Custom Highlight API against a real browser.
//
// The node suite reaches everything pure. What a highlight looks like painted, whether the walk
// scrolls only when a match is off screen, whether an `inert` panel stays out of the count, and
// what a streaming reply costs an open bar are only answerable here.

/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import { FindInPage } from "@/features/find-in-page";
import {
  FIND_SCOPE_ATTRIBUTE,
  type FindElementLike,
  buildTextIndex,
} from "@/features/find-in-page/lib/find-text-index.ts";
import { useFindInPageStore } from "@/features/find-in-page/stores/find-in-page-store.ts";
/* eslint-enable no-restricted-imports */
import "@/index.css";
import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

declare global {
  interface Window {
    // Optional: the app typechecks this entry, only the harness page installs it.
    __findSmoke?: {
      store: typeof useFindInPageStore;
      /** What the counter is showing, read straight out of the bar. */
      counter: () => string | null;
      /** Scroll offset of the conversation, so a test can see the walk move it. */
      scrollTop: () => number;
      /** Append a message, one character at a time, the way a reply streams in. */
      stream: (text: string, msPerChar?: number) => void;
      /** Time one flatten of the harness document, in milliseconds. */
      timeIndex: () => number;
      /** Stand in for a modal: Radix marks the shell aria-hidden for as long as one is up. */
      setModal: (open: boolean) => void;
      /** Swap the scroller from the scope to an outer container, as non-chat routes do. */
      setOuterScroll: (outer: boolean) => void;
    };
  }
}

/** Filler that mentions the search term often enough for a walk to be worth watching. */
const PARAGRAPHS = [
  "Unsloth Studio runs 500+ open models entirely offline, and this line exists so a search for unsloth has somewhere to land.",
  "Training with Unsloth is about twice as fast and uses roughly 70% less VRAM, or about 80% less on reinforcement learning.",
  "The Hub picker fills in hyperparameters for you; the export step writes GGUF, 16-bit safetensors or adapters.",
  "Data Recipes turns unstructured documents into a training-ready dataset through a graph of processing nodes.",
  "A long conversation is where find-in-page earns its keep, so this harness paints one that does not fit on a screen.",
];

function Conversation({ extra }: { extra: string[] }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-4 px-6 py-10">
      {Array.from({ length: 40 }, (_, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length static filler
          key={i}
          className="rounded-xl border border-border bg-card p-4 text-card-foreground text-sm"
        >
          <p className="mb-1 font-medium text-muted-foreground text-xs">
            Message {i + 1}
          </p>
          <p>{PARAGRAPHS[i % PARAGRAPHS.length]}</p>
          {/* A markdown soft wrap: one text node with a newline, rendered as one line. */}
          {i === 0 ? <p>{"A soft wrapped\n          phrase about unsloth."}</p> : null}
        </div>
      ))}
      {extra.map((text, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: append-only list
          key={`extra-${i}`}
          data-streamed=""
          className="rounded-xl border border-primary/40 bg-card p-4 text-card-foreground text-sm"
        >
          <p>{text}</p>
        </div>
      ))}
    </div>
  );
}

function Harness() {
  const [streamed, setStreamed] = useState<string[]>([]);
  const [modal, setModal] = useState(false);
  const [outerScroll, setOuterScroll] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const scopeRef = useRef<HTMLDivElement>(null);

  const stream = useCallback((text: string, msPerChar = 12) => {
    setStreamed((rows) => [...rows, ""]);
    let at = 0;
    const tick = () => {
      at += 1;
      setStreamed((rows) => {
        const next = [...rows];
        next[next.length - 1] = text.slice(0, at);
        return next;
      });
      if (at < text.length) setTimeout(tick, msPerChar);
    };
    setTimeout(tick, msPerChar);
  }, []);

  useEffect(() => {
    window.__findSmoke = {
      store: useFindInPageStore,
      counter: () =>
        document.querySelector('[role="search"] [aria-live="polite"]')
          ?.textContent ?? null,
      scrollTop: () => scrollerRef.current?.scrollTop ?? -1,
      stream,
      setModal,
      setOuterScroll,
      // What one flatten of a conversation costs, against the same function the bar calls.
      timeIndex: () => {
        const scope = scopeRef.current;
        if (!scope) return -1;
        const started = performance.now();
        buildTextIndex(scope as unknown as FindElementLike);
        return performance.now() - started;
      },
    };
    return () => {
      window.__findSmoke = undefined;
    };
  }, [stream]);

  return (
    <div className="dark flex h-dvh flex-col bg-background text-foreground">
      <div className="flex shrink-0 items-center gap-3 border-border border-b px-4 py-2 text-sm">
        <strong>find-in-page smoke</strong>
        <span className="text-muted-foreground">
          Press {navigator.platform.toLowerCase().includes("mac") ? "⌘F" : "Ctrl+F"} and search for
          "unsloth"
        </span>
        <button
          type="button"
          className="rounded-md border border-border px-2 py-1"
          onClick={() => stream("A streamed reply mentioning unsloth as it arrives.")}
        >
          Stream a reply
        </button>
      </div>
      {/* The shell's content region: relative because the bar floats inside it, and the scope. */}
      {/* `outerScroll` mirrors a non-chat route, where SidebarInset scrolls and the scope is
          taller than the window. Chat routes are the default: the scope is the fixed-height
          container and the thread viewport inside it scrolls. */}
      <div
        ref={outerScroll ? scrollerRef : undefined}
        className={
          outerScroll
            ? "min-h-0 flex-1 overflow-y-auto"
            : "flex min-h-0 flex-1 flex-col overflow-hidden"
        }
      >
        <div
          ref={scopeRef}
          {...{ [FIND_SCOPE_ATTRIBUTE]: "" }}
          aria-hidden={modal || undefined}
          className={
            outerScroll
              ? "relative flex flex-col"
              : "relative flex min-h-0 flex-1 flex-col overflow-hidden"
          }
        >
          <FindInPage />
          <div
            ref={outerScroll ? undefined : scrollerRef}
            className={outerScroll ? "" : "min-h-0 flex-1 overflow-y-auto"}
          >
            <Conversation extra={streamed} />
          </div>
          {/* A workspace parked off-route, as `__root.tsx` parks one. Never counted. */}
          <div hidden={true} inert={true}>
            <p>unsloth unsloth unsloth from a workspace nobody is looking at</p>
          </div>
          {/* Hidden by a CLASS, not an attribute: the case attributes alone miss. Tailwind's
              `hidden` is `display: none`, which is what a responsive `hidden lg:flex` resolves to
              at a breakpoint that is not active. */}
          <div className="hidden">
            <p>unsloth from a breakpoint that is not active</p>
          </div>
        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Harness />);
