// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for the find bar: real bar, index and CSS Custom Highlight API; no backend.

/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { FindInPage } from "@/features/find-in-page";
import {
  FIND_SCOPE_ATTRIBUTE,
  type FindElementLike,
  buildTextIndex,
} from "@/features/find-in-page/lib/find-text-index.ts";
/* eslint-enable no-restricted-imports */
import "@/index.css";
import { StrictMode, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

declare global {
  interface Window {
    __findSmoke?: {
      state: () => { open: boolean; focused: boolean };
      counter: () => string | null;
      scrollTop: () => number;
      stream: (text: string, msPerChar?: number) => void;
      timeIndex: () => number;
      setModal: (open: boolean) => void;
      setOuterScroll: (outer: boolean) => void;
      prepend: (count: number) => void;
      setWorkspace: (which: "chat" | "other") => void;
      setShell: (mounted: boolean) => void;
    };
  }
}

const PARAGRAPHS = [
  "Unsloth Studio runs 500+ open models entirely offline, and this line exists so a search for unsloth has somewhere to land.",
  "Training with Unsloth is about twice as fast and uses roughly 70% less VRAM, or about 80% less on reinforcement learning.",
  "The Hub picker fills in hyperparameters for you; the export step writes GGUF, 16-bit safetensors or adapters.",
  "Data Recipes turns unstructured documents into a training-ready dataset through a graph of processing nodes.",
  "A long conversation is where find-in-page earns its keep, so this harness paints one that does not fit on a screen.",
];

function Conversation({ extra, older }: { extra: string[]; older: string[] }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-4 px-6 py-10">
      {older.map((text, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: append-only list
          key={`older-${i}`}
          data-older=""
          className="rounded-xl border border-border bg-card p-4 text-card-foreground text-sm"
        >
          <p>{text}</p>
        </div>
      ))}
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
          {i === 0 ? (
            <p>{"A soft wrapped\n          phrase about unsloth."}</p>
          ) : null}
        </div>
      ))}
      {/* Skipped, not hidden: what a Hub README and a maths-bearing thread use. It sits below the
          fold of a long list, so the engine really is skipping it, and it has to stay searchable. */}
      <div
        data-skipped=""
        style={{
          contentVisibility: "auto",
          containIntrinsicSize: "auto 400px",
        }}
      >
        <p>unsloth inside a skipped subtree</p>
      </div>
      <span className="sr-only">unsloth data input handle</span>
      <details data-collapsible="">
        <summary>Release notes</summary>
        <p>unsloth inside a collapsible</p>
      </details>
      <div className="group/row">
        <span
          data-find-skip=""
          className="opacity-0 transition-opacity group-hover/row:opacity-100"
        >
          unsloth hover only badge
        </span>
      </div>
      <p data-greek="">{"\u039f\u0394\u039f\u03a3 \u0130 \u039f\u03a3"}</p>
      <p data-greek-split="">
        {"\u039f"}
        <em>{"\u03a3"}</em>
      </p>
      <pre className="whitespace-pre rounded-lg bg-muted p-3 text-xs">
        {"def train(model):\n    unsloth   fast = True\n"}
      </pre>
      <div data-css-blocks="">
        <span className="block">Open</span>
        <span className="block">AI models</span>
      </div>
      {/* An inline SVG, as a Mermaid diagram renders. Its tag name is lowercase in an HTML
          document, which is how it slipped past a skip list spelled in HTML casing. */}
      <svg viewBox="0 0 200 40" role="img" aria-label="diagram">
        <title>diagram</title>
        <text x="0" y="20">
          unsloth drawn into a diagram
        </text>
      </svg>
      {/* Boxless, not hidden: `display: contents` is how the shell and the training page hand a
          grid its children, and `checkVisibility` calls a wrapper with no box invisible. */}
      <div className="contents">
        <p>unsloth inside a display contents wrapper</p>
      </div>
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

function OtherWorkspace({ active }: { active: boolean }) {
  return (
    <div hidden={!active} inert={!active || undefined} data-workspace="other">
      {Array.from({ length: 12 }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length static filler
        <div key={i} className="rounded-xl border border-border p-4 text-sm">
          <p>Another workspace, unsloth line {i + 1}.</p>
        </div>
      ))}
    </div>
  );
}

function Harness() {
  const [streamed, setStreamed] = useState<string[]>([]);
  const [workspace, setWorkspace] = useState<"chat" | "other">("chat");
  const [shell, setShell] = useState(true);
  const [older, setOlder] = useState<string[]>([]);
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
      state: () => {
        const input = document.querySelector<HTMLInputElement>(
          '[role="search"] input',
        );
        return {
          open: input !== null,
          focused: document.activeElement === input,
        };
      },
      counter: () =>
        document.querySelector('[role="search"] [aria-live="polite"]')
          ?.textContent ?? null,
      scrollTop: () => scrollerRef.current?.scrollTop ?? -1,
      stream,
      setModal,
      setOuterScroll,
      setWorkspace,
      setShell,
      prepend: (count) =>
        setOlder(
          Array.from(
            { length: count },
            (_, i) => `An older message about unsloth, number ${i + 1}.`,
          ),
        ),
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
          Press{" "}
          {navigator.platform.toLowerCase().includes("mac") ? "⌘F" : "Ctrl+F"}{" "}
          and search for "unsloth"
        </span>
        <button
          type="button"
          className="rounded-md border border-border px-2 py-1"
          onClick={() =>
            stream("A streamed reply mentioning unsloth as it arrives.")
          }
        >
          Stream a reply
        </button>
        <Popover>
          <PopoverTrigger className="rounded-md border border-border px-2 py-1">
            Model picker
          </PopoverTrigger>
          <PopoverContent data-picker="">
            <p>unsloth inside a portaled popover</p>
          </PopoverContent>
        </Popover>
      </div>
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
          {shell ? <FindInPage /> : null}
          <div
            ref={outerScroll ? undefined : scrollerRef}
            className={outerScroll ? "" : "min-h-0 flex-1 overflow-y-auto"}
          >
            <OtherWorkspace active={workspace === "other"} />
            <div
              hidden={workspace !== "chat"}
              inert={workspace !== "chat" || undefined}
            >
              <Conversation extra={streamed} older={older} />
            </div>
          </div>
          <form
            data-find-skip=""
            className="shrink-0 border-border border-t p-2"
          >
            <textarea
              className="aui-composer-input w-full resize-none bg-transparent text-sm outline-none"
              placeholder="Message"
              rows={2}
            />
            <span className="text-muted-foreground text-xs">
              unsloth pill label
            </span>
          </form>
          <div hidden={true} inert={true}>
            <p>unsloth unsloth unsloth from a workspace nobody is looking at</p>
          </div>
          <div className="hidden">
            <p>unsloth from a breakpoint that is not active</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// StrictMode as in main.tsx: effects mount, unmount and remount in development.
createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
);
