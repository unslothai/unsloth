// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { HighlightResult } from "@streamdown/code";
import {
  memo,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";

import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { type FenceMode, resolveFenceMode } from "./code-fence-mode";
import {
  isBlankLine,
  type LineWindow,
  lineIsWindowed,
  plainLineText,
  selectLineWindow,
} from "./code-fence-window";
import { normalizeLanguage } from "./code-plugin";
import { WINDOW_CAP_LINES } from "./code-fence-window";

/* One-way: a fence renders as a plain shell until it first nears the viewport, then stays highlighted. */

/* One viewport of lookahead. The percentage resolves against the root's HEIGHT in all engines
 * (spec says width, w3c/IntersectionObserver#391); pf9462_parity.py guards it. */
const REACH_MARGIN = "100% 0px";

export { type FenceMode, resolveFenceMode, SHIP_DEFAULT } from "./code-fence-mode";

export const fenceMode = (): FenceMode =>
  resolveFenceMode(
    (globalThis as Record<string, unknown>).__UNSLOTH_DEFER_FENCE_HIGHLIGHT__,
    readBuildFlag(),
  );

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_DEFER_FENCE_HIGHLIGHT ?? "";
  } catch {
    return "";
  }
};

// Streamdown trims trailing newlines before rendering, so the shell must too or heights differ.
export const trimmedLength = (text: string): number => {
  let end = text.length;
  while (end > 0 && text[end - 1] === "\n") end -= 1;
  return end;
};

export const trimTrailingNewlines = (text: string): string =>
  text.slice(0, trimmedLength(text));

/* An empty fence is one line tall in streamdown, so the shell must render one line box too. */
const shellBody = (source: string): string => {
  const trimmed = trimTrailingNewlines(source);
  return trimmed === "" ? "\n" : trimmed;
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
          <code>{shellBody(source)}</code>
        </pre>
      </div>
    </div>
  );
}

// Read once at module scope: a state write from an effect would cascade a render per fence.
const CAN_OBSERVE =
  typeof IntersectionObserver !== "undefined" &&
  typeof globalThis !== "undefined";

/* Fences not yet reached, so a discontinuous scroll or a print can latch them all without a render. */
type FenceGate = {
  node: HTMLElement;
  near: HTMLElement | null;
  outer: HTMLElement | null;
  language: string | null;
  chars: number;
  warm: (tokens: boolean) => void;
  latch: () => void;
  /** A no-op state write that gives React sync work to do (see latchNow). */
  poke: () => void;
};

const unreached = new Set<FenceGate>();

const gateOpen = (gate: FenceGate): boolean =>
  inBand(gate.node, gate.near)
  && (gate.near === gate.outer || inBand(gate.near as HTMLElement, gate.outer));

/*
 * Upgrade inside this task so the browser never paints the plain shell. All three steps are needed:
 * warm(true) so render is a cache hit, flushSync(latch), and flushSync(poke) nested in an outer
 * flushSync so React runs the pending passive effect at discrete priority.
 */
const latchNow = (arrived: readonly FenceGate[]): void => {
  if (arrived.length === 0) return;
  for (const gate of arrived) {
    unreached.delete(gate);
    gate.warm(true);
  }
  flushSync(() => {
    flushSync(() => {
      for (const gate of arrived) gate.latch();
    });
    flushSync(() => {
      for (const gate of arrived) gate.poke();
    });
  });
};

/*
 * A jump (scrollbar drag, Ctrl+End, anchor) can outrun the lookahead with no render, and IO records
 * arrive after paint. Scroll events fire before paint in the same frame; one capturing listener.
 */
const lastScrollTop = new WeakMap<EventTarget, number>();
let scrollWatched = false;

const onScroll = (event: Event): void => {
  if (unreached.size === 0) return;
  const target = event.target;
  if (target === null) return;
  const element = target === document ? null : (target as HTMLElement);
  const top = element ? element.scrollTop : window.scrollY;
  const height = element ? element.clientHeight : window.innerHeight;
  const before = lastScrollTop.get(target);
  lastScrollTop.set(target, top);
  // An unseen scroller has no previous position, so its first event counts as a jump.
  if (before !== undefined && Math.abs(top - before) <= height) return;
  const arrived: FenceGate[] = [];
  for (const gate of unreached) {
    if (gateOpen(gate)) arrived.push(gate);
  }
  latchNow(arrived);
};

const watchScrolling = (): void => {
  if (scrollWatched || typeof document === "undefined") return;
  scrollWatched = true;
  document.addEventListener("scroll", onScroll, { capture: true, passive: true });
};

const unwatchScrolling = (): void => {
  if (!scrollWatched || unreached.size > 0 || typeof document === "undefined") return;
  scrollWatched = false;
  document.removeEventListener("scroll", onScroll, { capture: true });
};

/*
 * A print lays out the whole document at paper width, so latch every fence on the page now.
 * Both beforeprint and the print media query are needed: page.pdf() fires only the latter.
 */
const upgradeEverythingForPrint = (): void => {
  latchNow([...unreached]);
};

/*
 * Warm one fence per grammar on real text at idle, so latchNow never meets a loading grammar.
 * Loads all start at once; only tokenizations yield. Warms are capped at MAX_HIGHLIGHT_CHARS.
 */
const grammarsWarmed = new Set<string>();
const grammarsLoaded = new Set<string>();
let warmScheduled = false;

// Keyed the way `highlight` keys it, or `py` and `Python` are two keys for one grammar.
const grammarOf = (gate: FenceGate): string =>
  normalizeLanguage(gate.language ?? "text");

const warmGrammars = (): void => {
  warmScheduled = false;
  // Every grammar starts loading in the first task; only the tokenizations are worth yielding for.
  for (const gate of unreached) {
    const language = grammarOf(gate);
    if (grammarsLoaded.has(language)) continue;
    grammarsLoaded.add(language);
    gate.warm(false);
  }
  for (const gate of unreached) {
    const language = grammarOf(gate);
    if (grammarsWarmed.has(language)) continue;
    // Empty or over-cap fences do not mark the grammar warmed, so a later fence still warms it.
    if (gate.chars === 0 || gate.chars > MAX_HIGHLIGHT_CHARS) continue;
    grammarsWarmed.add(language);
    // True: real text is what takes the one-off tokenizer cost off the scroll.
    gate.warm(true);
    scheduleGrammarWarm();
    return;
  }
};

const scheduleGrammarWarm = (): void => {
  if (warmScheduled || typeof globalThis === "undefined") return;
  warmScheduled = true;
  const idle = (globalThis as Record<string, unknown>).requestIdleCallback as
    | ((cb: () => void, options?: { timeout: number }) => number)
    | undefined;
  if (typeof idle === "function") idle(warmGrammars, { timeout: 2000 });
  else setTimeout(warmGrammars, 500);
};

if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("beforeprint", () => {
    upgradeEverythingForPrint();
    setPrinting(true);
  });
  window.addEventListener("afterprint", () => setPrinting(false));
  window.matchMedia?.("print")?.addEventListener?.("change", (event) => {
    if (event.matches) {
      upgradeEverythingForPrint();
      setPrinting(true);
    } else {
      setPrinting(false);
    }
  });
}

/* Nearest scrolling ancestor (e.g. the reasoning pane), found by overflow, so lookahead applies there. */
const isScrollable = (el: HTMLElement): boolean => {
  const overflowY = getComputedStyle(el).overflowY;
  return (
    (overflowY === "auto" || overflowY === "scroll" || overflowY === "overlay")
    && el.scrollHeight > el.clientHeight
  );
};

const scrollerOf = (node: HTMLElement): HTMLElement | null => {
  for (let el = node.parentElement; el !== null; el = el.parentElement) {
    if (isScrollable(el)) return el;
  }
  return null;
};

/*
 * The outermost scroller too: a nested root ignores clipping above it, so the latch needs both
 * the fence vs the nearest scroller and the pane vs the outermost. One-way: can only withhold.
 */
const outermostScrollerOf = (node: HTMLElement): HTMLElement | null => {
  let found: HTMLElement | null = null;
  for (let el = node.parentElement; el !== null; el = el.parentElement) {
    if (isScrollable(el)) found = el;
  }
  return found;
};

/** Is `node` inside `scroller`'s box grown by one of its own heights, the observer's margin? */
const inBand = (node: HTMLElement, scroller: HTMLElement | null): boolean => {
  const bounds = scroller?.getBoundingClientRect();
  const top = bounds ? bounds.top : 0;
  const height = bounds ? bounds.height : window.innerHeight;
  const rect = node.getBoundingClientRect();
  return rect.bottom > top - height && rect.top < top + height * 2;
};

/**
 * @param enabled  false on the shipped default: no state, observers or layout reads.
 * @param warm  held in a ref so an unmemoized caller cannot rebuild every observer per render.
 */
export function useFenceReached(
  host: RefObject<HTMLElement | null>,
  enabled: boolean,
  streaming: boolean,
  language: string | null,
  chars: number,
  warm: (tokens: boolean) => void,
): boolean {
  const [latched, setLatched] = useState(false);
  // Bumped when the resolved scroller stops scrolling, rebuilding the gates below.
  const [generation, setGeneration] = useState(0);
  const reached = !enabled || !CAN_OBSERVE || streaming || latched;
  const warmRef = useRef(warm);
  useEffect(() => {
    warmRef.current = warm;
  }, [warm]);

  /* A completing stream must not downgrade the fence back to the shell; layout effect so it never paints. */
  useLayoutEffect(() => {
    if (!enabled || latched || !streaming) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLatched(true);
  }, [enabled, latched, streaming]);

  /* THE FIRST FRAME, which the observer cannot cover: IO delivers async, so latch an already-visible
   * fence before paint. Re-runs on rebind (`generation`). */
  useLayoutEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    const near = scrollerOf(node);
    const outer = outermostScrollerOf(node);
    if (inBand(node, near) && (near === outer || inBand(near as HTMLElement, outer))) {
      // The cascading render is intended: once per on-screen fence, versus painted frames of plain code.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLatched(true);
    }
  }, [reached, host, generation]);

  // The one-way edge. Once `reached` is true this effect early-returns and observes nothing again.
  useEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    // Rooted at the thread's scroller: with no root, rootMargin expands the document viewport instead.
    const near = scrollerOf(node);
    const outer = outermostScrollerOf(node);
    // Watch the pane, not the fence, against the outer root: the fence would be clipped by the pane.
    const gates: [Element, Element | null][] = near === outer
      ? [[node, near]]
      : [[node, near], [near as HTMLElement, outer]];
    const seen = gates.map(() => false);
    const observers = gates.map(([, root], i) => new IntersectionObserver(
      (entries) => {
        seen[i] = entries.some((entry) => entry.isIntersecting);
        if (!seen.every(Boolean)) return;
        for (const each of observers) each.disconnect();
        setLatched(true);
      },
      { root, rootMargin: REACH_MARGIN },
    ));
    gates.forEach(([target], i) => observers[i].observe(target));

    const registered: FenceGate = {
      node,
      near,
      outer,
      language,
      chars,
      warm: (tokens) => warmRef.current(tokens),
      latch: () => setLatched(true),
      // Reuses `generation`: this fence just latched, so the bump costs one render.
      poke: () => setGeneration((n) => n + 1),
    };
    unreached.add(registered);
    watchScrolling();
    scheduleGrammarWarm();

    // reasoning.tsx drops max-h-64 at stream end but keeps overflow-y-auto, so watch and rebuild.
    let resize: ResizeObserver | undefined;
    if (near !== null && near !== outer && typeof ResizeObserver !== "undefined") {
      resize = new ResizeObserver(() => {
        if (!isScrollable(near)) setGeneration((n) => n + 1);
      });
      resize.observe(near);
    }

    return () => {
      for (const observer of observers) observer.disconnect();
      resize?.disconnect();
      unreached.delete(registered);
      unwatchScrolling();
    };
  }, [reached, host, generation]);

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);

/*
 * Rendered here, not by streamdown, whose memo never hits (fresh result object per call), so one
 * memoized component per line. DOM must stay identical: playwright_code_block_flicker.py reads it.
 */

export type FenceTokens = HighlightResult;
type TokenLine = HighlightResult["tokens"][number];
type FenceToken = TokenLine[number];

/* Streamdown's class lists copied verbatim, including its unbalanced spellings: same DOM is the goal. */
/* Differs from streamdown: raw pixel text utilities are forbidden by test_ui_font_scale_contract.py. */
const LINE_CLASS =
  "block before:content-[counter(line)] before:inline-block before:[counter-increment:line] before:w-6 before:mr-4 before:text-ui-13 before:text-right before:text-muted-foreground/50 before:font-mono before:select-none";
const CODE_CLASS = "[counter-increment:line_0] [counter-reset:line]";
const PRE_CLASS =
  "bg-[var(--sdm-bg,inherit] dark:bg-[var(--shiki-dark-bg,var(--sdm-bg,inherit)]";
const TOKEN_CLASS =
  "text-[var(--sdm-c,inherit)] dark:text-[var(--shiki-dark,var(--sdm-c,inherit))]";
const TOKEN_BG_CLASS =
  "bg-[var(--sdm-tbg)] dark:bg-[var(--shiki-dark-bg,var(--sdm-tbg))]";

/* Keep `language-<info>` on body and pre: a published rendering contract streamdown also emits. */
const joinClasses = (...parts: (string | null)[]): string =>
  parts.filter((part): part is string => Boolean(part)).join(" ");

/* Split on the FIRST colon only, as streamdown does, so a `url(data:...)` value survives. */
const parseDeclarations = (text: string): Record<string, string> => {
  const style: Record<string, string> = {};
  for (const declaration of text.split(";")) {
    const colon = declaration.indexOf(":");
    if (colon <= 0) continue;
    const property = declaration.slice(0, colon).trim();
    const value = declaration.slice(colon + 1).trim();
    if (property && value) style[property] = value;
  }
  return style;
};

const tokenStyle = (
  token: FenceToken,
): { style: Record<string, string>; hasBackground: boolean } => {
  const style: Record<string, string> = {};
  let hasBackground = Boolean(token.bgColor);
  if (token.color) style["--sdm-c"] = token.color;
  if (token.bgColor) style["--sdm-tbg"] = token.bgColor;
  if (token.htmlStyle) {
    for (const [property, value] of Object.entries(token.htmlStyle)) {
      if (property === "color") style["--sdm-c"] = value;
      else if (property === "background-color") {
        style["--sdm-tbg"] = value;
        hasBackground = true;
      } else style[property] = value;
    }
  }
  return { style, hasBackground };
};

/** One line of a fence; `line` is never mutated after commit, so the shallow memo holds. */
export const FenceLine = memo(function FenceLine({
  line,
  windowed,
  inline = false,
}: {
  line: TokenLine;
  windowed: boolean;
  inline?: boolean;
}) {
  if (!inline && isBlankLine(line)) {
    return <span className={LINE_CLASS}>{"\n"}</span>;
  }
  if (!windowed) {
    return (
      <span className={inline ? "inline" : LINE_CLASS}>
        {plainLineText(line)}
      </span>
    );
  }
  return (
    <span className={inline ? "inline" : LINE_CLASS}>
      {line.map((token, index) => {
        const { style, hasBackground } = tokenStyle(token);
        return (
          <span
            className={
              hasBackground ? `${TOKEN_CLASS} ${TOKEN_BG_CLASS}` : TOKEN_CLASS
            }
            key={index}
            style={style}
            {...token.htmlAttrs}
          >
            {token.content}
          </span>
        );
      })}
    </span>
  );
});

// One scroll listener and one frame for every windowed fence: per-fence layout reads cost more.
const windowedFences = new Set<() => void>();
let windowFrame = 0;
let windowWatched = false;

/* A print colours every line; unlike the latch this reverts, so one Ctrl+P does not un-window forever. */
let printing = false;

const remeasureWindows = (): void => {
  windowFrame = 0;
  for (const measure of windowedFences) measure();
};

export const fencePrinting = (): boolean => printing;

/* Inside flushSync: there is no next paint before the print snapshot. */
const setPrinting = (value: boolean): void => {
  // Record state before the window check, or tokens arriving mid-preview would window it.
  if (printing === value) return;
  printing = value;
  if (windowFrame !== 0) {
    cancelAnimationFrame(windowFrame);
    windowFrame = 0;
  }
  if (windowedFences.size === 0) return;
  flushSync(remeasureWindows);
};

const scheduleRemeasure = (): void => {
  if (windowFrame !== 0 || windowedFences.size === 0) return;
  windowFrame = requestAnimationFrame(remeasureWindows);
};

const watchWindows = (): void => {
  if (windowWatched || typeof document === "undefined") return;
  windowWatched = true;
  // Capturing: scroll does not bubble, so this sees the thread scroller and nested panes.
  document.addEventListener("scroll", scheduleRemeasure, {
    capture: true,
    passive: true,
  });
  window.addEventListener("resize", scheduleRemeasure, { passive: true });
};

const unwatchWindows = (): void => {
  if (!windowWatched || windowedFences.size > 0 || typeof document === "undefined") {
    return;
  }
  windowWatched = false;
  document.removeEventListener("scroll", scheduleRemeasure, { capture: true });
  window.removeEventListener("resize", scheduleRemeasure);
  if (windowFrame !== 0) {
    cancelAnimationFrame(windowFrame);
    windowFrame = 0;
  }
};

function useLineWindow(
  code: RefObject<HTMLElement | null>,
  /* Outermost element, used only to find the scroller: the code block's overflow-x forces overflow-y auto. */
  frame: RefObject<HTMLElement | null>,
  lineCount: number,
  enabled: boolean,
): LineWindow | null {
  const [lineWindow, setLineWindow] = useState<LineWindow | null>(null);
  // Read via ref, not an effect dependency, so window moves do not rebuild the listener.
  const current = useRef<LineWindow | null>(null);
  const lines = useRef(lineCount);
  lines.current = lineCount;
  const hasBody = lineCount > 0;
  const measure = useRef<() => void>(() => {});

  measure.current = () => {
    const node = code.current;
    const outer = frame.current;
    if (!node || !outer) return;
    // Under the cap a window never applies; a fence growing past it re-registers via ResizeObserver.
    if (lines.current <= WINDOW_CAP_LINES && current.current === null) return;
    if (printing) {
      if (current.current === null) return;
      current.current = null;
      setLineWindow(null);
      return;
    }
    const scroller = scrollerOf(outer);
    const bounds = scroller?.getBoundingClientRect();
    const rect = node.getBoundingClientRect();
    const count = lines.current;
    const next = selectLineWindow({
      lineCount: count,
      // Measured: font size scales inside a container query, so line height is not knowable here.
      lineHeight: count > 0 ? rect.height / count : 0,
      contentTop: rect.top,
      viewportTop: bounds ? bounds.top : 0,
      viewportHeight: bounds ? bounds.height : window.innerHeight,
      previous: current.current,
    });
    // selectLineWindow returns the previous object when nothing moved, so this skips the render.
    if (next === current.current) return;
    current.current = next;
    setLineWindow(next);
  };

  /* Layout effect keyed on the body existing: at mount there is no <code>, and passive would paint spans. */
  useLayoutEffect(() => {
    if (!enabled) {
      current.current = null;
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLineWindow(null);
      return;
    }
    const run = () => measure.current();
    windowedFences.add(run);
    watchWindows();
    run();
    /* A ResizeObserver sees the fence's own growth, font loads, width changes and pane expansion. */
    /* Observed on the wrapper: ResizeObserver never fires for the inline <code>. */
    let resize: ResizeObserver | undefined;
    const box = frame.current;
    if (box && typeof ResizeObserver !== "undefined") {
      resize = new ResizeObserver(scheduleRemeasure);
      resize.observe(box);
    }
    return () => {
      resize?.disconnect();
      windowedFences.delete(run);
      unwatchWindows();
    };
    // `hasBody`, not `lineCount`: re-registering per streamed line would defeat the coalescing.
  }, [enabled, hasBody, code, frame]);

  return enabled ? lineWindow : null;
}

/** A fence's body, highlighted, with spans bounded to what is on screen; plain shell until tokens arrive. */
export const FenceBody = memo(function FenceBody({
  isIncomplete,
  language,
  result,
  source,
  windowing,
}: {
  isIncomplete: boolean | undefined;
  language: string | null;
  result: FenceTokens | null;
  source: string;
  windowing: boolean;
}) {
  const code = useRef<HTMLElement | null>(null);
  const frame = useRef<HTMLDivElement | null>(null);
  const tokens = result?.tokens ?? null;
  const lineWindow = useLineWindow(code, frame, tokens?.length ?? 0, windowing);
  const languageClass = language === null ? null : `language-${language}`;

  const rootStyle = useMemo(() => {
    const style: Record<string, string> = {};
    if (!result) return style;
    if (result.bg) style["--sdm-bg"] = result.bg;
    if (result.fg) style["--sdm-fg"] = result.fg;
    if (result.rootStyle) Object.assign(style, parseDeclarations(result.rootStyle));
    return style;
  }, [result]);

  // The shell is one line tall for an empty body, where an empty <code> is nothing at all.
  if (!tokens || tokens.length === 0) {
    return <FenceShell language={language} source={source} />;
  }

  return (
    <div
      className="my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2"
      data-incomplete={isIncomplete || undefined}
      data-language={language ?? undefined}
      data-streamdown="code-block"
      ref={frame}
      // Reproduced from streamdown so the computed cascade matches what the flicker probe reads.
      style={{ containIntrinsicSize: "auto 200px", contentVisibility: "auto" }}
    >
      <div
        className="flex h-8 items-center text-muted-foreground text-xs"
        data-language={language ?? undefined}
        data-streamdown="code-block-header"
      >
        <span className="ml-1 font-mono lowercase">{language}</span>
      </div>
      <div
        className={joinClasses(
          languageClass,
          "overflow-x-auto rounded-md border border-border bg-background p-4 text-sm",
        )}
        data-language={language ?? undefined}
        data-streamdown="code-block-body"
        data-unsloth-fence-windowed={lineWindow === null ? undefined : "true"}
      >
        <pre className={joinClasses(languageClass, PRE_CLASS)} style={rootStyle}>
          <code className={CODE_CLASS} ref={code}>
            {tokens.map((line, index) => (
              <FenceLine
                key={index}
                line={line}
                windowed={lineIsWindowed(lineWindow, index)}
              />
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
});
