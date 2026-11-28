// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Temporary quote-source paint. The CSS Custom Highlight API is preferred
// because it never mutates React-owned message DOM. Browsers without both
// required pieces fall back to fixed overlay rectangles.

import { normalizeQuoteText } from "./message-quote-model.ts";

export const QUOTE_HIGHLIGHT_NAME = "unsloth-quote";
export const QUOTE_HIGHLIGHT_DURATION_MS = 4000;

type HighlightLike = { priority: number; clear(): void };
type HighlightConstructor = new (...ranges: Range[]) => HighlightLike;
type HighlightRegistry = {
  get(name: string): HighlightLike | undefined;
  set(name: string, highlight: HighlightLike): void;
  delete(name: string): void;
};

type HighlightScope = {
  CSS?: { highlights?: HighlightRegistry };
  Highlight?: HighlightConstructor;
  document?: Document;
};

type QuoteHighlightSession = {
  range: Range;
  readonly expectedText: string;
  readonly viewport: HTMLElement | null;
  readonly row: HTMLElement | null;
  readonly onClear: (() => void) | undefined;
  readonly resolveRange: (() => Range | null) | undefined;
  timer: ReturnType<typeof setTimeout> | null;
  /** Overlay re-measure frame; separate so it cannot swallow a validation. */
  frame: number;
  validationFrame: number;
  registry: HighlightRegistry | null;
  overlay: HTMLElement | null;
  overlayBoxes: HTMLElement[];
  overlaySetRange: ((range: Range) => void) | null;
  contentObserver: MutationObserver | null;
  boundaryObserver: MutationObserver | null;
  viewportObserver: MutationObserver | null;
  scrollListener: (() => void) | null;
  resizeListener: (() => void) | null;
};

let activeSession: QuoteHighlightSession | null = null;
let supportCache: boolean | null = null;
let liveRegion: HTMLElement | null = null;
let liveRegionTimer: ReturnType<typeof setTimeout> | null = null;

export function detectQuoteHighlightSupport(
  scope: HighlightScope = globalThis as HighlightScope,
): boolean {
  return Boolean(
    scope.document !== undefined &&
      scope.CSS?.highlights !== undefined &&
      typeof scope.Highlight === "function",
  );
}

function quoteHighlightSupport(): boolean {
  if (typeof document === "undefined") {
    // Never cache a server/module-evaluation miss; a browser may initialize
    // the registry after this module is loaded.
    return false;
  }
  if (supportCache === null) {
    supportCache = detectQuoteHighlightSupport(
      globalThis as HighlightScope,
    );
  }
  return supportCache;
}

function registryFor(scope: HighlightScope): HighlightRegistry | null {
  return scope.CSS?.highlights ?? null;
}

export function quoteRangeStillMatches(
  range: Range,
  expectedText: string,
): boolean {
  try {
    if (
      !range.startContainer.isConnected ||
      !range.endContainer.isConnected ||
      range.collapsed
    ) {
      return false;
    }
    return (
      normalizeQuoteText(range.toString()) ===
      normalizeQuoteText(expectedText)
    );
  } catch {
    return false;
  }
}

export type QuoteHighlightBox = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

/** Pure geometry used by the overlay fallback and its tests. */
export function quoteHighlightBoxes(
  rects: Iterable<Pick<DOMRectReadOnly, "left" | "top" | "width" | "height">>,
): QuoteHighlightBox[] {
  const boxes: QuoteHighlightBox[] = [];
  for (const rect of rects) {
    if (
      !Number.isFinite(rect.left) ||
      !Number.isFinite(rect.top) ||
      !Number.isFinite(rect.width) ||
      !Number.isFinite(rect.height) ||
      rect.width <= 0 ||
      rect.height <= 0
    ) {
      continue;
    }
    boxes.push({
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    });
  }
  return boxes;
}

function createOverlay(range: Range): {
  overlay: HTMLElement;
  boxes: HTMLElement[];
  setRange: (range: Range) => void;
  update: () => void;
  cleanup: () => void;
} | null {
  if (typeof document === "undefined") {
    return null;
  }
  const overlay = document.createElement("div");
  overlay.className = "message-quote-highlight-layer";
  overlay.setAttribute("aria-hidden", "true");
  const boxes: HTMLElement[] = [];
  let current = range;
  const update = () => {
    if (!overlay.isConnected) {
      return;
    }
    for (const box of boxes) {
      box.remove();
    }
    boxes.length = 0;
    let rects: DOMRectList | DOMRect[];
    try {
      rects = current.getClientRects();
    } catch {
      return;
    }
    for (const rect of quoteHighlightBoxes(rects)) {
      const box = document.createElement("div");
      box.className = "message-quote-highlight-box";
      box.style.left = `${rect.left}px`;
      box.style.top = `${rect.top}px`;
      box.style.width = `${rect.width}px`;
      box.style.height = `${rect.height}px`;
      overlay.append(box);
      boxes.push(box);
    }
  };
  const setRange = (next: Range) => {
    current = next;
    update();
  };
  document.body.append(overlay);
  update();
  return {
    overlay,
    boxes,
    setRange,
    update,
    cleanup: () => {
      overlay.remove();
      boxes.length = 0;
    },
  };
}

function removeRegisteredHighlight(
  registry: HighlightRegistry,
  name: string,
): void {
  const highlight = registry.get(name);
  if (highlight !== undefined) {
    highlight.clear();
    registry.delete(name);
  }
}

function clearSession(session: QuoteHighlightSession): void {
  if (activeSession !== session) {
    return;
  }
  activeSession = null;
  if (session.timer !== null) {
    clearTimeout(session.timer);
    session.timer = null;
  }
  if (session.frame !== 0) {
    cancelAnimationFrame(session.frame);
    session.frame = 0;
  }
  if (session.validationFrame !== 0) {
    cancelAnimationFrame(session.validationFrame);
    session.validationFrame = 0;
  }
  session.contentObserver?.disconnect();
  session.boundaryObserver?.disconnect();
  session.viewportObserver?.disconnect();
  session.contentObserver = null;
  session.boundaryObserver = null;
  session.viewportObserver = null;
  if (session.scrollListener !== null && session.viewport !== null) {
    session.viewport.removeEventListener("scroll", session.scrollListener);
    session.scrollListener = null;
  }
  if (session.resizeListener !== null) {
    window.removeEventListener("resize", session.resizeListener);
    session.resizeListener = null;
  }
  if (session.registry !== null) {
    removeRegisteredHighlight(session.registry, QUOTE_HIGHLIGHT_NAME);
  }
  if (session.overlay !== null) {
    session.overlay.remove();
    session.overlay = null;
  }
  session.overlayBoxes.length = 0;
  session.overlaySetRange = null;
  session.row?.removeAttribute("data-quote-source-highlight");
  session.onClear?.();
}

export function clearQuoteHighlight(): void {
  if (activeSession !== null) {
    clearSession(activeSession);
  }
}

/** Clears the pending/current screen-reader status without touching the paint. */
export function clearQuoteAnnouncement(): void {
  if (liveRegionTimer !== null) {
    clearTimeout(liveRegionTimer);
    liveRegionTimer = null;
  }
  if (liveRegion !== null) {
    liveRegion.textContent = "";
  }
}

export type PaintQuoteHighlightOptions = {
  readonly viewport?: HTMLElement | null;
  readonly row?: HTMLElement | null;
  readonly expectedText: string;
  /** Re-resolves the exact anchor after React replaces the content root. */
  readonly resolveRange?: () => Range | null;
  readonly onClear?: () => void;
  readonly durationMs?: number;
};

export function paintQuoteHighlight(
  range: Range,
  options: PaintQuoteHighlightOptions,
): void {
  clearQuoteHighlight();
  if (typeof document === "undefined") {
    return;
  }

  const viewport = options.viewport ?? null;
  const row = options.row ?? null;
  const session: QuoteHighlightSession = {
    range,
    expectedText: options.expectedText,
    viewport,
    row,
    onClear: options.onClear,
    resolveRange: options.resolveRange,
    timer: null,
    frame: 0,
    validationFrame: 0,
    registry: null,
    overlay: null,
    overlayBoxes: [],
    overlaySetRange: null,
    contentObserver: null,
    boundaryObserver: null,
    viewportObserver: null,
    scrollListener: null,
    resizeListener: null,
  };
  activeSession = session;
  row?.setAttribute("data-quote-source-highlight", "");

  const installOverlay = (next: Range): boolean => {
    const overlay = createOverlay(next);
    if (overlay === null) {
      return false;
    }
    session.overlay = overlay.overlay;
    session.overlayBoxes = overlay.boxes;
    session.overlaySetRange = overlay.setRange;
    const schedule = () => {
      if (activeSession !== session || session.frame !== 0) {
        return;
      }
      session.frame = requestAnimationFrame(() => {
        session.frame = 0;
        if (activeSession !== session) {
          return;
        }
        overlay.update();
      });
    };
    if (viewport !== null) {
      viewport.addEventListener("scroll", schedule, { passive: true });
      session.scrollListener = schedule;
    }
    window.addEventListener("resize", schedule);
    session.resizeListener = schedule;
    return true;
  };

  const registerRange = (next: Range): boolean => {
    if (session.registry === null) {
      return false;
    }
    const Highlight = (globalThis as HighlightScope).Highlight;
    if (typeof Highlight !== "function") {
      session.registry = null;
      return false;
    }
    try {
      removeRegisteredHighlight(session.registry, QUOTE_HIGHLIGHT_NAME);
      const highlight = new Highlight(next);
      highlight.priority = 2;
      session.registry.set(QUOTE_HIGHLIGHT_NAME, highlight);
      session.range = next;
      return true;
    } catch {
      removeRegisteredHighlight(session.registry, QUOTE_HIGHLIGHT_NAME);
      session.registry = null;
      return false;
    }
  };

  const paintRange = (next: Range): boolean => {
    if (registerRange(next)) {
      return true;
    }
    if (session.overlaySetRange !== null) {
      session.overlaySetRange(next);
      session.range = next;
      return true;
    }
    if (installOverlay(next)) {
      session.range = next;
      return true;
    }
    return false;
  };

  let painted = false;
  if (quoteHighlightSupport()) {
    const registry = registryFor(globalThis as HighlightScope);
    if (registry !== null) {
      session.registry = registry;
      painted = registerRange(range);
    }
  }
  if (!painted) {
    installOverlay(range);
  }

  function observeContentRoot(): void {
    session.contentObserver?.disconnect();
    session.contentObserver = null;
    if (row === null || !row.isConnected) {
      return;
    }
    const contentRoot =
      row.querySelector<HTMLElement>(".aui-assistant-message-content");
    if (contentRoot === null) {
      return;
    }
    session.contentObserver = new MutationObserver(validate);
    session.contentObserver.observe(contentRoot, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }

  function validate(): void {
    if (activeSession !== session || session.validationFrame !== 0) {
      return;
    }
    session.validationFrame = requestAnimationFrame(() => {
      session.validationFrame = 0;
      if (activeSession !== session) {
        return;
      }
      // A Range can physically cross pruned chrome such as a code-block
      // language header. The Range text still includes that label, so a label
      // mutation conservatively clears the highlight even when the quoted
      // semantic passage is unchanged.
      if (quoteRangeStillMatches(session.range, session.expectedText)) {
        // The fallback overlay is geometry, not live CSS paint. Re-measure it
        // after reflow even when the selected text itself did not change.
        session.overlaySetRange?.(session.range);
        return;
      }
      // The text may still be present after React replaces its nodes. Resolve
      // the full anchor again; ambiguous or changed passages still clear.
      const replacement = session.resolveRange?.() ?? null;
      if (
        replacement === null ||
        !quoteRangeStillMatches(replacement, session.expectedText)
      ) {
        clearSession(session);
        return;
      }
      if (!paintRange(replacement)) {
        clearSession(session);
        return;
      }
      observeContentRoot();
    });
  }

  // The content boundary catches changes to the selected text. The stable row
  // boundary catches a replaced ancestor even when the outer row stays.
  observeContentRoot();
  if (row !== null) {
    session.boundaryObserver = new MutationObserver(validate);
    session.boundaryObserver.observe(row, {
      childList: true,
      subtree: true,
    });
  }
  if (viewport !== null) {
    session.viewportObserver = new MutationObserver(() => {
      if (activeSession !== session) {
        return;
      }
      if (row !== null && !row.isConnected) {
        clearSession(session);
      }
    });
    session.viewportObserver.observe(viewport, { childList: true });
  }

  session.timer = setTimeout(() => {
    session.timer = null;
    if (activeSession === session) {
      clearSession(session);
    }
  }, options.durationMs ?? QUOTE_HIGHLIGHT_DURATION_MS);
}

/**
 * One polite live region for the whole app, created lazily and reused. A
 * repeated announcement clears first so screen readers re-announce it.
 */
export function announceQuoteNavigation(message: string): void {
  if (typeof document === "undefined") {
    return;
  }
  if (liveRegion === null) {
    liveRegion = document.createElement("div");
    liveRegion.className = "sr-only";
    liveRegion.setAttribute("role", "status");
    liveRegion.setAttribute("aria-live", "polite");
    liveRegion.setAttribute("data-quote-navigation-status", "");
    document.body.append(liveRegion);
  }
  if (liveRegionTimer !== null) {
    clearTimeout(liveRegionTimer);
  }
  liveRegion.textContent = "";
  // A timer, unlike rAF, still announces when the tab/compositor is throttled.
  liveRegionTimer = setTimeout(() => {
    liveRegionTimer = null;
    if (liveRegion !== null) {
      liveRegion.textContent = message;
    }
  }, 0);
}
