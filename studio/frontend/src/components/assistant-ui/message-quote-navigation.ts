// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Navigates a live quote card back to its source. Anchors are resolved only
// against the source assistant message; missing or ambiguous passages fail
// rather than searching the rest of the conversation.

import { useChatRuntimeStore } from "@/features/chat";
import {
  buildTextIndex,
  type FindElementLike,
  type FindTextIndex,
} from "../../features/find-in-page/lib/find-text-index.ts";
import { rangeForMatch } from "../../features/find-in-page/lib/find-dom.ts";
import {
  normalizeQuoteText,
  resolveQuoteAnchor,
  type QuoteAnchor,
} from "./message-quote-model";
import {
  clearQuoteAnnouncement,
  clearQuoteHighlight,
  paintQuoteHighlight,
} from "./message-quote-highlight";
import { QUOTE_PRUNE_SELECTOR } from "./message-selection-model";
import { restoreMessageWindowBody } from "./message-window-controller";
import {
  completeProgressiveMounts,
  revealProgressiveRow,
} from "./progressive-messages";

export type QuoteNavigationResult =
  | "highlighted"
  | "source-missing"
  | "passage-missing"
  | "cancelled";

type QuoteSourceMessage = {
  readonly id?: string;
  readonly role?: string;
};

type CollapsedTracker = {
  readonly root: HTMLElement;
  readonly opener: HTMLElement | null;
  readonly details: boolean;
  userTouched: boolean;
  readonly markUserTouched: (event: Event) => void;
};

class CollapsedSession {
  private readonly trackers: CollapsedTracker[] = [];
  private restored = false;

  /** Opens only roots that were closed before this navigation touched them. */
  async openClosed(
    content: HTMLElement,
    isActive: () => boolean,
  ): Promise<boolean> {
    let openedAny = false;
    for (let pass = 0; pass < 3; pass += 1) {
      if (this.restored || !isActive()) {
        return openedAny;
      }
      const roots = closedQuoteRoots(content).filter(
        (root) => !this.trackers.some((tracker) => tracker.root === root),
      );
      if (roots.length === 0) {
        break;
      }
      for (const root of roots) {
        if (this.restored || !isActive()) {
          return openedAny;
        }
        const details = root instanceof HTMLDetailsElement;
        const opener = details
          ? root.querySelector<HTMLElement>("summary")
          : (root.querySelector<HTMLElement>(
              '[data-slot="collapsible-trigger"]',
            ) ?? root.querySelector<HTMLElement>("button[aria-controls]"));
        if (!details && opener === null) {
          continue;
        }
        const tracker: CollapsedTracker = {
          root,
          opener,
          details,
          userTouched: false,
          markUserTouched: (event) => {
            if (event.isTrusted) {
              tracker.userTouched = true;
            }
          },
        };
        this.trackers.push(tracker);
        if (details) {
          (root as HTMLDetailsElement).open = true;
        } else {
          opener?.click();
        }
        // Capture trusted interactions before a descendant stops propagation.
        root.addEventListener("pointerdown", tracker.markUserTouched, true);
        root.addEventListener("keydown", tracker.markUserTouched, true);
        root.addEventListener("click", tracker.markUserTouched, true);
        openedAny = true;
      }
      await nextFrame();
    }
    return openedAny;
  }

  /**
   * Closes only blocks this session opened, unless the user has interacted
   * with that block (or with a tracked descendant). Idempotent.
   */
  restore(): void {
    if (this.restored) {
      return;
    }
    this.restored = true;
    const touched = new Set<Element>();
    for (const tracker of this.trackers) {
      if (!tracker.userTouched) {
        continue;
      }
      let node: HTMLElement | null = tracker.root;
      while (node !== null) {
        if (this.trackers.some((candidate) => candidate.root === node)) {
          touched.add(node);
        }
        node = node.parentElement;
      }
    }
    for (const tracker of [...this.trackers].reverse()) {
      tracker.root.removeEventListener(
        "pointerdown",
        tracker.markUserTouched,
        true,
      );
      tracker.root.removeEventListener(
        "keydown",
        tracker.markUserTouched,
        true,
      );
      tracker.root.removeEventListener(
        "click",
        tracker.markUserTouched,
        true,
      );
      if (
        touched.has(tracker.root) ||
        !tracker.root.isConnected ||
        isClosedRoot(tracker.root)
      ) {
        continue;
      }
      if (tracker.details) {
        (tracker.root as HTMLDetailsElement).open = false;
      } else if (
        tracker.root.getAttribute("data-state") === "open" &&
        tracker.opener?.isConnected
      ) {
        tracker.opener.click();
      }
    }
    this.trackers.length = 0;
  }
}

type NavigationSession = {
  readonly token: number;
  /** Collection bucket captured when navigation started. */
  readonly threadKey: string;
  readonly activeThreadId: string | null;
  readonly abort: AbortController;
  /** Live collection bucket, so a stale render cannot paint a switched thread. */
  readonly readThreadKey: (() => string) | undefined;
  readonly isCurrentScope: (() => boolean) | undefined;
  collapsed: CollapsedSession | null;
  readonly escape: (event: KeyboardEvent) => void;
};

let navigationGeneration = 0;
let activeSession: NavigationSession | null = null;
let watchedThreadId: string | null = null;
let unwatchThread: (() => void) | null = null;

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const finish = () => {
      if (settled) return;
      settled = true;
      if (timer !== null) {
        clearTimeout(timer);
      }
      resolve();
    };
    // Background tabs can throttle rAF to zero; navigation must still settle.
    requestAnimationFrame(finish);
    timer = setTimeout(finish, 50);
  });
}

async function nextFrames(count: number, session: NavigationSession): Promise<boolean> {
  for (let index = 0; index < count; index += 1) {
    if (!isSessionCurrent(session)) {
      return false;
    }
    await nextFrame();
  }
  return isSessionCurrent(session);
}

function isSessionCurrent(session: NavigationSession): boolean {
  return (
    activeSession === session &&
    !session.abort.signal.aborted &&
    session.token === navigationGeneration &&
    useChatRuntimeStore.getState().activeThreadId === session.activeThreadId &&
    (session.readThreadKey?.() ?? session.threadKey) === session.threadKey &&
    (session.isCurrentScope?.() ?? true)
  );
}

function startThreadWatch(): void {
  if (unwatchThread !== null) {
    return;
  }
  watchedThreadId = useChatRuntimeStore.getState().activeThreadId;
  unwatchThread = useChatRuntimeStore.subscribe((state) => {
    if (state.activeThreadId !== watchedThreadId) {
      watchedThreadId = state.activeThreadId;
      if (activeSession !== null) {
        cancelQuoteNavigation();
      }
    }
  });
}

function stopThreadWatch(): void {
  unwatchThread?.();
  unwatchThread = null;
  watchedThreadId = null;
}

function finishSession(session: NavigationSession): void {
  session.abort.abort();
  session.collapsed?.restore();
  session.collapsed = null;
  document.removeEventListener("keydown", session.escape, true);
  if (activeSession === session) {
    activeSession = null;
  }
  if (activeSession === null) {
    stopThreadWatch();
  }
}

/**
 * Cancels the current navigation. A collection unmount/cleanup passes its own
 * thread key so a Compare pane cannot clear another pane's live highlight.
 */
export function cancelQuoteNavigation(threadKey?: string): void {
  if (threadKey !== undefined && activeSession?.threadKey !== threadKey) {
    return;
  }
  navigationGeneration += 1;
  const session = activeSession;
  activeSession = null;
  if (session !== null) {
    session.abort.abort();
    document.removeEventListener("keydown", session.escape, true);
  }
  clearQuoteHighlight();
  clearQuoteAnnouncement();
  session?.collapsed?.restore();
  if (session) {
    session.collapsed = null;
  }
  stopThreadWatch();
}

export function quoteSourceAvailable(
  messages: readonly QuoteSourceMessage[],
  messageId: string,
): boolean {
  return messages.some(
    (message) => message.id === messageId && message.role === "assistant",
  );
}

export type QuoteScrollBox = {
  readonly top: number;
  readonly bottom: number;
};

export type QuoteViewportBox = {
  readonly top: number;
  readonly height: number;
  readonly paddingTop?: number;
};

/** Pure passage alignment: upper third of the usable viewport. */
export function quoteScrollDelta(
  passage: QuoteScrollBox,
  viewport: QuoteViewportBox,
  inset = 24,
): number {
  const top = viewport.top + (viewport.paddingTop ?? 0) + inset;
  const bottom = viewport.top + viewport.height - inset;
  if (bottom <= top) {
    return 0;
  }
  if (passage.top >= top && passage.bottom <= bottom) {
    return 0;
  }
  const target = top + (bottom - top) * 0.28;
  return passage.top - target;
}

function closedQuoteRoots(content: HTMLElement): HTMLElement[] {
  return [
    ...content.querySelectorAll<HTMLElement>(
      '[data-slot="collapsible"][data-state="closed"], [data-slot="reasoning-root"][data-state="closed"], [data-slot="tool-group-root"][data-state="closed"], [data-slot="tool-fallback-root"][data-state="closed"], details:not([open])',
    ),
  ];
}

function isClosedRoot(root: HTMLElement): boolean {
  return root instanceof HTMLDetailsElement
    ? !root.open
    : root.getAttribute("data-state") === "closed";
}

function buildQuoteIndex(content: HTMLElement): FindTextIndex {
  return buildTextIndex(
    content as unknown as FindElementLike,
    [],
    QUOTE_PRUNE_SELECTOR,
  );
}

function quoteRangeForContent(
  content: HTMLElement,
  anchor: QuoteAnchor,
): Range | null {
  const index = buildQuoteIndex(content);
  const match = resolveQuoteAnchor(index, anchor);
  return match === null ? null : rangeForMatch(index, match);
}

function assistantContentForRow(row: HTMLElement): HTMLElement | null {
  return row.querySelector<HTMLElement>(".aui-assistant-message-content");
}

async function locateQuoteRow(
  viewport: HTMLElement,
  messageId: string,
  session: NavigationSession,
): Promise<HTMLElement | null> {
  const selector = `[data-role="assistant"][data-message-id="${CSS.escape(messageId)}"]`;
  let row = viewport.querySelector<HTMLElement>(selector);
  if (row === null) {
    revealProgressiveRow(viewport, messageId);
    row = viewport.querySelector<HTMLElement>(selector);
    if (row === null) {
      await nextFrame();
      if (!isSessionCurrent(session)) {
        return null;
      }
      row = viewport.querySelector<HTMLElement>(selector);
    }
  }
  if (row === null) {
    await completeProgressiveMounts(
      (candidate) => candidate === viewport,
      session.abort.signal,
    );
    if (!isSessionCurrent(session)) {
      return null;
    }
    row = viewport.querySelector<HTMLElement>(selector);
  }
  return row;
}

async function scrollRangeIntoView(
  viewport: HTMLElement,
  range: Range,
  row: HTMLElement,
  session: NavigationSession,
): Promise<boolean> {
  let sawRect = false;
  for (let frame = 0; frame < 6; frame += 1) {
    if (!isSessionCurrent(session)) {
      return false;
    }
    const rects = [...range.getClientRects()].filter(
      (rect) => rect.width > 0 && rect.height > 0,
    );
    sawRect = sawRect || rects.length > 0;
    const viewportRect = viewport.getBoundingClientRect();
    const box = {
      top: viewportRect.top,
      height: viewport.clientHeight,
      paddingTop: Number.parseFloat(getComputedStyle(viewport).paddingTop) || 0,
    };
    // content-visibility: auto reports no Range rects until the row itself is
    // near the viewport. Move the row first, then align the exact passage.
    const delta =
      rects.length === 0
        ? quoteScrollDelta(row.getBoundingClientRect(), box, 24)
        : quoteScrollDelta(rects[0], box, 24);
    if (Math.abs(delta) >= 1) {
      viewport.scrollBy({ top: delta, behavior: "instant" });
    }
    await nextFrame();
  }
  return sawRect && isSessionCurrent(session);
}

function focusQuoteRow(row: HTMLElement): void {
  if (!row.hasAttribute("tabindex")) {
    row.setAttribute("tabindex", "-1");
  }
  row.focus({ preventScroll: true });
}

export type NavigateQuoteOptions = {
  readonly viewport: HTMLElement | null;
  readonly threadKey: string;
  readonly anchor: QuoteAnchor;
  readonly messages: readonly QuoteSourceMessage[];
  /** Live bucket for the mounted collection; older navigations are stale. */
  readonly readThreadKey?: () => string;
  /** The viewport must still be the mounted one for this Thread. */
  readonly isCurrentScope?: () => boolean;
};

export async function navigateToQuote(
  options: NavigateQuoteOptions,
): Promise<QuoteNavigationResult> {
  cancelQuoteNavigation();
  const token = ++navigationGeneration;
  const { viewport, threadKey, anchor, messages } = options;
  if (
    viewport === null ||
    !viewport.isConnected ||
    !quoteSourceAvailable(messages, anchor.messageId)
  ) {
    return "source-missing";
  }

  const activeThreadId = useChatRuntimeStore.getState().activeThreadId;
  const session: NavigationSession = {
    token,
    threadKey,
    activeThreadId,
    abort: new AbortController(),
    readThreadKey: options.readThreadKey,
    isCurrentScope: options.isCurrentScope,
    collapsed: null,
    escape: (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.defaultPrevented) {
        cancelQuoteNavigation();
      }
    },
  };
  activeSession = session;
  startThreadWatch();
  document.addEventListener("keydown", session.escape, true);

  const fail = (result: QuoteNavigationResult): QuoteNavigationResult => {
    finishSession(session);
    return result;
  };

  try {
    const row = await locateQuoteRow(viewport, anchor.messageId, session);
    if (!isSessionCurrent(session)) {
      return fail("cancelled");
    }
    if (row === null) {
      return fail("source-missing");
    }

    restoreMessageWindowBody(viewport, anchor.messageId);
    if (!(await nextFrames(2, session))) {
      return fail("cancelled");
    }

    let content = assistantContentForRow(row);
    if (content === null) {
      return fail("passage-missing");
    }

    let range = quoteRangeForContent(content, anchor);
    if (range === null) {
      const collapsed = new CollapsedSession();
      session.collapsed = collapsed;
      const opened = await collapsed.openClosed(content, () =>
        isSessionCurrent(session),
      );
      if (!(await nextFrames(opened ? 2 : 1, session))) {
        return fail("cancelled");
      }
      content = assistantContentForRow(row);
      if (content === null) {
        return fail("passage-missing");
      }
      range = quoteRangeForContent(content, anchor);
    }
    if (range === null) {
      return fail("passage-missing");
    }
    if (!isSessionCurrent(session)) {
      return fail("cancelled");
    }

    // First scroll gets lazy code/markdown near the viewport. Re-resolve after
    // layout, then paint the final range and align it once more.
    if (!(await scrollRangeIntoView(viewport, range, row, session))) {
      return fail(isSessionCurrent(session) ? "passage-missing" : "cancelled");
    }
    content = assistantContentForRow(row);
    const finalRange =
      content === null ? null : quoteRangeForContent(content, anchor);
    if (finalRange === null) {
      return fail("passage-missing");
    }
    if (!(await scrollRangeIntoView(viewport, finalRange, row, session))) {
      return fail(isSessionCurrent(session) ? "passage-missing" : "cancelled");
    }
    if (!isSessionCurrent(session)) {
      return fail("cancelled");
    }

    paintQuoteHighlight(finalRange, {
      viewport,
      row,
      expectedText: normalizeQuoteText(finalRange.toString()),
      // React can replace the content root during an unrelated stream. Resolve
      // the exact anchor again instead of dropping a still-correct highlight.
      resolveRange: () => {
        const current = assistantContentForRow(row);
        return current === null ? null : quoteRangeForContent(current, anchor);
      },
      onClear: () => finishSession(session),
    });
    focusQuoteRow(row);
    return "highlighted";
  } catch {
    return fail("passage-missing");
  }
}
