// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ActionBarMorePrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { Bookmark02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type FC,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type RefObject,
  type UIEvent,
  memo,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { create } from "zustand";

import {
  completeProgressiveMounts,
  hasPendingProgressiveMounts,
} from "@/components/assistant-ui/progressive-messages";
import {
  threadTurns,
  turnOpenerIdAt,
} from "@/components/assistant-ui/thread-turns";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { useDetachThreadFromBottom } from "@/components/assistant-ui/use-intent-aware-autoscroll";
import {
  useBookmarkedTurnsStore,
  useChatPreferencesStore,
  useChatRuntimeStore,
} from "@/features/chat";
import { FIND_SKIP_ATTRIBUTE } from "@/features/find-in-page";
import { prefersReducedMotion } from "@/features/settings";
import { useT } from "@/i18n";
import { scheduleIdleTask } from "@/lib/schedule-idle-task";
import { cn } from "@/lib/utils";

const MIN_NAVIGATOR_TURNS = 5;
// end fade of an overflowing rail, scaled by how much is hidden past each end
const RAIL_FADE_PX = 24;
const RAIL_LEAD = 1.15;
// rail clearance from the thread's top and bottom
const RAIL_INSET_PX = 48;
const PROMPT_PREVIEW_CHARS = 240;
const REPLY_PREVIEW_CHARS = 480;
// long enough to cross from a marker onto the card
const PREVIEW_HIDE_DELAY_MS = 150;
const PYRAMID_REACH = 3;
// math blocks above the target settle from placeholder heights, so the jump re-aligns briefly
const JUMP_ALIGN_FRAMES = 4;

// keyed by thread (compare panes mount several); settled after a switch mounts, so it adds no cost
interface TurnIndex {
  turns: ReadonlyMap<string, number>;
  settled: boolean;
}
const useTurnIndexStore = create<Record<string, TurnIndex>>(() => ({}));

function useTurnIndexSettled(threadKey: string): boolean {
  return useTurnIndexStore((state) => state[threadKey]?.settled === true);
}

function useIsTurnBookmarked(
  threadId: string | undefined,
  openerId: string | undefined,
): boolean {
  return useBookmarkedTurnsStore((state) =>
    threadId && openerId
      ? (state.bookmarkedByThread[threadId]?.includes(openerId) ?? false)
      : false,
  );
}

// ownTurn: a user message opens its own turn, so its action bar skips the message scan
function useTurnBookmark(
  ownTurn: boolean,
): { bookmarked: boolean; toggle: () => void } | null {
  const incognito = useChatRuntimeStore((state) => state.incognito);
  // one primitive selector: it runs on every store write for every mounted message
  const key = useAuiState(
    ({ thread, message, threadListItem }) =>
      `${threadListItem.remoteId ?? ""}\n${ownTurn ? message.id : (turnOpenerIdAt(thread.messages, message.index) ?? "")}`,
  );
  const [threadId, openerId] = key.split("\n");
  const bookmarked = useIsTurnBookmarked(threadId, openerId);
  const toggleBookmarkedTurn = useBookmarkedTurnsStore(
    (state) => state.toggleBookmarkedTurn,
  );
  if (incognito || !threadId || !openerId) {
    return null;
  }
  return { bookmarked, toggle: () => toggleBookmarkedTurn(threadId, openerId) };
}

// gated so the default (off) mounts no thread subscriptions per message
export const BookmarkTurnButton: FC = () => {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  return enabled ? <BookmarkTurnButtonInner /> : null;
};

const BookmarkTurnButtonInner: FC = () => {
  const t = useT();
  const turnBookmark = useTurnBookmark(true);
  if (!turnBookmark) {
    return null;
  }
  return (
    <TooltipIconButton
      tooltip={t(
        turnBookmark.bookmarked ? "turns.removeBookmark" : "turns.bookmark",
      )}
      aria-pressed={turnBookmark.bookmarked}
      onClick={turnBookmark.toggle}
    >
      <HugeiconsIcon
        icon={Bookmark02Icon}
        strokeWidth={1.75}
        className={cn("size-icon", turnBookmark.bookmarked && "fill-current")}
      />
    </TooltipIconButton>
  );
};

export const BookmarkTurnMenuItem: FC<{ className?: string }> = ({
  className,
}) => {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  return enabled ? <BookmarkTurnMenuItemInner className={className} /> : null;
};

const BookmarkTurnMenuItemInner: FC<{ className?: string }> = ({
  className,
}) => {
  const t = useT();
  const turnBookmark = useTurnBookmark(false);
  if (!turnBookmark) {
    return null;
  }
  return (
    <ActionBarMorePrimitive.Item
      onSelect={turnBookmark.toggle}
      className={className}
    >
      <HugeiconsIcon
        icon={Bookmark02Icon}
        strokeWidth={1.75}
        className={cn("size-icon", turnBookmark.bookmarked && "fill-current")}
      />
      {t(turnBookmark.bookmarked ? "turns.removeBookmark" : "turns.bookmark")}
    </ActionBarMorePrimitive.Item>
  );
};

// memoized: the thread re-renders on composer resizes
export const TurnNavigator: FC<{
  viewportRef: RefObject<HTMLElement | null>;
}> = memo(({ viewportRef }) => {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  return enabled ? <TurnIndexPublisher viewportRef={viewportRef} /> : null;
});
TurnNavigator.displayName = "TurnNavigator";

const TurnIndexPublisher: FC<{
  viewportRef: RefObject<HTMLElement | null>;
}> = ({ viewportRef }) => {
  const threadKey = useAuiState(({ threadListItem }) => threadListItem.id);
  const signature = useAuiState(
    ({ thread }) => threadTurns(thread.messages).signature,
  );
  const settled = useTurnIndexSettled(threadKey);
  useEffect(() => {
    const turns = new Map(
      signature
        ? signature.split("\n").map((id, index) => [id, index + 1])
        : [],
    );
    useTurnIndexStore.setState((state) => ({
      [threadKey]: { turns, settled: state[threadKey]?.settled ?? false },
    }));
  }, [threadKey, signature]);
  useEffect(() => {
    let cancel = () => {};
    const waitForSettle = () => {
      cancel = scheduleIdleTask(() => {
        if (hasPendingProgressiveMounts()) {
          waitForSettle();
          return;
        }
        useTurnIndexStore.setState((state) => ({
          [threadKey]: {
            turns: state[threadKey]?.turns ?? new Map(),
            settled: true,
          },
        }));
      });
    };
    waitForSettle();
    return () => {
      cancel();
      useTurnIndexStore.setState((state) => {
        const { [threadKey]: _, ...rest } = state;
        return rest;
      }, true);
    };
  }, [threadKey]);
  return settled ? <TurnRail viewportRef={viewportRef} /> : null;
};

function alignTurnTop(
  viewport: HTMLElement,
  target: HTMLElement,
  frames: number,
): void {
  const inset = Number.parseFloat(getComputedStyle(viewport).paddingTop) || 0;
  const offset =
    target.getBoundingClientRect().top -
    viewport.getBoundingClientRect().top -
    inset;
  if (Math.abs(offset) < 1) {
    return;
  }
  viewport.scrollBy({ top: offset, behavior: "instant" });
  if (frames > 1) {
    requestAnimationFrame(() => alignTurnTop(viewport, target, frames - 1));
  }
}

const SHINE_GRADIENT = {
  light:
    "linear-gradient(110deg, transparent 36%, rgb(0 0 0 / 0.07) 44%, rgb(255 255 255 / 0.95) 50%, rgb(0 0 0 / 0.07) 56%, transparent 64%)",
  dark: "linear-gradient(110deg, transparent 36%, rgb(255 255 255 / 0.03) 44%, rgb(255 255 255 / 0.24) 50%, rgb(255 255 255 / 0.03) 56%, transparent 64%)",
};

function shineTurn(target: HTMLElement): void {
  const bubble = target.querySelector<HTMLElement>(".aui-user-message-content");
  if (!bubble || prefersReducedMotion()) {
    return;
  }
  const shine = {
    backgroundImage: document.documentElement.classList.contains("dark")
      ? SHINE_GRADIENT.dark
      : SHINE_GRADIENT.light,
    backgroundSize: "250% 100%",
    backgroundRepeat: "no-repeat",
  };
  bubble.animate(
    [
      { ...shine, backgroundPosition: "100% 0" },
      { ...shine, backgroundPosition: "0% 0" },
    ],
    { duration: 900, easing: "cubic-bezier(0.4, 0, 0.2, 1)" },
  );
}

type PreviewPart = { type: string; text?: string };

function previewText(content: readonly PreviewPart[], maxChars: number) {
  const text = content
    .map((part) => (part.type === "text" ? (part.text ?? "") : ""))
    .join("\n")
    // the regexes only need what can survive the cut
    .slice(0, maxChars * 4)
    .replace(/```[^\n]*/g, " ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^\s*(?:#{1,6}|>|[-*+]|\d+[.)])\s+/gm, "")
    .replace(/\*\*|__|`/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > maxChars ? `${text.slice(0, maxChars)}…` : text;
}

type PreviewMessage = {
  id: string;
  role: string;
  content: readonly PreviewPart[];
};

// skips tool-only steps so the card is not blank
function turnReplyText(
  messages: readonly PreviewMessage[],
  start: number,
): string {
  for (let index = start + 1; index < messages.length; index++) {
    const message = messages[index];
    if (message.role === "user") {
      break;
    }
    if (message.role === "assistant") {
      const text = previewText(message.content, REPLY_PREVIEW_CHARS);
      if (text) {
        return text;
      }
    }
  }
  return "";
}

interface TurnPreview {
  openerId: string;
  turn: number;
  prompt: string;
  reply: string;
  // viewport px: portaled, so its text changes stay out of the thread's autoscroll MutationObserver
  top: number;
  right: number;
}

const TurnRail: FC<{ viewportRef: RefObject<HTMLElement | null> }> = ({
  viewportRef,
}) => {
  const t = useT();
  const aui = useAui();
  const detachFromBottom = useDetachThreadFromBottom();
  // a string, so a streamed token that adds no turn does not render the rail
  const signature = useAuiState(
    ({ thread }) => threadTurns(thread.messages).signature,
  );
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const incognito = useChatRuntimeStore((state) => state.incognito);
  const bookmarkedIds = useBookmarkedTurnsStore((state) =>
    threadId ? state.bookmarkedByThread[threadId] : undefined,
  );
  const toggleBookmarkedTurn = useBookmarkedTurnsStore(
    (state) => state.toggleBookmarkedTurn,
  );
  const openerIds = useMemo(
    () => (signature ? signature.split("\n") : []),
    [signature],
  );
  const bookmarked = useMemo(() => new Set(bookmarkedIds), [bookmarkedIds]);
  const anchorRef = useRef<HTMLDivElement>(null);
  const railRef = useRef<HTMLElement>(null);
  // rebuilt only when the messages array changes, so a hover is one lookup
  const messageIndexRef = useRef<{
    messages: readonly PreviewMessage[] | null;
    index: Map<string, number>;
  }>({ messages: null, index: new Map() });
  const previewId = useId();
  // closing keeps the last preview, so the hidden card keeps its text nodes
  const [preview, setPreview] = useState<TurnPreview | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  // non-empty from the start, so the first hover edits text nodes instead of inserting them
  const lastReplyRef = useRef(" ");
  // only the latest turn can stream, so an open card follows it and the scan stops at its prompt
  const previewOpenerId = previewOpen ? preview?.openerId : undefined;
  const liveReply = useAuiState(({ thread }) => {
    if (!previewOpenerId) {
      return null;
    }
    const messages = thread.messages;
    for (let index = messages.length - 1; index >= 0; index--) {
      if (messages[index].role === "user") {
        return messages[index].id === previewOpenerId
          ? turnReplyText(messages, index)
          : null;
      }
    }
    return null;
  });
  const hideTimerRef = useRef<number | undefined>(undefined);
  const cancelHide = useCallback(
    () => window.clearTimeout(hideTimerRef.current),
    [],
  );
  useEffect(() => cancelHide, [cancelHide]);
  // set on the elements, not in state, so hovering never re-renders the memoized markers
  const raisedRef = useRef<HTMLElement[]>([]);
  const setActiveMarker = useCallback((marker: HTMLButtonElement | null) => {
    for (const element of raisedRef.current) {
      element.removeAttribute("data-dist");
      element.removeAttribute("data-hovering");
    }
    raisedRef.current = [];
    const rail = marker?.parentElement;
    if (!marker || !rail) {
      return;
    }
    rail.setAttribute("data-hovering", "");
    raisedRef.current.push(rail);
    const siblings = rail.children;
    const index = Array.prototype.indexOf.call(siblings, marker);
    for (let dist = 0; dist <= PYRAMID_REACH; dist++) {
      for (const neighbour of new Set([
        siblings[index - dist],
        siblings[index + dist],
      ])) {
        if (neighbour instanceof HTMLElement) {
          neighbour.setAttribute("data-dist", String(dist));
          raisedRef.current.push(neighbour);
        }
      }
    }
  }, []);

  const jumpToTurn = useCallback(
    async (messageId: string) => {
      const viewport = viewportRef.current;
      if (!viewport) {
        return;
      }
      const selector = `[data-message-id="${CSS.escape(messageId)}"]`;
      let target = viewport.querySelector<HTMLElement>(selector);
      if (!target) {
        await completeProgressiveMounts((candidate) => candidate === viewport);
        target = viewport.querySelector<HTMLElement>(selector);
      }
      if (!target) {
        return;
      }
      detachFromBottom();
      alignTurnTop(viewport, target, JUMP_ALIGN_FRAMES);
      shineTurn(target);
    },
    [viewportRef, detachFromBottom],
  );

  const onMarkerClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const messageId = event.currentTarget.dataset.turnId;
      if (messageId) {
        void jumpToTurn(messageId);
      }
    },
    [jumpToTurn],
  );

  // read on hover so the rail never holds stale message text
  const showPreview = useCallback(
    (
      event: PointerEvent<HTMLButtonElement> | FocusEvent<HTMLButtonElement>,
    ) => {
      const marker = event.currentTarget;
      const anchor = anchorRef.current;
      const openerId = marker.dataset.turnId;
      if (!anchor || !openerId) {
        return;
      }
      cancelHide();
      setActiveMarker(marker);
      const messages = aui.thread().getState().messages;
      if (messageIndexRef.current.messages !== messages) {
        messageIndexRef.current = {
          messages,
          index: new Map(messages.map((message, index) => [message.id, index])),
        };
      }
      const start = messageIndexRef.current.index.get(openerId);
      const box = marker.getBoundingClientRect();
      setPreviewOpen(true);
      setPreview({
        openerId,
        turn: Number(marker.dataset.turn),
        prompt:
          start === undefined
            ? ""
            : previewText(messages[start].content, PROMPT_PREVIEW_CHARS),
        reply: start === undefined ? "" : turnReplyText(messages, start),
        top: box.top + box.height / 2,
        right:
          document.documentElement.clientWidth -
          anchor.getBoundingClientRect().right,
      });
    },
    [aui, cancelHide, setActiveMarker],
  );
  const magnifyFrameRef = useRef(0);
  const magnifyYRef = useRef(0);
  const magnifiedRef = useRef<HTMLElement[]>([]);
  const clearMagnify = useCallback(() => {
    cancelAnimationFrame(magnifyFrameRef.current);
    magnifyFrameRef.current = 0;
    for (const dash of magnifiedRef.current) {
      dash.style.removeProperty("width");
      dash.style.removeProperty("transition-property");
    }
    magnifiedRef.current = [];
  }, []);
  useEffect(() => clearMagnify, [clearMagnify]);
  // dock-style: widths follow pointer distance every frame, so fast sweeps skip the data-dist steps
  const magnify = useCallback((rail: HTMLElement) => {
    magnifyFrameRef.current = 0;
    const markers = Array.from(rail.children) as HTMLElement[];
    const spacing = rail.scrollHeight / Math.max(markers.length, 1);
    const reach = (PYRAMID_REACH + 1) * spacing;
    const pointerY =
      magnifyYRef.current - rail.getBoundingClientRect().top + rail.scrollTop;
    // read every position before writing any width, so a frame lays out once
    const mags = markers.map((marker) => {
      const distance = Math.abs(
        marker.offsetTop + marker.offsetHeight / 2 - pointerY,
      );
      return distance < reach
        ? (1 + Math.cos((Math.PI * distance) / reach)) / 2
        : null;
    });
    const next: HTMLElement[] = [];
    markers.forEach((marker, index) => {
      const dash = marker.firstElementChild;
      const mag = mags[index];
      if (mag === null || !(dash instanceof HTMLElement)) {
        return;
      }
      dash.style.width = `calc(var(--spacing) * ${1.5 + mag * 5.5})`;
      dash.style.transitionProperty = "background-color";
      next.push(dash);
    });
    for (const dash of magnifiedRef.current) {
      if (!next.includes(dash)) {
        dash.style.removeProperty("width");
        dash.style.removeProperty("transition-property");
      }
    }
    magnifiedRef.current = next;
  }, []);
  const onRailPointerMove = useCallback(
    (event: PointerEvent<HTMLElement>) => {
      if (event.pointerType !== "mouse" || prefersReducedMotion()) {
        return;
      }
      magnifyYRef.current = event.clientY;
      if (!magnifyFrameRef.current) {
        const rail = event.currentTarget;
        magnifyFrameRef.current = requestAnimationFrame(() => magnify(rail));
      }
    },
    [magnify],
  );
  // the dashes move under a still pointer when the rail itself scrolls
  const onRailScroll = useCallback(
    (event: UIEvent<HTMLElement>) => {
      const rail = event.currentTarget;
      if (magnifiedRef.current.length > 0 && !magnifyFrameRef.current) {
        magnifyFrameRef.current = requestAnimationFrame(() => magnify(rail));
      }
      // the rail follows a keyboard jump, so the card stays on its still-focused marker
      const focused = document.activeElement;
      if (focused instanceof HTMLButtonElement && rail.contains(focused)) {
        const box = focused.getBoundingClientRect();
        setPreview((current) =>
          current && current.openerId === focused.dataset.turnId
            ? { ...current, top: box.top + box.height / 2 }
            : current,
        );
      }
    },
    [magnify],
  );
  // the rail is overflow-hidden so the wheel reaches the thread; it follows the thread instead
  const turnCount = openerIds.length;
  useEffect(() => {
    const viewport = viewportRef.current;
    const rail = railRef.current;
    if (!viewport || !rail || turnCount < MIN_NAVIGATOR_TURNS) {
      return;
    }
    let frame = 0;
    // refit only after a resize; scrolling just moves the rail
    let fitPending = true;
    let railRange = 0;
    let railFull = 0;
    let railView = 0;
    const setStyle = (name: string, value: string) => {
      if (rail.style.getPropertyValue(name) !== value) {
        rail.style.setProperty(name, value);
      }
    };
    const fade = (hidden: number) =>
      `${Math.min(Math.max(hidden, 0), RAIL_FADE_PX)}px`;
    const setFades = (top: number) => {
      setStyle("--rail-fade-top", fade(top));
      setStyle("--rail-fade-bottom", fade(railRange - top));
    };
    // reads before writes, no-op writes skipped: one layout per frame (autoscroll, resize drag)
    const sync = () => {
      frame = 0;
      const scrollTop = viewport.scrollTop;
      const range = viewport.scrollHeight - viewport.clientHeight;
      const anchor = anchorRef.current;
      let fit: Record<string, string> | null = null;
      if (fitPending && anchor) {
        const box = viewport.getBoundingClientRect();
        // centred on the anchor, so twice the tighter side's room; not the composer, it grows with the queue
        const centre = anchor.getBoundingClientRect().top;
        const room =
          Math.min(centre - box.top, box.bottom - centre) - RAIL_INSET_PX;
        // snap line, pitch and start to device pixels, or 125% scaling renders uneven dashes
        const dpr = window.devicePixelRatio || 1;
        const dash = rail.querySelector("span");
        const shift = Number.parseFloat(rail.style.marginTop) || 0;
        const top = dash ? (dash.getBoundingClientRect().top - shift) * dpr : 0;
        fit = {
          "max-height": `${Math.max(0, Math.floor(room * 2))}px`,
          "margin-top": `${(Math.round(top) - top) / dpr}px`,
          "--rail-row": `${Math.round(11 * dpr) / dpr}px`,
          "--rail-line": `${Math.max(1, Math.round(2 * dpr)) / dpr}px`,
        };
      }
      fitPending = false;
      if (fit) {
        for (const [name, value] of Object.entries(fit)) {
          setStyle(name, value);
        }
        // forces a layout, so only after a resize
        railFull = rail.scrollHeight;
        railView = rail.clientHeight;
        railRange = Math.max(0, railFull - railView);
      }
      // thread position maps to mid-rail, a touch ahead, so the ends come into view early
      const progress = range > 0 ? scrollTop / range : 0;
      const ahead = 0.5 + (progress - 0.5) * RAIL_LEAD;
      const railTop = Math.min(
        Math.max(ahead * railFull - railView / 2, 0),
        railRange,
      );
      rail.scrollTop = railTop;
      setFades(railTop);
    };
    const schedule = () => {
      if (!frame) {
        frame = requestAnimationFrame(sync);
      }
    };
    const refit = () => {
      fitPending = true;
      schedule();
    };
    const fadeEnds = () => setFades(rail.scrollTop);
    // focus scrolls the rail to reveal its dash; once focus leaves, follow the thread again
    const onFocusOut = (event: globalThis.FocusEvent) => {
      if (!rail.contains(event.relatedTarget as Node | null)) {
        schedule();
      }
    };
    sync();
    viewport.addEventListener("scroll", schedule, { passive: true });
    rail.addEventListener("scroll", fadeEnds, { passive: true });
    rail.addEventListener("focusout", onFocusOut);
    // the rail height eases, so observing it keeps the fades in step
    const resize = new ResizeObserver(refit);
    resize.observe(rail);
    resize.observe(viewport);
    return () => {
      viewport.removeEventListener("scroll", schedule);
      rail.removeEventListener("scroll", fadeEnds);
      rail.removeEventListener("focusout", onFocusOut);
      resize.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [viewportRef, turnCount]);
  const hidePreview = useCallback(() => {
    cancelHide();
    hideTimerRef.current = window.setTimeout(() => {
      setPreviewOpen(false);
      setActiveMarker(null);
    }, PREVIEW_HIDE_DELAY_MS);
  }, [cancelHide, setActiveMarker]);
  // WebView2 sends no pointerleave on exit onto the window frame, so close on leave or resize
  useEffect(() => {
    if (!previewOpen) {
      return;
    }
    const close = () => {
      cancelHide();
      clearMagnify();
      setPreviewOpen(false);
      setActiveMarker(null);
    };
    const onOut = (event: globalThis.MouseEvent) => {
      if (!event.relatedTarget) {
        close();
      }
    };
    document.addEventListener("mouseout", onOut);
    window.addEventListener("resize", close);
    window.addEventListener("blur", close);
    return () => {
      document.removeEventListener("mouseout", onOut);
      window.removeEventListener("resize", close);
      window.removeEventListener("blur", close);
    };
  }, [previewOpen, cancelHide, clearMagnify, setActiveMarker]);

  const onRailKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.key === "Escape") {
        cancelHide();
        setPreviewOpen(false);
        setActiveMarker(null);
        return;
      }
      const markers = Array.from(
        event.currentTarget.querySelectorAll<HTMLButtonElement>(
          "button[data-turn-id]",
        ),
      );
      const current = markers.indexOf(
        document.activeElement as HTMLButtonElement,
      );
      if (current < 0) {
        return;
      }
      let next: number;
      switch (event.key) {
        case "ArrowUp":
          next = current - 1;
          break;
        case "ArrowDown":
          next = current + 1;
          break;
        case "Home":
          next = 0;
          break;
        case "End":
          next = markers.length - 1;
          break;
        default:
          return;
      }
      event.preventDefault();
      markers[Math.min(Math.max(next, 0), markers.length - 1)]?.focus();
    },
    [cancelHide, setActiveMarker],
  );

  const markers = useMemo(
    () =>
      openerIds.map((openerId, index) => {
        const isBookmarked = bookmarked.has(openerId);
        return (
          <button
            key={openerId}
            type="button"
            data-turn-id={openerId}
            data-turn={index + 1}
            tabIndex={index === 0 ? 0 : -1}
            aria-describedby={previewId}
            aria-label={t(
              isBookmarked ? "turns.bookmarkedLabel" : "turns.label",
              {
                number: index + 1,
              },
            )}
            onClick={onMarkerClick}
            onPointerEnter={showPreview}
            onPointerLeave={hidePreview}
            onFocus={showPreview}
            onBlur={hidePreview}
            className="group flex h-[var(--rail-row,11px)] w-full shrink-0 cursor-pointer items-center justify-end rounded-sm pr-1.5 outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <span
              className={cn(
                "h-[var(--rail-line,2px)] w-1.5 rounded-full transition-[width,background-color] duration-150 ease-out group-data-[dist=0]:w-7 group-data-[dist=1]:w-5.5 group-data-[dist=2]:w-3.75 group-data-[dist=3]:w-2.75 motion-reduce:transition-none",
                isBookmarked
                  ? "bg-primary"
                  : "bg-muted-foreground/40 group-data-[dist=0]:bg-foreground",
              )}
            />
          </button>
        );
      }),
    [
      openerIds,
      bookmarked,
      t,
      previewId,
      onMarkerClick,
      showPreview,
      hidePreview,
    ],
  );

  if (openerIds.length < MIN_NAVIGATOR_TURNS) {
    return null;
  }

  const previewBookmarked = preview ? bookmarked.has(preview.openerId) : false;
  const previewReply = preview ? (liveReply ?? preview.reply) : "";
  if (previewReply) {
    lastReplyRef.current = previewReply;
  }
  // the rail is not a scroller, so wheel scrolling over it reaches the thread
  return (
    <div
      ref={anchorRef}
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-turn-navigator-anchor pointer-events-none select-none sticky top-1/2 z-10 h-0 w-full shrink-0"
    >
      {/* gutter beside the message column; if too narrow the dashes sit in the thread padding */}
      <div
        style={{
          width:
            "calc((100% - min(100%, var(--thread-content-max-width))) / 2)",
        }}
        className="aui-turn-navigator-gutter @container/turn-gutter absolute top-0 right-0 h-0"
      >
        <nav
          aria-label={t("turns.navigator")}
          style={{
            height: `min(calc(var(--rail-row, 11px) * ${openerIds.length} + 8px), 40dvh)`,
          }}
          ref={railRef}
          onKeyDown={onRailKeyDown}
          onPointerMove={onRailPointerMove}
          onPointerLeave={clearMagnify}
          onScroll={onRailScroll}
          className="aui-turn-navigator group/rail pointer-events-auto absolute top-0 right-[-1.125rem] flex w-9 -translate-y-1/2 flex-col overflow-y-hidden py-1 transition-[height] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] [contain:layout_paint] [mask-image:linear-gradient(to_bottom,transparent,#000_var(--rail-fade-top,0px),#000_calc(100%_-_var(--rail-fade-bottom,0px)),transparent)] motion-reduce:transition-none"
        >
          {markers}
        </nav>
      </div>
      {/* portaled out of the autoscroll observer; invisible, not unmounted: no :has() restyle */}
      {createPortal(
        <div
          id={previewId}
          role="tooltip"
          {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
          style={
            preview
              ? {
                  top: preview.top,
                  right: `calc(${preview.right}px + 1.5rem)`,
                  // keeps the card on screen at phone widths or a larger ui text scale
                  maxWidth: `calc(100vw - ${preview.right}px - 2rem)`,
                }
              : undefined
          }
          onPointerEnter={cancelHide}
          onPointerLeave={hidePreview}
          className={cn(
            "aui-turn-preview pointer-events-auto fixed z-50 flex w-84 -translate-y-1/2 select-none flex-col gap-1.5 rounded-2xl border border-sidebar-border bg-sidebar py-3 pr-2.5 pl-4 text-sidebar-foreground text-ui-13 shadow-md",
            !previewOpen && "invisible",
          )}
        >
          <div className="flex items-center gap-2">
            <p className="flex min-w-0 flex-1 items-baseline gap-2 font-medium text-ui-13p5">
              <span className="shrink-0 text-muted-foreground/80 text-ui-11">
                {preview ? t("turns.label", { number: preview.turn }) : " "}
              </span>
              <span className="min-w-0 truncate">{preview?.prompt ?? " "}</span>
            </p>
            {threadId && !incognito && (
              <button
                type="button"
                tabIndex={-1}
                aria-label={t(
                  previewBookmarked ? "turns.removeBookmark" : "turns.bookmark",
                )}
                onClick={() =>
                  preview && toggleBookmarkedTurn(threadId, preview.openerId)
                }
                className={cn(
                  "flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-md hover:bg-sidebar-accent",
                  previewBookmarked
                    ? "text-primary"
                    : "text-muted-foreground hover:text-sidebar-foreground",
                )}
              >
                <HugeiconsIcon
                  icon={Bookmark02Icon}
                  strokeWidth={1.75}
                  className={cn("size-4", previewBookmarked && "fill-current")}
                />
              </button>
            )}
          </div>
          {/* collapsed by class, not `hidden`: display toggles restyle the document via :has() */}
          <p
            className={cn(
              "line-clamp-3 pr-1.5 text-muted-foreground leading-relaxed",
              !previewReply && "invisible -mt-1.5 h-0",
            )}
          >
            {lastReplyRef.current}
          </p>
        </div>,
        document.body,
      )}
    </div>
  );
};
