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

const MIN_NAVIGATOR_TURNS = 3;
const RAIL_FADE_PX = 24;
const RAIL_LEAD = 1.15;
const RAIL_INSET_PX = 48;
const RAIL_ROW_PX = 8;
const PROMPT_PREVIEW_CHARS = 240;
const REPLY_PREVIEW_CHARS = 480;
const PREVIEW_HIDE_DELAY_MS = 150;
const PYRAMID_REACH = 2;
const JUMP_ALIGN_FRAMES = 4;

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

function useTurnBookmark(
  ownTurn: boolean,
): { bookmarked: boolean; toggle: () => void } | null {
  const incognito = useChatRuntimeStore((state) => state.incognito);
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
  prompt: string;
  reply: string;
  top: number;
  left: number;
  edge: "start" | "end" | null;
}

const TurnRail: FC<{ viewportRef: RefObject<HTMLElement | null> }> = ({
  viewportRef,
}) => {
  const t = useT();
  const aui = useAui();
  const detachFromBottom = useDetachThreadFromBottom();
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
  const messageIndexRef = useRef<{
    messages: readonly PreviewMessage[] | null;
    index: Map<string, number>;
  }>({ messages: null, index: new Map() });
  const previewId = useId();
  const [preview, setPreview] = useState<TurnPreview | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const lastReplyRef = useRef(" ");
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
    }
    raisedRef.current = [];
    const rail = marker?.parentElement;
    if (!marker || !rail) {
      return;
    }
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

  const showPreview = useCallback(
    (
      event: PointerEvent<HTMLButtonElement> | FocusEvent<HTMLButtonElement>,
    ) => {
      const marker = event.currentTarget;
      const rail = railRef.current;
      const openerId = marker.dataset.turnId;
      if (!rail || !openerId) {
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
      const turn = Number(marker.dataset.turn);
      setPreviewOpen(true);
      setPreview({
        openerId,
        prompt:
          start === undefined
            ? ""
            : previewText(messages[start].content, PROMPT_PREVIEW_CHARS),
        reply: start === undefined ? "" : turnReplyText(messages, start),
        top: box.top + box.height / 2,
        left: rail.getBoundingClientRect().right,
        edge: turn === 1 ? "start" : turn === openerIds.length ? "end" : null,
      });
    },
    [aui, cancelHide, setActiveMarker, openerIds.length],
  );
  const onRailScroll = useCallback((event: UIEvent<HTMLElement>) => {
    const rail = event.currentTarget;
    const focused = document.activeElement;
    if (focused instanceof HTMLButtonElement && rail.contains(focused)) {
      const box = focused.getBoundingClientRect();
      setPreview((current) =>
        current && current.openerId === focused.dataset.turnId
          ? { ...current, top: box.top + box.height / 2 }
          : current,
      );
    }
  }, []);
  // the rail is overflow-hidden so the wheel reaches the thread; it follows the thread instead
  const turnCount = openerIds.length;
  useEffect(() => {
    const viewport = viewportRef.current;
    const rail = railRef.current;
    if (!viewport || !rail || turnCount < MIN_NAVIGATOR_TURNS) {
      return;
    }
    let frame = 0;
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
          "--rail-row": `${Math.round(RAIL_ROW_PX * dpr) / dpr}px`,
          "--rail-line": `${Math.max(1, Math.round(2 * dpr)) / dpr}px`,
        };
      }
      fitPending = false;
      if (fit) {
        for (const [name, value] of Object.entries(fit)) {
          setStyle(name, value);
        }
        railFull = rail.scrollHeight;
        railView = rail.clientHeight;
        railRange = Math.max(0, railFull - railView);
      }
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
    const onFocusOut = (event: globalThis.FocusEvent) => {
      if (!rail.contains(event.relatedTarget as Node | null)) {
        schedule();
      }
    };
    sync();
    viewport.addEventListener("scroll", schedule, { passive: true });
    rail.addEventListener("scroll", fadeEnds, { passive: true });
    rail.addEventListener("focusout", onFocusOut);
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
  useEffect(() => {
    const viewport = viewportRef.current;
    const rail = railRef.current;
    if (!viewport || !rail || turnCount < MIN_NAVIGATOR_TURNS) {
      return;
    }
    const markerFor = new Map<Element, HTMLElement>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const marker = markerFor.get(entry.target);
          if (marker) marker.toggleAttribute("data-in-view", entry.isIntersecting);
        }
      },
      { root: viewport },
    );
    // One pass over the thread: a lookup per marker walks the whole DOM once per turn.
    const messages = new Map<string, Element>();
    for (const message of viewport.querySelectorAll("[data-message-id]")) {
      messages.set(message.getAttribute("data-message-id") ?? "", message);
    }
    for (const marker of rail.querySelectorAll<HTMLElement>("button[data-turn-id]")) {
      const message = messages.get(marker.dataset.turnId ?? "");
      if (message) {
        markerFor.set(message, marker);
        observer.observe(message);
      }
    }
    return () => observer.disconnect();
  }, [viewportRef, signature, turnCount]);
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
  }, [previewOpen, cancelHide, setActiveMarker]);

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
            className="group flex h-[var(--rail-row,8px)] w-full shrink-0 cursor-pointer items-center justify-start rounded-sm pl-1.5 outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {/* scaled, not resized: hover and the in-view tint move constantly, so only transform and colour animate */}
            <span
              className={cn(
                "h-[var(--rail-line,2px)] w-6 origin-left scale-x-[0.333] rounded-full transition-[transform,background-color] duration-150 ease-out group-data-[dist=0]:scale-x-100 group-data-[dist=1]:scale-x-[0.667] group-data-[dist=2]:scale-x-[0.417] motion-reduce:transition-none",
                isBookmarked
                  ? "bg-primary"
                  : "bg-muted-foreground/30 group-data-[in-view]:bg-muted-foreground group-data-[dist=0]:bg-foreground",
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
  return (
    <div
      ref={anchorRef}
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-turn-navigator-anchor pointer-events-none select-none sticky top-1/2 z-10 h-0 w-full shrink-0"
    >
      <div
        style={{
          width:
            "calc((100% - min(100%, var(--thread-content-max-width))) / 2)",
        }}
        className="aui-turn-navigator-gutter @container/turn-gutter absolute top-0 left-0 h-0"
      >
        <div className="absolute top-0 left-[-0.75rem] flex w-9 -translate-y-1/2 flex-col items-start opacity-0 transition-opacity duration-150 hover:opacity-100 focus-within:opacity-100 @min-[2.25rem]/turn-gutter:opacity-100">
          <nav
            aria-label={t("turns.navigator")}
            style={{
              height: `min(calc(var(--rail-row, 8px) * ${openerIds.length} + 8px), 40dvh)`,
            }}
            ref={railRef}
            onKeyDown={onRailKeyDown}
            onScroll={onRailScroll}
            className="aui-turn-navigator pointer-events-auto flex w-9 shrink-0 flex-col overflow-y-hidden py-1 transition-[height] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] [contain:layout_paint] [mask-image:linear-gradient(to_bottom,transparent,#000_var(--rail-fade-top,0px),#000_calc(100%_-_var(--rail-fade-bottom,0px)),transparent)] motion-reduce:transition-none"
          >
            {markers}
          </nav>
        </div>
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
                  left: `calc(${preview.left}px + 0.5rem)`,
                  maxWidth: `calc(100vw - ${preview.left}px - 1.5rem)`,
                  transform: `translateY(${preview.edge === "start" ? "-1rem" : preview.edge === "end" ? "calc(-100% + 1rem)" : "-50%"})`,
                }
              : undefined
          }
          onPointerEnter={cancelHide}
          onPointerLeave={hidePreview}
          className={cn(
            "aui-turn-preview pointer-events-auto fixed z-50 flex w-80 select-none flex-col gap-1 rounded-xl border border-border bg-popover py-3 pr-2.5 pl-3.5 text-popover-foreground text-ui-13 shadow-xl shadow-black/20",
            !previewOpen && "invisible",
          )}
        >
          <div className="flex items-center gap-2">
            <p className="min-w-0 flex-1 truncate font-medium text-ui-14">
              {preview?.prompt || " "}
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
                  "flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-md hover:bg-accent",
                  previewBookmarked
                    ? "text-primary"
                    : "text-muted-foreground hover:text-foreground",
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
              "line-clamp-3 pr-1.5 text-muted-foreground text-ui-13p5 leading-relaxed",
              !previewReply && "invisible -mt-1 h-0",
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
