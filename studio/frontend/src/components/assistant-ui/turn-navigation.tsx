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

import { completeProgressiveMounts } from "@/components/assistant-ui/progressive-messages";
import {
  threadTurns,
  turnNumberAt,
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
import { cn } from "@/lib/utils";

const MIN_NAVIGATOR_TURNS = 3;
// how far an overflowing rail's ends fade out, scaled by how much is hidden past each end
const RAIL_FADE_PX = 24;
const PROMPT_PREVIEW_CHARS = 240;
const REPLY_PREVIEW_CHARS = 480;
// long enough to cross from a marker onto the card
const PREVIEW_HIDE_DELAY_MS = 150;
const PYRAMID_REACH = 3;
// math blocks above the target settle from placeholder heights once reached, so the jump re-aligns briefly
const JUMP_ALIGN_FRAMES = 4;

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

function useTurnBookmark(): { bookmarked: boolean; toggle: () => void } | null {
  const incognito = useChatRuntimeStore((state) => state.incognito);
  // one primitive selector: it runs on every store write (keystrokes, streamed tokens) for every mounted message
  const key = useAuiState(
    ({ thread, message, threadListItem }) =>
      `${threadListItem.remoteId ?? ""}\n${turnOpenerIdAt(thread.messages, message.index) ?? ""}`,
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
  const turnBookmark = useTurnBookmark();
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
  const turnBookmark = useTurnBookmark();
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

export const UserTurnLabel: FC = () => {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  return enabled ? <UserTurnLabelText /> : null;
};

const UserTurnLabelText: FC = () => {
  const t = useT();
  const key = useAuiState(
    ({ thread, message, threadListItem }) =>
      `${turnNumberAt(thread.messages, message.index)}\n${threadListItem.remoteId ?? ""}\n${message.id}`,
  );
  const [turnText, threadId, messageId] = key.split("\n");
  const turn = Number(turnText);
  const bookmarked = useIsTurnBookmarked(threadId, messageId);
  if (turn === 0) {
    return null;
  }
  return (
    <div
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-user-turn-label flex select-none items-center gap-1 font-medium text-muted-foreground/80 text-ui-11"
    >
      {bookmarked && (
        <HugeiconsIcon
          icon={Bookmark02Icon}
          strokeWidth={2}
          className="size-3 fill-current"
          role="img"
          aria-label={t("turns.bookmarked")}
        />
      )}
      <span>{t("turns.label", { number: turn })}</span>
    </div>
  );
};

// memoized: the thread re-renders on composer resizes
export const TurnNavigator: FC<{
  viewportRef: RefObject<HTMLElement | null>;
}> = memo(({ viewportRef }) => {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  return enabled ? <TurnRail viewportRef={viewportRef} /> : null;
});
TurnNavigator.displayName = "TurnNavigator";

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

// keep the core wider than one frame of travel, or a thin line strobes
const SHINE_GRADIENT = {
  light:
    "linear-gradient(110deg, transparent 34%, rgb(0 0 0 / 0.03) 40%, rgb(255 255 255 / 0.3) 45.5%, rgb(255 255 255 / 0.8) 48.5%, rgb(255 255 255) 50%, rgb(255 255 255 / 0.8) 51.5%, rgb(255 255 255 / 0.3) 54.5%, rgb(0 0 0 / 0.03) 60%, transparent 66%)",
  dark: "linear-gradient(110deg, transparent 34%, rgb(255 255 255 / 0.03) 40%, rgb(255 255 255 / 0.1) 45.5%, rgb(255 255 255 / 0.28) 48.5%, rgb(255 255 255 / 0.36) 50%, rgb(255 255 255 / 0.28) 51.5%, rgb(255 255 255 / 0.1) 54.5%, rgb(255 255 255 / 0.03) 60%, transparent 66%)",
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
    { duration: 1100, easing: "cubic-bezier(0.4, 0, 0.2, 1)" },
  );
}

type PreviewPart = { type: string; text?: string };

function previewText(content: readonly PreviewPart[], maxChars: number) {
  const text = content
    .map((part) => (part.type === "text" ? (part.text ?? "") : ""))
    .join("\n")
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
  // marker centre relative to the anchor
  markerTop: number;
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
  const [preview, setPreview] = useState<TurnPreview | null>(null);
  // only the latest turn can still stream, so an open card follows it and the scan stops at its prompt
  const previewOpenerId = preview?.openerId;
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
      setPreview({
        openerId,
        turn: Number(marker.dataset.turn),
        prompt:
          start === undefined
            ? ""
            : previewText(messages[start].content, PROMPT_PREVIEW_CHARS),
        reply: start === undefined ? "" : turnReplyText(messages, start),
        markerTop:
          box.top + box.height / 2 - anchor.getBoundingClientRect().top,
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
  // dock-style: each dash's width follows its distance to the pointer every frame,
  // so fast sweeps stay smooth instead of stepping through the data-dist widths
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
      dash.style.width = `${0.5 + mag * 0.75}rem`;
      dash.style.transitionProperty = "height, background-color";
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
      if (magnifiedRef.current.length > 0 && !magnifyFrameRef.current) {
        const rail = event.currentTarget;
        magnifyFrameRef.current = requestAnimationFrame(() => magnify(rail));
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
    const fade = (hidden: number) =>
      `${Math.min(Math.max(hidden, 0), RAIL_FADE_PX)}px`;
    const fadeEnds = () => {
      const railRange = Math.max(0, rail.scrollHeight - rail.clientHeight);
      rail.style.setProperty("--rail-fade-top", fade(rail.scrollTop));
      rail.style.setProperty(
        "--rail-fade-bottom",
        fade(railRange - rail.scrollTop),
      );
    };
    const sync = () => {
      frame = 0;
      const railRange = Math.max(0, rail.scrollHeight - rail.clientHeight);
      const range = viewport.scrollHeight - viewport.clientHeight;
      rail.scrollTop =
        railRange > 0 && range > 0
          ? (viewport.scrollTop / range) * railRange
          : 0;
      fadeEnds();
    };
    const schedule = () => {
      if (!frame) {
        frame = requestAnimationFrame(sync);
      }
    };
    // keyboard focus scrolls the rail to reveal its dash; keep the ends right, then follow the thread again once focus leaves
    const onFocusOut = (event: globalThis.FocusEvent) => {
      if (!rail.contains(event.relatedTarget as Node | null)) {
        schedule();
      }
    };
    sync();
    viewport.addEventListener("scroll", schedule, { passive: true });
    rail.addEventListener("scroll", fadeEnds, { passive: true });
    rail.addEventListener("focusout", onFocusOut);
    // the rail eases to a new height on resize, so keep its ends in step
    const resize = new ResizeObserver(schedule);
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
      setPreview(null);
      setActiveMarker(null);
    }, PREVIEW_HIDE_DELAY_MS);
  }, [cancelHide, setActiveMarker]);

  const onRailKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.key === "Escape") {
        cancelHide();
        setPreview(null);
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
            className="group flex h-3 w-full shrink-0 cursor-pointer items-center justify-end rounded-sm pr-1.5 outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <span
              className={cn(
                "h-1 w-2 rounded-full transition-[width,height,background-color] duration-150 ease-out group-data-[dist=0]:w-5 group-data-[dist=1]:w-4 group-data-[dist=2]:w-3 group-data-[dist=3]:w-2.5 group-data-[hovering]/rail:h-[3px] motion-reduce:transition-none",
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
  // sticky, and the rail is not a scroller, so wheel scrolling over it reaches the thread
  return (
    <div
      ref={anchorRef}
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-turn-navigator-anchor pointer-events-none select-none sticky top-1/2 z-10 h-0 w-full shrink-0"
    >
      {/* gutter beside the message column: the rail hides when it would overlap messages */}
      <div
        style={{
          width:
            "calc((100% - min(100%, var(--thread-content-max-width))) / 2)",
        }}
        className="aui-turn-navigator-gutter @container/turn-gutter absolute top-0 right-0 h-0"
      >
        <nav
          aria-label={t("turns.navigator")}
          style={{ height: `min(${openerIds.length * 0.75 + 0.5}rem, 40dvh)` }}
          ref={railRef}
          onKeyDown={onRailKeyDown}
          onPointerMove={onRailPointerMove}
          onPointerLeave={clearMagnify}
          onScroll={onRailScroll}
          className="aui-turn-navigator group/rail pointer-events-auto absolute top-0 right-[-1.125rem] hidden w-8 -translate-y-1/2 flex-col overflow-y-hidden py-1 transition-[height] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] [contain:layout_paint] [mask-image:linear-gradient(to_bottom,transparent,#000_var(--rail-fade-top,0px),#000_calc(100%_-_var(--rail-fade-bottom,0px)),transparent)] motion-reduce:transition-none @[1.5rem]/turn-gutter:flex"
        >
          {markers}
        </nav>
      </div>
      {preview && (
        <div
          id={previewId}
          role="tooltip"
          style={{ top: preview.markerTop }}
          onPointerEnter={cancelHide}
          onPointerLeave={hidePreview}
          className="aui-turn-preview pointer-events-auto absolute right-6 flex w-84 -translate-y-1/2 flex-col gap-1.5 rounded-2xl border border-sidebar-border bg-sidebar py-3 pr-2.5 pl-4 text-sidebar-foreground text-ui-13 shadow-md"
        >
          <div className="flex items-center gap-2">
            <p className="min-w-0 flex-1 truncate font-medium text-ui-13p5">
              {preview.prompt || t("turns.label", { number: preview.turn })}
            </p>
            {threadId && !incognito && (
              <button
                type="button"
                tabIndex={-1}
                aria-label={t(
                  previewBookmarked ? "turns.removeBookmark" : "turns.bookmark",
                )}
                onClick={() => toggleBookmarkedTurn(threadId, preview.openerId)}
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
          {(liveReply ?? preview.reply) && (
            <p className="line-clamp-3 pr-1.5 text-muted-foreground leading-relaxed">
              {liveReply ?? preview.reply}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
