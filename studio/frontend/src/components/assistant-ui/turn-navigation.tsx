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
  memo,
  useCallback,
  useEffect,
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
// the card shows one line of the prompt and three of the reply, so a pasted log is cut well before that
const PROMPT_PREVIEW_CHARS = 240;
const REPLY_PREVIEW_CHARS = 480;
// long enough to cross from a marker onto the card
const PREVIEW_HIDE_DELAY_MS = 150;
// markers this far either side of the hovered one widen with it
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

// bookmark state for the turn holding the current message, null when turn bookmarks are unavailable
function useTurnBookmark(): { bookmarked: boolean; toggle: () => void } | null {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  const incognito = useChatRuntimeStore((state) => state.incognito);
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const openerId = useAuiState(({ thread, message }) =>
    enabled ? turnOpenerIdAt(thread.messages, message.index) : undefined,
  );
  const bookmarked = useIsTurnBookmarked(threadId, openerId);
  const toggleBookmarkedTurn = useBookmarkedTurnsStore(
    (state) => state.toggleBookmarkedTurn,
  );
  if (!enabled || incognito || !threadId || !openerId) {
    return null;
  }
  return { bookmarked, toggle: () => toggleBookmarkedTurn(threadId, openerId) };
}

export const BookmarkTurnButton: FC = () => {
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
  const turn = useAuiState(({ thread, message }) =>
    turnNumberAt(thread.messages, message.index),
  );
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const messageId = useAuiState(({ message }) => message.id);
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

// memoized: the thread re-renders on composer resizes and the rail props never change
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

// a metallic band swept across the bubble's own background: a bright core between darker edges, per theme
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

// plain text for the card: markdown markers are dropped and every run of whitespace becomes one space
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

// the first reply in the turn that has text, so a tool-only step does not blank the card
function turnReplyText(
  messages: readonly {
    id: string;
    role: string;
    content: readonly PreviewPart[];
  }[],
  openerId: string,
): string {
  const start = messages.findIndex((message) => message.id === openerId);
  if (start < 0) {
    return "";
  }
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
  // offset of the marker's centre from the anchor, which sits at the rail's centre
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
  const [preview, setPreview] = useState<TurnPreview | null>(null);
  const hideTimerRef = useRef<number | undefined>(undefined);
  const cancelHide = useCallback(
    () => window.clearTimeout(hideTimerRef.current),
    [],
  );
  useEffect(() => cancelHide, [cancelHide]);
  // the pyramid is marked on the elements rather than held in state, so the memoized markers do not re-render on hover
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

  // read on hover rather than per render, so the rail holds no message text and never goes stale
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
      const opener = messages.find((candidate) => candidate.id === openerId);
      const box = marker.getBoundingClientRect();
      setPreview({
        openerId,
        turn: Number(marker.dataset.turn),
        prompt: opener ? previewText(opener.content, PROMPT_PREVIEW_CHARS) : "",
        reply: turnReplyText(messages, openerId),
        markerTop:
          box.top + box.height / 2 - anchor.getBoundingClientRect().top,
      });
    },
    [aui, cancelHide, setActiveMarker],
  );
  const hidePreview = useCallback(() => {
    cancelHide();
    hideTimerRef.current = window.setTimeout(() => {
      setPreview(null);
      setActiveMarker(null);
    }, PREVIEW_HIDE_DELAY_MS);
  }, [cancelHide, setActiveMarker]);

  const onRailKeyDown = useCallback((event: KeyboardEvent<HTMLElement>) => {
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
  }, []);

  // memoized so showing the card re-renders the rail without touching a marker
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
            className="group flex min-h-1 w-full flex-1 cursor-pointer items-center justify-end rounded-sm pr-1.5 outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
    [openerIds, bookmarked, t, onMarkerClick, showPreview, hidePreview],
  );

  if (openerIds.length < MIN_NAVIGATOR_TURNS) {
    return null;
  }

  const previewBookmarked = preview ? bookmarked.has(preview.openerId) : false;
  // sticky inside the viewport so wheel scrolling over the rail still reaches the thread; the rail sits 2px clear of the scrollbar
  return (
    <div
      ref={anchorRef}
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-turn-navigator-anchor pointer-events-none sticky top-1/2 z-10 hidden h-0 w-full shrink-0 @[52rem]:block"
    >
      <nav
        aria-label={t("turns.navigator")}
        // markers shrink to fit the cap before the rail has to scroll
        style={{ height: `min(${openerIds.length * 0.75 + 0.5}rem, 40dvh)` }}
        onKeyDown={onRailKeyDown}
        className="aui-turn-navigator group/rail pointer-events-auto absolute top-0 right-[-1.125rem] flex w-8 -translate-y-1/2 flex-col overflow-y-auto py-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {markers}
      </nav>
      {preview && (
        <div
          aria-hidden={true}
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
          {preview.reply && (
            <p className="line-clamp-3 pr-1.5 text-muted-foreground leading-relaxed">
              {preview.reply}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
