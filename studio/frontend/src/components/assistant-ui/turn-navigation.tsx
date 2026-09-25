// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ActionBarMorePrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { PinIcon, PinOffIcon } from "@hugeicons/core-free-icons";
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
  useChatPreferencesStore,
  useChatRuntimeStore,
  usePinnedTurnsStore,
} from "@/features/chat";
import { FIND_SKIP_ATTRIBUTE } from "@/features/find-in-page";
import { prefersReducedMotion } from "@/features/settings";
import { useT } from "@/i18n";

const MIN_NAVIGATOR_TURNS = 3;
// bounds the text a pasted log can put in the tooltip, which scrolls past its height cap
const TURN_PREVIEW_CHARS = 4000;
// long enough to cross from a marker onto the tooltip to scroll it
const PREVIEW_HIDE_DELAY_MS = 150;
// how far the tooltip's near edge sits past the marker it opens from
const PREVIEW_OFFSET_PX = 14;
// math blocks above the target settle from placeholder heights once reached, so the jump re-aligns briefly
const JUMP_ALIGN_FRAMES = 4;

function useIsTurnPinned(
  threadId: string | undefined,
  openerId: string | undefined,
): boolean {
  return usePinnedTurnsStore((state) =>
    threadId && openerId
      ? (state.pinnedByThread[threadId]?.includes(openerId) ?? false)
      : false,
  );
}

// pin state for the turn holding the current message, null when turn pins are unavailable
function useTurnPin(): { pinned: boolean; toggle: () => void } | null {
  const enabled = useChatPreferencesStore((state) => state.showTurnNavigation);
  const incognito = useChatRuntimeStore((state) => state.incognito);
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const openerId = useAuiState(({ thread, message }) =>
    enabled ? turnOpenerIdAt(thread.messages, message.index) : undefined,
  );
  const pinned = useIsTurnPinned(threadId, openerId);
  const togglePinnedTurn = usePinnedTurnsStore(
    (state) => state.togglePinnedTurn,
  );
  if (!enabled || incognito || !threadId || !openerId) {
    return null;
  }
  return { pinned, toggle: () => togglePinnedTurn(threadId, openerId) };
}

export const PinTurnButton: FC = () => {
  const t = useT();
  const turnPin = useTurnPin();
  if (!turnPin) {
    return null;
  }
  return (
    <TooltipIconButton
      tooltip={t(turnPin.pinned ? "turns.unpin" : "turns.pin")}
      aria-pressed={turnPin.pinned}
      onClick={turnPin.toggle}
    >
      <HugeiconsIcon
        icon={turnPin.pinned ? PinOffIcon : PinIcon}
        strokeWidth={1.75}
        className="size-icon"
      />
    </TooltipIconButton>
  );
};

export const PinTurnMenuItem: FC<{ className?: string }> = ({ className }) => {
  const t = useT();
  const turnPin = useTurnPin();
  if (!turnPin) {
    return null;
  }
  return (
    <ActionBarMorePrimitive.Item
      onSelect={turnPin.toggle}
      className={className}
    >
      <HugeiconsIcon
        icon={turnPin.pinned ? PinOffIcon : PinIcon}
        strokeWidth={1.75}
        className="size-icon"
      />
      {t(turnPin.pinned ? "turns.unpin" : "turns.pin")}
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
  const pinned = useIsTurnPinned(threadId, messageId);
  if (turn === 0) {
    return null;
  }
  return (
    <div
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-user-turn-label flex select-none items-center gap-1 font-medium text-muted-foreground/80 text-ui-11"
    >
      {pinned && (
        <HugeiconsIcon
          icon={PinIcon}
          strokeWidth={2}
          className="size-3"
          role="img"
          aria-label={t("turns.pinned")}
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

function turnPreview(content: readonly { type: string; text?: string }[]) {
  const text = content
    .map((part) => (part.type === "text" ? (part.text ?? "") : ""))
    .join("\n\n")
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return text.length > TURN_PREVIEW_CHARS
    ? `${text.slice(0, TURN_PREVIEW_CHARS)}…`
    : text;
}

interface TurnPreview {
  turn: number;
  pinned: boolean;
  text: string;
  // the anchor sits at the rail's centre, so a marker below it opens the tooltip upward
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
  const pinnedIds = usePinnedTurnsStore((state) =>
    threadId ? state.pinnedByThread[threadId] : undefined,
  );
  const openerIds = useMemo(
    () => (signature ? signature.split("\n") : []),
    [signature],
  );
  const pinned = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  const anchorRef = useRef<HTMLDivElement>(null);
  const [preview, setPreview] = useState<TurnPreview | null>(null);
  const hideTimerRef = useRef<number | undefined>(undefined);
  const cancelHide = useCallback(
    () => window.clearTimeout(hideTimerRef.current),
    [],
  );
  useEffect(() => cancelHide, [cancelHide]);
  // marked on the element rather than in state, so the memoized markers do not re-render on hover
  const activeMarkerRef = useRef<HTMLButtonElement | null>(null);
  const setActiveMarker = useCallback((marker: HTMLButtonElement | null) => {
    activeMarkerRef.current?.removeAttribute("data-active");
    marker?.setAttribute("data-active", "");
    activeMarkerRef.current = marker;
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
      if (!anchor) {
        return;
      }
      cancelHide();
      setActiveMarker(marker);
      const message = aui
        .thread()
        .getState()
        .messages.find((candidate) => candidate.id === marker.dataset.turnId);
      const box = marker.getBoundingClientRect();
      setPreview({
        turn: Number(marker.dataset.turn),
        pinned: marker.dataset.pinned !== undefined,
        text: message ? turnPreview(message.content) : "",
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

  // memoized so showing the tooltip re-renders the rail without touching a marker
  const markers = useMemo(
    () =>
      openerIds.map((openerId, index) => {
        const isPinned = pinned.has(openerId);
        return (
          <button
            key={openerId}
            type="button"
            data-turn-id={openerId}
            data-turn={index + 1}
            data-pinned={isPinned || undefined}
            tabIndex={index === 0 ? 0 : -1}
            aria-label={t(isPinned ? "turns.pinnedLabel" : "turns.label", {
              number: index + 1,
            })}
            onClick={onMarkerClick}
            onPointerEnter={showPreview}
            onPointerLeave={hidePreview}
            onFocus={showPreview}
            onBlur={hidePreview}
            className="group flex min-h-1 w-8 flex-1 cursor-pointer items-center justify-end rounded-sm pr-1.5 outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <span className="h-0.5 w-4 rounded-full bg-muted-foreground/35 transition-[width,background-color] duration-100 group-hover:w-6 group-hover:bg-muted-foreground group-data-[active]:w-6 group-data-[active]:bg-muted-foreground group-data-[pinned]:h-[3px] group-data-[pinned]:bg-primary" />
          </button>
        );
      }),
    [openerIds, pinned, t, onMarkerClick, showPreview, hidePreview],
  );

  if (openerIds.length < MIN_NAVIGATOR_TURNS) {
    return null;
  }

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
        className="aui-turn-navigator pointer-events-auto absolute top-0 right-[-1.125rem] flex w-8 -translate-y-1/2 flex-col overflow-y-auto py-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {markers}
      </nav>
      {preview && (
        <div
          aria-hidden={true}
          style={{
            top:
              preview.markerTop > 0
                ? preview.markerTop + PREVIEW_OFFSET_PX
                : preview.markerTop - PREVIEW_OFFSET_PX,
          }}
          onPointerEnter={cancelHide}
          onPointerLeave={hidePreview}
          data-placement={preview.markerTop > 0 ? "above" : "below"}
          className="aui-turn-preview pointer-events-auto data-[placement=above]:-translate-y-full absolute right-[1.125rem] flex max-h-[min(36dvh,22rem)] w-max max-w-[28rem] flex-col gap-1 rounded-[14px] border border-sidebar-border bg-sidebar px-3.5 py-2.5 font-medium text-sidebar-foreground text-ui-13 leading-snug shadow-md"
        >
          <div className="flex shrink-0 items-center gap-1 text-muted-foreground text-ui-11">
            {preview.pinned && (
              <HugeiconsIcon
                icon={PinIcon}
                strokeWidth={2}
                className="size-3"
              />
            )}
            {t("turns.label", { number: preview.turn })}
          </div>
          {preview.text && (
            <p className="min-h-0 overflow-y-auto overscroll-contain whitespace-pre-line pr-1 [scrollbar-width:thin]">
              {preview.text}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
