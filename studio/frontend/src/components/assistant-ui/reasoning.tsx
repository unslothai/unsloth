// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/* eslint-disable react-refresh/only-export-components */

import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  resolveReasoningGroupDuration,
  resolveReasoningOpen,
  resolveReasoningToggle,
  startsNewReasoningRound,
  useChatPreferencesStore,
} from "@/features/chat";
import {
  nextReasoningExpandEnd,
  nextReasoningWindowStart,
  widenReasoningWindowStart,
} from "@/features/chat/utils/reasoning-window";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { cn } from "@/lib/utils";
import {
  type ReasoningGroupComponent,
  type ReasoningMessagePartComponent,
  useAuiState,
  useMessagePartText,
} from "@assistant-ui/react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { type VariantProps, cva } from "class-variance-authority";
import { ChevronDownIcon, CopyIcon } from "lucide-react";
import { BulbIcon } from "@/lib/bulb-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type ComponentProps,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
const ANIMATION_DURATION = 200;
const AUTO_SCROLL_THRESHOLD_PX = 24;

export const reasoningVariants = cva("aui-reasoning-root mt-3 mb-4 w-full", {
  variants: {
    variant: {
      outline: "rounded-lg border px-3 py-2",
      ghost: "",
      muted: "rounded-lg bg-muted/50 px-3 py-2",
    },
  },
  defaultVariants: {
    variant: "outline",
  },
});

export type ReasoningRootProps = Omit<
  ComponentProps<typeof Collapsible>,
  "open" | "onOpenChange"
> &
  VariantProps<typeof reasoningVariants> & {
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
    defaultOpen?: boolean;
  };

function ReasoningRoot({
  className,
  variant,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  defaultOpen = false,
  children,
  ...props
}: ReasoningRootProps) {
  const collapsibleRef = useRef<HTMLDivElement>(null);
  const [uncontrolledOpen, setUncontrolledOpen] = useState(defaultOpen);
  const lockScroll = useCollapseScrollLock(collapsibleRef, ANIMATION_DURATION);

  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : uncontrolledOpen;

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) {
        lockScroll();
      }
      if (!isControlled) {
        setUncontrolledOpen(open);
      }
      controlledOnOpenChange?.(open);
    },
    [lockScroll, isControlled, controlledOnOpenChange],
  );

  return (
    <Collapsible
      ref={collapsibleRef}
      data-slot="reasoning-root"
      data-variant={variant}
      open={isOpen}
      onOpenChange={handleOpenChange}
      className={cn(
        "group/reasoning-root",
        reasoningVariants({ variant, className }),
      )}
      style={
        {
          "--animation-duration": `${ANIMATION_DURATION}ms`,
        } as CSSProperties
      }
      {...props}
    >
      {children}
    </Collapsible>
  );
}

function ReasoningTrigger({
  active,
  duration,
  className,
  ...props
}: ComponentProps<typeof CollapsibleTrigger> & {
  active?: boolean;
  duration?: number;
}) {
  return (
    <CollapsibleTrigger
      data-slot="reasoning-trigger"
      className={cn(
        "aui-reasoning-trigger group/trigger flex min-w-0 flex-1 cursor-pointer items-center gap-2 py-1 text-muted-foreground text-sm transition-colors hover:text-foreground",
        className,
      )}
      {...props}
    >
      <BulbIcon className="aui-reasoning-trigger-icon size-4 shrink-0" />
      <span
        data-slot="reasoning-trigger-label"
        className="aui-reasoning-trigger-label-wrapper relative inline-block leading-none"
      >
        {active ? (
          <span className="text-sm">Thinking...</span>
        ) : (
          <span>Thought for {duration ?? 0} {duration === 1 ? "second" : "seconds"}</span>
        )}
      </span>
      <ChevronDownIcon
        data-slot="reasoning-trigger-chevron"
        className={cn(
          "aui-reasoning-trigger-chevron mt-0.5 size-3.5 shrink-0",
          "transition-transform duration-(--animation-duration) ease-out",
          "group-data-[state=closed]/trigger:-rotate-90",
          "group-data-[state=open]/trigger:rotate-0",
        )}
      />
    </CollapsibleTrigger>
  );
}

function ReasoningContent({
  className,
  children,
  streaming,
  ...props
}: ComponentProps<typeof CollapsibleContent> & { streaming?: boolean }) {
  return (
    <CollapsibleContent
      data-slot="reasoning-content"
      className={cn(
        "aui-reasoning-content relative overflow-hidden text-foreground/85 text-ui-13p5 outline-none",
        "group/collapsible-content ease-out",
        "data-[state=closed]:animate-collapsible-up",
        "data-[state=open]:animate-collapsible-down",
        "data-[state=closed]:fill-mode-forwards",
        "data-[state=closed]:pointer-events-none",
        "data-[state=open]:duration-(--animation-duration)",
        "data-[state=closed]:duration-(--animation-duration)",
        className,
      )}
      {...props}
    >
      {children}
    </CollapsibleContent>
  );
}

function ReasoningText({
  className,
  streaming,
  children,
  ...props
}: ComponentProps<"div"> & { streaming?: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const detachedFromBottomRef = useRef(false);
  const lastScrollTopRef = useRef(0);

  useEffect(() => {
    if (!(streaming && scrollRef.current)) {
      return;
    }
    const el = scrollRef.current;
    const updateAutoScroll = () => {
      const currentScrollTop = el.scrollTop;
      if (currentScrollTop < lastScrollTopRef.current) {
        detachedFromBottomRef.current = true;
      }
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (
        detachedFromBottomRef.current &&
        distanceFromBottom <= AUTO_SCROLL_THRESHOLD_PX
      ) {
        detachedFromBottomRef.current = false;
      }
      shouldAutoScrollRef.current = !detachedFromBottomRef.current;
      lastScrollTopRef.current = currentScrollTop;
    };
    const handleWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) {
        detachedFromBottomRef.current = true;
        shouldAutoScrollRef.current = false;
      }
    };
    const observer = new MutationObserver(() => {
      if (shouldAutoScrollRef.current) {
        el.scrollTop = el.scrollHeight;
      }
    });
    el.addEventListener("scroll", updateAutoScroll);
    el.addEventListener("wheel", handleWheel, { passive: true });
    observer.observe(el, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    lastScrollTopRef.current = el.scrollTop;
    detachedFromBottomRef.current = false;
    updateAutoScroll();
    return () => {
      observer.disconnect();
      el.removeEventListener("scroll", updateAutoScroll);
      el.removeEventListener("wheel", handleWheel);
    };
  }, [streaming]);

  return (
    <div
      ref={scrollRef}
      data-slot="reasoning-text"
      className={cn(
        "aui-reasoning-text relative z-0 overflow-y-auto pt-2 pb-0 pl-0 leading-relaxed",
        streaming ? "max-h-64" : "",
        "transform-gpu transition-[transform,opacity]",
        "group-data-[state=open]/collapsible-content:animate-in",
        "group-data-[state=closed]/collapsible-content:animate-out",
        "group-data-[state=open]/collapsible-content:fade-in-0",
        "group-data-[state=closed]/collapsible-content:fade-out-0",
        "group-data-[state=open]/collapsible-content:slide-in-from-top-4",
        "group-data-[state=closed]/collapsible-content:slide-out-to-top-4",
        "group-data-[state=open]/collapsible-content:duration-(--animation-duration)",
        "group-data-[state=closed]/collapsible-content:duration-(--animation-duration)",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

/**
 * How long after the stream ends before a finished block starts mounting the rest of itself.
 *
 * The group closes itself when the round finishes and Radix unmounts closed content, so in the
 * ordinary case this timer is cleaned up before it fires and the rest of the body never mounts at
 * all. That is where most of the win is: mounting 90,000 characters of thinking on the completion
 * frame, only to unmount it 200ms later, was measured at a 5.8 SECOND frame on a machine with no
 * headroom.
 *
 * It fires for the reader who opened the block by hand and is therefore still looking at it.
 */
const SETTLE_DELAY_MS = ANIMATION_DURATION + 50;

/** How near the top of the pane the reader has to get before the window widens, in pixels. */
const WIDEN_TRIGGER_PX = 200;

/** How near the bottom still counts as following the stream, in pixels. */
const AT_BOTTOM_PX = 24;

/**
 * How long after a widen the reader's position keeps being restored, in milliseconds.
 *
 * Not one frame. The content mounted above the reader reaches its final height only once Shiki
 * has highlighted its fences, which is asynchronous and lands over several frames, so a single
 * correction is made against a body that is still growing and drifts as it settles.
 */
const ANCHOR_SETTLE_MS = 400;

/**
 * The thinking body: as much of it as is worth mounting, and all of it whenever the reader asks.
 *
 * Three states, and they are genuinely different problems:
 *
 *   STREAMING, READER AT THE END. Mount a tail window. The reader is watching the newest text and
 *   cannot see what is above it, so the start may advance. This is where the cost is and where
 *   the win is.
 *
 *   STREAMING, READER SCROLLED BACK. Stop advancing, and WIDEN when they reach the top of what is
 *   mounted. Nothing above the reader is ever unmounted, because unmounting above a reader is
 *   what produces scroll jumps.
 *
 *   FINISHED AND EXPANDED BY HAND. Mount from the head and grow downward a step per frame. A
 *   finished group has no height cap -- `streaming` is false -- so it is not a 256px scroller at
 *   all, and a reader opening a finished thinking block wants its beginning, not its end. Growing
 *   downward also needs no anchoring, since appending below the reader cannot move what is above.
 *
 * In every state, repeated asking reaches the whole body. That is a correctness property rather
 * than a performance one, and features/chat/utils/reasoning-window.ts carries the tests for it.
 */
function ReasoningBody() {
  const { text, status } = useMessagePartText();
  const isRunning = status.type === "running";

  const hostRef = useRef<HTMLDivElement>(null);
  const startRef = useRef(0);
  // Set once the reader scrolls back, which freezes the start. Cleared when they return to the
  // bottom, which lets it follow the stream again.
  const pinnedRef = useRef(false);
  // How much of the text the pane shows while the reader is scrolled back.
  //
  // The tail is FROZEN for as long as they are detached, and that is not only politeness. It is
  // what makes the scroll anchor valid: the place a widen has to hold still is measured as a
  // distance from the BOTTOM of the body, and while the stream is still appending below, the
  // bottom moves, so holding a distance from it drags the reader downwards. Measured before this
  // was added, the reader's distance from the bottom walked 17,042px to 46,125px across eight
  // widens. Freezing the tail makes the bottom static, which makes the distance exact.
  //
  // Nothing is lost: the text is still in the part, and returning to the bottom shows all of it.
  const frozenEndRef = useRef<number | null>(null);
  // What the restore loop last wrote, so a scroll it did not cause can be told from one it did.
  const wroteRef = useRef<number | null>(null);
  const settleRef = useRef<number | null>(null);
  const [, forceRender] = useState(0);

  const [settled, setSettled] = useState(!isRunning);
  const [expandEnd, setExpandEnd] = useState<number | null>(null);

  useEffect(() => {
    if (isRunning) {
      setSettled(false);
      setExpandEnd(null);
      return;
    }
    const timeout = window.setTimeout(() => setSettled(true), SETTLE_DELAY_MS);
    return () => window.clearTimeout(timeout);
  }, [isRunning]);

  const scroller = useCallback(
    () => hostRef.current?.closest<HTMLElement>('[data-slot="reasoning-text"]') ?? null,
    [],
  );

  // Grow a finished, hand-expanded block one step per frame until it is whole. Stepping rather
  // than committing it all at once is what keeps the expand off a single long frame, and it
  // finishes on its own, so the reader never has to scroll to be given the rest.
  useEffect(() => {
    if (!settled || isRunning) return;
    if (expandEnd !== null && expandEnd >= text.length) return;
    const handle = window.requestAnimationFrame(() => {
      setExpandEnd((current) => nextReasoningExpandEnd(text, current ?? 0));
    });
    return () => window.cancelAnimationFrame(handle);
  }, [settled, isRunning, expandEnd, text]);

  /**
   * Hold the reader's place across a widen, measured as DISTANCE FROM THE BOTTOM.
   *
   * PR 9058 holds a mounted row still in viewport space, and that is the right technique there
   * because its rows survive the commit that inserts more above them. Here they do not: widening
   * hands `IncrementalMarkdownCache` a string that is not an extension of the last one, which
   * drops its retained blocks and re-keys Streamdown, so the entire body remounts and any node
   * captured beforehand is gone by the time a layout effect could measure it. `isConnected` would
   * be false and the correction would silently never run.
   *
   * Distance from the bottom survives that, because widening changes nothing below the reader:
   * the tail is character-for-character what it was. It is also immune to the double-correction
   * that makes document space wrong in 9058, since a full remount leaves the browser's own scroll
   * anchoring nothing to anchor to.
   *
   * Re-applied for ANCHOR_SETTLE_MS rather than once, because the newly mounted fences reach
   * their real height only as Shiki finishes with them, and it stops the moment the reader
   * scrolls themselves.
   */
  const holdPlace = useCallback(
    (distanceFromBottom: number) => {
      if (settleRef.current !== null) window.cancelAnimationFrame(settleRef.current);
      const deadline = performance.now() + ANCHOR_SETTLE_MS;
      const step = () => {
        const element = scroller();
        if (!element) {
          settleRef.current = null;
          return;
        }
        // The reader moved it themselves since the last write, so stop: their scroll wins.
        if (wroteRef.current !== null && Math.abs(element.scrollTop - wroteRef.current) > 1) {
          settleRef.current = null;
          wroteRef.current = null;
          return;
        }
        const target = Math.max(
          0,
          element.scrollHeight - element.clientHeight - distanceFromBottom,
        );
        if (Math.abs(element.scrollTop - target) >= 1) element.scrollTop = target;
        wroteRef.current = element.scrollTop;
        settleRef.current =
          performance.now() < deadline ? window.requestAnimationFrame(step) : null;
      };
      wroteRef.current = null;
      settleRef.current = window.requestAnimationFrame(step);
    },
    [scroller],
  );

  useEffect(
    () => () => {
      if (settleRef.current !== null) window.cancelAnimationFrame(settleRef.current);
    },
    [],
  );

  useEffect(() => {
    const element = scroller();
    if (!(element && isRunning)) return;
    const onScroll = () => {
      // Our own correction fires scroll events too. Without this the settle loop's write reads as
      // the reader arriving at the top again and widens once more, which widens again, and the
      // pane walks back to its whole size in three gestures: measured at 2,690 pane elements
      // before the first widen and 12,875 after the third, with a 671ms frame in the middle.
      if (
        wroteRef.current !== null &&
        Math.abs(element.scrollTop - wroteRef.current) <= 1
      ) {
        return;
      }
      const distanceFromBottom =
        element.scrollHeight - element.scrollTop - element.clientHeight;
      const wasPinned = pinnedRef.current;
      pinnedRef.current = distanceFromBottom > AT_BOTTOM_PX;
      if (pinnedRef.current && !wasPinned) {
        // Just detached. Freeze what is shown at exactly what is on screen now.
        frozenEndRef.current = text.length;
      } else if (!pinnedRef.current && wasPinned) {
        // Back at the bottom: follow the stream again, and show all of it.
        frozenEndRef.current = null;
        forceRender((n) => n + 1);
      }
      if (!pinnedRef.current || startRef.current <= 0) return;
      if (element.scrollTop > WIDEN_TRIGGER_PX) return;
      // One widen at a time. While the settle loop is still holding the reader's place the body
      // is still reaching its final height, so a second widen would be triggered against a
      // measurement that has not finished.
      if (settleRef.current !== null) return;
      const next = widenReasoningWindowStart(text, startRef.current);
      if (next >= startRef.current) return;
      startRef.current = next;
      // Captured here, synchronously, so the reader's own scroll is already inside the baseline
      // rather than spanning it. 9058 skips any frame a gesture landed in; that guard cannot be
      // reused here, because here the gesture is what TRIGGERS the widen and skipping would skip
      // every correction.
      holdPlace(distanceFromBottom);
      forceRender((n) => n + 1);
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => element.removeEventListener("scroll", onScroll);
  }, [scroller, isRunning, text, holdPlace]);

  const body = (() => {
    if (isRunning) {
      // Advanced during render, not in an effect: while the reader is at the end the start is a
      // pure function of the text, and computing it after the commit would render one frame of
      // the previous window against the new text every time it moves.
      if (!pinnedRef.current) {
        startRef.current = nextReasoningWindowStart(text, startRef.current);
      }
      const end = pinnedRef.current ? (frozenEndRef.current ?? text.length) : text.length;
      if (startRef.current === 0 && end >= text.length) return text;
      return text.slice(startRef.current, end);
    }
    if (!settled) {
      // The completion frame. Keep exactly what was mounted rather than growing on the frame the
      // collapse animation is running.
      return startRef.current === 0 ? text : text.slice(startRef.current);
    }
    const end = expandEnd ?? 0;
    return end >= text.length ? text : text.slice(0, end);
  })();

  return (
    <div ref={hostRef} className="contents">
      <MarkdownText text={body === text ? undefined : body} />
    </div>
  );
}

const ReasoningImpl: ReasoningMessagePartComponent = () => <ReasoningBody />;

const COPY_RESET_MS = 2000;

function ReasoningCopyButton({ startIndex, endIndex }: { startIndex: number; endIndex: number }) {
  const [copied, setCopied] = useState(false);
  const resetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reasoningText = useAuiState(({ message }) => {
    return message.parts
      .slice(startIndex, endIndex + 1)
      .filter((p) => p.type === "reasoning")
      .map((p) => ("text" in p ? (p as { text: string }).text : ""))
      .join("\n");
  });

  const handleCopy = useCallback(async () => {
    if (await copyToClipboard(reasoningText)) {
      setCopied(true);
      if (resetRef.current) clearTimeout(resetRef.current);
      resetRef.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  }, [reasoningText]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:text-foreground hover:bg-muted"
      aria-label="Copy reasoning"
    >
      {copied ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3" />
      ) : (
        <CopyIcon className="size-3" />
      )}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

const ReasoningGroupImpl: ReasoningGroupComponent = ({
  children,
  startIndex,
  endIndex,
}) => {
  const isReasoningStreaming = useAuiState(({ message }) => {
    if (message.status?.type !== "running") {
      return false;
    }
    const parts = message.parts;
    const len = parts.length;
    if (len === 0) {
      return false;
    }

    let groupHasReasoning = false;
    for (let i = startIndex; i <= endIndex && i < len; i += 1) {
      if (parts[i]?.type === "reasoning") {
        groupHasReasoning = true;
        break;
      }
    }
    if (!groupHasReasoning) {
      return false;
    }
    for (let i = endIndex + 1; i < len; i += 1) {
      if (parts[i]?.type !== "tool-call") {
        return false;
      }
    }
    return true;
  });

  const persistedDuration = useAuiState(({ message }) => {
    return resolveReasoningGroupDuration(
      message.parts,
      startIndex,
      message.metadata?.custom as Record<string, unknown> | undefined,
    );
  });

  const collapseByDefault = useChatPreferencesStore(
    (state) => state.collapseThinkingByDefault,
  );

  const [manualOpen, setManualOpen] = useState(false);
  const [dismissedWhileStreaming, setDismissedWhileStreaming] = useState(false);
  const [retainStreamingHeight, setRetainStreamingHeight] = useState(false);
  const [duration, setDuration] = useState<number>(0);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (isReasoningStreaming) {
      if (startTimeRef.current === null) {
        startTimeRef.current = Date.now();
      }
    } else if (startTimeRef.current !== null) {
      const elapsed = Math.round((Date.now() - startTimeRef.current) / 1000);
      setDuration(elapsed);
      startTimeRef.current = null;
    }
  }, [isReasoningStreaming]);

  // Reset per-round open state. manualOpen is sticky and regenerate reuses this
  // instance, so a hand-opened block would stay pinned open and never collapse.
  // Adjusted during render, not in an effect: React re-runs this component
  // before committing, so a stale open never reaches the DOM.
  const [wasStreaming, setWasStreaming] = useState(isReasoningStreaming);
  if (wasStreaming !== isReasoningStreaming) {
    setWasStreaming(isReasoningStreaming);
    if (startsNewReasoningRound(isReasoningStreaming, wasStreaming)) {
      setDismissedWhileStreaming(false);
      setManualOpen(false);
    }
  }

  // Keep the streaming height cap until the automatic close finishes. Removing
  // it on the completion frame expands long reasoning to its full height before
  // the collapsible can close, which makes the entire chat jump.
  useEffect(() => {
    const timeout = window.setTimeout(
      () => setRetainStreamingHeight(isReasoningStreaming),
      isReasoningStreaming ? 0 : ANIMATION_DURATION,
    );
    return () => window.clearTimeout(timeout);
  }, [isReasoningStreaming]);

  // Open while streaming (unless dismissed), or once manually opened. With
  // collapse by default on, only a manual open shows the block.
  const isOpen = resolveReasoningOpen({
    isStreaming: isReasoningStreaming,
    collapseByDefault,
    dismissedWhileStreaming,
    manualOpen,
  });
  const variant = isOpen ? "outline" : "ghost";

  // Allow closing during streaming (matches ChatGPT).
  const handleOpenChange = useCallback(
    (open: boolean) => {
      const next = resolveReasoningToggle(open, {
        isStreaming: isReasoningStreaming,
        collapseByDefault,
      });
      if (next.releaseStreamingHeight) {
        setRetainStreamingHeight(false);
      }
      setManualOpen(next.manualOpen);
      if (next.dismissedWhileStreaming !== undefined) {
        setDismissedWhileStreaming(next.dismissedWhileStreaming);
      }
    },
    [isReasoningStreaming, collapseByDefault],
  );

  return (
    <ReasoningRoot
      open={isOpen}
      onOpenChange={handleOpenChange}
      variant={variant}
    >
      <div className="flex min-w-0 items-center gap-2">
        <ReasoningTrigger
          className="min-w-0 flex-1"
          active={isReasoningStreaming}
          // Prefer server timing when available.
          duration={persistedDuration ?? duration}
        />
        <div className="flex w-16 shrink-0 justify-end">
          {isOpen && !isReasoningStreaming && (
            <ReasoningCopyButton startIndex={startIndex} endIndex={endIndex} />
          )}
        </div>
      </div>
      <ReasoningContent
        aria-busy={isReasoningStreaming}
        streaming={isReasoningStreaming}
      >
        <ReasoningText
          streaming={isReasoningStreaming || retainStreamingHeight}
        >
          {children}
        </ReasoningText>
      </ReasoningContent>
    </ReasoningRoot>
  );
};

const Reasoning = memo(
  ReasoningImpl,
) as unknown as ReasoningMessagePartComponent & {
  Root: typeof ReasoningRoot;
  Trigger: typeof ReasoningTrigger;
  Content: typeof ReasoningContent;
  Text: typeof ReasoningText;
};

Reasoning.displayName = "Reasoning";
Reasoning.Root = ReasoningRoot;
Reasoning.Trigger = ReasoningTrigger;
Reasoning.Content = ReasoningContent;
Reasoning.Text = ReasoningText;

const ReasoningGroup = memo(ReasoningGroupImpl);
ReasoningGroup.displayName = "ReasoningGroup";

export {
  Reasoning,
  ReasoningGroup,
  ReasoningRoot,
  ReasoningTrigger,
  ReasoningContent,
  ReasoningText,
};
