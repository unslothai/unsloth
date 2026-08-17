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
import { nextReasoningWindowStart } from "@/features/chat/utils/reasoning-window";
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

/** How near the bottom still counts as following the stream, in pixels. */
const AT_BOTTOM_PX = 24;

/**
 * How long the reader's place keeps being held after the body is restored, in milliseconds.
 *
 * Not one frame. The content mounted above them reaches its final height only as Shiki finishes
 * with its fences, which lands over several frames, so a single correction is made against a body
 * that is still growing.
 */
const RESTORE_SETTLE_MS = 400;

/**
 * The thinking body: a bounded tail of it while it streams and the reader is watching the end,
 * and all of it the moment that stops being true.
 *
 * The window exists for exactly one situation, which happens to be the one the whole complaint is
 * about: a block streaming into a 256px pane that auto-follows its own bottom, where the reader
 * cannot see what is above and the mounted nodes above them are pure cost. Measured on the
 * capture's own fixture, the page's frame rate tracks that node count with r = -0.88 in sample
 * windows where almost nothing is mutating, so it is the nodes existing that costs, not the work
 * of building them.
 *
 * Two things end the window, and both restore the body whole and leave it whole:
 *
 *   THE READER SCROLLS BACK. They have said they want to read it, so give them all of it, once,
 *   and stop windowing for the rest of the round. Their place is held across the restore.
 *
 *   THE ROUND FINISHES. A finished block is never windowed. It mounts exactly what it mounts
 *   today.
 *
 * It restores in one step rather than widening progressively because Streamdown 2.5.0 keys blocks
 * positionally, so prepending remounts the whole body however little it adds; see
 * features/chat/utils/reasoning-window.ts. Widening was built and measured first, and it cost
 * frames of 207, 280, 646 and 846ms across four gestures. One restore is one of those, and it is
 * the last one: afterwards the pane is what it would have been without any of this.
 */
function ReasoningBody() {
  const { text, status } = useMessagePartText();
  const isRunning = status.type === "running";

  const hostRef = useRef<HTMLDivElement>(null);
  const startRef = useRef(0);
  // Once set, the whole body is mounted for the rest of this round.
  const restoredRef = useRef(false);
  const settleRef = useRef<number | null>(null);
  const [, forceRender] = useState(0);

  const scroller = useCallback(
    () => hostRef.current?.closest<HTMLElement>('[data-slot="reasoning-text"]') ?? null,
    [],
  );

  // A regenerate reuses this instance, so a new round has to start windowed again rather than
  // inheriting the last round's restore.
  const [wasRunning, setWasRunning] = useState(isRunning);
  if (wasRunning !== isRunning) {
    setWasRunning(isRunning);
    if (isRunning && !wasRunning) {
      startRef.current = 0;
      restoredRef.current = false;
    }
  }

  useEffect(
    () => () => {
      if (settleRef.current !== null) cancelAnimationFrame(settleRef.current);
    },
    [],
  );

  /**
   * Hold the reader's place across the restore, measured as a distance from the BOTTOM.
   *
   * PR 9058 holds a mounted row still in viewport space, which is right for the thread because
   * its rows survive the commit that inserts more above them. They do not survive here: restoring
   * hands `IncrementalMarkdownCache` a string that is not an extension of the last one, which
   * drops its retained blocks and re-keys Streamdown, so anything captured beforehand is detached
   * by the time a layout effect could measure it and the correction would silently never run.
   *
   * Distance from the bottom survives that, because the restore only adds content ABOVE the
   * reader: everything below them is character-for-character what it was. It is also immune to
   * the double-correction that makes document space wrong in 9058, since a full remount leaves
   * the browser's own scroll anchoring nothing to anchor to.
   */
  const holdPlace = useCallback(
    (distanceFromBottom: number) => {
      const deadline = performance.now() + RESTORE_SETTLE_MS;
      const step = () => {
        const element = scroller();
        if (!element) {
          settleRef.current = null;
          return;
        }
        const target = Math.max(
          0,
          element.scrollHeight - element.clientHeight - distanceFromBottom,
        );
        if (Math.abs(element.scrollTop - target) >= 1) element.scrollTop = target;
        settleRef.current =
          performance.now() < deadline ? requestAnimationFrame(step) : null;
      };
      settleRef.current = requestAnimationFrame(step);
    },
    [scroller],
  );

  useEffect(() => {
    const element = scroller();
    if (!(element && isRunning)) return;
    const onScroll = () => {
      if (restoredRef.current || startRef.current <= 0) return;
      const distanceFromBottom =
        element.scrollHeight - element.scrollTop - element.clientHeight;
      if (distanceFromBottom <= AT_BOTTOM_PX) return;
      // Scrolled back. Give them everything and stop windowing for this round. Nothing here can
      // run twice: `restoredRef` is checked first and set before the correction starts, so the
      // hold's own scroll writes cannot re-enter this.
      restoredRef.current = true;
      holdPlace(distanceFromBottom);
      forceRender((n) => n + 1);
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => element.removeEventListener("scroll", onScroll);
  }, [scroller, isRunning, holdPlace]);

  // Advanced during render, not in an effect: while the reader is at the end the start is a pure
  // function of the text, and computing it after the commit would render one frame of the
  // previous window against the new text every time it moves.
  if (isRunning && !restoredRef.current) {
    startRef.current = nextReasoningWindowStart(text, startRef.current);
  }
  const windowed = isRunning && !restoredRef.current && startRef.current > 0;

  return (
    <div ref={hostRef} className="contents">
      <MarkdownText text={windowed ? text.slice(startRef.current) : undefined} />
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
