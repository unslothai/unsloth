// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/* eslint-disable react-refresh/only-export-components */

import {
  MarkdownCodeHighlightingContext,
  MarkdownText,
  SearchImagesEnabledContext,
} from "@/components/assistant-ui/markdown-text";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { GRID_COLLAPSE_REASONING_ENABLED } from "@/components/assistant-ui/thread-feature-flags";
import {
  CLOSE_FALLBACK_MARGIN_MS,
  UnmeasuredCollapsible,
  UnmeasuredCollapsibleContent,
  UnmeasuredCollapsibleTrigger,
} from "@/components/ui/unmeasured-collapsible";
import {
  resolveReasoningGroupDuration,
  resolveReasoningOpen,
  resolveReasoningToggle,
  startsNewReasoningRound,
  useChatPreferencesStore,
} from "@/features/chat";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { cn } from "@/lib/utils";
import {
  type ReasoningGroupComponent,
  type ReasoningMessagePartComponent,
  useAuiState,
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
  // The lock starts in the click handler; the grid transition only starts once React has
  // committed the `0fr` class, so an exact ANIMATION_DURATION releases the scroll container
  // while the row is still shrinking and lets the remaining height change shift the thread.
  // Same margin as the collapse backstop, and only on the grid path: `tool-group` and
  // `tool-fallback` still animate height and keep the plain duration.
  const lockScroll = useCollapseScrollLock(
    collapsibleRef,
    GRID_COLLAPSE_REASONING_ENABLED
      ? ANIMATION_DURATION + CLOSE_FALLBACK_MARGIN_MS
      : ANIMATION_DURATION,
  );

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

  const rootProps = {
    ref: collapsibleRef,
    "data-slot": "reasoning-root",
    "data-variant": variant,
    open: isOpen,
    onOpenChange: handleOpenChange,
    className: cn("group/reasoning-root", reasoningVariants({ variant, className })),
    style: {
      "--animation-duration": `${ANIMATION_DURATION}ms`,
    } as CSSProperties,
    ...props,
  };

  // Same props either way. The only difference is which primitive receives them, and the
  // unmeasured one is a drop-in for the subset of Radix's surface this pane uses.
  return GRID_COLLAPSE_REASONING_ENABLED ? (
    <UnmeasuredCollapsible {...rootProps}>{children}</UnmeasuredCollapsible>
  ) : (
    <Collapsible {...rootProps}>{children}</Collapsible>
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
  const Trigger = GRID_COLLAPSE_REASONING_ENABLED
    ? UnmeasuredCollapsibleTrigger
    : CollapsibleTrigger;

  return (
    <Trigger
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
    </Trigger>
  );
}

function ReasoningContent({
  className,
  children,
  streaming,
  ...props
}: ComponentProps<typeof CollapsibleContent> & { streaming?: boolean }) {
  const shared = cn(
    "aui-reasoning-content relative overflow-hidden text-foreground/85 text-ui-13p5 outline-none",
    "group/collapsible-content ease-out",
    "data-[state=closed]:pointer-events-none",
  );

  if (GRID_COLLAPSE_REASONING_ENABLED) {
    return (
      <UnmeasuredCollapsibleContent
        data-slot="reasoning-content"
        closeDurationMs={ANIMATION_DURATION}
        className={cn(
          shared,
          // No `animate-collapsible-*`, so nothing consumes
          // `--radix-collapsible-content-height` and nothing needs to know the content's height.
          // `1fr` resolves against the content on every frame, which is also what makes this
          // correct while reasoning is still streaming into an open pane: the row simply tracks
          // the growing content instead of holding a height captured at toggle time.
          //
          // The duration is unconditional here. The height keyframes needed a per-state duration
          // because they were two different animations; this is one transition run in both
          // directions. `prefers-reduced-motion` still reaches it: index.css forces
          // `transition-duration: 0.01ms !important` on every element, and this is a transition.
          "duration-(--animation-duration)",
          className,
        )}
        {...props}
      >
        {children}
      </UnmeasuredCollapsibleContent>
    );
  }

  return (
    <CollapsibleContent
      data-slot="reasoning-content"
      className={cn(
        shared,
        "data-[state=closed]:animate-collapsible-up",
        "data-[state=open]:animate-collapsible-down",
        "data-[state=closed]:fill-mode-forwards",
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

const ReasoningImpl: ReasoningMessagePartComponent = () => (
  <SearchImagesEnabledContext.Provider value={false}>
    {/*
     * Thinking traces are skimmed, not read, and they are mostly code. Highlighting them is what
     * makes opening the panes on a long thread slow: 5.4 effective fps at 92% busy against 25.5
     * fps at 61% busy, measured on a 100K-character thread in WebKitGTK on a gfx1151. The code is
     * still all there and still copyable; it just has no colours.
     */}
    <MarkdownCodeHighlightingContext.Provider value="plain">
      <MarkdownText />
    </MarkdownCodeHighlightingContext.Provider>
  </SearchImagesEnabledContext.Provider>
);

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
  //
  // The grid path needs the same margin the collapsible's own backstop uses. The
  // height keyframes animate from a height captured at toggle time, so releasing
  // the cap mid-animation cannot change what they animate; `1fr` instead resolves
  // against the live content every frame, so an early release grows the row in the
  // middle of the collapse and produces exactly the jump this timer prevents. The
  // transition also starts a render after this timer is armed, so an exact
  // ANIMATION_DURATION lands inside it.
  useEffect(() => {
    const closeDelay = GRID_COLLAPSE_REASONING_ENABLED
      ? ANIMATION_DURATION + CLOSE_FALLBACK_MARGIN_MS
      : ANIMATION_DURATION;
    const timeout = window.setTimeout(
      () => setRetainStreamingHeight(isReasoningStreaming),
      isReasoningStreaming ? 0 : closeDelay,
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
