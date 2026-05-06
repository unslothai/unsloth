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
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { cn } from "@/lib/utils";
import {
  type ReasoningGroupComponent,
  type ReasoningMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Idea01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type VariantProps, cva } from "class-variance-authority";
import { ChevronDownIcon, CopyIcon, CheckIcon } from "lucide-react";
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

export const reasoningVariants = cva("aui-reasoning-root mb-4 w-full", {
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
        "aui-reasoning-trigger group/trigger flex min-w-0 flex-1 items-center gap-2 py-1 text-muted-foreground text-sm transition-colors hover:text-foreground",
        className,
      )}
      {...props}
    >
      <HugeiconsIcon
        icon={Idea01Icon}
        className="aui-reasoning-trigger-icon size-4 shrink-0"
      />
      <span
        data-slot="reasoning-trigger-label"
        className="aui-reasoning-trigger-label-wrapper relative inline-block leading-none"
      >
        {active ? (
          <span className="text-sm">Thinking...</span>
        ) : (
          <span>Thought for {duration ?? 0} seconds</span>
        )}
      </span>
      <ChevronDownIcon
        data-slot="reasoning-trigger-chevron"
        className={cn(
          "aui-reasoning-trigger-chevron mt-0.5 size-4 shrink-0",
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
        "aui-reasoning-content relative overflow-hidden text-foreground/85 text-[13.5px] outline-none",
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
        "aui-reasoning-text relative z-0 overflow-y-auto pt-2 pb-2 pl-0 leading-relaxed",
        streaming ? "max-h-32" : "max-h-64",
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

const ReasoningImpl: ReasoningMessagePartComponent = () => <MarkdownText />;

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
        <CheckIcon className="size-3" />
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
    const d = (message.metadata?.custom as Record<string, unknown>)
      ?.reasoningDuration;
    return typeof d === "number" ? d : 0;
  });

  const [manualOpen, setManualOpen] = useState(false);
  const [dismissedWhileStreaming, setDismissedWhileStreaming] = useState(false);
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

  // Reset dismissed flag when a new stream starts
  useEffect(() => {
    if (isReasoningStreaming) {
      setDismissedWhileStreaming(false);
    }
  }, [isReasoningStreaming]);

  // Derived: open during streaming (unless dismissed), or if user manually opened after
  const isOpen = (isReasoningStreaming && !dismissedWhileStreaming) || manualOpen;
  const variant = isOpen ? "outline" : "ghost";

  // Allow closing during streaming (matches ChatGPT)
  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (isReasoningStreaming) {
        setDismissedWhileStreaming(!open);
      } else {
        setManualOpen(open);
      }
    },
    [isReasoningStreaming],
  );

  return (
    <ReasoningRoot
      open={isOpen}
      onOpenChange={handleOpenChange}
      variant={variant}
    >
      <div className="flex items-center gap-2">
        <ReasoningTrigger
          className="min-w-0 flex-1"
          active={isReasoningStreaming}
          duration={duration || persistedDuration}
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
        <ReasoningText streaming={isReasoningStreaming}>
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
