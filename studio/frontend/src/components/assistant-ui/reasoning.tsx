// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/* eslint-disable react-refresh/only-export-components */

import {
  MarkdownText,
  SearchImagesEnabledContext,
} from "@/components/assistant-ui/markdown-text";
import {
  REASONING_TRANSCRIPT_THRESHOLD,
  type ReasoningReadingAnchor,
} from "./reasoning-transcript-index";
import { ReasoningTranscript } from "./reasoning-transcript";
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
  clearReasoningRound,
  foldIsActive,
  isRenderableRenderHtmlToolPart,
  resolveReasoningGroupDuration,
  resolveReasoningOpen,
  setReasoningRoundOpen,
  startsNewReasoningRound,
  useChatPreferencesStore,
  useChatRuntimeStore,
  useReasoningRoundStore,
} from "@/features/chat";
import {
  countFoldedToolParts,
  endsFoldedSpan,
  foldEnd,
  foldedToolSummary,
  foldedTurnDuration,
  isBlankTextPart,
  isFoldedReasoningGroup,
  leadReasoningEnd,
  reasoningRoundKey,
} from "@/components/assistant-ui/thinking-fold";
import { toolRunIsExempt } from "@/components/assistant-ui/tool-fold-exemptions";
import { useDetachThreadFromBottom } from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { formatWorkedFor } from "@/lib/format-worked-for";
import { cn } from "@/lib/utils";
import {
  type ReasoningGroupComponent,
  type ReasoningMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { type VariantProps, cva } from "class-variance-authority";
import { ChevronDownIcon } from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { IconActionButton } from "./icon-action-button";
import {
  type CSSProperties,
  type ComponentProps,
  type ReactNode,
  type RefObject,
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
const ANIMATION_DURATION = 200;

function selectionIntersectsElement(
  selection: Selection | null,
  element: Element | null,
): boolean {
  if (!selection || selection.isCollapsed || !element) {
    return false;
  }
  for (let index = 0; index < selection.rangeCount; index += 1) {
    if (selection.getRangeAt(index).intersectsNode(element)) {
      return true;
    }
  }
  return false;
}

// Plain text in the message column: a header line and, when open, the thoughts
// under it. No box, no icon. Outer spacing comes from the message body's rhythm.
export const reasoningVariants = cva("aui-reasoning-root w-full", {
  variants: {
    variant: {
      outline: "rounded-lg border px-3 py-2",
      ghost: "",
      muted: "rounded-lg bg-muted/50 px-3 py-2",
    },
  },
  defaultVariants: {
    variant: "ghost",
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
    className: cn(
      "group/reasoning-root",
      reasoningVariants({ variant, className }),
    ),
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
  foldedToolCount = 0,
  className,
  ...props
}: ComponentProps<typeof CollapsibleTrigger> & {
  active?: boolean;
  duration?: number;
  /** Tool calls hidden under this block, named so a closed block is not silent about them. */
  foldedToolCount?: number;
}) {
  const foldedSummary = foldedToolSummary(foldedToolCount);
  const Trigger = GRID_COLLAPSE_REASONING_ENABLED
    ? UnmeasuredCollapsibleTrigger
    : CollapsibleTrigger;

  return (
    <Trigger
      data-slot="reasoning-trigger"
      data-active={active ? "" : undefined}
      className={cn(
        "aui-reasoning-trigger group/trigger flex min-h-5 min-w-0 cursor-pointer items-center gap-2 text-muted-foreground text-sm transition-colors hover:text-foreground",
        className,
      )}
      {...props}
    >
      {/* No overflow clipping: with leading-none the line box is the font size, and hidden
          overflow cuts the descenders off "Thinking" and "Worked". */}
      <span
        data-slot="reasoning-trigger-label"
        className="aui-reasoning-trigger-label-wrapper relative inline-block whitespace-nowrap leading-none"
      >
        {active ? (
          <span>Thinking</span>
        ) : (
          <span>Worked for {formatWorkedFor(duration ?? 0)}</span>
        )}
        {active && (
          <span
            aria-hidden={true}
            data-slot="reasoning-trigger-shimmer"
            className="aui-reasoning-trigger-shimmer shimmer pointer-events-none absolute inset-0 motion-reduce:animate-none"
          >
            Thinking
          </span>
        )}
      </span>
      {/* Outside the label so it reads while the block is still working and closed. */}
      {foldedSummary ? (
        <span className="whitespace-nowrap leading-none text-muted-foreground/70">
          {"\u00b7 "}
          {foldedSummary}
        </span>
      ) : null}
      <ChevronDownIcon
        data-slot="reasoning-trigger-chevron"
        className={cn(
          "aui-reasoning-trigger-chevron size-3.5 shrink-0",
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
    "aui-reasoning-content relative overflow-hidden text-[#0d0d0d] dark:text-foreground outline-none",
    "group/collapsible-content ease-out",
    "data-[state=closed]:pointer-events-none",
  );

  if (GRID_COLLAPSE_REASONING_ENABLED) {
    return (
      <UnmeasuredCollapsibleContent
        data-slot="reasoning-content"
        data-streaming={streaming ? "" : undefined}
        closeDurationMs={ANIMATION_DURATION}
        className={cn(
          shared,
          // No `animate-collapsible-*`, so nothing consumes `--radix-collapsible-content-height`
          // and nothing needs to know the content's height. `1fr` resolves against the content on
          // every frame, which is also what makes this correct while reasoning is still streaming
          // into an open pane: the row simply tracks the growing content instead of holding a
          // height captured at toggle time. The duration is unconditional here. The height
          // keyframes needed a per-state duration because they were two different animations; this
          // is one transition run in both directions. `prefers-reduced-motion` still reaches it:
          // index.css forces `transition-duration: 0.01ms !important` on every element, and this is
          // a transition.
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
      data-streaming={streaming ? "" : undefined}
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
  virtualized = false,
  children,
  ...props
}: ComponentProps<"div"> & { streaming?: boolean; virtualized?: boolean }) {
  return (
    <div
      data-slot="reasoning-text"
      data-streaming={streaming ? "" : undefined}
      className={cn(
        // Reads like the answer: same size and colour, flush with the header, no cap and no
        // fade. The thread's own follow-scroll tracks it while it streams.
        "aui-reasoning-text relative z-0 pt-4 pb-0 leading-relaxed",
        "[&_p]:my-0 [&_p+p]:mt-4 [&_ul]:my-4 [&_ol]:my-4 [&_pre]:my-4",
        !virtualized &&
          cn(
            "transform-gpu transition-[transform,opacity]",
            "group-data-[state=open]/collapsible-content:animate-in",
            "group-data-[state=closed]/collapsible-content:animate-out",
            "group-data-[state=open]/collapsible-content:fade-in-0",
            "group-data-[state=closed]/collapsible-content:fade-out-0",
            "group-data-[state=open]/collapsible-content:slide-in-from-top-4",
            "group-data-[state=closed]/collapsible-content:slide-out-to-top-4",
            "group-data-[state=open]/collapsible-content:duration-(--animation-duration)",
            "group-data-[state=closed]/collapsible-content:duration-(--animation-duration)",
          ),
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
    <MarkdownText />
  </SearchImagesEnabledContext.Provider>
);

const COPY_RESET_MS = 2000;

function ReasoningCopyButton({
  startIndex,
  endIndex,
}: { startIndex: number; endIndex: number }) {
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
    <IconActionButton
      label={copied ? "Copied" : "Copy reasoning"}
      onClick={handleCopy}
    >
      {copied ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3" />
      ) : (
        <HugeiconsIcon icon={Copy01Icon} className="size-3" />
      )}
    </IconActionButton>
  );
}

// Keep an existing selection intact when a live trace crosses the windowing threshold.
// An already-long saved trace is bounded on its first render.
function useReasoningTranscriptMode({
  messageId,
  reasoningDocuments,
  reasoningContentRef,
}: {
  messageId: string;
  reasoningDocuments: readonly string[];
  reasoningContentRef: RefObject<HTMLDivElement | null>;
}) {
  const wantsWindow =
    reasoningDocuments.reduce((sum, text) => sum + text.length, 0) >
    REASONING_TRANSCRIPT_THRESHOLD;
  const [session, setSession] = useState<{
    messageId: string;
    active: boolean;
    anchor?: ReasoningReadingAnchor;
  }>({ messageId, active: wantsWindow });
  useEffect(() => {
    if (!session.anchor) return;
    // The transcript captures this only on its first mount. Reopening the block
    // later must not replay a reading position from the threshold transition.
    const timer = setTimeout(
      () => setSession((current) => ({ ...current, anchor: undefined })),
      0,
    );
    return () => clearTimeout(timer);
  }, [session.anchor]);
  if (session.messageId !== messageId || (!wantsWindow && session.active))
    setSession({ messageId, active: wantsWindow });
  useEffect(() => {
    if (!wantsWindow) return;
    if (session.active) return;
    const activate = () => {
      const content = reasoningContentRef.current;
      const viewport = content?.closest(".aui-thread-viewport");
      const top = viewport?.getBoundingClientRect().top ?? 0;
      const passage =
        content &&
        [
          ...content.querySelectorAll<HTMLElement>(
            "p, pre, li, h1, h2, h3, h4, h5, h6",
          ),
        ].find(
          (node) =>
            node.textContent &&
            node.getBoundingClientRect().top >= top &&
            node.getBoundingClientRect().top <
              top + (viewport?.clientHeight ?? 0),
        );
      setSession({
        messageId,
        active: true,
        anchor: passage
          ? {
              text: passage.textContent!.slice(0, 200),
              top: passage.getBoundingClientRect().top,
            }
          : undefined,
      });
    };
    if (
      !selectionIntersectsElement(
        window.getSelection(),
        reasoningContentRef.current,
      )
    ) {
      const timer = setTimeout(activate, 0);
      return () => clearTimeout(timer);
    }
    const handleSelectionChange = () => {
      if (
        !selectionIntersectsElement(
          window.getSelection(),
          reasoningContentRef.current,
        )
      )
        activate();
    };
    document.addEventListener("selectionchange", handleSelectionChange);
    return () =>
      document.removeEventListener("selectionchange", handleSelectionChange);
  }, [messageId, reasoningContentRef, session.active, wantsWindow]);
  return {
    active:
      wantsWindow && (session.messageId === messageId ? session.active : true),
    documents: reasoningDocuments,
    anchor: session.anchor,
  };
}

function ReasoningBody({
  transcript,
  messageId,
  messageHasRenderableRenderHtmlTool,
  isStreaming,
  textClassName,
  children,
}: {
  transcript: ReturnType<typeof useReasoningTranscriptMode>;
  messageId: string;
  messageHasRenderableRenderHtmlTool: boolean;
  isStreaming: boolean;
  textClassName?: string;
  children: ReactNode;
}) {
  return (
    <ReasoningText
      className={textClassName}
      streaming={isStreaming}
      virtualized={transcript.active}
    >
      {transcript.active ? (
        <ReasoningTranscript
          key={messageId}
          initialAnchor={transcript.anchor}
          documents={transcript.documents}
          messageId={messageId}
          messageHasRenderableRenderHtmlTool={
            messageHasRenderableRenderHtmlTool
          }
          streaming={isStreaming}
        />
      ) : (
        children
      )}
    </ReasoningText>
  );
}

// With the fold preference on, the first Thinking block of a turn heads everything before the
// answer. Later reasoning groups in that span render as plain rounds inside it, following its
// open state, and the tool groups between them do the same (see tool-group.tsx).
const ReasoningGroupImpl: ReasoningGroupComponent = (props) => {
  const foldToolActivity = useChatPreferencesStore((state) =>
    foldIsActive(state.foldToolActivityIntoThinking, state.toolVisibility),
  );
  const folded = useAuiState(
    ({ message }) =>
      foldToolActivity &&
      isFoldedReasoningGroup(message.parts, props.startIndex),
  );
  if (folded) {
    return <FoldedReasoningRound {...props} />;
  }
  return <ReasoningGroupBlock {...props} foldTurn={foldToolActivity} />;
};

// A thin line closing the trace when the answer comes right after it, so the two do not read
// as one text. Rendered inside whatever is last under the header, so it hides with it.
// Full --border, not 60% of it: at 60% the line was 14/255 off the dark background and
// 19/255 off the light one, which is under the rule rather than a quiet version of it.
function ReasoningEndRule() {
  return (
    <div
      data-slot="reasoning-end-rule"
      aria-hidden={true}
      className="mt-4 border-border border-t"
    />
  );
}

const FoldedReasoningRound: ReasoningGroupComponent = ({
  children,
  startIndex,
  endIndex,
}) => {
  const messageId = useAuiState(({ message }) => message.id);
  const roundKey = useAuiState(({ message }) => {
    const lead = leadReasoningEnd(message.parts, startIndex);
    return lead === null ? null : reasoningRoundKey(message.id, lead);
  });
  const open = useReasoningRoundStore(
    (state) => roundKey !== null && (state.open[roundKey] ?? false),
  );
  // Streaming while this round is the last thinking so far and only calls follow it.
  const isStreaming = useAuiState(({ message }) => {
    if (message.status?.type !== "running") return false;
    const parts = message.parts;
    for (let i = endIndex + 1; i < parts.length; i += 1) {
      if (parts[i]?.type !== "tool-call") return false;
    }
    return true;
  });
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  const reasoningContentRef = useRef<HTMLDivElement>(null);
  const reasoningDocuments = useAuiState(
    useShallow(({ message }) =>
      message.parts
        .slice(startIndex, endIndex + 1)
        .filter((part) => part.type === "reasoning")
        .map((part) => ("text" in part ? (part as { text: string }).text : "")),
    ),
  );
  const transcript = useReasoningTranscriptMode({
    messageId,
    reasoningDocuments,
    reasoningContentRef,
  });
  const closesTrace = useAuiState(({ message }) =>
    endsFoldedSpan(message.parts, endIndex),
  );
  // Hidden, not unmounted, so the text keeps streaming in while the lead is closed.
  return (
    <div
      ref={reasoningContentRef}
      data-slot="reasoning-folded-round"
      className={cn(!open && "hidden")}
    >
      <ReasoningBody
        transcript={transcript}
        messageId={messageId}
        messageHasRenderableRenderHtmlTool={messageHasRenderableRenderHtmlTool}
        isStreaming={isStreaming}
        textClassName="pt-0"
      >
        {children}
      </ReasoningBody>
      {closesTrace && <ReasoningEndRule />}
    </div>
  );
};

const ReasoningGroupBlock = ({
  children,
  startIndex,
  endIndex,
  foldTurn,
}: ComponentProps<ReasoningGroupComponent> & { foldTurn: boolean }) => {
  // The lead of a folded span: its first Thinking block.
  const foldLead = useAuiState(
    ({ message }) =>
      foldTurn && leadReasoningEnd(message.parts, endIndex) === endIndex,
  );
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
    // The lead of a folded span keeps working through later thinking and blank text too,
    // until the answer.
    for (let i = endIndex + 1; i < len; i += 1) {
      const part = parts[i];
      if (part?.type === "tool-call") continue;
      if (foldLead && (part?.type === "reasoning" || isBlankTextPart(part))) {
        continue;
      }
      return false;
    }
    return true;
  });

  const messageId = useAuiState(({ message }) => message.id);

  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );

  const reasoningContentRef = useRef<HTMLDivElement>(null);

  const reasoningDocuments = useAuiState(
    useShallow(({ message }) =>
      message.parts
        .slice(startIndex, endIndex + 1)
        .filter((part) => part.type === "reasoning")
        .map((part) => ("text" in part ? (part as { text: string }).text : "")),
    ),
  );
  const transcript = useReasoningTranscriptMode({
    messageId,
    reasoningDocuments,
    reasoningContentRef,
  });

  const persistedDuration = useAuiState(({ message }) => {
    const custom = message.metadata?.custom as
      | Record<string, unknown>
      | undefined;
    if (foldLead) {
      return foldedTurnDuration(message.parts, endIndex, (parts, start) =>
        resolveReasoningGroupDuration(parts, start, custom),
      );
    }
    return resolveReasoningGroupDuration(message.parts, startIndex, custom);
  });

  const visibility = useChatPreferencesStore(
    (state) => state.thinkingVisibility,
  );

  // null until toggled by hand, then it outranks the setting for the round.
  const [override, setOverride] = useState<boolean | null>(null);
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

  // Reset per-round open state. Regenerate reuses this instance, so a hand-opened block would
  // stay pinned open. Adjusted during render, not in an effect, so no stale open reaches the DOM.
  const [wasStreaming, setWasStreaming] = useState(isReasoningStreaming);
  if (wasStreaming !== isReasoningStreaming) {
    setWasStreaming(isReasoningStreaming);
    if (startsNewReasoningRound(isReasoningStreaming, wasStreaming)) {
      setOverride(null);
    }
  }

  // Changing the setting hands every block back to it, including those already on screen.
  const [lastVisibility, setLastVisibility] = useState(visibility);
  if (lastVisibility !== visibility) {
    setLastVisibility(visibility);
    setOverride(null);
  }

  // Whatever the setting says, until this block is toggled by hand.
  const isOpen = resolveReasoningOpen({
    isStreaming: isReasoningStreaming,
    visibility,
    override,
  });

  // Publish the lead's open state for everything folded under it, and count the tool calls for
  // its header. A layout effect, so the folded parts settle before the frame the user sees.
  const roundKey = reasoningRoundKey(messageId, endIndex);
  const toolConfirmations = useChatRuntimeStore((s) => s.toolConfirmations);
  const foldedToolCount = useAuiState(({ message }) =>
    foldLead
      ? countFoldedToolParts(message.parts, endIndex, (start, end) =>
          toolRunIsExempt(message.parts, start, end, toolConfirmations),
        )
      : 0,
  );
  // Copy reaches the rounds folded in here too: they have no Copy of their own.
  const copyEndIndex = useAuiState(({ message }) =>
    foldLead ? foldEnd(message.parts, endIndex) - 1 : endIndex,
  );
  // The answer follows this block directly, with nothing folded in between.
  const closesTrace = useAuiState(({ message }) =>
    foldLead
      ? endsFoldedSpan(message.parts, endIndex)
      : message.parts[endIndex + 1]?.type === "text",
  );
  useLayoutEffect(() => {
    if (!foldLead) return;
    setReasoningRoundOpen(roundKey, isOpen);
  }, [foldLead, isOpen, roundKey]);
  useLayoutEffect(() => () => clearReasoningRound(roundKey), [roundKey]);

  // Opening by hand grows the block downward and leaves the header where it is. The viewport
  // would otherwise treat the growth as new content and pin the bottom, which shoves the header
  // up. Streaming keeps following: the auto-open is not a click and the stream should track.
  const detachFromBottom = useDetachThreadFromBottom();
  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        detachFromBottom();
      }
      setOverride(open);
    },
    [detachFromBottom],
  );

  return (
    <ReasoningRoot open={isOpen} onOpenChange={handleOpenChange}>
      <div
        data-slot="reasoning-header"
        className="flex min-w-0 items-center gap-2"
      >
        <ReasoningTrigger
          className="min-w-0"
          active={isReasoningStreaming}
          // Prefer server timing when available.
          duration={persistedDuration ?? duration}
          foldedToolCount={isOpen ? 0 : foldedToolCount}
        />
        {isOpen && !isReasoningStreaming && (
          <span className="ml-auto flex items-center leading-none">
            <ReasoningCopyButton
              startIndex={startIndex}
              endIndex={copyEndIndex}
            />
          </span>
        )}
      </div>
      <ReasoningContent
        aria-busy={isReasoningStreaming}
        streaming={isReasoningStreaming}
        ref={reasoningContentRef}
      >
        <ReasoningBody
          transcript={transcript}
          messageId={messageId}
          messageHasRenderableRenderHtmlTool={
            messageHasRenderableRenderHtmlTool
          }
          isStreaming={isReasoningStreaming}
        >
          {children}
        </ReasoningBody>
        {closesTrace && <ReasoningEndRule />}
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
