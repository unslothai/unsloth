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
import { captureReasoningAnchor } from "./reasoning-reading-anchor";
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
  useAui,
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
import { shallow } from "zustand/shallow";
import { partsHaveRenderableRenderHtmlTool } from "@/components/assistant-ui/message-derived";
import { useMessageMemo } from "@/components/assistant-ui/use-message-memo";
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
  // The grid transition starts only after React commits `0fr`, so the lock needs the backstop
  // margin on this path; tool-group and tool-fallback keep the plain duration.
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
      // Native scroll anchoring can move the focused header when a long transcript opens below it.
      lockScroll();
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
      {/* No overflow clipping: it cuts the descenders off "Thinking" and "Worked". */}
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
  // Colour and size live on ReasoningText: the folded-round path skips this wrapper.
  const shared = cn(
    "aui-reasoning-content relative overflow-hidden outline-none",
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
          // `1fr` tracks content every frame, so streaming into an open pane needs no measured height.
          // Reduced motion still applies: index.css forces transition-duration on every element.
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
        "aui-reasoning-text relative z-0 pt-4 pb-0 text-sm text-muted-foreground leading-relaxed",
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

  const aui = useAui();

  // Read on click: as a selector it joined all the reasoning on every store write.
  const handleCopy = useCallback(async () => {
    const reasoningText = aui
      .message()
      .getState()
      .parts.slice(startIndex, endIndex + 1)
      .filter((p) => p.type === "reasoning")
      .map((p) => ("text" in p ? (p as { text: string }).text : ""))
      .join("\n");
    if (await copyToClipboard(reasoningText)) {
      setCopied(true);
      if (resetRef.current) clearTimeout(resetRef.current);
      resetRef.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  }, [aui, startIndex, endIndex]);

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
    // Captured only on first mount; reopening must not replay the threshold reading position.
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
      setSession({
        messageId,
        active: true,
        anchor:
          content && viewport
            ? captureReasoningAnchor(content, viewport)
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

// With the fold preference on, the first Thinking block heads everything before the answer;
// later reasoning and tool groups render inside it (see tool-group.tsx).
const ReasoningGroupImpl: ReasoningGroupComponent = (props) => {
  const foldToolActivity = useChatPreferencesStore((state) =>
    foldIsActive(state.foldToolActivityIntoThinking, state.toolVisibility),
  );
  const folded = useMessageMemo(
    (message) =>
      foldToolActivity &&
      isFoldedReasoningGroup(message.parts, props.startIndex),
    [foldToolActivity, props.startIndex],
  );
  if (folded) {
    return <FoldedReasoningRound {...props} />;
  }
  return <ReasoningGroupBlock {...props} foldTurn={foldToolActivity} />;
};

// Full --border: at 60% the line was nearly invisible on both themes.
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
  const roundKey = useMessageMemo(
    (message) => {
      const lead = leadReasoningEnd(message.parts, startIndex);
      return lead === null ? null : reasoningRoundKey(message.id, lead);
    },
    [startIndex],
  );
  const open = useReasoningRoundStore(
    (state) => roundKey !== null && (state.open[roundKey] ?? false),
  );
  const isStreaming = useAuiState(({ message }) => {
    if (message.status?.type !== "running") return false;
    const parts = message.parts;
    for (let i = endIndex + 1; i < parts.length; i += 1) {
      if (parts[i]?.type !== "tool-call") return false;
    }
    return true;
  });
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    partsHaveRenderableRenderHtmlTool(message.parts),
  );
  const reasoningContentRef = useRef<HTMLDivElement>(null);
  const reasoningDocuments = useMessageMemo(
    (message) =>
      message.parts
        .slice(startIndex, endIndex + 1)
        .filter((part) => part.type === "reasoning")
        .map((part) => ("text" in part ? (part as { text: string }).text : "")),
    [startIndex, endIndex],
    shallow,
  );
  const transcript = useReasoningTranscriptMode({
    messageId,
    reasoningDocuments,
    reasoningContentRef,
  });
  const closesTrace = useMessageMemo(
    (message) => endsFoldedSpan(message.parts, endIndex),
    [endIndex],
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
  const foldLead = useMessageMemo(
    (message) =>
      foldTurn && leadReasoningEnd(message.parts, endIndex) === endIndex,
    [foldTurn, endIndex],
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
    partsHaveRenderableRenderHtmlTool(message.parts),
  );

  const reasoningContentRef = useRef<HTMLDivElement>(null);

  const reasoningDocuments = useMessageMemo(
    (message) =>
      message.parts
        .slice(startIndex, endIndex + 1)
        .filter((part) => part.type === "reasoning")
        .map((part) => ("text" in part ? (part as { text: string }).text : "")),
    [startIndex, endIndex],
    shallow,
  );
  const transcript = useReasoningTranscriptMode({
    messageId,
    reasoningDocuments,
    reasoningContentRef,
  });

  const persistedDuration = useMessageMemo(
    (message) => {
      const custom = message.metadata?.custom as
        | Record<string, unknown>
        | undefined;
      if (foldLead) {
        return foldedTurnDuration(message.parts, endIndex, (parts, start) =>
          resolveReasoningGroupDuration(parts, start, custom),
        );
      }
      return resolveReasoningGroupDuration(message.parts, startIndex, custom);
    },
    [foldLead, startIndex, endIndex],
  );

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

  // Reset per round: regenerate reuses this instance. Adjusted during render so no stale open paints.
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

  const isOpen = resolveReasoningOpen({
    isStreaming: isReasoningStreaming,
    visibility,
    override,
  });

  // A layout effect, so the folded parts settle before the frame the user sees.
  const roundKey = reasoningRoundKey(messageId, endIndex);
  const toolConfirmations = useChatRuntimeStore((s) => s.toolConfirmations);
  const foldedToolCount = useMessageMemo(
    (message) =>
      foldLead
        ? countFoldedToolParts(message.parts, endIndex, (start, end) =>
            toolRunIsExempt(message.parts, start, end, toolConfirmations),
          )
        : 0,
    [foldLead, endIndex, toolConfirmations],
  );
  // Copy reaches the rounds folded in here too: they have no Copy of their own.
  const copyEndIndex = useMessageMemo(
    (message) => (foldLead ? foldEnd(message.parts, endIndex) - 1 : endIndex),
    [foldLead, endIndex],
  );
  // The answer follows this block directly, with nothing folded in between.
  const closesTrace = useMessageMemo(
    (message) =>
      foldLead
        ? endsFoldedSpan(message.parts, endIndex)
        : message.parts[endIndex + 1]?.type === "text",
    [foldLead, endIndex],
  );
  useLayoutEffect(() => {
    if (!foldLead) return;
    setReasoningRoundOpen(roundKey, isOpen);
  }, [foldLead, isOpen, roundKey]);
  useLayoutEffect(() => () => clearReasoningRound(roundKey), [roundKey]);

  // Detach from bottom on a manual open, or the viewport pins the bottom and shoves the header up.
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
