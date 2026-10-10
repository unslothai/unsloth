"use client";

import {
  memo,
  useCallback,
  useRef,
  useState,
  type FC,
  type PropsWithChildren,
} from "react";
import { useAuiState } from "@assistant-ui/react";
import { useMessageMemo } from "@/components/assistant-ui/use-message-memo";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
// eslint-disable-next-line no-restricted-imports -- this file is in the startup cycle; the chat barrel closes it.
import { useReasoningRoundStore } from "@/features/chat/stores/reasoning-round-store";
import {
  endsFoldedSpan,
  governingReasoningEnd,
  reasoningRoundKey,
} from "./thinking-fold";
import {
  toolOutputKey,
  useToolPaneScope,
  useUnresolvedToolPaneScope,
} from "@/features/chat/tool-output-scope";
import { ChevronDownIcon } from "lucide-react";
import { Wrench01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cva, type VariantProps } from "class-variance-authority";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useDetachThreadFromBottom } from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { cn } from "@/lib/utils";
import { Spinner } from "@/components/ui/spinner";
import { awaitsConfirmation, holdsOwnOutput } from "./tool-fold-exemptions";
import {
  syncToolActivityPreference,
  toolActivityOpen,
} from "./tool-activity-open-state";
// eslint-disable-next-line no-restricted-imports -- this file is in the startup cycle; the chat barrel closes it.
import { foldIsActive } from "@/features/chat/utils/display-visibility";

const ANIMATION_DURATION = 200;

const toolGroupVariants = cva("aui-tool-group-root group/tool-group w-full", {
  variants: {
    variant: {
      outline: "corner-squircle rounded-lg border py-3",
      ghost: "",
      muted:
        "corner-squircle rounded-lg border border-muted-foreground/30 bg-muted/30 py-3",
    },
  },
  defaultVariants: { variant: "ghost" },
});

export type ToolGroupRootProps = Omit<
  React.ComponentProps<typeof Collapsible>,
  "open" | "onOpenChange"
> &
  VariantProps<typeof toolGroupVariants> & {
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
    defaultOpen?: boolean;
  };

function ToolGroupRoot({
  className,
  variant,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  defaultOpen = false,
  children,
  ...props
}: ToolGroupRootProps) {
  const collapsibleRef = useRef<HTMLDivElement>(null);
  // Without this sync, a hand-expanded group stays open while every card inside it closes.
  const visibility = useChatPreferencesStore((state) => state.toolVisibility);
  const [uncontrolledState, setUncontrolledState] = useState(() => ({
    visibility,
    active: defaultOpen,
    override: null as boolean | null,
  }));
  const syncedUncontrolledState = syncToolActivityPreference(
    uncontrolledState,
    visibility,
    defaultOpen,
  );
  if (syncedUncontrolledState !== uncontrolledState) {
    setUncontrolledState(syncedUncontrolledState);
  }
  const lockScroll = useCollapseScrollLock(collapsibleRef, ANIMATION_DURATION);

  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled
    ? controlledOpen
    : toolActivityOpen(syncedUncontrolledState);

  // Opening by hand grows the group downward; see the same note in reasoning.tsx.
  const detachFromBottom = useDetachThreadFromBottom();
  const messageRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) {
        lockScroll();
      } else if (!messageRunning) {
        detachFromBottom();
      }
      if (!isControlled) {
        setUncontrolledState({ ...syncedUncontrolledState, override: open });
      }
      controlledOnOpenChange?.(open);
    },
    [
      syncedUncontrolledState,
      lockScroll,
      isControlled,
      controlledOnOpenChange,
      detachFromBottom,
      messageRunning,
    ],
  );

  return (
    <Collapsible
      ref={collapsibleRef}
      data-slot="tool-group-root"
      data-variant={variant ?? "ghost"}
      open={isOpen}
      onOpenChange={handleOpenChange}
      className={cn(
        toolGroupVariants({ variant }),
        "group/tool-group-root",
        className,
      )}
      style={
        {
          "--animation-duration": `${ANIMATION_DURATION}ms`,
        } as React.CSSProperties
      }
      {...props}
    >
      {children}
    </Collapsible>
  );
}

function ToolGroupTrigger({
  count,
  active = false,
  className,
  ...props
}: React.ComponentProps<typeof CollapsibleTrigger> & {
  count: number;
  active?: boolean;
}) {
  const label = `${count} tool ${count === 1 ? "call" : "calls"}`;

  return (
    <CollapsibleTrigger
      data-slot="tool-group-trigger"
      className={cn(
        "aui-tool-group-trigger group/trigger flex w-full cursor-pointer items-center gap-2 text-muted-foreground text-sm transition-colors hover:text-foreground",
        "group-data-[variant=outline]/tool-group-root:px-4",
        "group-data-[variant=muted]/tool-group-root:px-4",
        "group-data-[variant=ghost]/tool-group-root:px-0",
        className,
      )}
      {...props}
    >
      {active ? (
        <Spinner className="aui-tool-group-trigger-loader" />
      ) : (
        <HugeiconsIcon
          icon={Wrench01Icon}
          data-slot="tool-group-trigger-wrench"
          className="size-4 shrink-0 text-foreground"
          strokeWidth={2}
        />
      )}
      <span
        data-slot="tool-group-trigger-label"
        className={cn(
          "aui-tool-group-trigger-label-wrapper relative inline-block text-left font-medium leading-none",
        )}
      >
        <span>{label}</span>
        {active && (
          <span
            aria-hidden
            data-slot="tool-group-trigger-shimmer"
            className="aui-tool-group-trigger-shimmer shimmer pointer-events-none absolute inset-0 motion-reduce:animate-none"
          >
            {label}
          </span>
        )}
      </span>
      <ChevronDownIcon
        data-slot="tool-group-trigger-chevron"
        className={cn(
          "aui-tool-group-trigger-chevron size-3.5 shrink-0",
          "transition-transform duration-(--animation-duration) ease-out",
          "group-data-[state=closed]/trigger:-rotate-90",
          "group-data-[state=open]/trigger:rotate-0",
        )}
      />
    </CollapsibleTrigger>
  );
}

function ToolGroupContent({
  className,
  children,
  ...props
}: React.ComponentProps<typeof CollapsibleContent>) {
  return (
    <CollapsibleContent
      data-slot="tool-group-content"
      className={cn(
        "aui-tool-group-content relative overflow-hidden text-sm outline-none",
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
      <div
        className={cn(
          "mt-2 flex flex-col gap-2",
          "group-data-[variant=outline]/tool-group-root:mt-3 group-data-[variant=outline]/tool-group-root:border-t group-data-[variant=outline]/tool-group-root:px-4 group-data-[variant=outline]/tool-group-root:pt-3",
          "group-data-[variant=muted]/tool-group-root:mt-3 group-data-[variant=muted]/tool-group-root:border-t group-data-[variant=muted]/tool-group-root:px-4 group-data-[variant=muted]/tool-group-root:pt-3",
          "group-data-[variant=ghost]/tool-group-root:mt-3 group-data-[variant=ghost]/tool-group-root:gap-3",
        )}
      >
        {children}
      </div>
    </CollapsibleContent>
  );
}

type ToolGroupComponent = FC<
  PropsWithChildren<{ startIndex: number; endIndex: number }>
> & {
  Root: typeof ToolGroupRoot;
  Trigger: typeof ToolGroupTrigger;
  Content: typeof ToolGroupContent;
};

const ToolGroupImpl: FC<
  PropsWithChildren<{ startIndex: number; endIndex: number }>
> = ({ children, startIndex, endIndex }) => {
  const toolCount = endIndex - startIndex + 1;
  const containsUngroupedTool = useMessageMemo(
    (message) =>
      message.parts.slice(startIndex, endIndex + 1).some(holdsOwnOutput),
    [startIndex, endIndex],
  );
  // Force the group open while any call awaits confirmation, so the prompt is never hidden.
  const toolConfirmations = useChatRuntimeStore((s) => s.toolConfirmations);
  const hasPendingConfirmation = useMessageMemo(
    (message) =>
      message.parts
        .slice(startIndex, endIndex + 1)
        .some((part) => awaitsConfirmation(part, toolConfirmations)),
    [startIndex, endIndex, toolConfirmations],
  );
  const messageRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  // Still working: a part inherits the message status until it has a result, so the group goes
  // quiet once every call in it has one, without waiting for the rest of the turn.
  const groupRunning = useMessageMemo(
    (message) =>
      message.status?.type === "running" &&
      message.parts
        .slice(startIndex, endIndex + 1)
        .some(
          (part) =>
            part.type === "tool-call" &&
            (part as { result?: unknown }).result === undefined,
        ),
    [startIndex, endIndex],
  );
  // Only collapsed suppresses the forced opens below. Auto and expanded want it open anyway.
  const collapseByDefault = useChatPreferencesStore(
    (state) => state.toolVisibility === "collapsed",
  );
  const toolLiveOutput = useChatRuntimeStore((s) => s.toolLiveOutput);
  const paneScope = useToolPaneScope();
  const unresolvedScope = useUnresolvedToolPaneScope();
  const hasLiveOutput = useMessageMemo(
    (message) =>
      message.parts
        .slice(startIndex, endIndex + 1)
        .some(
          (part) =>
            part.type === "tool-call" &&
            // Either scope: a first turn writes under the unresolved one for its whole
            // life, even after the autosave assigns the id (see useToolOutputFor).
            (Object.prototype.hasOwnProperty.call(
              toolLiveOutput,
              toolOutputKey(paneScope, part.toolCallId),
            ) ||
              Object.prototype.hasOwnProperty.call(
                toolLiveOutput,
                toolOutputKey(unresolvedScope, part.toolCallId),
              )),
        ),
    [startIndex, endIndex, toolLiveOutput, paneScope, unresolvedScope],
  );
  // Keep the group open once forced so allow/deny does not snap it shut between calls. Only latch
  // what could have forced it, or turning collapsed off would snap them all open.
  const forcedOpenRef = useRef(false);
  if (hasPendingConfirmation || (hasLiveOutput && !collapseByDefault)) {
    forcedOpenRef.current = true;
  }
  const forceOpen =
    hasPendingConfirmation ||
    (!collapseByDefault &&
      ((hasLiveOutput && messageRunning) ||
        (forcedOpenRef.current && messageRunning)));

  // With the fold preference on, this run shows only while the turn's first thinking block is open;
  // own-output and awaiting calls are never folded.
  const foldToolActivity = useChatPreferencesStore((state) =>
    foldIsActive(state.foldToolActivityIntoThinking, state.toolVisibility),
  );
  const roundKey = useMessageMemo(
    (message) => {
      const reasoningEnd = governingReasoningEnd(message.parts, startIndex);
      return reasoningEnd === null
        ? null
        : reasoningRoundKey(message.id, reasoningEnd);
    },
    [startIndex],
  );
  const roundOpen = useReasoningRoundStore((state) =>
    roundKey === null ? true : (state.open[roundKey] ?? false),
  );
  const underThinking = foldToolActivity && roundKey !== null;
  const exempt = containsUngroupedTool || hasPendingConfirmation;
  // Last thing under the block, with the answer right after: close the trace with a rule.
  const closesTrace = useMessageMemo(
    (message) => endsFoldedSpan(message.parts, endIndex),
    [endIndex],
  );

  // Render these directly so their persistent content never hides in a collapsed group.
  const group =
    toolCount <= 1 || containsUngroupedTool ? (
      <>{children}</>
    ) : (
      <ToolGroupRoot open={forceOpen ? true : undefined} defaultOpen={groupRunning}>
        <ToolGroupTrigger count={toolCount} />
        <ToolGroupContent>{children}</ToolGroupContent>
      </ToolGroupRoot>
    );

  if (!underThinking) {
    return group;
  }

  // Hidden rather than unmounted, so opening keeps cards, scroll positions and live output intact.
  return (
    <div
      data-slot="tool-run-under-thinking"
      className={cn(!(roundOpen || exempt) && "hidden")}
    >
      {group}
      {closesTrace && (
        <div
          data-slot="reasoning-end-rule"
          aria-hidden={true}
          className={cn("mt-4 border-border border-t", !roundOpen && "hidden")}
        />
      )}
    </div>
  );
};

const ToolGroup = memo(ToolGroupImpl) as unknown as ToolGroupComponent;

ToolGroup.displayName = "ToolGroup";
ToolGroup.Root = ToolGroupRoot;
ToolGroup.Trigger = ToolGroupTrigger;
ToolGroup.Content = ToolGroupContent;

export {
  ToolGroup,
  ToolGroupRoot,
  ToolGroupTrigger,
  ToolGroupContent,
  toolGroupVariants,
};
