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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  toolOutputKey,
  useChatPreferencesStore,
  useToolPaneScope,
  useUnresolvedToolPaneScope,
} from "@/features/chat";
import { ChevronDownIcon } from "lucide-react";
import { Wrench01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cva, type VariantProps } from "class-variance-authority";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { cn } from "@/lib/utils";
import { Spinner } from "@/components/ui/spinner";
import { syncToolActivityPreference } from "./tool-activity-open-state";

const ANIMATION_DURATION = 200;

const toolGroupVariants = cva("aui-tool-group-root group/tool-group w-full", {
  variants: {
    variant: {
      outline: "corner-squircle rounded-lg border py-3",
      ghost: "py-2",
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
  // Same treatment as ToolFallbackRoot. Without it the group is the one
  // disclosure the preference cannot reach: a group the user expanded by hand
  // stays expanded when the setting is turned on, while every card inside it
  // closes. ToolGroupImpl hands us `undefined` whenever it is not forcing the
  // group open, so this uncontrolled state is what is on screen for most of a
  // group's life.
  const collapseByDefault = useChatPreferencesStore(
    (state) => state.collapseToolActivityByDefault,
  );
  const [uncontrolledState, setUncontrolledState] = useState(() => ({
    collapseByDefault,
    open: defaultOpen && !collapseByDefault,
  }));
  const syncedUncontrolledState = syncToolActivityPreference(
    uncontrolledState,
    collapseByDefault,
    defaultOpen,
  );
  if (syncedUncontrolledState !== uncontrolledState) {
    setUncontrolledState(syncedUncontrolledState);
  }
  const lockScroll = useCollapseScrollLock(collapsibleRef, ANIMATION_DURATION);

  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : syncedUncontrolledState.open;

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) {
        lockScroll();
      }
      if (!isControlled) {
        setUncontrolledState({ collapseByDefault, open });
      }
      controlledOnOpenChange?.(open);
    },
    [collapseByDefault, lockScroll, isControlled, controlledOnOpenChange],
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
        "aui-tool-group-trigger group/trigger flex w-full cursor-pointer items-center gap-2 text-sm transition-colors",
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
          className="size-4 shrink-0 text-muted-foreground"
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
          "group-data-[variant=ghost]/tool-group-root:mt-1 group-data-[variant=ghost]/tool-group-root:gap-1",
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
  const containsUngroupedTool = useAuiState(({ message }) =>
    message.parts
      .slice(startIndex, endIndex + 1)
      .some(
        (part) =>
          part.type === "tool-call" &&
          (part.toolName === "render_html" || part.toolName === "python"),
      ),
  );
  // A blocking allow/deny prompt must never be hidden inside a collapsed
  // group, so force the group open while any of its calls awaits confirmation.
  const toolConfirmations = useChatRuntimeStore((s) => s.toolConfirmations);
  const hasPendingConfirmation = useAuiState(({ message }) =>
    message.parts
      .slice(startIndex, endIndex + 1)
      .some(
        (part) =>
          part.type === "tool-call" &&
          Object.prototype.hasOwnProperty.call(
            toolConfirmations,
            part.toolCallId,
          ),
      ),
  );
  const messageRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const collapseByDefault = useChatPreferencesStore(
    (state) => state.collapseToolActivityByDefault,
  );
  // Force the group open when any call is receiving tool_output events.
  const toolLiveOutput = useChatRuntimeStore((s) => s.toolLiveOutput);
  const paneScope = useToolPaneScope();
  const unresolvedScope = useUnresolvedToolPaneScope();
  const hasLiveOutput = useAuiState(({ message }) =>
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
  );
  // Keep the group open once a confirmation or live output forced it (so an
  // allow/deny doesn't snap it shut between calls); reverts once the turn ends.
  // Only latch what could have forced the group open in the first place: a
  // latch set while collapsed is a force nobody saw, and turning the preference
  // off mid-turn would then snap every such group open at once.
  const forcedOpenRef = useRef(false);
  if (hasPendingConfirmation || (hasLiveOutput && !collapseByDefault)) {
    forcedOpenRef.current = true;
  }
  const forceOpen =
    hasPendingConfirmation ||
    (!collapseByDefault &&
      ((hasLiveOutput && messageRunning) ||
        (forcedOpenRef.current && messageRunning)));

  // Render single calls, canvases, and Python scripts directly so their
  // persistent content never hides in a collapsed group.
  if (toolCount <= 1 || containsUngroupedTool) {
    return <>{children}</>;
  }

  return (
    <ToolGroupRoot open={forceOpen ? true : undefined}>
      <ToolGroupTrigger count={toolCount} />
      <ToolGroupContent>{children}</ToolGroupContent>
    </ToolGroupRoot>
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
