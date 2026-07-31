// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Spinner } from "@/components/ui/spinner";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import { cn } from "@/lib/utils";
import {
  type ToolCallMessagePartComponent,
  type ToolCallMessagePartStatus,
} from "@assistant-ui/react";
import {
  AlertCircleIcon,
  ChevronDownIcon,
  LoaderIcon,
  XCircleIcon,
} from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type ComponentProps,
  type ElementType,
  memo,
  useCallback,
  useRef,
  useState,
} from "react";

const ANIMATION_DURATION = 200;

export type ToolFallbackRootProps = Omit<
  ComponentProps<typeof Collapsible>,
  "open" | "onOpenChange"
> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  defaultOpen?: boolean;
};

function ToolFallbackRoot({
  className,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  defaultOpen = false,
  children,
  ...props
}: ToolFallbackRootProps) {
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
      data-slot="tool-fallback-root"
      open={isOpen}
      onOpenChange={handleOpenChange}
      className={cn(
        "aui-tool-fallback-root group/tool-fallback-root w-full py-1",
        className,
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

type ToolStatus = ToolCallMessagePartStatus["type"];

// The shared app tick is icon data, not a component; wrap it to slot into the
// status map alongside the lucide icons.
function CompleteTickIcon(props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">) {
  return <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} {...props} />;
}

const statusIconMap: Record<ToolStatus, ElementType> = {
  running: LoaderIcon,
  complete: CompleteTickIcon,
  incomplete: XCircleIcon,
  "requires-action": AlertCircleIcon,
};

const MCP_TOOL_PREFIX = "mcp__";

function formatToolNameForDisplay(toolName: string): string {
  if (!toolName.startsWith(MCP_TOOL_PREFIX)) return toolName;
  const rest = toolName.slice(MCP_TOOL_PREFIX.length);
  const sep = rest.indexOf("__");
  if (sep <= 0) return toolName;
  return `${rest.slice(0, sep)} · ${rest.slice(sep + 2)}`;
}

function ToolFallbackTrigger({
  toolName,
  status,
  icon: ToolIcon,
  className,
  ...props
}: ComponentProps<typeof CollapsibleTrigger> & {
  toolName: string;
  status?: ToolCallMessagePartStatus;
  icon?: ElementType;
}) {
  const statusType = status?.type ?? "complete";
  const isRunning = statusType === "running";
  const isCancelled =
    status?.type === "incomplete" && status.reason === "cancelled";

  const StatusIcon = statusIconMap[statusType];
  const label = isCancelled ? "Cancelled tool" : "Used tool";
  const displayName = formatToolNameForDisplay(toolName);

  return (
    <CollapsibleTrigger
      data-slot="tool-fallback-trigger"
      className={cn(
        "aui-tool-fallback-trigger group/trigger flex w-full cursor-pointer items-center gap-2 py-1.5 text-sm transition-colors",
        className,
      )}
      {...props}
    >
      {isRunning ? (
        <Spinner className="aui-tool-fallback-trigger-icon" />
      ) : ToolIcon ? (
        <ToolIcon
          data-slot="tool-fallback-trigger-icon"
          className={cn(
            "aui-tool-fallback-trigger-icon size-4 shrink-0",
            isCancelled && "text-muted-foreground",
          )}
        />
      ) : (
        <StatusIcon
          data-slot="tool-fallback-trigger-icon"
          className={cn(
            "aui-tool-fallback-trigger-icon size-4 shrink-0",
            isCancelled && "text-muted-foreground",
          )}
        />
      )}
      <span
        data-slot="tool-fallback-trigger-label"
        className={cn(
          "aui-tool-fallback-trigger-label-wrapper relative min-w-0 text-left leading-none text-muted-foreground",
          isCancelled && "text-muted-foreground line-through",
        )}
      >
        <span
          className={cn(
            "block truncate leading-normal",
            "group-data-[state=open]/trigger:overflow-visible group-data-[state=open]/trigger:whitespace-normal group-data-[state=open]/trigger:break-words",
          )}
        >
          {label}:{" "}
          <span className="font-medium text-foreground/85">{displayName}</span>
        </span>
        {isRunning && (
          <span
            aria-hidden={true}
            data-slot="tool-fallback-trigger-shimmer"
            className={cn(
              "aui-tool-fallback-trigger-shimmer shimmer pointer-events-none absolute inset-0 block truncate leading-normal motion-reduce:animate-none",
              "group-data-[state=open]/trigger:overflow-visible group-data-[state=open]/trigger:whitespace-normal group-data-[state=open]/trigger:break-words",
            )}
          >
            {label}:{" "}
            <span className="font-medium text-foreground/85">{displayName}</span>
          </span>
        )}
      </span>
      <ChevronDownIcon
        data-slot="tool-fallback-trigger-chevron"
        className={cn(
          "aui-tool-fallback-trigger-chevron mr-1 size-3.5 shrink-0 self-center",
          "transition-[transform,opacity] duration-(--animation-duration) ease-out",
          "group-data-[state=closed]/trigger:-rotate-90",
          "group-data-[state=open]/trigger:rotate-0",
        )}
      />
    </CollapsibleTrigger>
  );
}

function ToolFallbackContent({
  className,
  children,
  ...props
}: ComponentProps<typeof CollapsibleContent>) {
  return (
    <CollapsibleContent
      data-slot="tool-fallback-content"
      className={cn(
        "aui-tool-fallback-content relative overflow-hidden text-sm outline-none",
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
      <div className="mt-1 flex flex-col gap-2 pl-5">{children}</div>
    </CollapsibleContent>
  );
}

function ToolFallbackArgs({
  argsText,
  className,
  ...props
}: ComponentProps<"div"> & {
  argsText?: string;
}) {
  if (!argsText) {
    return null;
  }

  return (
    <div
      data-slot="tool-fallback-args"
      className={cn("aui-tool-fallback-args", className)}
      {...props}
    >
      <pre className="aui-tool-fallback-args-value whitespace-pre-wrap">
        {argsText}
      </pre>
    </div>
  );
}

interface McpImageResult {
  text: string;
  images: { data: string; mimeType: string }[];
}

function isMcpImageResult(val: unknown): val is McpImageResult {
  if (typeof val !== "object" || val === null) {
    return false;
  }
  const v = val as { text?: unknown; images?: unknown };
  return (
    typeof v.text === "string" &&
    Array.isArray(v.images) &&
    v.images.length > 0 &&
    v.images.every(
      (img: unknown) =>
        typeof img === "object" &&
        img !== null &&
        typeof (img as { data?: unknown }).data === "string" &&
        typeof (img as { mimeType?: unknown }).mimeType === "string",
    )
  );
}

function ToolFallbackResult({
  result,
  className,
  ...props
}: ComponentProps<"div"> & {
  result?: unknown;
}) {
  if (result === undefined) {
    return null;
  }

  const imageResult = isMcpImageResult(result) ? result : null;

  return (
    <div
      data-slot="tool-fallback-result"
      className={cn("aui-tool-fallback-result pt-2", className)}
      {...props}
    >
      <p className="aui-tool-fallback-result-header font-semibold">Result:</p>
      {imageResult ? (
        <>
          {imageResult.text && (
            <pre className="aui-tool-fallback-result-content whitespace-pre-wrap">
              {imageResult.text}
            </pre>
          )}
          <div className="mt-2 flex flex-col gap-2">
            {imageResult.images.map((img, i) => (
              <img
                key={i}
                src={`data:${img.mimeType};base64,${img.data}`}
                alt={`Tool result ${i + 1}`}
                loading="lazy"
                className="max-w-full rounded border border-border"
              />
            ))}
          </div>
        </>
      ) : (
        <pre className="aui-tool-fallback-result-content whitespace-pre-wrap">
          {typeof result === "string" ? result : JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ToolFallbackError({
  status,
  className,
  ...props
}: ComponentProps<"div"> & {
  status?: ToolCallMessagePartStatus;
}) {
  if (status?.type !== "incomplete") {
    return null;
  }

  const error = status.error;
  const errorText = error
    ? typeof error === "string"
      ? error
      : JSON.stringify(error)
    : null;

  if (!errorText) {
    return null;
  }

  const isCancelled = status.reason === "cancelled";
  const headerText = isCancelled ? "Cancelled reason:" : "Error:";

  return (
    <div
      data-slot="tool-fallback-error"
      className={cn("aui-tool-fallback-error", className)}
      {...props}
    >
      <p className="aui-tool-fallback-error-header font-semibold text-muted-foreground">
        {headerText}
      </p>
      <p className="aui-tool-fallback-error-reason text-muted-foreground">
        {errorText}
      </p>
    </div>
  );
}

const ToolFallbackImpl: ToolCallMessagePartComponent = ({
  toolName,
  argsText,
  result,
  status,
}) => {
  // Allow/Deny confirmation controls are rendered uniformly for every tool
  // card (built-in and fallback) by the `withToolConfirmation` wrapper in
  // thread.tsx, so this renderer stays purely presentational.
  const isCancelled =
    status?.type === "incomplete" && status.reason === "cancelled";

  return (
    <ToolFallbackRoot className={cn(isCancelled && "bg-muted/30")}>
      <ToolFallbackTrigger toolName={toolName} status={status} />
      <ToolFallbackContent>
        <ToolFallbackError status={status} />
        <ToolFallbackArgs
          argsText={argsText}
          className={cn(isCancelled && "opacity-60")}
        />
        {!isCancelled && <ToolFallbackResult result={result} />}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

const ToolFallback = memo(
  ToolFallbackImpl,
) as unknown as ToolCallMessagePartComponent & {
  Root: typeof ToolFallbackRoot;
  Trigger: typeof ToolFallbackTrigger;
  Content: typeof ToolFallbackContent;
  Args: typeof ToolFallbackArgs;
  Result: typeof ToolFallbackResult;
  Error: typeof ToolFallbackError;
};

ToolFallback.displayName = "ToolFallback";
ToolFallback.Root = ToolFallbackRoot;
ToolFallback.Trigger = ToolFallbackTrigger;
ToolFallback.Content = ToolFallbackContent;
ToolFallback.Args = ToolFallbackArgs;
ToolFallback.Result = ToolFallbackResult;
ToolFallback.Error = ToolFallbackError;

export {
  ToolFallback,
  ToolFallbackRoot,
  ToolFallbackTrigger,
  ToolFallbackContent,
  ToolFallbackArgs,
  ToolFallbackResult,
  ToolFallbackError,
};
