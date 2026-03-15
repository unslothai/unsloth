"use client";

import { useMessageTiming } from "@assistant-ui/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { FC } from "react";

const formatTimingMs = (ms: number | undefined): string => {
  if (ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

/**
 * Shows streaming stats (TTFT, total time, chunks) as a badge with a
 * hover/focus tooltip. Renders nothing until the stream completes.
 *
 * Place it inside `ActionBarPrimitive.Root` in your `thread.tsx` so it
 * inherits the action bar's autohide behaviour:
 *
 * ```tsx
 * import { MessageTiming } from "@/components/assistant-ui/message-timing";
 *
 * <ActionBarPrimitive.Root >
 *   <ActionBarPrimitive.Copy />
 *   <ActionBarPrimitive.Reload />
 *   <MessageTiming />  // <-- add this
 * </ActionBarPrimitive.Root>
 * ```
 *
 * @param side - Side of the tooltip relative to the badge trigger. Defaults to `"right"`.
 */
export const MessageTiming: FC<{
  className?: string;
  side?: "top" | "right" | "bottom" | "left";
}> = ({ className, side = "right" }) => {
  const timing = useMessageTiming();
  if (timing?.totalStreamTime === undefined) return null;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-slot="message-timing-trigger"
          aria-label="Message timing"
          className={cn(
            "flex items-center rounded-md p-1 font-mono text-muted-foreground text-xs tabular-nums transition-colors hover:bg-accent hover:text-accent-foreground",
            className,
          )}
        >
          {formatTimingMs(timing.totalStreamTime)}
        </button>
      </TooltipTrigger>
      <TooltipContent
        side={side}
        sideOffset={8}
        data-slot="message-timing-popover"
        className="[&_span>svg]:hidden! rounded-lg border bg-popover px-3 py-2 text-popover-foreground shadow-md"
      >
        <div className="grid min-w-35 gap-1.5 text-xs">
          {timing.firstTokenTime !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">First token</span>
              <span className="font-mono tabular-nums">
                {formatTimingMs(timing.firstTokenTime)}
              </span>
            </div>
          )}
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">Total</span>
            <span className="font-mono tabular-nums">
              {formatTimingMs(timing.totalStreamTime)}
            </span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">Chunks</span>
            <span className="font-mono tabular-nums">{timing.totalChunks}</span>
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
};
