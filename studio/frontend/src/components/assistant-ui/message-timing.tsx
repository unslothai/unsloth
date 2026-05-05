"use client";

import { useMessageTiming, useMessage } from "@assistant-ui/react";
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

const formatNumber = (n: number): string => {
  return n.toLocaleString();
};

/**
 * Shows streaming stats as a badge with hover tooltip.
 * When server timings are available (GGUF), shows prompt eval, generation,
 * speed, tokens, and cache hits. Falls back to client-side metrics otherwise.
 */
export const MessageTiming: FC<{
  className?: string;
  side?: "top" | "right" | "bottom" | "left";
}> = ({ className, side = "right" }) => {
  const timing = useMessageTiming();
  const message = useMessage();

  if (timing?.totalStreamTime === undefined) return null;

  const serverTimings = (
    message.metadata as Record<string, unknown> | undefined
  )?.custom as { serverTimings?: Record<string, number> } | undefined;
  const st = serverTimings?.serverTimings;

  // Badge text: show tok/s if available, otherwise total time
  const badgeText = st?.predicted_per_second != null
    ? `${st.predicted_per_second.toFixed(1)} tok/s`
    : formatTimingMs(timing.totalStreamTime);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-slot="message-timing-trigger"
          aria-label="Message timing"
          className={cn(
            "flex items-center rounded-[10px] p-1 font-mono text-chat-icon-fg text-[13px] tabular-nums transition-colors hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
            className,
          )}
        >
          {badgeText}
        </button>
      </TooltipTrigger>
      <TooltipContent
        side={side}
        sideOffset={8}
        data-slot="message-timing-popover"
        variant="rich"
        className="[&_span>svg]:hidden!"
      >
        <div className="grid min-w-40 gap-1.5 text-xs">
          {st ? (
            <>
              {/* Server-side metrics (GGUF) */}
              {st?.prompt_ms != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Prompt eval</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.prompt_ms)}
                  </span>
                </div>
              )}
              {(st?.prompt_n ?? 0) > 1 && st?.prompt_per_second != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Prompt speed</span>
                  <span className="font-mono tabular-nums">
                    {st.prompt_per_second.toFixed(1)} tok/s
                  </span>
                </div>
              )}
              {st?.predicted_ms != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Generation</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.predicted_ms)}
                  </span>
                </div>
              )}
              {st?.predicted_per_second != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Speed</span>
                  <span className="font-mono tabular-nums">
                    {st.predicted_per_second.toFixed(1)} tok/s
                  </span>
                </div>
              )}
              {timing.tokenCount !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Tokens</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(timing.tokenCount)}
                  </span>
                </div>
              )}
              {(st?.cache_n ?? 0) > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Cache hits</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st!.cache_n)}
                  </span>
                </div>
              )}
              <div className="my-0.5 border-t border-border/40" />
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">Total</span>
                <span className="font-mono tabular-nums">
                  {formatTimingMs(timing.totalStreamTime)}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">Chunks</span>
                <span className="font-mono tabular-nums">
                  {timing.totalChunks}
                </span>
              </div>
            </>
          ) : (
            <>
              {/* Client-side metrics (safetensors fallback) */}
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
                <span className="font-mono tabular-nums">
                  {timing.totalChunks}
                </span>
              </div>
            </>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};
