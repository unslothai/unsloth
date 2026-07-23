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

const formatRate = (r: number | undefined): string => {
  if (r === undefined || !Number.isFinite(r)) return "—";
  return `${Math.round(r).toLocaleString()} tok/s`;
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

  const custom = (
    message.metadata as Record<string, unknown> | undefined
  )?.custom as
    | {
        serverTimings?: Record<string, number>;
        contextUsage?: {
          cachedTokens?: number;
          cacheWriteTokens?: number;
        };
      }
    | undefined;
  const st = custom?.serverTimings;
  // `??` (not `||`) so an explicit cache_n=0 isn't replaced by a stale
  // contextUsage.cachedTokens from a prior turn.
  const cacheHits =
    st?.cache_n ?? custom?.contextUsage?.cachedTokens ?? 0;
  // Anthropic-only cache-write count.
  const cacheWrites = custom?.contextUsage?.cacheWriteTokens ?? 0;
  // DiffusionGemma reports separately-labelled throughput (no prefill, so no "prompt
  // speed"), matching the CLI: in-step parallel, effective (canvas*blocks/wall), and
  // output (answer tokens/wall).
  const isDiffusion = (st as { diffusion?: boolean } | undefined)?.diffusion === true;

  // Guard unphysical tok/s: llama.cpp emits predicted_ms=0 on no-op turns,
  // blowing the rate up to Infinity. Require >=1 token, a non-zero decode
  // window, and a finite rate. Fast cached sub-10ms responses are legit.
  const hasPredicted =
    (st?.predicted_n ?? 0) >= 1 && (st?.predicted_ms ?? 0) > 0;
  const predictedRate =
    hasPredicted &&
    st?.predicted_per_second != null &&
    Number.isFinite(st.predicted_per_second)
      ? st.predicted_per_second
      : undefined;

  // Badge text: show tok/s if available, otherwise total time
  const badgeText = predictedRate != null
    ? `${predictedRate.toFixed(1)} tok/s`
    : formatTimingMs(timing.totalStreamTime);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-slot="message-timing-trigger"
          aria-label="Message timing"
          className={cn(
            "flex items-center rounded-[10px] p-1 font-mono text-chat-icon-fg text-ui-13 tabular-nums transition-colors hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
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
            isDiffusion ? (
            <>
              {/* DiffusionGemma: honest throughput (no autoregressive prompt speed) */}
              {timing.firstTokenTime !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">First token</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(timing.firstTokenTime)}
                  </span>
                </div>
              )}
              {st?.diffusion_parallel_tok_s != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Speed (in-step)</span>
                  <span className="font-mono tabular-nums">
                    {formatRate(st.diffusion_parallel_tok_s)}
                  </span>
                </div>
              )}
              {st?.diffusion_effective_tok_s != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Effective</span>
                  <span className="font-mono tabular-nums">
                    {formatRate(st.diffusion_effective_tok_s)}
                  </span>
                </div>
              )}
              {st?.diffusion_output_tok_s != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Output</span>
                  <span className="font-mono tabular-nums">
                    {formatRate(st.diffusion_output_tok_s)}
                  </span>
                </div>
              )}
              {st?.diffusion_steps != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Denoising</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_steps)} steps
                    {st?.diffusion_blocks != null
                      ? `, ${formatNumber(st.diffusion_blocks)} block${st.diffusion_blocks === 1 ? "" : "s"}`
                      : ""}
                  </span>
                </div>
              )}
              {st?.diffusion_canvas != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Canvas</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_canvas)} tokens
                  </span>
                </div>
              )}
              {(st?.diffusion_wall_ms ?? st?.predicted_ms) != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Generation</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.diffusion_wall_ms ?? st.predicted_ms)}
                  </span>
                </div>
              )}
              {timing.tokenCount !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Answer tokens</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(timing.tokenCount)}
                  </span>
                </div>
              )}
              {(st?.diffusion_prompt_n ?? st?.prompt_n) != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Prompt</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_prompt_n ?? st.prompt_n)} tokens
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
              {hasPredicted && st?.predicted_ms != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Generation</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.predicted_ms)}
                  </span>
                </div>
              )}
              {predictedRate != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Speed</span>
                  <span className="font-mono tabular-nums">
                    {predictedRate.toFixed(1)} tok/s
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
              {timing.firstTokenTime !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">First token</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(timing.firstTokenTime)}
                  </span>
                </div>
              )}
              {st?.diffusion_steps != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Denoising steps</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_steps)}
                  </span>
                </div>
              )}
              {st?.diffusion_blocks != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Blocks</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_blocks)}
                  </span>
                </div>
              )}
              {cacheHits > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Cache hits</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheHits)}
                  </span>
                </div>
              )}
              {cacheWrites > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Cache writes</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheWrites)}
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
            )
          ) : (
            <>
              {/* Client-side metrics (safetensors + external provider fallback) */}
              {timing.firstTokenTime !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">First token</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(timing.firstTokenTime)}
                  </span>
                </div>
              )}
              {cacheHits > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Cache hits</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheHits)}
                  </span>
                </div>
              )}
              {cacheWrites > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Cache writes</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheWrites)}
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
