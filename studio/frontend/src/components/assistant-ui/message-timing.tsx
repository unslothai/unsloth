"use client";

import { useMessageTiming, useMessage } from "@assistant-ui/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { FC } from "react";

type Translate = ReturnType<typeof useT>;

const formatTimingMs = (ms: number | undefined): string => {
  if (ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const formatNumber = (n: number): string => {
  return n.toLocaleString();
};

/** A user stop wins over the stream's last finish_reason. */
function stopReasonLabel(
  t: Translate,
  finishReason: unknown,
  incompleteReason?: unknown,
): string | null {
  if (incompleteReason === "cancelled") return t("generationStats.timing.stoppedByYou");
  switch (finishReason) {
    case "stop":
      return t("generationStats.timing.endOfTurn");
    case "length":
      return t("generationStats.timing.lengthLimit");
    case "tool_calls":
    case "function_call":
      return t("generationStats.timing.toolCall");
    case "content_filter":
      return t("generationStats.timing.contentFilter");
    default:
      return typeof finishReason === "string" && finishReason ? finishReason : null;
  }
}

const tokens = (t: Translate, n: number): string =>
  t(n === 1 ? "generationStats.tokensOne" : "generationStats.tokens", { count: formatNumber(n) });

const formatRate = (r: number | undefined): string => {
  if (r === undefined || !Number.isFinite(r)) return "—";
  return `${Math.round(r).toLocaleString()} tok/s`;
};

export const MessageTiming: FC<{
  className?: string;
  side?: "top" | "right" | "bottom" | "left";
}> = ({ className, side = "right" }) => {
  const timing = useMessageTiming();
  const message = useMessage();
  const t = useT();

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
        stopReason?: unknown;
        incomplete?: { reason?: unknown };
      }
    | undefined;
  const st = custom?.serverTimings;
  const stopReason = stopReasonLabel(t, custom?.stopReason, custom?.incomplete?.reason);
  const draftN = st?.draft_n ?? 0;
  const draftAccepted = st?.draft_n_accepted;
  const draftText =
    draftN > 0 && typeof draftAccepted === "number" && Number.isFinite(draftAccepted)
      ? `${formatNumber(draftAccepted)} / ${formatNumber(draftN)} (${Math.round((draftAccepted / draftN) * 100)}%)`
      : null;
  // `??` (not `||`) so an explicit cache_n=0 isn't replaced by a stale
  // contextUsage.cachedTokens from a prior turn.
  const cacheHits =
    st?.cache_n ?? custom?.contextUsage?.cachedTokens ?? 0;
  const cacheWrites = custom?.contextUsage?.cacheWriteTokens ?? 0;
  const isDiffusion = (st as { diffusion?: boolean } | undefined)?.diffusion === true;

  // llama.cpp emits predicted_ms=0 on no-op turns; guard against an Infinity rate.
  const hasPredicted =
    (st?.predicted_n ?? 0) >= 1 && (st?.predicted_ms ?? 0) > 0;
  const predictedRate =
    hasPredicted &&
    st?.predicted_per_second != null &&
    Number.isFinite(st.predicted_per_second)
      ? st.predicted_per_second
      : undefined;

  const badgeText = predictedRate != null
    ? `${predictedRate.toFixed(1)} tok/s`
    : formatTimingMs(timing.totalStreamTime);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-slot="message-timing-trigger"
          aria-label={t("generationStats.timing.trigger")}
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
              {timing.firstTokenTime !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.firstToken")}</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(timing.firstTokenTime)}
                  </span>
                </div>
              )}
              {st?.diffusion_parallel_tok_s != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.speedInStep")}</span>
                  <span className="font-mono tabular-nums">
                    {formatRate(st.diffusion_parallel_tok_s)}
                  </span>
                </div>
              )}
              {st?.diffusion_effective_tok_s != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.effective")}</span>
                  <span className="font-mono tabular-nums">
                    {formatRate(st.diffusion_effective_tok_s)}
                  </span>
                </div>
              )}
              {st?.diffusion_output_tok_s != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.output")}</span>
                  <span className="font-mono tabular-nums">
                    {formatRate(st.diffusion_output_tok_s)}
                  </span>
                </div>
              )}
              {st?.diffusion_steps != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.denoising")}</span>
                  <span className="font-mono tabular-nums">
                    {t(
                      st.diffusion_steps === 1
                        ? "generationStats.timing.stepsOne"
                        : "generationStats.timing.steps",
                      { count: formatNumber(st.diffusion_steps) },
                    )}
                    {st?.diffusion_blocks != null
                      ? `, ${t(
                          st.diffusion_blocks === 1
                            ? "generationStats.timing.blocksOne"
                            : "generationStats.timing.blocks",
                          { count: formatNumber(st.diffusion_blocks) },
                        )}`
                      : ""}
                  </span>
                </div>
              )}
              {st?.diffusion_canvas != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.canvas")}</span>
                  <span className="font-mono tabular-nums">
                    {tokens(t, st.diffusion_canvas)}
                  </span>
                </div>
              )}
              {(st?.diffusion_wall_ms ?? st?.predicted_ms) != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.generation")}</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.diffusion_wall_ms ?? st.predicted_ms)}
                  </span>
                </div>
              )}
              {timing.tokenCount !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.answerTokens")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(timing.tokenCount)}
                  </span>
                </div>
              )}
              {(st?.diffusion_prompt_n ?? st?.prompt_n) != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.prompt")}</span>
                  <span className="font-mono tabular-nums">
                    {tokens(t, st.diffusion_prompt_n ?? st.prompt_n)}
                  </span>
                </div>
              )}
              <div className="my-0.5 border-t border-border/40" />
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{t("generationStats.timing.total")}</span>
                <span className="font-mono tabular-nums">
                  {formatTimingMs(timing.totalStreamTime)}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{t("generationStats.timing.chunks")}</span>
                <span className="font-mono tabular-nums">
                  {timing.totalChunks}
                </span>
              </div>
            </>
            ) : (
            <>
              {st?.prompt_ms != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.promptEval")}</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.prompt_ms)}
                  </span>
                </div>
              )}
              {(st?.prompt_n ?? 0) > 1 &&
                st?.prompt_per_second != null &&
                st.prompt_per_second > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.promptSpeed")}</span>
                  <span className="font-mono tabular-nums">
                    {st.prompt_per_second.toFixed(1)} tok/s
                  </span>
                </div>
              )}
              {hasPredicted && st?.predicted_ms != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.generation")}</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(st.predicted_ms)}
                  </span>
                </div>
              )}
              {predictedRate != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.speed")}</span>
                  <span className="font-mono tabular-nums">
                    {predictedRate.toFixed(1)} tok/s
                  </span>
                </div>
              )}
              {timing.tokenCount !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.tokens")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(timing.tokenCount)}
                  </span>
                </div>
              )}
              {draftText !== null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.draftAccepted")}</span>
                  <span className="font-mono tabular-nums">{draftText}</span>
                </div>
              )}
              {timing.firstTokenTime !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.firstToken")}</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(timing.firstTokenTime)}
                  </span>
                </div>
              )}
              {st?.diffusion_steps != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.denoisingSteps")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_steps)}
                  </span>
                </div>
              )}
              {st?.diffusion_blocks != null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.blocksLabel")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(st.diffusion_blocks)}
                  </span>
                </div>
              )}
              {cacheHits > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.cacheHits")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheHits)}
                  </span>
                </div>
              )}
              {cacheWrites > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.cacheWrites")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheWrites)}
                  </span>
                </div>
              )}
              <div className="my-0.5 border-t border-border/40" />
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{t("generationStats.timing.total")}</span>
                <span className="font-mono tabular-nums">
                  {formatTimingMs(timing.totalStreamTime)}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{t("generationStats.timing.chunks")}</span>
                <span className="font-mono tabular-nums">
                  {timing.totalChunks}
                </span>
              </div>
              {stopReason !== null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.stopReason")}</span>
                  <span>{stopReason}</span>
                </div>
              )}
            </>
            )
          ) : (
            <>
              {timing.firstTokenTime !== undefined && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.firstToken")}</span>
                  <span className="font-mono tabular-nums">
                    {formatTimingMs(timing.firstTokenTime)}
                  </span>
                </div>
              )}
              {cacheHits > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.cacheHits")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheHits)}
                  </span>
                </div>
              )}
              {cacheWrites > 0 && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.cacheWrites")}</span>
                  <span className="font-mono tabular-nums">
                    {formatNumber(cacheWrites)}
                  </span>
                </div>
              )}
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{t("generationStats.timing.total")}</span>
                <span className="font-mono tabular-nums">
                  {formatTimingMs(timing.totalStreamTime)}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{t("generationStats.timing.chunks")}</span>
                <span className="font-mono tabular-nums">
                  {timing.totalChunks}
                </span>
              </div>
              {stopReason !== null && (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">{t("generationStats.timing.stopReason")}</span>
                  <span>{stopReason}</span>
                </div>
              )}
            </>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};
