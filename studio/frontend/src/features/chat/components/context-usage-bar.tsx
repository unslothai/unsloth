"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { FC } from "react";

const formatTokenCount = (n: number): string => {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
};

const formatTokenCountFull = (n: number): string => {
  return n.toLocaleString();
};

/**
 * Which limit warning the tooltip carries, if any.
 *
 * llama.cpp stops generating at the window, so its advice is to raise the limit before
 * hitting it. MLX generates straight past instead, so the same wording would promise a
 * stop that never comes -- and once a conversation is over the window there is something
 * different to say about it. Read from the unclamped ratio: the displayed figure caps at
 * 100%, which is exactly the state being reported on.
 */
function contextLimitAdvice(
  used: number,
  total: number | null | undefined,
  isMlx: boolean | undefined,
): "none" | "stops-at-limit" | "mlx-near-limit" | "mlx-past-limit" {
  if (typeof total !== "number" || total <= 0) return "none";
  if ((used / total) * 100 <= 85) return "none";
  if (!isMlx) return "stops-at-limit";
  return used > total ? "mlx-past-limit" : "mlx-near-limit";
}

function getSeverityColor(percent: number): {
  bar: string;
  text: string;
} {
  if (percent > 85) return { bar: "bg-red-500", text: "text-red-500" };
  if (percent > 65) return { bar: "bg-amber-500", text: "text-amber-500" };
  return { bar: "bg-control-accent", text: "text-control-accent" };
}

export const ContextUsageBar: FC<{
  used: number;
  // null on external providers (unknown window); bar hides the ratio.
  total?: number | null;
  cached?: number;
  // Anthropic-only (billed at the write premium).
  cacheWrites?: number;
  promptTokens?: number;
  completionTokens?: number;
  // MLX keeps generating past the window instead of stopping there, so it needs the
  // opposite advice from llama.cpp once a conversation outgrows the limit.
  isMlx?: boolean;
  className?: string;
}> = ({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
  isMlx,
  className,
}) => {
  const hasKnownLimit = typeof total === "number" && total > 0;
  const hasUsageDetails =
    promptTokens !== undefined ||
    completionTokens !== undefined ||
    (cached !== undefined && cached > 0) ||
    (cacheWrites !== undefined && cacheWrites > 0);

  // Nothing to show: no limit and no per-turn counters.
  if (!hasKnownLimit && used <= 0 && !hasUsageDetails) return null;

  const percent = hasKnownLimit
    ? Math.min((used / (total as number)) * 100, 100)
    : null;
  const advice = contextLimitAdvice(used, total, isMlx);
  const severity = getSeverityColor(percent ?? 0);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={
            hasKnownLimit
              ? `Context usage: ${formatTokenCount(used)} of ${formatTokenCount(total as number)} tokens`
              : `Token usage: ${formatTokenCount(used)} tokens`
          }
          className={cn(
            "flex items-center gap-2 rounded-[10px] px-2.5 py-1 font-mono text-chat-icon-fg text-ui-13 tabular-nums transition-colors hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
            className,
          )}
        >
          <span>
            {hasKnownLimit
              ? `${formatTokenCount(used)} / ${formatTokenCount(total as number)}`
              : `${formatTokenCount(used)} tokens`}
          </span>
          {hasKnownLimit && percent !== null ? (
            <div className="h-1.5 w-16 rounded-full bg-black/10 dark:bg-white/15 overflow-hidden">
              <div
                className={cn("h-full rounded-full transition-all", severity.bar)}
                style={{ width: `${percent}%` }}
              />
            </div>
          ) : null}
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="bottom"
        sideOffset={8}
        variant="rich"
        className="[&_span>svg]:hidden!"
      >
        <div className="grid min-w-44 gap-1.5 text-xs">
          {hasKnownLimit && percent !== null ? (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Context usage</span>
              <span className={cn("font-mono tabular-nums font-medium", severity.text)}>
                {percent.toFixed(1)}%
              </span>
            </div>
          ) : null}
          {promptTokens !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Prompt tokens</span>
              <span className="font-mono tabular-nums">
                {formatTokenCountFull(promptTokens)}
              </span>
            </div>
          )}
          {completionTokens !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Completion</span>
              <span className="font-mono tabular-nums">
                {formatTokenCountFull(completionTokens)}
              </span>
            </div>
          )}
          {cached !== undefined && cached > 0 && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Cache hits</span>
              <span className="font-mono tabular-nums">
                {formatTokenCountFull(cached)}
              </span>
            </div>
          )}
          {cacheWrites !== undefined && cacheWrites > 0 && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Cache writes</span>
              <span className="font-mono tabular-nums">
                {formatTokenCountFull(cacheWrites)}
              </span>
            </div>
          )}
          <div className="my-0.5 border-t border-border/40" />
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">
              {hasKnownLimit ? "Total" : "Total tokens"}
            </span>
            <span className="font-mono tabular-nums">
              {hasKnownLimit
                ? `${formatTokenCountFull(used)} / ${formatTokenCountFull(total as number)}`
                : formatTokenCountFull(used)}
            </span>
          </div>
          {advice !== "none" ? (
            <div className="mt-1 max-w-64 text-ui-11 leading-snug text-muted-foreground/90">
              {advice === "mlx-past-limit" ? (
                <>
                  Past the context limit. The chat keeps working — replies
                  won't be cut off — but the model can no longer hold the whole
                  conversation, so answers get slower and less accurate the
                  further past it you go. Increase{" "}
                  <span className="font-medium">Context Length</span> in the
                  chat Settings panel to fit it all.
                </>
              ) : advice === "mlx-near-limit" ? (
                <>
                  Close to the context limit. Past it the chat keeps working,
                  but answers get slower and less accurate. Increase{" "}
                  <span className="font-medium">Context Length</span> in the
                  chat Settings panel to fit the whole conversation.
                </>
              ) : (
                <>
                  Close to the context limit. Generation will stop at 100%.
                  Increase <span className="font-medium">Context Length</span> in
                  the chat Settings panel to keep going.
                </>
              )}
            </div>
          ) : null}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};
