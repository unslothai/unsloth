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
  className?: string;
}> = ({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
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
            "flex items-center gap-2 rounded-[10px] px-2.5 py-1 font-mono text-chat-icon-fg text-[0.8125rem] tabular-nums transition-colors hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
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
          {hasKnownLimit && percent !== null && percent > 85 ? (
            <div className="mt-1 max-w-64 text-[0.6875rem] leading-snug text-muted-foreground/90">
              Close to the context limit. Generation will stop at 100%.
              Increase <span className="font-medium">Context Length</span> in
              the chat Settings panel to keep going.
            </div>
          ) : null}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};
