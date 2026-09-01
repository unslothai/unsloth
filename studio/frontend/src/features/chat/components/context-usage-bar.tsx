"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { FC } from "react";
import {
  type ContextUsageBarInput,
  deriveContextUsageBar,
  formatTokenCountFull,
} from "../lib/context-usage-bar-state";

function getSeverityColor(percent: number): {
  bar: string;
  text: string;
} {
  if (percent > 85) return { bar: "bg-red-500", text: "text-red-500" };
  if (percent > 65) return { bar: "bg-amber-500", text: "text-amber-500" };
  return { bar: "bg-control-accent", text: "text-control-accent" };
}

export const ContextUsageBar: FC<
  ContextUsageBarInput & { className?: string }
> = ({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
  className,
}) => {
  const state = deriveContextUsageBar({
    used,
    total,
    cached,
    cacheWrites,
    promptTokens,
    completionTokens,
  });
  if (!state) return null;

  const { percent } = state;
  const severity = getSeverityColor(percent ?? 0);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={state.label}
          className={cn(
            "flex items-center gap-2 rounded-[10px] px-2.5 py-1 font-mono text-chat-icon-fg text-ui-13 tabular-nums transition-colors hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
            className,
          )}
        >
          <span>{state.face}</span>
          {percent !== null ? (
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
          {percent !== null ? (
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
          {percent !== null || state.hasUsageDetails ? (
            <div className="my-0.5 border-t border-border/40" />
          ) : null}
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">{state.totalRowName}</span>
            <span className="font-mono tabular-nums">{state.totalRowValue}</span>
          </div>
          {percent !== null && percent > 85 ? (
            <div className="mt-1 max-w-64 text-ui-11 leading-snug text-muted-foreground/90">
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
