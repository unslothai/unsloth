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
  isMlx,
  contextEnforced,
  className,
}) => {
  const state = deriveContextUsageBar({
    used,
    total,
    cached,
    cacheWrites,
    promptTokens,
    completionTokens,
    isMlx,
    contextEnforced,
  });
  if (!state) return null;

  const { percent, advice } = state;
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
          {advice !== "none" ? (
            <div className="mt-1 max-w-64 text-ui-11 leading-snug text-muted-foreground/90">
              {advice === "mlx-past-limit" ? (
                <>
                  Past the context limit. The chat keeps going rather than
                  stopping here, but the model can no longer hold the whole
                  conversation: answers get slower and less accurate, and a
                  long enough chat can still run out of memory. Increase{" "}
                  <span className="font-medium">Context Length</span> in the
                  chat Settings panel to fit it all.
                </>
              ) : advice === "mlx-near-limit" ? (
                <>
                  Close to the context limit. Past it the chat keeps going, but
                  answers get slower and less accurate. Increase{" "}
                  <span className="font-medium">Context Length</span> in the
                  chat Settings panel to fit the whole conversation.
                </>
              ) : advice === "unenforced-limit" ? (
                <>
                  This model builds its own cache, so{" "}
                  <span className="font-medium">Context Length</span> is the
                  window it was sized for, not a limit on it: the cache keeps
                  growing and lowering the setting will not save memory.
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
