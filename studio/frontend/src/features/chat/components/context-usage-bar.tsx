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
  return { bar: "bg-emerald-500", text: "text-emerald-500" };
}

export const ContextUsageBar: FC<{
  used: number;
  total: number;
  cached?: number;
  promptTokens?: number;
  completionTokens?: number;
  className?: string;
}> = ({ used, total, cached, promptTokens, completionTokens, className }) => {
  if (total <= 0) return null;

  const percent = Math.min((used / total) * 100, 100);
  const severity = getSeverityColor(percent);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`Context usage: ${formatTokenCount(used)} of ${formatTokenCount(total)} tokens`}
          className={cn(
            "flex items-center gap-2 rounded-md px-2 py-1 text-xs font-mono tabular-nums text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
            className,
          )}
        >
          <span>
            {formatTokenCount(used)} / {formatTokenCount(total)}
          </span>
          <div className="h-1.5 w-16 rounded-full bg-muted overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", severity.bar)}
              style={{ width: `${percent}%` }}
            />
          </div>
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="bottom"
        sideOffset={8}
        className="[&_span>svg]:hidden! rounded-lg border bg-popover px-3 py-2 text-popover-foreground shadow-md"
      >
        <div className="grid min-w-44 gap-1.5 text-xs">
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">Context usage</span>
            <span className={cn("font-mono tabular-nums font-medium", severity.text)}>
              {percent.toFixed(1)}%
            </span>
          </div>
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
          <div className="my-0.5 border-t border-border/40" />
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">Total</span>
            <span className="font-mono tabular-nums">
              {formatTokenCountFull(used)} / {formatTokenCountFull(total)}
            </span>
          </div>
          {percent > 85 && (
            <div className="mt-1 max-w-64 text-[11px] leading-snug text-muted-foreground/90">
              Close to the context limit. Generation will stop at 100%.
              Increase <span className="font-medium">Context Length</span> in
              the chat Settings panel to keep going.
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};
