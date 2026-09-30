"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { CSSProperties, FC } from "react";
import {
  type ContextUsageBarInput,
  deriveContextUsageBar,
  formatTokenCountFull,
} from "../lib/context-usage-bar-state";

function getSeverityColor(percent: number): {
  bar: string;
  stroke: string;
  text: string;
} {
  if (percent > 85) {
    return { bar: "bg-red-500", stroke: "stroke-red-500", text: "text-red-500" };
  }
  if (percent > 65) {
    return { bar: "bg-amber-500", stroke: "stroke-amber-500", text: "text-amber-500" };
  }
  return {
    bar: "bg-control-accent",
    stroke: "stroke-control-accent",
    text: "text-control-accent",
  };
}

const RING_RADIUS = 5.5;
const RING_LENGTH = 2 * Math.PI * RING_RADIUS;

const UsageRing: FC<{ percent: number | null; stroke: string }> = ({
  percent,
  stroke,
}) => (
  <svg viewBox="0 0 16 16" aria-hidden={true} className="size-3.5 shrink-0 -rotate-90">
    <circle cx={8} cy={8} r={RING_RADIUS} fill="none" strokeWidth={2.5} className="stroke-(--track)" />
    {percent ? (
      <circle
        cx={8}
        cy={8}
        r={RING_RADIUS}
        fill="none"
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeDasharray={RING_LENGTH}
        strokeDashoffset={RING_LENGTH * (1 - percent / 100)}
        className={cn("transition-[stroke-dashoffset]", stroke)}
      />
    ) : null}
  </svg>
);

export const ContextUsageBar: FC<
  ContextUsageBarInput & { className?: string }
> = ({ className, ...input }) => {
  const state = deriveContextUsageBar(input);
  if (!state) return null;

  const { cached, cacheWrites, promptTokens, completionTokens } = input;
  const { percent, advice, face, compactFace } = state;
  const severity = getSeverityColor(percent ?? 0);
  const ring = percent !== null || compactFace === null;
  // Mono text, so widths are exact in ch. Full: face, gap, 4rem bar. Label: gap, text.
  const fullWidth = `calc(${face.length}ch${percent !== null ? " + 4.5rem" : ""})`;
  const labelWidth = `calc(${compactFace?.length ?? 0}ch${ring ? " + 0.375rem" : ""})`;
  const ringWidth = ring ? "0.875rem" : "0px";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={state.label}
          style={
            {
              "--full": fullWidth,
              "--label": labelWidth,
              "--ring": ringWidth,
              // A fixed-size track, so the header can squeeze the button down to the ring alone.
              gridTemplateColumns: `minmax(${ring ? "var(--ring)" : "var(--label)"}, var(--full))`,
            } as CSSProperties
          }
          className={cn(
            "grid shrink! items-center overflow-hidden rounded-[10px] px-2.5 font-mono text-chat-icon-fg text-ui-13 tabular-nums whitespace-nowrap transition-colors hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
            // ring and bar track
            "[--track:rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:[--track:rgb(255_255_255_/_calc(0.15*var(--contrast-wash-gain,1)))]",
            className,
          )}
        >
          {/* Exactly one of these has width: compact once the button is narrower than the full face. */}
          <span
            className="col-start-1 row-start-1 flex items-center justify-center justify-self-center overflow-hidden"
            style={{ width: "clamp(0px, (var(--full) - 100% - 1px) * 999, 100%)" }}
          >
            {ring ? <UsageRing percent={percent} stroke={severity.stroke} /> : null}
            {compactFace !== null ? (
              // Same trick: the label drops when only the ring fits.
              <span
                className="shrink-0 overflow-hidden"
                style={{ width: "clamp(0px, (100% - var(--ring) - var(--label) + 1px) * 999, var(--label))" }}
              >
                <span className={ring ? "ps-1.5" : undefined}>{compactFace}</span>
              </span>
            ) : null}
          </span>
          <span
            className="col-start-1 row-start-1 flex items-center gap-2 overflow-hidden"
            style={{ width: "clamp(0px, (100% - var(--full) + 1px) * 999, var(--full))" }}
          >
            <span>{face}</span>
            {percent !== null ? (
              <span className="h-1.5 w-16 shrink-0 overflow-hidden rounded-full bg-(--track)">
                <span
                  className={cn("block h-full rounded-full transition-all", severity.bar)}
                  style={{ width: `${percent}%` }}
                />
              </span>
            ) : null}
          </span>
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
              ) : advice === "mlx-refuses-past-limit" ? (
                <>
                  Close to the context limit. This model quantizes its cache
                  instead of capping it, so{" "}
                  <span className="font-medium">Context Length</span> is applied
                  to each request rather than to the cache: past it the request
                  is refused instead of the chat slowing down. Increase it in
                  the chat Settings panel, or shorten the conversation.
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
