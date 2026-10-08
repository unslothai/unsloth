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

// Outer edge matches the header's icon glyphs.
const RING_RADIUS = 6.25;
const RING_LENGTH = 2 * Math.PI * RING_RADIUS;

const UsageRing: FC<{ percent: number | null; stroke: string }> = ({
  percent,
  stroke,
}) => (
  <svg viewBox="0 0 16 16" aria-hidden={true} className="size-icon shrink-0 -rotate-90">
    <circle cx={8} cy={8} r={RING_RADIUS} fill="none" strokeWidth={2} className="stroke-(--track)" />
    {percent ? (
      <circle
        cx={8}
        cy={8}
        r={RING_RADIUS}
        fill="none"
        strokeWidth={2}
        strokeLinecap="round"
        strokeDasharray={RING_LENGTH}
        strokeDashoffset={RING_LENGTH * (1 - percent / 100)}
        className={cn("transition-[stroke-dashoffset]", stroke)}
      />
    ) : null}
  </svg>
);

const DOT_TRIM_CH = 0.15;

// Visible gap each side of the slash, less each glyph's side bearing (JetBrains Mono, in ch).
const SLASH_GAP_CH = 0.55;
function slashMargins(before: string, after: string): [number, number] {
  const beforeBearing = before === "k" ? 0.03 : before === "—" ? 0 : 0.11;
  const afterBearing = after === "1" ? 0.17 : 0.11;
  return [SLASH_GAP_CH - beforeBearing - 0.08, SLASH_GAP_CH - 0.12 - afterBearing];
}

// " / " becomes a slash with slashMargins either side, and each "." is pulled in by DOT_TRIM_CH.
function faceWidthCh(face: string): number {
  const dots = face.split(".").length - 1;
  const at = face.indexOf(" / ");
  const [left, right] = at >= 0 ? slashMargins(face[at - 1] ?? "", face[at + 3] ?? "") : [1, 1];
  return face.length + left + right - 2 - dots * 2 * DOT_TRIM_CH;
}

const TokenCount: FC<{ text: string }> = ({ text }) => (
  <>
    {text.split(".").map((part, i) => (
      <span key={i}>
        {/* 0.85em dot, margins in its own ch so it still takes 1 - 2 * DOT_TRIM_CH of the parent's. */}
        {i > 0 ? <span className="-mx-[0.088ch] text-[0.85em]">.</span> : null}
        {part}
      </span>
    ))}
  </>
);

const Face: FC<{ face: string }> = ({ face }) => {
  const slash = face.indexOf(" / ");
  if (slash < 0) return <span><TokenCount text={face} /></span>;
  const [left, right] = slashMargins(face[slash - 1] ?? "", face[slash + 3] ?? "");
  return (
    <span>
      <TokenCount text={face.slice(0, slash)} />
      <span style={{ marginLeft: `${left}ch`, marginRight: `${right}ch` }}>/</span>
      <TokenCount text={face.slice(slash + 3)} />
    </span>
  );
};

export const ContextUsageBar: FC<
  ContextUsageBarInput & { className?: string }
> = ({ className, ...input }) => {
  const state = deriveContextUsageBar(input);
  if (!state) return null;

  const { cached, cacheWrites, promptTokens, completionTokens } = input.estimated
    ? {}
    : input;
  const { percent, advice, face, compactFace } = state;
  const severity = getSeverityColor(percent ?? 0);
  // The ring shows whenever there is a window to fill, even before anything is counted.
  const showRing = compactFace === null;
  // Mono text, so widths are exact in ch. Full: padding, face, and the gap and ring. Compact: an icon button.
  const fullWidth = showRing
    ? `calc(${faceWidthCh(face)}ch + var(--icon-size) + 7 * var(--spacing))`
    : `calc(${faceWidthCh(face)}ch + 5 * var(--spacing))`;
  const compactWidth =
    compactFace === null ? "calc(30px * var(--ui-space-scale, 1))" : `calc(${faceWidthCh(compactFace)}ch + 5 * var(--spacing))`;
  const hover = "rounded-[10px] transition-colors group-hover:bg-chat-icon-bg-hover";

  return (
    <Tooltip>
      {/* The header squeezes this wrapper; the button inside takes only the face it shows. */}
      <div
        style={
          {
            "--full": fullWidth,
            "--compact": compactWidth,
            gridTemplateColumns: "minmax(var(--compact), var(--full))",
          } as CSSProperties
        }
        // Mono here too, so ch in the widths resolves the same as in the button.
        className={cn("grid shrink! items-center font-mono text-ui-13", className)}
      >
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={state.label}
            // The full face where it fits, else the compact one, right-aligned against the icons.
            style={{
              width:
                "calc(clamp(0px, (100% - var(--full) + 1px) * 999, var(--full)) + clamp(0px, (var(--full) - 100% - 1px) * 999, var(--compact)))",
            }}
            className={cn(
              "group grid h-full items-center justify-self-end overflow-hidden rounded-[10px] text-chat-icon-fg tabular-nums whitespace-nowrap hover:text-chat-icon-fg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              // ring track
              "[--track:rgb(0_0_0_/_calc(0.14*var(--contrast-wash-gain,1)))] dark:[--track:rgb(255_255_255_/_calc(0.2*var(--contrast-wash-gain,1)))]",
            )}
          >
            {/* Exactly one of these has width, matching the button's. */}
            <span
              className="col-start-1 row-start-1 flex h-full items-center justify-center overflow-hidden"
              style={{ width: "clamp(0px, (var(--full) - 100% - 1px) * 999, 100%)" }}
            >
              {compactFace === null ? (
                <span className={cn("flex size-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center justify-center", hover)}>
                  <UsageRing percent={percent} stroke={severity.stroke} />
                </span>
              ) : (
                <span className={cn("flex h-full shrink-0 items-center px-2.5", hover)}><Face face={compactFace} /></span>
              )}
            </span>
            <span
              className="col-start-1 row-start-1 h-full overflow-hidden"
              style={{ width: "clamp(0px, (100% - var(--full) + 1px) * 999, var(--full))" }}
            >
              <span className={cn("flex h-full w-(--full) items-center gap-2 px-2.5", hover)}>
                <Face face={face} />
                {showRing ? <UsageRing percent={percent} stroke={severity.stroke} /> : null}
              </span>
            </span>
          </button>
        </TooltipTrigger>
      </div>
      <TooltipContent
        side="bottom"
        sideOffset={8}
        variant="rich"
        className="[&_span>svg]:hidden!"
      >
        <div className="grid min-w-44 gap-1.5 text-xs">
          {percent !== null ? (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">
                {input.estimated ? "Estimated context usage" : "Context usage"}
              </span>
              <span className={cn("font-mono tabular-nums font-medium", severity.text)}>
                {input.estimated ? "~" : ""}
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
          {input.compactionCount !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Automatic compactions</span>
              <span className="font-mono tabular-nums">
                {input.compactionCount}
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
          {input.estimated ? (
            <div className="mt-1 max-w-64 text-ui-11 leading-snug text-muted-foreground/90">
              Estimated from the chat&apos;s text, without attachments. A
              loaded model replaces it with an exact count.
            </div>
          ) : null}
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
