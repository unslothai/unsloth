// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ModelLoadDescriptionProps = {
  title?: string | null;
  message?: string | null;
  progressPercent?: number | null;
  progressLabel?: string | null;
  // Extra classes for the root row (e.g. a titleless caller dropping min-h-12).
  className?: string;
  /** "floating": the Images / Video progress card. Ring spinner, brighter text on its dark card. */
  variant?: "default" | "floating";
};

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}

/** Split a composed progress label into a primary chunk, next to the percent, and a secondary chunk
 *  on its own row, so neither line wraps raggedly once rate and ETA appear mid-download. Labels
 *  without the separator return primary-only, so the secondary row does not render. */
function splitProgressLabel(
  label: string | null | undefined,
): { primary: string; secondary: string } {
  if (!label) return { primary: "", secondary: "" };
  const idx = label.indexOf(" \u2022 ");
  if (idx < 0) return { primary: label, secondary: "" };
  return {
    primary: label.slice(0, idx),
    secondary: label.slice(idx + 3),
  };
}

export function ModelLoadDescription({
  title,
  message,
  progressPercent,
  progressLabel,
  className,
  variant = "default",
}: ModelLoadDescriptionProps) {
  const hasProgress = typeof progressPercent === "number";
  const floating = variant === "floating";
  // Split once at the top so the JSX below stays flat (no IIFE).
  const { primary: labelPrimary, secondary: labelSecondary } =
    splitProgressLabel(progressLabel);

  return (
    <div className={cn("relative flex min-h-12 w-full items-stretch gap-2", className)}>
      <div className="flex h-full shrink-0 items-center self-center">
        <Spinner
          variant={floating ? "ring" : "arc"}
          className={cn(
            "size-3.5 text-muted-foreground",
            floating && "size-4 dark:text-foreground/70",
          )}
        />
      </div>
      <div className="flex min-w-0 flex-1 flex-col justify-center">
        {title ? <p className="text-foreground leading-tight font-semibold">{title}</p> : null}
        {hasProgress ? (
          <div className="w-full pt-1">
            <div
              className={cn(
                "flex items-center justify-between gap-2 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/80",
                floating && "dark:text-foreground/85",
              )}
            >
              <span className="min-w-0 truncate">{labelPrimary}</span>
              <span className="shrink-0 tabular-nums">
                {Math.round(clampProgress(progressPercent))}%
              </span>
            </div>
            {labelSecondary ? (
              <div
                className={cn(
                  "truncate pt-0.5 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/60",
                  floating && "dark:text-foreground/65",
                )}
              >
                {labelSecondary}
              </div>
            ) : null}
            <Progress
              value={clampProgress(progressPercent)}
              className={cn(
                "mt-1 h-1 bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]",
                // The lighter card would swallow the 8% track.
                floating &&
                  "dark:bg-[color-mix(in_oklab,var(--foreground)_calc(14%*var(--contrast-wash-gain,1)),transparent)]",
              )}
            />
          </div>
        ) : message ? (
          <p
            className={cn(
              "pt-1 text-xs leading-relaxed text-muted-foreground",
              floating && "dark:text-foreground/85",
            )}
          >
            {message}
          </p>
        ) : null}
      </div>
    </div>
  );
}

type ModelLoadInlineStatusProps = {
  label: string;
  title: string;
  progressPercent?: number | null;
  progressLabel?: string | null;
  onStop?: () => void;
};

export function ModelLoadInlineStatus({
  label,
  title,
  progressPercent,
  progressLabel,
  onStop,
}: ModelLoadInlineStatusProps) {
  const hasProgress = typeof progressPercent === "number";

  return (
    <div className="flex min-w-[20rem] items-center gap-2.5 text-muted-foreground" title={title}>
      <div className="flex items-center gap-1.5 shrink-0">
        <Spinner className="size-3.5 shrink-0" />
        <span className="text-xs">{label}</span>
      </div>
      {hasProgress ? (
        <div className="flex min-w-0 flex-[1.35] items-center gap-2.5">
          <div className="min-w-[7rem] flex-1">
            <Progress value={clampProgress(progressPercent)} className="h-1 bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]" />
          </div>
          <div
            className="flex shrink-0 items-center gap-1 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/80"
            title={progressLabel ?? undefined}
          >
            {/* Tight inline layout: show only the primary (bytes) chunk;
                full label (rate/ETA) stays in the tooltip. */}
            <span>{splitProgressLabel(progressLabel).primary}</span>
            <span className="tabular-nums">
              {Math.round(clampProgress(progressPercent))}%
            </span>
          </div>
        </div>
      ) : null}
      {onStop ? (
        <Button
          type="button"
          size="xs"
          variant="outline"
          className="shrink-0 text-ui-11"
          onClick={onStop}
        >
          Stop
        </Button>
      ) : null}
    </div>
  );
}
