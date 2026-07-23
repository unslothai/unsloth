// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Button } from "@/components/ui/button";

type ModelLoadDescriptionProps = {
  title?: string | null;
  message?: string | null;
  progressPercent?: number | null;
  progressLabel?: string | null;
};

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}

/**
 * Split a composed progress label like "22.8 of 122.3 GB • 330.1 MB/s • 5m 9s
 * left" into a primary chunk (next to the percent) and a secondary chunk (its
 * own row), so neither line wraps raggedly once rate/ETA appears mid-download.
 * Labels without " • " return primary-only, so the secondary row doesn't render.
 */
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
}: ModelLoadDescriptionProps) {
  const hasProgress = typeof progressPercent === "number";
  // Split once at the top so the JSX below stays flat (no IIFE).
  const { primary: labelPrimary, secondary: labelSecondary } =
    splitProgressLabel(progressLabel);

  return (
    <div className="relative flex min-h-12 w-full items-stretch gap-2">
      <div className="flex h-full shrink-0 items-center self-center">
        <Spinner className="size-3.5 text-muted-foreground" />
      </div>
      <div className="flex min-w-0 flex-1 flex-col justify-center">
        {title ? <p className="text-foreground leading-tight font-semibold">{title}</p> : null}
        {hasProgress ? (
          <div className="w-full pt-1">
            <div className="flex items-center justify-between gap-2 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/80">
              <span className="min-w-0 truncate">{labelPrimary}</span>
              <span className="shrink-0 tabular-nums">
                {Math.round(clampProgress(progressPercent))}%
              </span>
            </div>
            {labelSecondary ? (
              <div className="truncate pt-0.5 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/60">
                {labelSecondary}
              </div>
            ) : null}
            <Progress
              value={clampProgress(progressPercent)}
              className="mt-1 h-1 bg-foreground/[0.08]"
            />
          </div>
        ) : message ? (
          <p className="pt-1 text-xs leading-relaxed text-muted-foreground">{message}</p>
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
            <Progress value={clampProgress(progressPercent)} className="h-1 bg-foreground/[0.08]" />
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
