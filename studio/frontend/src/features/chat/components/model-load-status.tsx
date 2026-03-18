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
  onStop?: () => void;
};

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export function ModelLoadDescription({
  title,
  message,
  progressPercent,
  progressLabel,
  onStop,
}: ModelLoadDescriptionProps) {
  const hasProgress = typeof progressPercent === "number";

  return (
    <div className="relative flex min-h-12 w-full items-stretch gap-2">
      <div className="flex h-full shrink-0 items-center self-center">
        <Spinner className="size-4 text-foreground" />
      </div>
      <div className="min-w-0 flex-1 pr-5">
        {title ? <p className="text-foreground leading-5 font-semibold">{title}</p> : null}
        {hasProgress ? (
          <div className="w-full pt-1">
            <div className="flex items-center justify-between text-[10px] font-medium tracking-[0.08em] text-muted-foreground/80">
              <span>{progressLabel}</span>
              <span>{Math.round(clampProgress(progressPercent))}%</span>
            </div>
            <Progress value={clampProgress(progressPercent)} className="h-1 bg-foreground/[0.08]" />
          </div>
        ) : message ? (
          <p className="pt-1 text-xs leading-relaxed text-muted-foreground">{message}</p>
        ) : null}
      </div>
      {onStop ? (
        <Button
          type="button"
          size="xs"
          variant="ghost"
          aria-label="Stop model loading"
          className="h-auto self-stretch shrink-0 !rounded-none !border-0 bg-transparent px-1 text-[10px] text-muted-foreground hover:bg-transparent hover:text-destructive focus-visible:text-destructive"
          onClick={onStop}
        >
          Cancel
        </Button>
      ) : null}
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
          <div className="flex shrink-0 items-center gap-1 text-[10px] font-medium tracking-[0.08em] text-muted-foreground/80">
            <span>{progressLabel}</span>
            <span>{Math.round(clampProgress(progressPercent))}%</span>
          </div>
        </div>
      ) : null}
      {onStop ? (
        <Button
          type="button"
          size="xs"
          variant="outline"
          className="shrink-0 text-[11px]"
          onClick={onStop}
        >
          Stop
        </Button>
      ) : null}
    </div>
  );
}
