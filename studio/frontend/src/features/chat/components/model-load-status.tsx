// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Button } from "@/components/ui/button";

type ModelLoadDescriptionProps = {
  message?: string | null;
  progressPercent?: number | null;
  progressLabel?: string | null;
  onStop?: () => void;
};

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export function ModelLoadDescription({
  message,
  progressPercent,
  progressLabel,
  onStop,
}: ModelLoadDescriptionProps) {
  const hasProgress = typeof progressPercent === "number";

  return (
    <div className="flex items-center gap-1.5">
      <div className="min-w-0 flex-1">
        {hasProgress ? (
          <div className="w-[12.5rem] max-w-full">
            <div className="flex items-center justify-between text-[10px] font-medium tracking-[0.08em] text-muted-foreground/80">
              <span>{progressLabel}</span>
              <span>{Math.round(clampProgress(progressPercent))}%</span>
            </div>
            <Progress value={clampProgress(progressPercent)} className="h-1 bg-foreground/[0.08]" />
          </div>
        ) : message ? (
          <p className="text-xs leading-relaxed text-muted-foreground">{message}</p>
        ) : null}
      </div>
      {onStop ? (
        <Button
          type="button"
          size="xs"
          variant="outline"
          className="h-5 shrink-0 px-2 text-[10px]"
          onClick={onStop}
        >
          Stop
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
