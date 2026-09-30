// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { useT, type TranslationKey } from "@/i18n";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  AlertCircleIcon,
  CheckmarkCircle01Icon,
  CancelCircleIcon,
} from "@hugeicons/core-free-icons";
import { getBenchmarkLogLineClass } from "../lib/log-style";
import { useBenchmarkRuntimeStore } from "../stores/benchmark-runtime-store";

const PHASE_LABEL_KEY: Record<string, TranslationKey> = {
  starting: "benchmark.runPanel.starting",
  running: "benchmark.runPanel.running",
  success: "benchmark.runPanel.complete",
  error: "benchmark.runPanel.failed",
  canceled: "benchmark.runPanel.cancelled",
};

function formatDuration(s: string | null | undefined): string {
  if (!s) return "--";
  const parts = s.split(":").map(Number);
  if (parts.length === 2) {
    const [m, sec] = parts;
    if (m === 0) return `${sec}s`;
    return `${m}m ${sec}s`;
  }
  if (parts.length === 3) {
    const [h, m, sec] = parts;
    if (h === 0) return `${m}m ${sec}s`;
    return `${h}h ${m}m ${sec}s`;
  }
  return s;
}

export function BenchmarkRunPanel({
  onClose,
}: {
  onClose: () => void;
}) {
  const t = useT();
  const phase = useBenchmarkRuntimeStore((s) => s.phase);
  const error = useBenchmarkRuntimeStore((s) => s.error);
  const stage = useBenchmarkRuntimeStore((s) => s.stage);
  const logLines = useBenchmarkRuntimeStore((s) => s.logLines);
  const requestCancel = useBenchmarkRuntimeStore((s) => s.requestCancel);
  const progress = useBenchmarkRuntimeStore((s) => s.progressPercent);
  const progressDetail = useBenchmarkRuntimeStore((s) => s.progressDetail);

  const isTerminal = phase === "success" || phase === "error" || phase === "canceled";
  const isActive = phase === "starting" || phase === "running";

  const phaseLabel = isTerminal
    ? PHASE_LABEL_KEY[phase]
    : progress > 0
      ? PHASE_LABEL_KEY.running
      : PHASE_LABEL_KEY.starting;

  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isActive && <Spinner className="size-4" />}
          {phase === "success" && (
            <HugeiconsIcon icon={CheckmarkCircle01Icon} className="size-4 text-green-500" />
          )}
          {phase === "error" && (
            <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-red-500" />
          )}
          {phase === "canceled" && (
            <HugeiconsIcon icon={CancelCircleIcon} className="size-4 text-muted-foreground" />
          )}
          <span className="text-sm font-medium">
            {t(phaseLabel)}
          </span>
          {isActive && (
            <>
              {progressDetail && (
                <span className="rounded-full border border-border/60 px-2.5 py-1 text-[10px] font-medium tabular-nums text-muted-foreground">
                  {progressDetail.current}/{progressDetail.total}
                </span>
              )}
              <span className="rounded-full border border-border/60 px-2.5 py-1 text-[10px] font-medium tabular-nums text-muted-foreground">
                {progress.toFixed(1)}%
              </span>
              {progressDetail && (
                <span className="text-[10px] tabular-nums text-muted-foreground/70">
                  {formatDuration(progressDetail.elapsed)}
                </span>
              )}
              {progressDetail && (
                <span className="text-[10px] tabular-nums text-muted-foreground/50">
                  eta {formatDuration(progressDetail.eta)}
                </span>
              )}
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isActive && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void requestCancel()}
            >
              {t("common.cancel")}
            </Button>
          )}
          {isTerminal && (
            <Button type="button" variant="outline" size="sm" onClick={onClose}>
              {t("common.close")}
            </Button>
          )}
        </div>
      </div>

      {isActive && (
        <Progress
          value={progress}
          className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
          indicatorClassName={undefined}
        />
      )}

      {!isActive && stage && (
        <p className="text-xs text-muted-foreground">{stage}</p>
      )}

      {logLines.filter((e) => e.stream === "stdout" || e.stream === "stderr").length > 0 && (
        <div className="max-h-48 overflow-y-auto rounded-lg bg-black/80 p-3 font-mono text-xs leading-relaxed">
          {logLines.filter((e) => e.stream === "stdout" || e.stream === "stderr").map((entry, i) => (
            <div
              key={i}
              className={getBenchmarkLogLineClass(entry)}
            >
              {entry.line}
            </div>
          ))}
        </div>
      )}

      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}
    </div>
  );
}
