// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The live run card, drawn like Config sweeps' so both tabs read the same while a run is on.

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useT, type TranslationKey } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ChartAverageIcon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { getBenchmarkLogLineClass } from "../lib/log-style";
import { useBenchmarkRuntimeStore } from "../stores/benchmark-runtime-store";

const PHASE_LABEL_KEY: Record<string, TranslationKey> = {
  starting: "benchmark.runPanel.starting",
  running: "benchmark.runPanel.running",
  success: "benchmark.runPanel.complete",
  error: "benchmark.runPanel.failed",
  canceled: "benchmark.runPanel.cancelled",
};

const BADGE: Record<string, string> = {
  active: "bg-muted text-foreground",
  success:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  error: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400",
  canceled: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400",
};

function formatDuration(s: string | null | undefined): string {
  if (!s) return "--";
  const parts = s.split(":").map(Number);
  if (parts.length === 2) {
    const [m, sec] = parts;
    return m === 0 ? `${sec}s` : `${m}m ${sec}s`;
  }
  if (parts.length === 3) {
    const [h, m, sec] = parts;
    return h === 0 ? `${m}m ${sec}s` : `${h}h ${m}m ${sec}s`;
  }
  return s;
}

export function BenchmarkRunPanel({
  task,
  onClose,
}: {
  task?: string | null;
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
  const [showLog, setShowLog] = useState(false);

  const isTerminal =
    phase === "success" || phase === "error" || phase === "canceled";
  const isActive = phase === "starting" || phase === "running";
  const phaseLabel = isTerminal
    ? PHASE_LABEL_KEY[phase]
    : progress > 0
      ? PHASE_LABEL_KEY.running
      : PHASE_LABEL_KEY.starting;
  const lines = logLines.filter(
    (e) => e.stream === "stdout" || e.stream === "stderr",
  );

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title={task ? `Evals · ${task}` : "Evals"}
      description={stage || t(phaseLabel)}
      className="shadow-border border border-border/60 bg-card/90 ring-0"
      headerAction={
        isActive ? (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => void requestCancel()}
            className="h-8 rounded-full px-3.5 text-xs shadow-sm"
          >
            <HugeiconsIcon icon={StopIcon} className="size-3" />
            {t("common.cancel")}
          </Button>
        ) : isTerminal ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 rounded-full px-3.5 text-xs"
          >
            {t("common.close")}
          </Button>
        ) : undefined
      }
    >
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={cn(
              "rounded-full px-2.5 py-1 text-ui-10 font-semibold",
              BADGE[isActive ? "active" : phase] ?? BADGE.active,
            )}
          >
            {t(phaseLabel)}
          </span>
          {progressDetail && (
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10 font-medium tabular-nums text-foreground/80">
              {progressDetail.current} of {progressDetail.total}
            </span>
          )}
          {progressDetail && (
            <span className="text-ui-10 tabular-nums text-muted-foreground">
              {formatDuration(progressDetail.elapsed)} elapsed
              {isActive && ` · ${formatDuration(progressDetail.eta)} left`}
            </span>
          )}
        </div>
        {isActive && (
          <>
            <div className="flex justify-end text-xs tabular-nums text-muted-foreground">
              {Math.round(progress)}%
            </div>
            <Progress
              value={progress}
              className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
            />
          </>
        )}
        {error && (
          <p className="rounded-xl bg-destructive/10 px-3 py-2 text-ui-12 text-destructive">
            {error}
          </p>
        )}
        {lines.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => setShowLog((v) => !v)}
              className="mr-auto flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-11 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
            >
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                strokeWidth={1.75}
                className={cn(
                  "size-3.5 transition-transform",
                  showLog && "rotate-180",
                )}
              />
              {showLog ? "Hide log" : `Show log (${lines.length} lines)`}
            </button>
            {showLog && (
              <div className="max-h-48 overflow-y-auto rounded-xl bg-black/80 p-3 font-mono text-xs leading-relaxed">
                {lines.map((entry, i) => (
                  <div key={i} className={getBenchmarkLogLineClass(entry)}>
                    {entry.line}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </SectionCard>
  );
}
