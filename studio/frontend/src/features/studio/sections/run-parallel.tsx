// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  PARALLEL_MODE_LABEL_KEYS,
  runGpuCount,
} from "@/features/training/lib/parallel-plan";
import type { RunParallelInfo } from "@/features/training/types/parallel";
import type { GpuUtilization } from "@/hooks/use-gpu-utilization";
import { useLocale, useT } from "@/i18n";
import type { ReactElement } from "react";

export function RunParallelChip({
  parallel,
}: {
  parallel: RunParallelInfo;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const mode = t(PARALLEL_MODE_LABEL_KEYS[parallel.mode]);
  const count = runGpuCount(parallel);
  return (
    <span
      className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10 font-medium text-foreground/80"
      data-testid="run-parallel-chip"
    >
      {count === 1
        ? mode
        : t("studio.gpus.runChip", {
            mode,
            count: new Intl.NumberFormat(locale).format(count),
          })}
    </span>
  );
}

/** One line per GPU of the run, so a rank running out of memory is visible without switching the
 * monitor's GPU picker. Ranks follow `gpu_ids` order (rank r = r-th GPU of the run). */
export function RankMemoryStrip({
  parallel,
  devices,
}: {
  parallel: RunParallelInfo;
  devices: readonly GpuUtilization[];
}): ReactElement | null {
  const t = useT();
  if (parallel.gpuIds.length < 2) return null;
  const multiProcess =
    parallel.mode !== "layer_split" && parallel.mode !== "single";
  return (
    <div className="flex flex-col gap-1.5" data-testid="rank-memory-strip">
      {parallel.gpuIds.map((gpuIndex, rank) => {
        const device = devices.find((d) => d.index === gpuIndex);
        const used = device?.vram_used_gb ?? null;
        const total = device?.vram_total_gb ?? null;
        const pct =
          used !== null && total !== null && total > 0
            ? Math.min(100, Math.max(0, (used / total) * 100))
            : 0;
        return (
          <div key={gpuIndex} className="flex items-center gap-2.5 text-ui-11">
            <span className="w-24 shrink-0 truncate text-muted-foreground">
              {multiProcess
                ? t("studio.gpus.rankGpu", { rank, index: gpuIndex })
                : t("studio.gpus.gpuIndex", { index: gpuIndex })}
            </span>
            <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-muted/80">
              <div
                className={`h-full rounded-full transition-all duration-300 ${pct >= 95 ? "bg-red-500" : pct >= 60 ? "bg-amber-500" : "bg-control-accent"}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="w-28 shrink-0 text-right font-mono tabular-nums text-muted-foreground">
              {used !== null && total !== null
                ? `${used} / ${total} GiB`
                : "--"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
