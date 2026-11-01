// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowExpandDiagonal01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { computeLayerStats, type LayerStats } from "./activation-stats";
import { ReplayControls } from "./neuron-heatmap-section";
import type { ActivationRecord } from "@/features/training";
import { type ReactElement, useMemo, useState } from "react";

interface NeuronHealthTableProps {
  stats: LayerStats[];
}

function NeuronHealthTable({ stats }: NeuronHealthTableProps): ReactElement {
  const t = useT();
  if (stats.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
        {t("studio.interpretability.noDataYet")}
      </div>
    );
  }

  // Scale bars to the max value in the data so differences are visible
  const maxPct = Math.max(...stats.map((s) => Math.max(s.deadPct, s.constantPct)), 1);

  return (
    <div className="overflow-y-auto h-full">
      <table className="w-full text-xs border-separate border-spacing-0">
        <thead className="sticky top-0 bg-card z-10">
          <tr>
            <th className="text-left font-normal text-muted-foreground pb-1 pr-2 w-8">{t("studio.interpretability.layer")}</th>
            <th className="text-left font-normal text-muted-foreground pb-1 pr-1 w-[50%]">
              <span className="text-red-500/70">{t("studio.interpretability.dead")}</span>
            </th>
            <th className="text-left font-normal text-muted-foreground pb-1 w-[50%]">
              <span className="text-amber-500/70">{t("studio.interpretability.constant")}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {stats.map((row) => (
            <tr key={row.layer} className="group">
              <td className="pr-2 py-0.5 text-muted-foreground font-mono leading-none">
                {row.layer}
              </td>
              <td className="pr-2 py-0.5">
                <div className="flex items-center gap-1.5">
                  <div className="flex-1 h-2 rounded-sm overflow-hidden bg-muted/40">
                    <div
                      className={cn(
                        "h-full rounded-sm transition-all duration-300",
                        row.deadPct > 20 ? "bg-red-500" : "bg-red-500/50",
                      )}
                      style={{ width: `${(row.deadPct / maxPct) * 100}%` }}
                    />
                  </div>
                  <span
                    className={cn(
                      "w-8 text-right tabular-nums leading-none shrink-0",
                      row.deadPct > 20
                        ? "text-red-500 font-medium"
                        : "text-muted-foreground",
                    )}
                  >
                    {row.deadPct.toFixed(1)}%
                  </span>
                </div>
              </td>
              <td className="py-0.5">
                <div className="flex items-center gap-1.5">
                  <div className="flex-1 h-2 rounded-sm overflow-hidden bg-muted/40">
                    <div
                      className={cn(
                        "h-full rounded-sm transition-all duration-300",
                        row.constantPct > 30 ? "bg-amber-500" : "bg-amber-500/50",
                      )}
                      style={{ width: `${(row.constantPct / maxPct) * 100}%` }}
                    />
                  </div>
                  <span
                    className={cn(
                      "w-8 text-right tabular-nums leading-none shrink-0",
                      row.constantPct > 30
                        ? "text-amber-500 font-medium"
                        : "text-muted-foreground",
                    )}
                  >
                    {row.constantPct.toFixed(1)}%
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="pt-1.5 text-ui-10 text-muted-foreground">
        {stats[0].fullCoverage
          ? t("studio.interpretability.health.coverageNote", { channels: stats[0].totalChannels })
          : t("studio.interpretability.health.sampledNote", { channels: stats[0].totalChannels })}
      </p>
    </div>
  );
}

interface DeadNeuronPanelProps {
  isTraining: boolean;
  records: ActivationRecord[];
  stepIndex: number;
  onStepChange: (idx: number) => void;
}

export function DeadNeuronPanel({ isTraining, records, stepIndex, onStepChange }: DeadNeuronPanelProps): ReactElement {
  const t = useT();
  const [expanded, setExpanded] = useState(false);

  // Only consider records up to the current step so the table stays in sync with the heatmap
  const visibleRecords = records.slice(0, stepIndex + 1);
  const stats = useMemo(() => computeLayerStats(visibleRecords), [visibleRecords]);

  const totalDead = useMemo(() => {
    if (stats.length === 0) return null;
    const avg = stats.reduce((s, l) => s + l.deadPct, 0) / stats.length;
    return avg.toFixed(1);
  }, [stats]);

  const totalConstant = useMemo(() => {
    if (stats.length === 0) return null;
    const avg = stats.reduce((s, l) => s + l.constantPct, 0) / stats.length;
    return avg.toFixed(1);
  }, [stats]);

  const currentStep = records[stepIndex]?.step ?? 0;

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center justify-between">
          <div className="flex items-center gap-3">
            {t("studio.interpretability.health.title")}
            {totalDead !== null && (
              <span className="flex gap-2 items-center text-xs font-normal text-muted-foreground">
                <span className="flex items-center gap-1">
                  <span className="text-red-500 font-medium">{totalDead}%</span>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="inline-flex cursor-default items-center rounded-full border px-1.5 py-0.5 text-ui-10 font-medium select-none border-red-500/40 bg-red-500/10 text-red-500"
                      >
                        {t("studio.interpretability.health.deadAvg")}
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-[calc(240px*var(--ui-space-scale,1))]">
                      <p className="font-medium mb-1">{t("studio.interpretability.health.deadTitle")}</p>
                      <p className="text-xs/relaxed font-normal">
                        {t("studio.interpretability.health.deadBody")}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </span>
                <span className="flex items-center gap-1">
                  <span className="text-amber-500 font-medium">{totalConstant}%</span>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="inline-flex cursor-default items-center rounded-full border px-1.5 py-0.5 text-ui-10 font-medium select-none border-amber-500/40 bg-amber-500/10 text-amber-500"
                      >
                        {t("studio.interpretability.health.constantAvg")}
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-[calc(240px*var(--ui-space-scale,1))]">
                      <p className="font-medium mb-1">{t("studio.interpretability.health.constantTitle")}</p>
                      <p className="text-xs/relaxed font-normal">
                        {t("studio.interpretability.health.constantBody")}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </span>
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="rounded p-1 text-muted-foreground opacity-40 transition-opacity hover:opacity-100 hover:bg-muted/60 hover:text-foreground focus:opacity-100"
            title={t("studio.interpretability.health.expand")}
            aria-label={t("studio.interpretability.health.expandAria")}
          >
            <HugeiconsIcon icon={ArrowExpandDiagonal01Icon} className="size-3.5" />
          </button>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 pt-0">
        <NeuronHealthTable stats={stats} />
      </CardContent>

      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent className="w-[90vw] max-w-none sm:max-w-none">
          <DialogHeader>
            <DialogTitle className="font-bold">{t("studio.interpretability.health.title")}</DialogTitle>
          </DialogHeader>
          <div className="mt-2 w-full flex flex-col gap-3">
            <div style={{ height: Math.min(window.innerHeight * 0.65, 520) }}>
              <NeuronHealthTable stats={stats} />
            </div>
            {!isTraining && records.length > 1 && (
              <ReplayControls
                stepIndex={stepIndex}
                totalSteps={records.length}
                onStepChange={onStepChange}
                currentStep={currentStep}
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
