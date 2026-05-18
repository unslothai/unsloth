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
import { cn } from "@/lib/utils";
import { ArrowExpandDiagonal01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ReplayControls } from "./neuron-heatmap-section";
import type { ActivationRecord } from "@/features/training/api/train-api";
import { type ReactElement, useMemo, useState, useSyncExternalStore } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

// A neuron channel is "dead" if its max mean_abs across all captured steps is below this.
const DEAD_THRESHOLD = 0.01;
// A channel is "constant" if its coefficient of variation (std/mean) is below this.
const CONSTANT_CV_THRESHOLD = 0.05;

function subscribe(cb: () => void): () => void {
  const observer = new MutationObserver(cb);
  observer.observe(document.documentElement, { attributeFilter: ["class"] });
  return () => observer.disconnect();
}
function getSnapshot(): boolean {
  return document.documentElement.classList.contains("dark");
}
function useIsDark(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

interface LayerStats {
  layer: string;
  deadPct: number;
  constantPct: number;
  totalChannels: number;
}

function computeLayerStats(records: ActivationRecord[]): LayerStats[] {
  if (records.length === 0) return [];

  // Collect all layer keys across all records
  const layerKeys = new Set<string>();
  for (const rec of records) {
    for (const k of Object.keys(rec.layers)) layerKeys.add(k);
  }

  const sorted = [...layerKeys].sort((a, b) => Number(a) - Number(b));

  return sorted.map((layerKey) => {
    // Build channel → [mean_abs values across steps]
    const channelSeries = new Map<number, number[]>();

    for (const rec of records) {
      const layer = rec.layers[layerKey];
      if (!layer) continue;
      layer.mean_abs.forEach((val, idx) => {
        if (!channelSeries.has(idx)) channelSeries.set(idx, []);
        channelSeries.get(idx)!.push(val);
      });
    }

    const totalChannels = channelSeries.size;
    let deadCount = 0;
    let constantCount = 0;

    for (const vals of channelSeries.values()) {
      const maxVal = Math.max(...vals);
      if (maxVal < DEAD_THRESHOLD) {
        deadCount++;
        continue; // dead implies constant too, don't double-count
      }
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      const std = Math.sqrt(
        vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length,
      );
      const cv = mean > 0 ? std / mean : 0;
      if (cv < CONSTANT_CV_THRESHOLD) constantCount++;
    }

    return {
      layer: `L${layerKey}`,
      deadPct: totalChannels > 0 ? (deadCount / totalChannels) * 100 : 0,
      constantPct: totalChannels > 0 ? (constantCount / totalChannels) * 100 : 0,
      totalChannels,
    };
  });
}

interface NeuronHealthChartProps {
  stats: LayerStats[];
  isDark: boolean;
}

function NeuronHealthChart({ stats, isDark }: NeuronHealthChartProps): ReactElement {
  const tooltipStyle = {
    fontSize: 11,
    borderRadius: 8,
    background: isDark ? "#000" : "#fff",
    color: isDark ? "#fff" : "#000",
    border: `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)"}`,
    padding: "6px 10px",
  };

  if (stats.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
        No activation data yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={stats}
        layout="vertical"
        margin={{ top: 0, right: 12, bottom: 18, left: 28 }}
        barSize={6}
        barGap={2}
      >
        <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-border" />
        <XAxis
          type="number"
          domain={[0, 100]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fontSize: 10 }}
          className="fill-muted-foreground"
          label={{ value: "% of channels", position: "insideBottom", offset: -2, fontSize: 10, fill: "rgba(120,120,120,0.75)" }}
        />
        <YAxis
          type="category"
          dataKey="layer"
          tick={{ fontSize: 9 }}
          width={28}
          className="fill-muted-foreground"
          label={{ value: "Layer", angle: -90, position: "insideLeft", offset: 12, style: { fontSize: 10, fill: "rgba(120,120,120,0.75)", textAnchor: "middle" } }}
        />
        <RechartsTooltip
          formatter={(_v, name, item) => [
            `${(Number(item?.value) ?? 0).toFixed(1)}%`,
            name === "deadPct" ? "Dead neurons" : "Constant neurons",
          ]}
          contentStyle={tooltipStyle}
          wrapperStyle={{ outline: "none" }}
          cursor={{ fill: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)" }}
        />
        <Bar dataKey="deadPct" name="deadPct" radius={[0, 2, 2, 0]}>
          {stats.map((entry) => (
            <Cell
              key={entry.layer}
              fill={entry.deadPct > 20 ? "rgb(239,68,68)" : "rgb(239,68,68,0.6)"}
            />
          ))}
        </Bar>
        <Bar dataKey="constantPct" name="constantPct" radius={[0, 2, 2, 0]}>
          {stats.map((entry) => (
            <Cell
              key={entry.layer}
              fill={entry.constantPct > 30 ? "rgb(245,158,11)" : "rgb(245,158,11,0.55)"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

interface DeadNeuronPanelProps {
  isTraining: boolean;
  records: ActivationRecord[];
  stepIndex: number;
  onStepChange: (idx: number) => void;
}

export function DeadNeuronPanel({ isTraining, records, stepIndex, onStepChange }: DeadNeuronPanelProps): ReactElement {
  const isDark = useIsDark();
  const [expanded, setExpanded] = useState(false);

  // Only consider records up to the current step so the chart stays in sync with the heatmap
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
            Neuron Health
            {totalDead !== null && (
              <span className="flex gap-2 items-center text-xs font-normal text-muted-foreground">
                <span className="flex items-center gap-1">
                  <span className="text-red-500 font-medium">{totalDead}%</span>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="inline-flex cursor-default items-center rounded-full border px-1.5 py-0.5 text-[10px] font-medium select-none border-red-500/40 bg-red-500/10 text-red-500"
                      >
                        dead (avg)
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className={cn("max-w-[240px]", isDark ? "dark" : "light")}>
                      <p className="font-medium mb-1">Dead neurons</p>
                      <p className="text-xs/relaxed font-normal">
                        A channel is dead when its mean absolute activation stays below 0.01 across all
                        captured steps. Dead neurons contribute nothing to model output — often caused
                        by too-high learning rates, poor initialisation, or excessive regularisation.
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
                        className="inline-flex cursor-default items-center rounded-full border px-1.5 py-0.5 text-[10px] font-medium select-none border-amber-500/40 bg-amber-500/10 text-amber-500"
                      >
                        constant (avg)
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className={cn("max-w-[240px]", isDark ? "dark" : "light")}>
                      <p className="font-medium mb-1">Constant neurons</p>
                      <p className="text-xs/relaxed font-normal">
                        A channel is constant when its activation barely changes over training steps
                        (coefficient of variation &lt; 5%). Unlike dead neurons these do fire, but carry
                        no dynamic information — limiting the model&apos;s expressive capacity.
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
            title="Expand chart"
            aria-label="Expand Neuron Health chart"
          >
            <HugeiconsIcon icon={ArrowExpandDiagonal01Icon} className="size-3.5" />
          </button>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 pt-0">
        <NeuronHealthChart stats={stats} isDark={isDark} />
      </CardContent>

      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent className="w-[90vw] max-w-none sm:max-w-none">
          <DialogHeader>
            <DialogTitle className="font-bold">Neuron Health</DialogTitle>
          </DialogHeader>
          <div className="mt-2 w-full flex flex-col gap-3">
            <div style={{ height: Math.min(window.innerHeight * 0.55, 480) }}>
              <NeuronHealthChart stats={stats} isDark={isDark} />
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
