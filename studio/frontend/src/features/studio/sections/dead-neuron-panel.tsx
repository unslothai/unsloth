// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useActivationData } from "@/features/training/hooks/use-activation-data";
import type { ActivationRecord } from "@/features/training/api/train-api";
import { type ReactElement, useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// A neuron channel is "dead" if its max mean_abs across all captured steps is below this.
const DEAD_THRESHOLD = 0.01;
// A channel is "constant" if its coefficient of variation (std/mean) is below this.
const CONSTANT_CV_THRESHOLD = 0.05;

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

interface DeadNeuronPanelProps {
  isTraining: boolean;
  jobId?: string | null;
}

export function DeadNeuronPanel({ isTraining, jobId }: DeadNeuronPanelProps): ReactElement {
  const { records } = useActivationData({ isTraining, jobId });
  const stats = useMemo(() => computeLayerStats(records), [records]);

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

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-3">
          Neuron Health
          {totalDead !== null && (
            <span className="flex gap-3 text-xs font-normal text-muted-foreground">
              <span>
                <span className="text-red-500 font-medium">{totalDead}%</span> dead (avg)
              </span>
              <span>
                <span className="text-amber-500 font-medium">{totalConstant}%</span> constant (avg)
              </span>
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 pt-0">
        {stats.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
            No activation data yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={stats}
              layout="vertical"
              margin={{ top: 0, right: 12, bottom: 0, left: 28 }}
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
              />
              <YAxis
                type="category"
                dataKey="layer"
                tick={{ fontSize: 9 }}
                width={28}
                className="fill-muted-foreground"
              />
              <Tooltip
                formatter={(_v, name, item) => [
                  `${(Number(item?.value) ?? 0).toFixed(1)}%`,
                  name === "deadPct" ? "Dead neurons" : "Constant neurons",
                ]}
                contentStyle={{ fontSize: 11 }}
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
        )}
      </CardContent>
    </Card>
  );
}
