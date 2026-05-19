// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ActivationRecord } from "@/features/training/api/train-api";
import { type ReactElement, useMemo, useSyncExternalStore } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { computeTrendData } from "./activation-stats";

function subscribe(cb: () => void): () => void {
  const observer = new MutationObserver(cb);
  observer.observe(document.documentElement, { attributeFilter: ["class"] });
  return () => observer.disconnect();
}
function useIsDark(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => document.documentElement.classList.contains("dark"),
    () => false,
  );
}

export interface NeuronHealthTrendProps {
  isTraining: boolean;
  records: ActivationRecord[];
  stepIndex: number;
  onStepChange: (idx: number) => void;
}

const DEAD_COLOR    = "rgb(96,165,250)";   // blue-400
const CONSTANT_COLOR = "rgb(251,146,60)";  // orange-400

export function NeuronHealthTrend({
  records,
  stepIndex,
  onStepChange,
}: NeuronHealthTrendProps): ReactElement {
  const isDark = useIsDark();

  const trendData = useMemo(() => computeTrendData(records), [records]);

  const currentStep = records[stepIndex]?.step ?? 0;

  const tooltipStyle = {
    fontSize: 11,
    borderRadius: 8,
    background: isDark ? "hsl(var(--popover))" : "hsl(var(--popover))",
    color: isDark ? "#fff" : "#000",
    border: `1px solid hsl(var(--border))`,
    padding: "6px 10px",
  };

  if (records.length < 2) {
    return (
      <Card className="flex flex-col h-full">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Neuron Health Trend</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <p className="text-xs text-muted-foreground text-center max-w-[180px]">
            Need at least 2 captured steps to show trend
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-3">
          Neuron Health Trend
          <span className="flex gap-3 text-[10px] font-normal text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-3 rounded-sm" style={{ backgroundColor: DEAD_COLOR }} />
              Dead
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-3 rounded-sm" style={{ backgroundColor: CONSTANT_COLOR }} />
              Constant
            </span>
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 pt-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={trendData}
            margin={{ top: 4, right: 12, bottom: 18, left: 0 }}
            onClick={(data) => {
              const idx = (data as { activeTooltipIndex?: number })?.activeTooltipIndex;
              if (idx != null) onStepChange(idx);
            }}
            style={{ cursor: "crosshair" }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10 }}
              className="fill-muted-foreground"
              label={{
                value: "Step →",
                position: "insideBottom",
                offset: -2,
                fontSize: 10,
                fill: "rgba(120,120,120,0.75)",
              }}
            />
            <YAxis
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              tick={{ fontSize: 10 }}
              width={36}
              className="fill-muted-foreground"
              domain={[0, "auto"]}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              wrapperStyle={{ outline: "none" }}
              formatter={(value: number | undefined, name: string | undefined) => [
                `${(value ?? 0).toFixed(1)}%`,
                name === "deadPct" ? "Dead" : "Constant",
              ]}
              labelFormatter={(label: unknown) => `Step ${label}`}
            />
            {/* Cursor line at current step */}
            <ReferenceLine
              x={currentStep}
              stroke={isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.2)"}
              strokeWidth={1.5}
              strokeDasharray="4 2"
            />
            <Line
              type="monotone"
              dataKey="deadPct"
              stroke={DEAD_COLOR}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="constantPct"
              stroke={CONSTANT_COLOR}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
