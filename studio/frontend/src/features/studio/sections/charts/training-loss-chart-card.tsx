// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import type { ReactElement } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import type { ScaleMode } from "./types";
import {
  CHART_SYNC_ID,
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
  fromLog1p,
} from "./utils";

const lossConfig = {
  displayLoss: { label: "Loss", color: "#3b82f6" },
  displaySmoothed: { label: "Smoothed", color: "#f59e0b" },
} satisfies ChartConfig;

interface LossChartPoint {
  step: number;
  loss: number;
  smoothed: number;
  displayLoss: number;
  displaySmoothed: number;
}

export function TrainingLossChartCard({
  data,
  domain,
  visibleStepDomain,
  xAxisTicks,
  avgRaw,
  avgDisplay,
  showRaw,
  showSmoothed,
  showAvgLine,
  scale,
}: {
  data: LossChartPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  avgRaw: number;
  avgDisplay: number;
  showRaw: boolean;
  showSmoothed: boolean;
  showAvgLine: boolean;
  scale: ScaleMode;
}): ReactElement {
  const showPoint = data.length <= 1 ? { r: 3, strokeWidth: 0 } : false;

  return (
    <Card data-tour="studio-training-loss" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Training Loss</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={lossConfig} className={CHART_CONTAINER_CLASS}>
          <LineChart
            data={data}
            syncId={CHART_SYNC_ID}
            syncMethod="value"
            accessibilityLayer={true}
            margin={DEFAULT_CHART_MARGIN}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              type="number"
              domain={visibleStepDomain}
              ticks={xAxisTicks}
              allowDataOverflow={true}
              allowDecimals={false}
              minTickGap={28}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              fontSize={10}
              tickFormatter={(value) => formatStepTick(Number(value))}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={domain}
              allowDataOverflow={true}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickCount={5}
              fontSize={10}
              width={DEFAULT_Y_AXIS_WIDTH}
              tickFormatter={(value) => {
                const num = Number(value);
                if (!Number.isFinite(num)) {
                  return "0";
                }
                const shown = scale === "log" ? fromLog1p(num) : num;
                return formatAxisMetric(shown);
              }}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_value, payload) =>
                    `Step ${payload?.[0]?.payload?.step ?? ""}`
                  }
                  formatter={(_value, name, item) => {
                    if (name === "displaySmoothed") {
                      return [
                        formatMetric(Number(item?.payload?.smoothed)),
                        "Smoothed",
                      ];
                    }
                    return [formatMetric(Number(item?.payload?.loss)), "Loss"];
                  }}
                />
              }
            />
            {showAvgLine && (
              <ReferenceLine
                y={avgDisplay}
                stroke="#3b82f6"
                strokeDasharray="4 4"
                strokeOpacity={0.5}
                label={{
                  value: `avg ${formatMetric(avgRaw)}`,
                  position: "insideTopRight",
                  fontSize: 10,
                  fill: "#3b82f6",
                }}
              />
            )}
            {showRaw && (
              <Line
                type="linear"
                dataKey="displayLoss"
                stroke="var(--color-displayLoss)"
                strokeWidth={1.2}
                strokeOpacity={showSmoothed ? 0.35 : 1}
                dot={showPoint}
                activeDot={{ r: 3, strokeWidth: 0 }}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
            )}
            {showSmoothed && (
              <Line
                type="linear"
                dataKey="displaySmoothed"
                stroke="var(--color-displaySmoothed)"
                strokeWidth={2.2}
                dot={showPoint}
                activeDot={{ r: 3, strokeWidth: 0 }}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
            )}
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
