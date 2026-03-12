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
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
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

const gradNormConfig = {
  displayGradNorm: { label: "Grad Norm", color: "#f97316" },
} satisfies ChartConfig;

interface GradNormPoint {
  step: number;
  gradNorm: number;
  displayGradNorm: number;
}

export function GradNormChartCard({
  data,
  domain,
  visibleStepDomain,
  xAxisTicks,
  scale,
}: {
  data: GradNormPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  scale: ScaleMode;
}): ReactElement {
  const showPoint = data.length <= 1 ? { r: 3, strokeWidth: 0 } : false;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Gradient Norm</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={gradNormConfig} className={CHART_CONTAINER_CLASS}>
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
                  formatter={(_value, _name, item) => {
                    const raw = Number(item?.payload?.gradNorm);
                    return [formatMetric(raw), "Grad Norm"];
                  }}
                />
              }
            />
            <Line
              type="linear"
              dataKey="displayGradNorm"
              stroke="var(--color-displayGradNorm)"
              strokeWidth={2}
              dot={showPoint}
              activeDot={{ r: 3, strokeWidth: 0 }}
              connectNulls={true}
              strokeLinecap="round"
              strokeLinejoin="round"
              isAnimationActive={false}
            />
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
