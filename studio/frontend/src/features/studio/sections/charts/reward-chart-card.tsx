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
import { useT } from "@/i18n";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  CHART_CONTAINER_CLASS,
  CHART_SYNC_ID,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
} from "./utils";

interface RewardPoint {
  step: number;
  reward: number;
}

export function RewardChartCard({
  data,
  domain,
  visibleStepDomain,
  xAxisTicks,
}: {
  data: RewardPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
}): ReactElement {
  const t = useT();
  const rewardConfig = {
    reward: { label: t("studio.charts.reward"), color: "#10b981" },
  } satisfies ChartConfig;
  const showPoint = data.length <= 1 ? { r: 3, strokeWidth: 0 } : false;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">
          {t("studio.charts.meanReward")}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={rewardConfig} className={CHART_CONTAINER_CLASS}>
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
                return Number.isFinite(num) ? formatAxisMetric(num) : "0";
              }}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_value, payload) =>
                    t("studio.charts.step", {
                      step: payload?.[0]?.payload?.step ?? "",
                    })
                  }
                  formatter={(_value, _name, item) => {
                    const raw = Number(item?.payload?.reward);
                    return [formatMetric(raw), t("studio.charts.reward")];
                  }}
                />
              }
            />
            <Line
              type="linear"
              dataKey="reward"
              stroke="var(--color-reward)"
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
