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
import { ChartAverageIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { useI18n } from "@/features/i18n";
import {
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
  placeholderEvalData,
} from "./utils";

export function EvalLossChartCard({
  data,
  domain,
  ticks,
  isTraining,
  evalEnabled,
}: {
  data: { step: number; loss: number }[];
  domain: [number, number];
  ticks?: number[];
  isTraining: boolean;
  evalEnabled: boolean;
}): ReactElement {
  const { t } = useI18n();
  const evalLossConfig = {
    loss: { label: t("studio.charts.evalLoss"), color: "#ef4444" },
  } satisfies ChartConfig;

  return (
    <Card data-tour="studio-eval-loss" size="sm">
      <CardHeader>
        <CardTitle className={`text-sm${data.length > 0 ? "" : " text-muted-foreground"}`}>
          {t("studio.charts.evalLoss")}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {data.length > 0 ? (
          <ChartContainer config={evalLossConfig} className={CHART_CONTAINER_CLASS}>
            <LineChart
              data={data}
              accessibilityLayer={true}
              margin={DEFAULT_CHART_MARGIN}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="step"
                type="number"
                domain={["dataMin", "dataMax"]}
                ticks={ticks}
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
                tickFormatter={(value) => formatAxisMetric(Number(value))}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                  labelFormatter={(_value, payload) =>
                      `${t("studio.progress.step")} ${payload?.[0]?.payload?.step ?? ""}`
                    }
                    formatter={(_value, _name, item) => [
                      formatMetric(Number(item?.payload?.loss)),
                      t("studio.charts.evalLoss"),
                    ]}
                  />
                }
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-loss)"
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0, fill: "#ef4444" }}
                activeDot={{ r: 4, strokeWidth: 0 }}
                connectNulls={true}
                isAnimationActive={false}
              />
              <ChartLegend content={<ChartLegendContent />} />
            </LineChart>
          </ChartContainer>
        ) : (
          <div className="relative">
            <ChartContainer
              config={evalLossConfig}
              className={`${CHART_CONTAINER_CLASS} blur`}
            >
              <LineChart
                data={placeholderEvalData}
                accessibilityLayer={true}
                margin={DEFAULT_CHART_MARGIN}
              >
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  fontSize={10}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  tickCount={5}
                  fontSize={10}
                  width={DEFAULT_Y_AXIS_WIDTH}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="var(--color-loss)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ChartContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
              <HugeiconsIcon
                icon={ChartAverageIcon}
                className="size-5 text-muted-foreground/50"
              />
              <p className="text-sm font-medium text-muted-foreground">
                {isTraining && evalEnabled
                  ? t("studio.charts.eval.waitingFirstStep")
                  : t("studio.charts.eval.notConfigured")}
              </p>
              <p className="text-xs text-muted-foreground/60">
                {isTraining && evalEnabled
                  ? t("studio.charts.eval.waitingHint")
                  : t("studio.charts.eval.configureHint")}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
