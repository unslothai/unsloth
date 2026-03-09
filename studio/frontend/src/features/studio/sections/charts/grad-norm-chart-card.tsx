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
        <CardTitle className="text-sm pl-1">Gradient Norm</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={gradNormConfig}
          className="-ml-3 h-[220px] w-full"
        >
          <LineChart
            data={data}
            syncId={CHART_SYNC_ID}
            syncMethod="value"
            accessibilityLayer={true}
            margin={{ left: 0, right: 8 }}
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
              tickMargin={4}
              fontSize={10}
              width={80}
              tickFormatter={(value) => {
                const num = Number(value);
                if (!Number.isFinite(num)) {
                  return "0";
                }
                const shown = scale === "log" ? fromLog1p(num) : num;
                return formatMetric(shown);
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
