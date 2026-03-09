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
import { formatMetric, formatStepTick, placeholderEvalData } from "./utils";

const evalLossConfig = {
  loss: { label: "Eval Loss", color: "#ef4444" },
} satisfies ChartConfig;

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
  return (
    <Card data-tour="studio-eval-loss" size="sm">
      <CardHeader>
        <CardTitle
          className={`text-sm pl-1${data.length > 0 ? "" : " text-muted-foreground"}`}
        >
          Eval Loss
        </CardTitle>
      </CardHeader>
      <CardContent>
        {data.length > 0 ? (
          <ChartContainer
            config={evalLossConfig}
            className="-ml-3 h-[220px] w-full"
          >
            <LineChart
              data={data}
              accessibilityLayer={true}
              margin={{ left: 0, right: 8 }}
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
                tickMargin={4}
                fontSize={10}
                width={80}
                tickFormatter={(value) => formatMetric(Number(value))}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) =>
                      `Step ${payload?.[0]?.payload?.step ?? ""}`
                    }
                    formatter={(_value, _name, item) => [
                      formatMetric(Number(item?.payload?.loss)),
                      "Eval Loss",
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
              className="-ml-3 h-[220px] w-full blur"
            >
              <LineChart
                data={placeholderEvalData}
                accessibilityLayer={true}
                margin={{ left: 0, right: 8 }}
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
                  tickMargin={4}
                  fontSize={10}
                  width={40}
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
                  ? "Waiting for first evaluation step…"
                  : "Evaluation not configured"}
              </p>
              <p className="text-xs text-muted-foreground/60">
                {isTraining && evalEnabled
                  ? "Chart will appear once eval_steps is reached"
                  : "Set eval dataset & eval_steps to track eval loss"}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
