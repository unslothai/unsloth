import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { TrainingMetrics } from "@/types/training";
import { ChartAverageIcon, Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

const lossConfig = {
  loss: { label: "Loss", color: "#3b82f6" },
  smoothed: { label: "Smoothed", color: "#f59e0b" },
} satisfies ChartConfig;

const lrConfig = {
  lr: { label: "LR", color: "#8b5cf6" },
} satisfies ChartConfig;
const gradNormConfig = {
  gradNorm: { label: "Grad Norm", color: "#f97316" },
} satisfies ChartConfig;
const evalLossConfig = {
  loss: { label: "Eval Loss", color: "#ef4444" },
} satisfies ChartConfig;

const placeholderEvalData = [
  { step: 0, loss: 2.8 },
  { step: 50, loss: 2.4 },
  { step: 100, loss: 2.0 },
  { step: 150, loss: 1.7 },
  { step: 200, loss: 1.5 },
];

type LossHistoryItem = { step: number; loss: number };
type SmoothedLossItem = LossHistoryItem & { smoothed: number };

function ema(data: LossHistoryItem[], alpha: number): SmoothedLossItem[] {
  if (data.length === 0) {
    return [];
  }
  let s = data[0].loss;
  return data.map((d) => {
    s = alpha * d.loss + (1 - alpha) * s;
    return { ...d, smoothed: +s.toFixed(4) };
  });
}

export function ChartsContent({
  metrics,
}: { metrics: TrainingMetrics }): ReactElement {
  const [smoothing, setSmoothing] = useState(0.6);
  const [showRaw, setShowRaw] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showAvgLine, setShowAvgLine] = useState(true);

  const lossHistory = metrics.lossHistory;
  const smoothedData = useMemo(
    () => (lossHistory ? ema(lossHistory, 1 - smoothing) : []),
    [lossHistory, smoothing],
  );

  const avg =
    metrics.lossHistory.length > 0
      ? +(
          metrics.lossHistory.reduce((a, b) => a + b.loss, 0) /
          metrics.lossHistory.length
        ).toFixed(4)
      : 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Training Loss */}
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">Training Loss</CardTitle>
          <CardAction>
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <button
                  type="button"
                  className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
                >
                  <HugeiconsIcon icon={Settings02Icon} className="size-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel className="text-xs">
                  Chart Settings
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <div className="flex flex-col gap-1.5 px-2 py-1.5">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">Smoothing</Label>
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {smoothing.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    value={[smoothing]}
                    onValueChange={([v]) => setSmoothing(v)}
                    min={0}
                    max={0.99}
                    step={0.01}
                  />
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuCheckboxItem
                  checked={showRaw}
                  onCheckedChange={setShowRaw}
                >
                  Show raw loss
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem
                  checked={showSmoothed}
                  onCheckedChange={setShowSmoothed}
                >
                  Show smoothed loss
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem
                  checked={showAvgLine}
                  onCheckedChange={setShowAvgLine}
                >
                  Show average line
                </DropdownMenuCheckboxItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={lossConfig}
            className="h-[200px] w-full -ml-3"
          >
            <LineChart
              data={smoothedData}
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
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) =>
                      `Step ${payload?.[0]?.payload?.step ?? ""}`
                    }
                  />
                }
              />
              {showAvgLine && (
                <ReferenceLine
                  y={avg}
                  stroke="#3b82f6"
                  strokeDasharray="4 4"
                  strokeOpacity={0.5}
                  label={{
                    value: `avg ${avg}`,
                    position: "insideTopRight",
                    fontSize: 10,
                    fill: "#3b82f6",
                  }}
                />
              )}
              {showRaw && (
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="var(--color-loss)"
                  strokeWidth={1.5}
                  strokeOpacity={showSmoothed ? 0.3 : 1}
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              {showSmoothed && (
                <Line
                  type="monotone"
                  dataKey="smoothed"
                  stroke="var(--color-smoothed)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              <ChartLegend content={<ChartLegendContent />} />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Grad Norm */}
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">Gradient Norm</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={gradNormConfig}
            className="h-[200px] w-full -ml-3"
          >
            <LineChart
              data={metrics.gradNormHistory}
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
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) =>
                      `Step ${payload?.[0]?.payload?.step ?? ""}`
                    }
                  />
                }
              />
              <Line
                type="monotone"
                dataKey="gradNorm"
                stroke="var(--color-gradNorm)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <ChartLegend content={<ChartLegendContent />} />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Learning Rate */}
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">Learning Rate</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={lrConfig}
            className="h-[200px] w-full -ml-1.5"
          >
            <LineChart
              data={metrics.lrHistory}
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
                tickFormatter={(v) => v.toExponential(0)}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) =>
                      `Step ${payload?.[0]?.payload?.step ?? ""}`
                    }
                    formatter={(value) => [
                      Number(value).toExponential(3),
                      "LR",
                    ]}
                  />
                }
              />
              <Line
                type="monotone"
                dataKey="lr"
                stroke="var(--color-lr)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <ChartLegend content={<ChartLegendContent />} />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Eval Loss (disabled/blurred) */}
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm text-muted-foreground">
            Eval Loss
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <ChartContainer
              config={evalLossConfig}
              className="h-[200px] w-full -ml-3 blur"
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
                Evaluation not configured
              </p>
              <p className="text-xs text-muted-foreground/60">
                Set eval dataset & eval_steps to track eval loss
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
