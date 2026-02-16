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

interface TrainingChartSeries {
  lossHistory: LossHistoryItem[];
  lrHistory: { step: number; lr: number }[];
  gradNormHistory: { step: number; gradNorm: number }[];
  evalLossHistory: { step: number; loss: number }[];
}

const CHART_SYNC_ID = "train-metrics-sync";
const MAX_RENDER_POINTS = 800;
const DEFAULT_VISIBLE_POINTS = 160;

function formatStepTick(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return String(Math.round(value));
}

function compressSeries<T>(data: T[], maxPoints: number): T[] {
  if (data.length <= maxPoints) {
    return data;
  }

  const stride = Math.ceil(data.length / maxPoints);
  return data.filter(
    (_item, index) => index % stride === 0 || index === data.length - 1,
  );
}

function buildStepTicks(min: number, max: number, targetCount = 6): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (max <= min) {
    return [min, max];
  }

  const stepSize = Math.max(1, Math.ceil((max - min) / (targetCount - 1)));
  const ticks: number[] = [];
  let current = min;

  while (current < max) {
    ticks.push(current);
    current += stepSize;
  }

  ticks.push(max);
  return Array.from(new Set(ticks));
}

function buildYDomain(values: number[]): [number, number] {
  if (values.length === 0) {
    return [0, 1];
  }

  const min = Math.min(...values);
  const max = Math.max(...values);

  if (min === max) {
    const base = Math.abs(min);
    const pad = base > 0 ? base * 0.08 : 0.1;
    return [min - pad, max + pad];
  }

  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

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
  isTraining,
  evalEnabled,
}: { metrics: TrainingChartSeries; isTraining: boolean; evalEnabled: boolean }): ReactElement {
  const [smoothing, setSmoothing] = useState(0.75);
  const [showRaw, setShowRaw] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showAvgLine, setShowAvgLine] = useState(true);

  const lossHistory = metrics.lossHistory;
  const smoothedData = useMemo(
    () => (lossHistory.length > 0 ? ema(lossHistory, 1 - smoothing) : []),
    [lossHistory, smoothing],
  );

  const reducedLossData = useMemo(
    () => compressSeries(smoothedData, MAX_RENDER_POINTS),
    [smoothedData],
  );

  const reducedGradNormData = useMemo(
    () => compressSeries(metrics.gradNormHistory, MAX_RENDER_POINTS),
    [metrics.gradNormHistory],
  );

  const reducedLrData = useMemo(
    () => compressSeries(metrics.lrHistory, MAX_RENDER_POINTS),
    [metrics.lrHistory],
  );

  const reducedEvalLossData = useMemo(
    () => compressSeries(metrics.evalLossHistory, MAX_RENDER_POINTS),
    [metrics.evalLossHistory],
  );

  const visibleStepDomain = useMemo<[number, number]>(() => {
    const allSteps = [
      ...reducedLossData.map((point) => point.step),
      ...reducedGradNormData.map((point) => point.step),
      ...reducedLrData.map((point) => point.step),
    ].sort((a, b) => a - b);

    if (allSteps.length === 0) {
      return [0, 1];
    }

    const minStep = allSteps[0] ?? 0;
    const endStep = allSteps[allSteps.length - 1] ?? 1;
    const startIndex = Math.max(0, allSteps.length - DEFAULT_VISIBLE_POINTS);
    const startStep = allSteps[startIndex] ?? minStep;
    if (startStep === endStep) {
      return [startStep, startStep + 4];
    }
    if (endStep - startStep < 6) {
      return [Math.max(minStep, endStep - 6), endStep];
    }
    return [startStep, endStep];
  }, [reducedGradNormData, reducedLossData, reducedLrData]);

  const xAxisTicks = useMemo(
    () => buildStepTicks(visibleStepDomain[0], visibleStepDomain[1]),
    [visibleStepDomain],
  );

  const visibleLossValues = useMemo(
    () =>
      reducedLossData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] && point.step <= visibleStepDomain[1],
        )
        .map((point) => point.loss),
    [reducedLossData, visibleStepDomain],
  );

  const visibleSmoothValues = useMemo(
    () =>
      reducedLossData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] && point.step <= visibleStepDomain[1],
        )
        .map((point) => point.smoothed),
    [reducedLossData, visibleStepDomain],
  );

  const visibleGradValues = useMemo(
    () =>
      reducedGradNormData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] && point.step <= visibleStepDomain[1],
        )
        .map((point) => point.gradNorm),
    [reducedGradNormData, visibleStepDomain],
  );

  const visibleLrValues = useMemo(
    () =>
      reducedLrData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] && point.step <= visibleStepDomain[1],
        )
        .map((point) => point.lr),
    [reducedLrData, visibleStepDomain],
  );

  const lossDomain = useMemo(
    () => buildYDomain([...visibleLossValues, ...visibleSmoothValues]),
    [visibleLossValues, visibleSmoothValues],
  );
  const gradDomain = useMemo(() => buildYDomain(visibleGradValues), [visibleGradValues]);
  const lrDomain = useMemo(() => buildYDomain(visibleLrValues), [visibleLrValues]);

  const evalLossDomain = useMemo(() => {
    const vals = reducedEvalLossData.map((p) => p.loss);
    return buildYDomain(vals);
  }, [reducedEvalLossData]);

  const evalLossStepTicks = useMemo(() => {
    if (reducedEvalLossData.length < 2) return undefined;
    const min = reducedEvalLossData[0].step;
    const max = reducedEvalLossData[reducedEvalLossData.length - 1].step;
    return buildStepTicks(min, max);
  }, [reducedEvalLossData]);

  const avg =
    metrics.lossHistory.length > 0
      ? +(
        metrics.lossHistory.reduce((a, b) => a + b.loss, 0) /
        metrics.lossHistory.length
      ).toFixed(4)
      : 0;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Card data-tour="studio-training-loss" size="sm">
        <CardHeader>
          <CardTitle className="text-sm pl-2">Training Loss</CardTitle>
          <CardAction>
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <button
                  type="button"
                  className="cursor-pointer rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
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
          <ChartContainer config={lossConfig} className="-ml-3 h-[220px] w-full">
            <LineChart
              data={reducedLossData}
              syncId={CHART_SYNC_ID}
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
                domain={lossDomain}
                allowDataOverflow={true}
                tickLine={false}
                axisLine={false}
                tickMargin={4}
                fontSize={10}
                width={52}
                tickFormatter={(value) => Number(value).toFixed(2)}
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
                  type="monotoneX"
                  dataKey="loss"
                  stroke="var(--color-loss)"
                  strokeWidth={1.2}
                  strokeOpacity={showSmoothed ? 0.35 : 1}
                  dot={false}
                  activeDot={{ r: 3, strokeWidth: 0 }}
                  connectNulls={true}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  isAnimationActive={false}
                />
              )}
              {showSmoothed && (
                <Line
                  type="monotoneX"
                  dataKey="smoothed"
                  stroke="var(--color-smoothed)"
                  strokeWidth={2.2}
                  dot={false}
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

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm pl-2">Gradient Norm</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={gradNormConfig}
            className="-ml-3 h-[220px] w-full"
          >
            <LineChart
              data={reducedGradNormData}
              syncId={CHART_SYNC_ID}
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
                domain={gradDomain}
                allowDataOverflow={true}
                tickLine={false}
                axisLine={false}
                tickMargin={4}
                fontSize={10}
                width={52}
                tickFormatter={(value) => Number(value).toFixed(2)}
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
                type="monotoneX"
                dataKey="gradNorm"
                stroke="var(--color-gradNorm)"
                strokeWidth={2}
                dot={false}
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

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm pl-2">Learning Rate</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer config={lrConfig} className="-ml-1.5 h-[220px] w-full">
            <LineChart
              data={reducedLrData}
              syncId={CHART_SYNC_ID}
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
                domain={lrDomain}
                allowDataOverflow={true}
                tickLine={false}
                axisLine={false}
                tickMargin={4}
                fontSize={10}
                width={52}
                tickFormatter={(value) => Number(value).toExponential(0)}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) =>
                      `Step ${payload?.[0]?.payload?.step ?? ""}`
                    }
                    formatter={(value) => [Number(value).toExponential(3), "LR"]}
                  />
                }
              />
              <Line
                type="monotoneX"
                dataKey="lr"
                stroke="var(--color-lr)"
                strokeWidth={2}
                dot={false}
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

      <Card data-tour="studio-eval-loss" size="sm">
        <CardHeader>
          <CardTitle className={`text-sm pl-2${reducedEvalLossData.length > 0 ? "" : " text-muted-foreground"}`}>
            Eval Loss
          </CardTitle>
        </CardHeader>
        <CardContent>
          {reducedEvalLossData.length > 0 ? (
            <ChartContainer
              config={evalLossConfig}
              className="-ml-3 h-[220px] w-full"
            >
              <LineChart
                data={reducedEvalLossData}
                accessibilityLayer={true}
                margin={{ left: 0, right: 8 }}
              >
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  ticks={evalLossStepTicks}
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
                  domain={evalLossDomain}
                  allowDataOverflow={true}
                  tickLine={false}
                  axisLine={false}
                  tickMargin={4}
                  fontSize={10}
                  width={52}
                  tickFormatter={(value) => Number(value).toFixed(2)}
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
                  {isTraining && evalEnabled ? "Waiting for first evaluation step…" : "Evaluation not configured"}
                </p>
                <p className="text-xs text-muted-foreground/60">
                  {isTraining && evalEnabled ? "Chart will appear once eval_steps is reached" : "Set eval dataset & eval_steps to track eval loss"}
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
