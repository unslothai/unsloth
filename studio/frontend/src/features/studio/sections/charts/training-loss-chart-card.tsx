import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";
import { SharedChartSettings } from "./shared-chart-settings";
import type { OutlierMode, ScaleMode, ViewSettingsState } from "./types";
import { CHART_SYNC_ID, formatMetric, formatStepTick, fromLog1p } from "./utils";

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
  smoothing,
  setSmoothing,
  showRaw,
  setShowRaw,
  showSmoothed,
  setShowSmoothed,
  showAvgLine,
  setShowAvgLine,
  viewSettings,
  scale,
  setScale,
  outlierMode,
  setOutlierMode,
}: {
  data: LossChartPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  avgRaw: number;
  avgDisplay: number;
  smoothing: number;
  setSmoothing: (value: number) => void;
  showRaw: boolean;
  setShowRaw: (value: boolean) => void;
  showSmoothed: boolean;
  setShowSmoothed: (value: boolean) => void;
  showAvgLine: boolean;
  setShowAvgLine: (value: boolean) => void;
  viewSettings: ViewSettingsState;
  scale: ScaleMode;
  setScale: (value: ScaleMode) => void;
  outlierMode: OutlierMode;
  setOutlierMode: (value: OutlierMode) => void;
}): ReactElement {
  return (
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
              <DropdownMenuLabel className="text-xs">Chart Settings</DropdownMenuLabel>
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
                onCheckedChange={(value) => setShowRaw(Boolean(value))}
              >
                Show raw loss
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                checked={showSmoothed}
                onCheckedChange={(value) => setShowSmoothed(Boolean(value))}
              >
                Show smoothed loss
              </DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem
                checked={showAvgLine}
                onCheckedChange={(value) => setShowAvgLine(Boolean(value))}
              >
                Show average line
              </DropdownMenuCheckboxItem>
              <SharedChartSettings
                view={viewSettings}
                scale={scale}
                setScale={setScale}
                outlierMode={outlierMode}
                setOutlierMode={setOutlierMode}
              />
            </DropdownMenuContent>
          </DropdownMenu>
        </CardAction>
      </CardHeader>
      <CardContent>
        <ChartContainer config={lossConfig} className="-ml-3 h-[220px] w-full">
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
              width={52}
              tickFormatter={(value) => {
                const num = Number(value);
                if (!Number.isFinite(num)) return "0";
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
                  formatter={(_value, name, item) => {
                    if (name === "displaySmoothed") {
                      return [formatMetric(Number(item?.payload?.smoothed)), "Smoothed"];
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
                type="monotoneX"
                dataKey="displayLoss"
                stroke="var(--color-displayLoss)"
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
                dataKey="displaySmoothed"
                stroke="var(--color-displaySmoothed)"
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
  );
}
