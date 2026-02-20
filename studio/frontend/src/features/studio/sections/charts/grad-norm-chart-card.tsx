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
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { SharedChartSettings } from "./shared-chart-settings";
import type { OutlierMode, ScaleMode, ViewSettingsState } from "./types";
import { CHART_SYNC_ID, formatMetric, formatStepTick, fromLog1p } from "./utils";

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
  setScale,
  outlierMode,
  setOutlierMode,
  viewSettings,
}: {
  data: GradNormPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  scale: ScaleMode;
  setScale: (value: ScaleMode) => void;
  outlierMode: OutlierMode;
  setOutlierMode: (value: OutlierMode) => void;
  viewSettings: ViewSettingsState;
}): ReactElement {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm pl-2">Gradient Norm</CardTitle>
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
        <ChartContainer config={gradNormConfig} className="-ml-3 h-[220px] w-full">
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
                  formatter={(_value, _name, item) => {
                    const raw = Number(item?.payload?.gradNorm);
                    return [formatMetric(raw), "Grad Norm"];
                  }}
                />
              }
            />
            <Line
              type="monotoneX"
              dataKey="displayGradNorm"
              stroke="var(--color-displayGradNorm)"
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
  );
}
