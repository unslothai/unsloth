// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { type RlMetricPoint, rlChartKeys } from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { type ReactElement, useMemo } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
} from "./utils";

const PALETTE = [
  "#10b981",
  "#8b5cf6",
  "#f97316",
  "#0ea5e9",
  "#ec4899",
  "#eab308",
];

// New TRL logs rewards/<fn>/mean; older TRL (and the notebook tables) log rewards/<fn>.
const PER_REWARD_KEY = /^rewards\/(.+?)(\/mean)?$/;
const PREFERENCE_REWARD_KEYS = new Set([
  "rewards/chosen",
  "rewards/rejected",
  "rewards/margins",
  "rewards/accuracies",
]);

interface SeriesSpec {
  key: string;
  label: string;
}

interface CardSpec {
  id: string;
  titleKey: TranslationKey;
  descriptionKey: TranslationKey;
  series: SeriesSpec[];
}

// TRL log keys: GRPO logs reward, reward_std, rewards/<fn>/mean, kl, completions/mean_length;
// DPO/ORPO log rewards/chosen, rewards/rejected, rewards/margins, rewards/accuracies.
function buildCards(
  keys: Set<string>,
  t: (k: TranslationKey) => string,
): CardSpec[] {
  const has = (k: string) => keys.has(k);
  const perReward = [...keys]
    .filter(
      (k) =>
        PER_REWARD_KEY.test(k) &&
        !k.endsWith("/std") &&
        !PREFERENCE_REWARD_KEYS.has(k),
    )
    .sort()
    .map((k) => ({ key: k, label: k.replace(PER_REWARD_KEY, "$1") }));
  const lengthKey = has("completions/mean_length")
    ? "completions/mean_length"
    : "completion_length";
  const cards: CardSpec[] = [];
  if (has("reward")) {
    cards.push({
      id: "reward",
      titleKey: "rl.charts.reward",
      descriptionKey: "rl.charts.rewardDescription",
      series: [{ key: "reward", label: t("rl.charts.total") }, ...perReward],
    });
  }
  if (has("reward_std")) {
    cards.push({
      id: "reward_std",
      titleKey: "rl.charts.rewardStd",
      descriptionKey: "rl.charts.rewardStdDescription",
      series: [{ key: "reward_std", label: t("rl.charts.rewardStd") }],
    });
  }
  if (has(lengthKey)) {
    cards.push({
      id: "length",
      titleKey: "rl.charts.completionLength",
      descriptionKey: "rl.charts.completionLengthDescription",
      series: [
        {
          key: lengthKey,
          label: t("rl.charts.completionLength"),
        },
      ],
    });
  }
  if (has("kl")) {
    cards.push({
      id: "kl",
      titleKey: "rl.charts.kl",
      descriptionKey: "rl.charts.klDescription",
      series: [{ key: "kl", label: t("rl.charts.kl") }],
    });
  }
  if (has("rewards/margins")) {
    cards.push({
      id: "margins",
      titleKey: "rl.charts.margin",
      descriptionKey: "rl.charts.marginDescription",
      series: [{ key: "rewards/margins", label: t("rl.charts.margin") }],
    });
  }
  if (has("rewards/chosen") || has("rewards/rejected")) {
    cards.push({
      id: "chosen_rejected",
      titleKey: "rl.charts.chosenRejected",
      descriptionKey: "rl.charts.chosenRejectedDescription",
      series: [
        { key: "rewards/chosen", label: t("rl.charts.chosen") },
        { key: "rewards/rejected", label: t("rl.charts.rejected") },
      ],
    });
  }
  if (has("rewards/accuracies")) {
    cards.push({
      id: "accuracy",
      titleKey: "rl.charts.accuracy",
      descriptionKey: "rl.charts.accuracyDescription",
      series: [{ key: "rewards/accuracies", label: t("rl.charts.accuracy") }],
    });
  }
  return cards;
}

function RlChartCard({
  spec,
  data,
}: {
  spec: CardSpec;
  data: Record<string, number>[];
}): ReactElement {
  const t = useT();
  const config = Object.fromEntries(
    spec.series.map((s, i) => [
      `s${i}`,
      { label: s.label, color: PALETTE[i % PALETTE.length] },
    ]),
  ) satisfies ChartConfig;
  const rows = data.map((row) => {
    const out: Record<string, number> = { step: row.step };
    spec.series.forEach((s, i) => {
      if (row[s.key] !== undefined) {
        out[`s${i}`] = row[s.key];
      }
    });
    return out;
  });
  const latest =
    data.length > 0 ? data[data.length - 1][spec.series[0].key] : undefined;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2 text-sm">
          <span className="flex items-center gap-1.5">
            {t(spec.titleKey)}
            <NewBadge />
          </span>
          {latest !== undefined && (
            <span className="font-mono text-xs text-muted-foreground">
              {formatMetric(latest)}
            </span>
          )}
        </CardTitle>
        <p className="text-ui-11p5 text-muted-foreground/85">
          {t(spec.descriptionKey)}
        </p>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className={CHART_CONTAINER_CLASS}>
          <LineChart
            data={rows}
            accessibilityLayer={true}
            margin={DEFAULT_CHART_MARGIN}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              type="number"
              domain={["dataMin", "dataMax"]}
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
                    t("studio.charts.step", {
                      step: payload?.[0]?.payload?.step ?? "",
                    })
                  }
                />
              }
            />
            {spec.series.map((s, i) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={`s${i}`}
                stroke={`var(--color-s${i})`}
                strokeWidth={i === 0 ? 2 : 1.5}
                strokeDasharray={i === 0 ? undefined : "4 3"}
                dot={false}
                connectNulls={true}
                isAnimationActive={false}
              />
            ))}
            {spec.series.length > 1 && (
              <ChartLegend content={<ChartLegendContent />} />
            )}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export function RlChartsGrid({
  history,
}: { history: RlMetricPoint[] }): ReactElement | null {
  const t = useT();
  const data = useMemo(
    () => history.map((point) => ({ step: point.step, ...point.values })),
    [history],
  );
  const cards = useMemo(
    () => buildCards(rlChartKeys(history), t),
    [history, t],
  );

  if (cards.length === 0) {
    return null;
  }
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {cards.map((spec) => (
        <RlChartCard key={spec.id} spec={spec} data={data} />
      ))}
    </div>
  );
}
