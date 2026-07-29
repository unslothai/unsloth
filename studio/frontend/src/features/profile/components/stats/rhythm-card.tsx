// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { useT } from "@/i18n";
import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts";
import type { ProfileStats } from "../../api/profile-stats";
import { formatFullNumber } from "../../utils/stats-format";
import { StatsCard } from "./stat-primitives";

const HOURS_IN_DAY = 24;
const CHART_CLASS = "h-[160px] w-full";

function chartConfig(label: string): ChartConfig {
  return { messages: { label, color: "var(--primary)" } } satisfies ChartConfig;
}

/** Hour-of-day histogram: when during the day the user actually works. */
export function HourRhythmCard({ stats }: { stats: ProfileStats }) {
  const t = useT();
  const data = useMemo(
    () =>
      Array.from({ length: HOURS_IN_DAY }, (_, hour) => ({
        hour,
        label: `${`${hour}`.padStart(2, "0")}:00`,
        messages: stats.hourly[hour] ?? 0,
      })),
    [stats.hourly],
  );

  const busiest = useMemo(
    () =>
      data.reduce(
        (best, entry) => (entry.messages > best.messages ? entry : best),
        data[0] ?? { hour: 0, label: "00:00", messages: 0 },
      ),
    [data],
  );

  return (
    <StatsCard
      title={t("settings.profile.stats.hourTitle")}
      description={
        busiest.messages > 0
          ? t("settings.profile.stats.hourDescription", { hour: busiest.label })
          : t("settings.profile.stats.noRhythm")
      }
    >
      <ChartContainer
        config={chartConfig(t("settings.profile.stats.messages"))}
        className={CHART_CLASS}
      >
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="hour"
            tickLine={false}
            axisLine={false}
            tickMargin={6}
            interval={5}
            tickFormatter={(hour: number) => `${hour}`}
            className="text-ui-11"
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(_label, payload) =>
                  `${payload?.[0]?.payload?.label ?? ""}`
                }
                formatter={(value) => formatFullNumber(Number(value))}
              />
            }
          />
          <Bar
            dataKey="messages"
            fill="var(--color-messages)"
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ChartContainer>
    </StatsCard>
  );
}

/** Weekday distribution, Monday-first to match the activity grid columns. */
export function WeekdayRhythmCard({ stats }: { stats: ProfileStats }) {
  const t = useT();
  const names = useMemo(() => {
    const formatter = new Intl.DateTimeFormat(navigator.language, {
      weekday: "short",
    });
    // 2024-01-01 was a Monday, so this walks Mon..Sun in the user's locale.
    return Array.from({ length: 7 }, (_, index) =>
      formatter.format(new Date(2024, 0, 1 + index)),
    );
  }, []);

  const data = useMemo(
    () =>
      names.map((name, index) => ({
        day: name,
        messages: stats.weekday[index] ?? 0,
      })),
    [names, stats.weekday],
  );

  const busiest = useMemo(
    () =>
      data.reduce(
        (best, entry) => (entry.messages > best.messages ? entry : best),
        data[0] ?? { day: "", messages: 0 },
      ),
    [data],
  );

  return (
    <StatsCard
      title={t("settings.profile.stats.weekdayTitle")}
      description={
        busiest.messages > 0
          ? t("settings.profile.stats.weekdayDescription", { day: busiest.day })
          : t("settings.profile.stats.noRhythm")
      }
    >
      <ChartContainer
        config={chartConfig(t("settings.profile.stats.messages"))}
        className={CHART_CLASS}
      >
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="day"
            tickLine={false}
            axisLine={false}
            tickMargin={6}
            className="text-ui-11"
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                formatter={(value) => formatFullNumber(Number(value))}
              />
            }
          />
          <Bar
            dataKey="messages"
            fill="var(--color-messages)"
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ChartContainer>
    </StatsCard>
  );
}
