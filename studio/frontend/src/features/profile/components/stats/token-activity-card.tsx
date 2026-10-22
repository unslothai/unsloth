// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts";
import type { ProfileStatsDay } from "../../api/profile-stats";
import {
  type ActivityCell,
  type ActivityMode,
  type HeatLevel,
  type WeekChartPoint,
  activitySummaryForMode,
  buildActivityColumns,
  formatCompactNumber,
  formatFullNumber,
  formatProfileCount,
  heatLevel,
  heatThresholds,
  monthLabelColumns,
  parseDayKey,
  seriesForMode,
  weekChartPoints,
} from "../../utils/stats-format";
import { StatsCard } from "./stat-primitives";
import { roundIfSingleLine } from "@/lib/tooltip-pill";

const DAYS_PER_WEEK = 7;
const WINDOW_WEEKS = 53;
const MIN_CELL = 9;
const CELL_GAP = 3;
const MIN_COLUMNS = 8;
const MONTH_LABEL_SPAN = 3;
const HEAT_OPACITY: Record<Exclude<HeatLevel, 0>, number> = {
  1: 0.28,
  2: 0.5,
  3: 0.75,
  4: 1,
};
const LEGEND_LEVELS: HeatLevel[] = [0, 1, 2, 3, 4];
const MODES: ActivityMode[] = ["daily", "weekly", "cumulative"];

type Cell = ActivityCell<ProfileStatsDay>;

function useVisibleColumns(maxColumns: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(maxColumns);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const width = element.clientWidth;
      if (width <= 0) return;
      const fits = Math.floor((width + CELL_GAP) / (MIN_CELL + CELL_GAP));
      setColumns(Math.max(MIN_COLUMNS, Math.min(maxColumns, fits)));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [maxColumns]);

  return { ref, columns };
}

function HeatSwatch({ level, title }: { level: HeatLevel | null; title?: string }) {
  return (
    <div
      title={title}
      className={cn(
        "aspect-square w-full rounded-[3px]",
        level === null
          ? "bg-transparent"
          : level === 0
            ? "bg-muted-foreground/12"
            : "bg-primary",
      )}
      style={level ? { opacity: HEAT_OPACITY[level] } : undefined}
    />
  );
}

function DayHeatmap({
  grid,
  dateFormatter,
}: {
  grid: Cell[][];
  dateFormatter: Intl.DateTimeFormat;
}) {
  const t = useT();
  const locale = useLocale();
  const thresholds = useMemo(
    () => heatThresholds(grid.flat().map((cell) => cell.day?.tokens ?? 0)),
    [grid],
  );
  const monthFormatter = useMemo(
    () => new Intl.DateTimeFormat(locale, { month: "short" }),
    [locale],
  );
  const months = useMemo(() => monthLabelColumns(grid), [grid]);
  const dayActivity = (day: ProfileStatsDay) => {
    const parts: string[] = [];
    if (day.messages > 0 || !day.apiRequests) {
      parts.push(formatProfileCount(day.messages, "message", locale));
    }
    if (day.apiRequests) {
      parts.push(
        t("settings.profile.stats.apiRequests", {
          count: formatFullNumber(day.apiRequests, locale),
        }),
      );
    }
    return parts.join(", ");
  };
  const columnTemplate = `repeat(${grid.length}, minmax(0, 1fr))`;

  return (
    <div className="flex flex-col gap-2">
      <div
        className="grid grid-flow-col"
        style={{
          gridTemplateColumns: columnTemplate,
          gridTemplateRows: `repeat(${DAYS_PER_WEEK}, auto)`,
          gap: CELL_GAP,
        }}
      >
        {grid.flatMap((column) =>
          column.map((cell) =>
            cell.day ? (
              <HeatSwatch
                key={cell.key}
                level={heatLevel(cell.day.tokens, thresholds)}
                title={
                  cell.day.tokens > 0
                    ? t("settings.profile.stats.cellTooltip", {
                        tokens: formatProfileCount(cell.day.tokens, "token", locale),
                        messages: dayActivity(cell.day),
                        date: dateFormatter.format(parseDayKey(cell.day.date)),
                      })
                    : t("settings.profile.stats.emptyDayTooltip", {
                        date: dateFormatter.format(parseDayKey(cell.day.date)),
                      })
                }
              />
            ) : (
              <HeatSwatch key={`${column[0]?.key}-${cell.key}`} level={null} />
            ),
          ),
        )}
      </div>
      {/* Same columns as the squares, so a month sits under the week its 1st falls in. One that
          starts in the last weeks ends flush with the grid instead of adding columns past it. */}
      <div
        className="grid text-ui-11 text-muted-foreground"
        style={{ gridTemplateColumns: columnTemplate, columnGap: CELL_GAP }}
      >
        {months.map((month) => {
          const tail = month.column > grid.length - MONTH_LABEL_SPAN;
          return (
            <span
              key={month.date}
              className="whitespace-nowrap"
              style={{
                gridRow: 1,
                gridColumn: tail
                  ? `${Math.max(1, grid.length - MONTH_LABEL_SPAN + 1)} / -1`
                  : `${month.column + 1} / span ${MONTH_LABEL_SPAN}`,
                justifySelf: tail ? "end" : "start",
              }}
            >
              {monthFormatter.format(parseDayKey(month.date))}
            </span>
          );
        })}
      </div>
      <div className="flex items-center justify-end gap-1.5 text-ui-11 text-muted-foreground">
        <span>{t("settings.profile.stats.less")}</span>
        {LEGEND_LEVELS.map((level) => (
          <div key={level} className="size-[11px]">
            <HeatSwatch level={level} />
          </div>
        ))}
        <span>{t("settings.profile.stats.more")}</span>
      </div>
    </div>
  );
}

const WEEK_CHART_CONFIG = {
  tokens: { color: "var(--primary)" },
} satisfies ChartConfig;

function WeekChart({
  grid,
  mode,
  dateFormatter,
}: {
  grid: Cell[][];
  mode: ActivityMode;
  dateFormatter: Intl.DateTimeFormat;
}) {
  const t = useT();
  const locale = useLocale();
  const data = useMemo(() => weekChartPoints(grid), [grid]);
  const tickFormatter = useMemo(
    () => new Intl.DateTimeFormat(locale, { month: "short", day: "numeric" }),
    [locale],
  );
  const xAxis = (
    <XAxis
      dataKey="week"
      tickLine={true}
      axisLine={true}
      tickMargin={6}
      minTickGap={24}
      interval="preserveStartEnd"
      tickFormatter={(value) =>
        tickFormatter.format(parseDayKey(String(value)))
      }
    />
  );
  const yAxis = (
    <YAxis
      tickLine={true}
      axisLine={true}
      tickMargin={4}
      tickCount={4}
      allowDecimals={false}
      width={48}
      tickFormatter={(value) => formatCompactNumber(Number(value), locale)}
    />
  );
  const tooltip = (
    <ChartTooltip
      cursor={mode === "weekly" ? undefined : false}
      content={({ active, payload }) => {
        const point = payload?.[0]?.payload as WeekChartPoint | undefined;
        if (!(active && point)) {
          return null;
        }
        return (
          <div ref={roundIfSingleLine} className="tooltip-compact">
            {t(
              mode === "weekly"
                ? "settings.profile.stats.weekTooltip"
                : "settings.profile.stats.cumulativeTooltip",
              {
                date: dateFormatter.format(parseDayKey(point.week)),
                tokens: formatProfileCount(point.tokens, "token", locale),
              },
            )}
          </div>
        );
      }}
    />
  );
  const margin = { top: 6, right: 8, bottom: 0, left: 0 };

  return (
    <ChartContainer
      config={WEEK_CHART_CONFIG}
      className="h-[calc(160px*var(--ui-space-scale,1))] w-full"
    >
      {mode === "weekly" ? (
        <BarChart data={data} margin={margin} accessibilityLayer={true}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          {xAxis}
          {yAxis}
          {tooltip}
          <Bar
            dataKey="tokens"
            fill="var(--color-tokens)"
            radius={[3, 3, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      ) : (
        <LineChart data={data} margin={margin} accessibilityLayer={true}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          {xAxis}
          {yAxis}
          {tooltip}
          <Line
            type="linear"
            dataKey="tokens"
            stroke="var(--color-tokens)"
            strokeWidth={2}
            dot={data.length <= 1 ? { r: 3, strokeWidth: 0 } : false}
            activeDot={{ r: 3, strokeWidth: 0 }}
            isAnimationActive={false}
          />
        </LineChart>
      )}
    </ChartContainer>
  );
}

export function TokenActivityCard({ daily }: { daily: ProfileStatsDay[] }) {
  const t = useT();
  const pillId = useId();
  const [mode, setMode] = useState<ActivityMode>("daily");
  const maxColumns = Math.min(
    WINDOW_WEEKS,
    Math.ceil(daily.length / DAYS_PER_WEEK) + 1,
  );
  const { ref, columns } = useVisibleColumns(maxColumns);

  const values = useMemo(() => seriesForMode(daily, mode), [daily, mode]);
  const grid = useMemo(
    () => buildActivityColumns(daily, values, columns, mode),
    [daily, values, columns, mode],
  );
  // The app language, not the browser's: those differ whenever the user picks
  // a language in Settings, and it has to be a dependency so switching while
  // the panel is open rebuilds the formatters.
  const locale = useLocale();
  const summaryTotal = useMemo(
    () => activitySummaryForMode(grid, mode),
    [grid, mode],
  );
  const firstShown = grid[0]?.find((cell) => cell.day !== null)?.day?.date;

  const dateFormatter = useMemo(
    () =>
      new Intl.DateTimeFormat(locale, {
        weekday: "short",
        month: "short",
        day: "numeric",
        year: "numeric",
      }),
    [locale],
  );
  const sinceFormatter = useMemo(
    () =>
      new Intl.DateTimeFormat(locale, {
        month: "short",
        day: "numeric",
        year: "numeric",
      }),
    [locale],
  );

  return (
    <StatsCard
      title={t("settings.profile.stats.activityTitle")}
      description={t(`settings.profile.stats.activityDescription.${mode}`, {
        total: formatProfileCount(
          summaryTotal,
          "token",
          locale,
          formatCompactNumber(summaryTotal, locale),
        ),
        since: firstShown ? sinceFormatter.format(parseDayKey(firstShown)) : "",
      })}
      action={
        <div className="hub-tab-toggle inline-flex h-8 w-fit items-center rounded-full">
          {MODES.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setMode(option)}
              aria-pressed={mode === option}
              className={cn(
                "relative inline-flex h-8 items-center rounded-full px-3 text-ui-13 font-medium transition-colors",
                mode === option
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {mode === option && <SegmentedPill group={`${pillId}-mode`} />}
              {t(`settings.profile.stats.mode.${option}`)}
            </button>
          ))}
        </div>
      }
    >
      <div ref={ref} className="w-full">
        {mode === "daily" ? (
          <DayHeatmap grid={grid} dateFormatter={dateFormatter} />
        ) : (
          <WeekChart grid={grid} mode={mode} dateFormatter={sinceFormatter} />
        )}
      </div>
    </StatsCard>
  );
}
