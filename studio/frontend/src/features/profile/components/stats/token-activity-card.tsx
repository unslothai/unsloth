// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useEffect, useMemo, useRef, useState } from "react";
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
import { useActivityChartStore } from "../../stores/activity-chart-store";
import {
  type ActivityMode,
  type WeekChartPoint,
  activitySummaryForMode,
  formatCompactNumber,
  formatProfileCount,
  heatLevel,
  parseDayKey,
  seriesForMode,
  weekChartPoints,
  windowBaseline,
} from "../../utils/stats-format";
import { StatsCard } from "./stat-primitives";

const DAYS_PER_WEEK = 7;
const CELL_SIZE = 11;
const CELL_GAP = 3;
const COLUMN_WIDTH = CELL_SIZE + CELL_GAP;
const MIN_COLUMNS = 8;
const HEAT_OPACITY = [0, 0.4, 0.62, 0.8, 1] as const;
// Weekly and cumulative are on/off, so they use one flat shade.
const SOLID_LEVEL = 4;
const MODES: ActivityMode[] = ["daily", "weekly", "cumulative"];

type Cell = {
  key: string;
  day: ProfileStatsDay | null;
  value: number;
};

/**
 * Trim the series to the most recent `columns` weeks, ending on a partial
 * current week. The grid never scrolls, so older days fall off the left.
 */
function buildColumns(
  daily: ProfileStatsDay[],
  values: number[],
  columns: number,
  mode: ActivityMode,
): Cell[][] {
  if (daily.length === 0 || columns <= 0) return [];

  const lastDay = daily.at(-1);
  if (!lastDay) return [];
  // Days after today in the final (partial) week.
  const trailing =
    DAYS_PER_WEEK -
    1 -
    ((parseDayKey(lastDay.date).getDay() + 6) % DAYS_PER_WEEK);
  const capacity = columns * DAYS_PER_WEEK - trailing;
  const start = Math.max(0, daily.length - capacity);
  const visible = daily.slice(start);
  // Cumulative is a running total over what the grid shows, so a narrow card
  // that drops older weeks has to rebase off the last hidden day.
  const baseline = windowBaseline(values, start, mode);

  const cells: Cell[] = [];
  // Pad so every column is a Monday-started week.
  const firstVisible = visible[0];
  if (!firstVisible) return [];
  const leading = (parseDayKey(firstVisible.date).getDay() + 6) % DAYS_PER_WEEK;
  for (let index = 0; index < leading; index += 1) {
    cells.push({ key: `pad-${index}`, day: null, value: 0 });
  }
  for (const [index, day] of visible.entries()) {
    cells.push({
      key: day.date,
      day,
      value: (values[start + index] ?? 0) - baseline,
    });
  }

  const grid: Cell[][] = [];
  for (let index = 0; index < cells.length; index += DAYS_PER_WEEK) {
    grid.push(cells.slice(index, index + DAYS_PER_WEEK));
  }
  return grid;
}

/** Month captions under the grid, one per column where the month turns over. */
function buildMonthLabels(grid: Cell[][], locale: string) {
  const formatter = new Intl.DateTimeFormat(locale, { month: "short" });
  const labels: Array<{ key: string; column: number; text: string }> = [];
  let lastMonth = -1;
  for (const [columnIndex, column] of grid.entries()) {
    const firstDay = column.find((cell) => cell.day !== null)?.day;
    if (!firstDay) continue;
    const date = parseDayKey(firstDay.date);
    if (date.getMonth() === lastMonth) continue;
    lastMonth = date.getMonth();
    // Skip a label that would collide with the previous one, or run off the end.
    const previous = labels.at(-1);
    if (previous && columnIndex - previous.column < 3) continue;
    if (columnIndex > grid.length - 3) continue;
    labels.push({
      key: firstDay.date,
      column: columnIndex,
      text: formatter.format(date),
    });
  }
  return labels;
}

/**
 * Per-column totals for the bar modes. Shading every day of an active week
 * instead would fill the grid solid and hide the shape.
 */
function columnSummary(column: Cell[]) {
  let value = 0;
  let tokens = 0;
  let firstDay: string | null = null;
  for (const cell of column) {
    if (!cell.day) continue;
    value = Math.max(value, cell.value);
    tokens += cell.day.tokens;
    firstDay ??= cell.day.date;
  }
  return { value, tokens, firstDay };
}

/** Bar height in cells, at least one for any activity. */
function barHeight(value: number, peakValue: number): number {
  if (value <= 0 || peakValue <= 0) return 0;
  return Math.max(1, Math.round((value / peakValue) * DAYS_PER_WEEK));
}

/** How many week columns fit the card's current width. */
function useVisibleColumns(maxColumns: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(maxColumns);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const width = element.clientWidth;
      if (width <= 0) return;
      // The final column carries no trailing gap.
      const fits = Math.floor((width + CELL_GAP) / COLUMN_WIDTH);
      setColumns(Math.max(MIN_COLUMNS, Math.min(maxColumns, fits)));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [maxColumns]);

  return { ref, columns };
}

const CELL_CLASS = "size-[11px] rounded-[3px]";

function Block({
  title,
  tone,
}: { title: string; tone: 0 | 1 | 2 | 3 | 4 | -1 }) {
  return (
    <div
      title={title}
      className={cn(
        CELL_CLASS,
        tone === -1
          ? "bg-transparent"
          : tone === 0
            ? "bg-muted-foreground/12"
            : "bg-primary",
      )}
      style={
        tone > 0 ? { opacity: HEAT_OPACITY[tone as 1 | 2 | 3 | 4] } : undefined
      }
    />
  );
}

/** Daily: each day shaded by its own volume. */
function DayColumn({
  column,
  peak,
  dateFormatter,
}: {
  column: Cell[];
  peak: number;
  dateFormatter: Intl.DateTimeFormat;
}) {
  const t = useT();
  const locale = useLocale();

  return (
    <div className="flex flex-col gap-[3px]">
      {column.map((cell) => {
        if (!cell.day) {
          return <Block key={cell.key} title="" tone={-1} />;
        }
        return (
          <Block
            key={cell.key}
            tone={heatLevel(cell.value, peak)}
            title={t("settings.profile.stats.cellTooltip", {
              tokens: formatProfileCount(cell.day.tokens, "token", locale),
              messages: formatProfileCount(
                cell.day.messages,
                "message",
                locale,
              ),
              date: dateFormatter.format(parseDayKey(cell.day.date)),
            })}
          />
        );
      })}
    </div>
  );
}

/** Weekly and cumulative: one flat bar per column, anchored to the bottom. */
function BarColumn({
  column,
  peak,
  dateFormatter,
}: {
  column: Cell[];
  peak: number;
  dateFormatter: Intl.DateTimeFormat;
}) {
  const t = useT();
  const locale = useLocale();
  const summary = columnSummary(column);
  const height = barHeight(summary.value, peak);
  // The bar is scaled by summary.value, which in cumulative mode is the
  // running total. The tooltip says "week of", so it reports that week.
  const title = summary.firstDay
    ? t("settings.profile.stats.weekTooltip", {
        date: dateFormatter.format(parseDayKey(summary.firstDay)),
        tokens: formatProfileCount(summary.tokens, "token", locale),
      })
    : "";

  return (
    <div className="flex flex-col gap-[3px]">
      {Array.from({ length: DAYS_PER_WEEK }, (_, row) => (
        <Block
          key={column[row]?.key ?? `slot-${row}`}
          title={title}
          tone={row >= DAYS_PER_WEEK - height ? SOLID_LEVEL : 0}
        />
      ))}
    </div>
  );
}

const WEEK_CHART_CONFIG = {
  tokens: { color: "var(--primary)" },
} satisfies ChartConfig;

/** Opt-in view: Weekly as bars, Cumulative as a line, on labelled token and date axes. */
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
          <div className="rounded-md border border-border/50 bg-background px-2 py-1 text-xs shadow-md">
            {t("settings.profile.stats.weekTooltip", {
              date: dateFormatter.format(parseDayKey(point.week)),
              tokens: formatProfileCount(point.tokens, "token", locale),
            })}
          </div>
        );
      }}
    />
  );
  const margin = { top: 6, right: 8, bottom: 0, left: 0 };

  return (
    <ChartContainer
      config={WEEK_CHART_CONFIG}
      className="h-[calc(140px*var(--ui-space-scale,1))] w-full"
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
  const [mode, setMode] = useState<ActivityMode>("daily");
  const maxColumns = Math.ceil(daily.length / DAYS_PER_WEEK) + 1;
  const { ref, columns } = useVisibleColumns(maxColumns);

  const shaded = mode === "daily";
  const showChart = useActivityChartStore((state) => state.showChart);
  const setShowChart = useActivityChartStore((state) => state.setShowChart);
  const charted = !shaded && showChart;
  const values = useMemo(() => seriesForMode(daily, mode), [daily, mode]);
  const grid = useMemo(
    () => buildColumns(daily, values, columns, mode),
    [daily, values, columns, mode],
  );
  // The app language, not the browser's: those differ whenever the user picks
  // a language in Settings, and it has to be a dependency so switching while
  // the panel is open rebuilds the formatters.
  const locale = useLocale();
  const monthLabels = useMemo(
    () => buildMonthLabels(grid, locale),
    [grid, locale],
  );
  // Daily scales against the busiest day, the bar modes against the busiest
  // column, so a full-height bar always means the peak week.
  const peak = useMemo(
    () =>
      shaded
        ? grid.reduce(
            (max, column) =>
              column.reduce(
                (best, cell) => Math.max(best, cell.day?.tokens ?? 0),
                max,
              ),
            0,
          )
        : grid.reduce(
            (max, column) => Math.max(max, columnSummary(column).value),
            0,
          ),
    [grid, shaded],
  );
  const summaryTotal = useMemo(
    () => activitySummaryForMode(grid, mode),
    [grid, mode],
  );

  const dateFormatter = useMemo(
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
        weeks: formatProfileCount(grid.length, "week", locale),
      })}
      action={
        <div className="flex flex-wrap items-center justify-end gap-2">
          {shaded ? null : (
            <div className="hub-tab-toggle inline-flex h-8 w-fit items-center rounded-full">
              <button
                type="button"
                onClick={() => setShowChart(!showChart)}
                aria-pressed={showChart}
                title={t("settings.profile.stats.chartToggleHint")}
                className={cn(
                  "inline-flex h-8 items-center rounded-full px-3 text-ui-13 font-medium transition-colors",
                  showChart
                    ? "hub-tab-toggle-pill text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {t("settings.profile.stats.chartToggle")}
              </button>
            </div>
          )}
          <div className="hub-tab-toggle inline-flex h-8 w-fit items-center rounded-full">
            {MODES.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setMode(option)}
                aria-pressed={mode === option}
                className={cn(
                  "inline-flex h-8 items-center rounded-full px-3 text-ui-13 font-medium transition-colors",
                  mode === option
                    ? "hub-tab-toggle-pill text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {t(`settings.profile.stats.mode.${option}`)}
              </button>
            ))}
          </div>
        </div>
      }
    >
      {/* Measured, never scrolled: the grid is trimmed to fit instead. */}
      <div ref={ref} className="w-full overflow-hidden">
        {charted ? (
          <WeekChart grid={grid} mode={mode} dateFormatter={dateFormatter} />
        ) : (
          <>
            <div className="flex gap-[3px]">
              {grid.map((column) =>
                shaded ? (
                  <DayColumn
                    key={column[0]?.key ?? "column"}
                    column={column}
                    peak={peak}
                    dateFormatter={dateFormatter}
                  />
                ) : (
                  <BarColumn
                    key={column[0]?.key ?? "column"}
                    column={column}
                    peak={peak}
                    dateFormatter={dateFormatter}
                  />
                ),
              )}
            </div>

            <div className="relative mt-2 h-4">
              {monthLabels.map((label) => (
                <span
                  key={label.key}
                  className="absolute top-0 text-ui-11 text-muted-foreground"
                  style={{ left: `${label.column * COLUMN_WIDTH}px` }}
                >
                  {label.text}
                </span>
              ))}
            </div>
          </>
        )}
      </div>
    </StatsCard>
  );
}
