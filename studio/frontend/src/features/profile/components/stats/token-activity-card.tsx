// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ProfileStatsDay } from "../../api/profile-stats";
import {
  type ActivityMode,
  formatCompactNumber,
  formatProfileCount,
  heatLevel,
  parseDayKey,
  seriesForMode,
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

export function TokenActivityCard({ daily }: { daily: ProfileStatsDay[] }) {
  const t = useT();
  const [mode, setMode] = useState<ActivityMode>("daily");
  const maxColumns = Math.ceil(daily.length / DAYS_PER_WEEK) + 1;
  const { ref, columns } = useVisibleColumns(maxColumns);

  const shaded = mode === "daily";
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
  const visibleTotal = useMemo(
    () =>
      grid.reduce(
        (sum, column) =>
          column.reduce((total, cell) => total + (cell.day?.tokens ?? 0), sum),
        0,
      ),
    [grid],
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
      description={t("settings.profile.stats.activityDescription", {
        total: formatProfileCount(
          visibleTotal,
          "token",
          locale,
          formatCompactNumber(visibleTotal, locale),
        ),
        weeks: formatProfileCount(grid.length, "week", locale),
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
      }
    >
      {/* Measured, never scrolled: the grid is trimmed to fit instead. */}
      <div ref={ref} className="w-full overflow-hidden">
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
      </div>
    </StatsCard>
  );
}
