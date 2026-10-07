// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Popover, PopoverAnchor, PopoverContent } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ATTACHMENT_KIND_ICONS, ATTACHMENT_KIND_ICON_CLASS, attachmentFileKind } from "@/features/chat";
import { useLocale, useT } from "@/i18n";
import { useSettingsDialogStore } from "@/features/settings";
import { isTauri } from "@/lib/api-base";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowRight01Icon,
  ArrowUp01Icon,
  Cancel01Icon,
  Delete02Icon,
  FilterMailIcon,
  Folder01Icon,
  InternetIcon,
  LinkSquare02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DateRange } from "react-day-picker";
import { hostOf } from "./address";
import { BookmarkEditPopover, bookmarkTitle, removeBookmarkWithUndo } from "./bookmarks";
import { type Bookmark, useBrowserBookmarksStore } from "./bookmarks-store";
import { ClearBrowsingDataDialog } from "./clear-data-dialog";
import { type DownloadItem, type HistoryItem, useBrowserHistoryStore } from "./history-store";
import { LinkContextMenu, MenuRow } from "./link-context-menu";
import { nativeDownloadsExist, revealNativeDownload } from "./native-downloads";
import { SiteFavicon } from "./site-favicon";
import { type InternalPage, useBrowserStore } from "./store";

function startOfDay(time: number): number {
  const date = new Date(time);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

/** Local midnight `days` calendar days after `day`; a fixed 24 h misses it across a daylight saving change. */
function addDays(day: number, days: number): number {
  const date = new Date(day);
  date.setDate(date.getDate() + days);
  return date.getTime();
}

function formatSize(bytes: number, locale: string): string {
  const units = ["byte", "kilobyte", "megabyte", "gigabyte"] as const;
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit++;
  }
  return new Intl.NumberFormat(locale, {
    style: "unit",
    unit: units[unit],
    unitDisplay: "short",
    maximumFractionDigits: value < 10 && unit > 0 ? 1 : 0,
  }).format(value);
}

function PageShell({
  title,
  query,
  onQueryChange,
  onClear,
  clearLabel,
  empty,
  children,
}: {
  title: string;
  query: string;
  onQueryChange: (query: string) => void;
  /** Left out where clearing everything at once would be a loss, as for bookmarks. */
  onClear?: () => void;
  clearLabel?: string;
  empty: boolean;
  children: ReactNode;
}) {
  const t = useT();
  return (
    <div className="size-full overflow-auto bg-background">
      <div className="mx-auto flex w-full max-w-2xl flex-col gap-5 px-6 pb-10 pt-8">
        <div className="flex items-center gap-3">
          <h1 className="min-w-0 flex-1 truncate text-ui-18 font-medium text-foreground">{title}</h1>
          {onClear ? (
            <Button type="button" variant="ghost" size="sm" disabled={empty} onClick={onClear}>
              {clearLabel}
            </Button>
          ) : null}
        </div>
        <label className="flex h-9 items-center gap-2 rounded-full border border-border/80 bg-card px-3.5 dark:border-transparent dark:bg-accent">
          <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4 shrink-0 text-muted-foreground" />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={t("browser.pages.search")}
            aria-label={t("browser.pages.search")}
            spellCheck={false}
            className="min-w-0 flex-1 bg-transparent text-ui-13 outline-none placeholder:text-muted-foreground"
          />
        </label>
        {children}
      </div>
    </div>
  );
}

function Row({
  icon,
  title,
  detail,
  time,
  url,
  onOpen,
  onRemove,
  menu,
  anchor,
}: {
  icon: ReactNode;
  title: string;
  detail: string;
  time: string;
  /** The page it came from, for the right-click menu; null for a file from chat. */
  url: string | null;
  onOpen?: () => void;
  onRemove: () => void;
  /** Rows of the right-click menu ahead of Remove, and what to do with focus as it closes. */
  menu?: { rows: ReactNode; onCloseAutoFocus?: (event: Event) => void };
  /** Something placed over the row, such as the anchor of a popover it opens. */
  anchor?: ReactNode;
}) {
  const t = useT();
  return (
    <LinkContextMenu
      url={url}
      onCloseAutoFocus={menu?.onCloseAutoFocus}
      extra={
        <>
          {menu?.rows}
          <MenuRow icon={Delete02Icon} onSelect={onRemove}>
            {t("browser.pages.remove")}
          </MenuRow>
        </>
      }
    >
      <li className="group/row relative flex items-center gap-1 rounded-xl pr-1 hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]">
        <button
          type="button"
          onClick={onOpen}
          disabled={!onOpen}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 rounded-xl px-3 py-2 text-start focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default"
        >
          {icon}
          <span className="min-w-0 flex-1">
            <span className="block truncate text-ui-13p5 text-foreground">{title}</span>
            <span className="block truncate text-ui-12 text-muted-foreground">{detail}</span>
          </span>
          <span className="shrink-0 text-ui-12 tabular-nums text-muted-foreground">{time}</span>
        </button>
        <button
          type="button"
          aria-label={t("browser.pages.remove")}
          onClick={onRemove}
          className="flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 hover:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:opacity-100 group-hover/row:opacity-100"
        >
          <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} className="size-3.5" />
        </button>
        {anchor}
      </li>
    </LinkContextMenu>
  );
}

function Groups<Item extends { id: string }>({
  groups,
  emptyLabel,
  render,
}: {
  groups: { label: string; items: Item[] }[];
  emptyLabel: string;
  render: (item: Item) => ReactNode;
}) {
  if (groups.length === 0) {
    return <p className="py-10 text-center text-ui-13 text-muted-foreground">{emptyLabel}</p>;
  }
  return (
    <>
      {groups.map((group) => (
        <section key={group.label} className="flex flex-col gap-1">
          <h2 className="px-3 text-ui-12 font-medium text-muted-foreground">{group.label}</h2>
          <ul className="flex flex-col">{group.items.map(render)}</ul>
        </section>
      ))}
    </>
  );
}

const downloadTime = (item: DownloadItem) => item.downloadedAt;

// Washes carry the contrast gain, as elsewhere in Studio. Written out so Tailwind sees them.
const CARD_WASH = "bg-[color-mix(in_oklab,var(--foreground)_calc(3%*var(--contrast-wash-gain,1)),transparent)]";
const SELECTED_WASH = "bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)]";
const HOVER_WASH = "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]";
const OPEN_WASH = "data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]";
const BUTTON_WASH = "bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]";

type HistoryRange = "all" | "today" | "week" | "month" | "custom";
type PresetRange = Exclude<HistoryRange, "custom">;

const HISTORY_PAGE_ROWS = 100;

const RANGE_LABELS = {
  all: { menu: "browser.pages.allTime", heading: "browser.pages.allTimeHistory" },
  today: { menu: "browser.pages.today", heading: "browser.pages.today" },
  week: { menu: "browser.pages.lastWeek", heading: "browser.pages.lastWeek" },
  month: { menu: "browser.pages.lastMonth", heading: "browser.pages.lastMonth" },
} as const;

/** The visit times a range keeps: from `since` up to but not including `until`. */
function rangeBounds(range: HistoryRange, dates: DateRange | undefined): { since: number; until: number } {
  const today = startOfDay(Date.now());
  if (range === "today") return { since: today, until: Infinity };
  if (range === "week") return { since: addDays(today, -6), until: Infinity };
  if (range === "month") return { since: addDays(today, -29), until: Infinity };
  if (range === "custom" && dates?.from) {
    const since = startOfDay(dates.from.getTime());
    const last = startOfDay((dates.to ?? dates.from).getTime());
    return { since, until: addDays(last, 1) };
  }
  return { since: 0, until: Infinity };
}

function HistoryRow({
  item,
  tabId,
  time,
  selected,
  onSelect,
}: {
  item: HistoryItem;
  tabId: string;
  time: string;
  selected: boolean;
  onSelect: (selected: boolean) => void;
}) {
  const t = useT();
  const host = hostOf(item.url);
  const title = item.title || host;
  return (
    <LinkContextMenu
      url={item.url}
      tabId={tabId}
      extra={
        <MenuRow
          icon={Delete02Icon}
          onSelect={() => useBrowserHistoryStore.getState().removeVisit(item.id)}
        >
          {t("browser.pages.removeFromHistory")}
        </MenuRow>
      }
    >
      <li
        className={cn(
          "relative flex h-12 items-center gap-3 px-4 before:absolute before:inset-x-4 before:top-0 before:h-px before:bg-border",
          selected ? SELECTED_WASH : HOVER_WASH,
          "data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]",
        )}
      >
        <Checkbox
          checked={selected}
          onCheckedChange={(checked) => onSelect(checked === true)}
          aria-label={t("browser.pages.select", { title })}
        />
        <button
          type="button"
          onClick={() => useBrowserStore.getState().navigate(tabId, { url: item.url })}
          className="flex h-full min-w-0 flex-1 cursor-pointer items-center gap-3 text-start focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
        >
          <span className="flex size-5 shrink-0 items-center justify-center">
            <SiteFavicon
              url={item.url}
              className="size-4 rounded-[3px]"
              fallbackClassName="size-4 text-muted-foreground"
            />
          </span>
          <span className="min-w-0 max-w-[65%] shrink-0 truncate text-ui-14 text-foreground">{title}</span>
          <span className="min-w-0 flex-1 truncate text-ui-13 text-muted-foreground">{host}</span>
          <span className="shrink-0 text-ui-13 tabular-nums text-muted-foreground">{time}</span>
        </button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={t("browser.pages.pageActions", { title })}
              className={cn(
                "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-lg text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                HOVER_WASH,
                OPEN_WASH,
                "data-[state=open]:text-foreground",
              )}
            >
              <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={2} className="size-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="browser-menu min-w-48">
            <DropdownMenuItem onSelect={() => useBrowserStore.getState().openUrl(item.url, { newTab: true })}>
              <HugeiconsIcon icon={LinkSquare02Icon} strokeWidth={1.75} />
              {t("browser.pages.openPage")}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => useBrowserHistoryStore.getState().removeVisit(item.id)}>
              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} />
              {t("browser.pages.removeFromHistory")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </li>
    </LinkContextMenu>
  );
}

/** A day's entries in a card that folds shut. */
function DayCard({ label, children }: { label: string; children: ReactNode }) {
  const t = useT();
  const [open, setOpen] = useState(true);
  return (
    <section className={cn("overflow-hidden rounded-xl border border-border", CARD_WASH)}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label={t(open ? "browser.pages.collapse" : "browser.pages.expand", { date: label })}
        className="flex h-13 w-full cursor-pointer items-center justify-between px-4 text-start focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
      >
        <h2 className="text-ui-14 font-medium text-foreground">{label}</h2>
        <HugeiconsIcon
          icon={open ? ArrowUp01Icon : ArrowDown01Icon}
          strokeWidth={2}
          className="size-4 text-muted-foreground"
        />
      </button>
      {open ? <ul>{children}</ul> : null}
    </section>
  );
}

/** Settings > Browser > `current`, each step back a link. */
function PageCrumbs({ current }: { current: string }) {
  const t = useT();
  const openSettings = (tab?: "browser") => useSettingsDialogStore.getState().openDialog(tab);
  const crumb =
    "cursor-pointer rounded-md text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
  return (
    <nav className="flex items-center gap-2 px-6 pt-4 text-ui-14" aria-label={current}>
      <button type="button" className={crumb} onClick={() => openSettings()}>
        {t("settings.dialog.title")}
      </button>
      <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-3.5 text-muted-foreground" />
      <button type="button" className={crumb} onClick={() => openSettings("browser")}>
        {t("browser.settingsTitle")}
      </button>
      <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-3.5 text-muted-foreground" />
      <span className="text-foreground" aria-current="page">
        {current}
      </span>
    </nav>
  );
}

/** The time range a history page shows, and the heading that names it. */
function useRangeFilter() {
  const t = useT();
  const locale = useLocale();
  const [range, setRange] = useState<HistoryRange>("all");
  const [dates, setDates] = useState<DateRange | undefined>();
  const bounds = useMemo(() => rangeBounds(range, dates), [range, dates]);
  const heading = useMemo(() => {
    if (range !== "custom") return t(RANGE_LABELS[range].heading);
    if (!dates?.from) return t("browser.pages.customDates");
    const format = new Intl.DateTimeFormat(locale, { dateStyle: "medium" });
    return format.formatRange(dates.from, dates.to ?? dates.from);
  }, [range, dates, locale, t]);
  return { range, setRange, dates, setDates, bounds, heading };
}

type RangeFilter = ReturnType<typeof useRangeFilter>;

/** The search field, with the range filter (presets or chosen dates) at its end. */
function SearchWithFilter({
  query,
  onQueryChange,
  placeholder,
  filter,
}: {
  query: string;
  onQueryChange: (query: string) => void;
  placeholder: string;
  filter: RangeFilter;
}) {
  const t = useT();
  const { range, setRange, dates, setDates } = filter;
  const [datesOpen, setDatesOpen] = useState(false);
  // Opened once the filter menu has closed, so its focus return doesn't dismiss the picker.
  const openDatesOnClose = useRef(false);
  const filterOption = (label: string, checked: boolean, onSelect: () => void) => (
    <DropdownMenuItem key={label} role="menuitemradio" aria-checked={checked} onSelect={onSelect}>
      <span className="flex-1">{label}</span>
      <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className={cn("ml-2 size-4", !checked && "invisible")} />
    </DropdownMenuItem>
  );
  return (
    <Popover open={datesOpen} onOpenChange={setDatesOpen}>
      <PopoverAnchor asChild>
        <label className="mt-8 flex h-11 items-center gap-2.5 rounded-full border border-border pl-4 pr-1.5">
          <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4.5 shrink-0 text-muted-foreground" />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={placeholder}
            aria-label={placeholder}
            spellCheck={false}
            className="min-w-0 flex-1 bg-transparent text-ui-14 outline-none placeholder:text-muted-foreground"
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                aria-label={t("browser.pages.filter")}
                className={cn(
                  "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  HOVER_WASH,
                  range !== "all" && "text-foreground",
                )}
              >
                <HugeiconsIcon icon={FilterMailIcon} strokeWidth={1.75} className="size-5" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              sideOffset={4}
              className="browser-menu w-max min-w-44"
              onCloseAutoFocus={(event) => {
                if (!openDatesOnClose.current) return;
                openDatesOnClose.current = false;
                event.preventDefault();
                setDatesOpen(true);
              }}
            >
              {(Object.keys(RANGE_LABELS) as PresetRange[]).map((id) =>
                filterOption(t(RANGE_LABELS[id].menu), range === id, () => setRange(id)),
              )}
              <DropdownMenuSeparator className="mx-3" />
              {filterOption(t("browser.pages.customDates"), range === "custom", () => {
                openDatesOnClose.current = true;
              })}
              {range !== "all" && (
                <>
                  <DropdownMenuSeparator className="mx-3" />
                  <DropdownMenuItem
                    onSelect={() => {
                      setRange("all");
                      setDates(undefined);
                    }}
                  >
                    <span className="text-muted-foreground">{t("browser.pages.clearFilter")}</span>
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </label>
      </PopoverAnchor>
      <PopoverContent align="end" sideOffset={6} className="browser-menu w-auto rounded-2xl p-0">
        <Calendar
          mode="range"
          selected={dates}
          onSelect={(next) => {
            setDates(next);
            setRange(next?.from ? "custom" : "all");
          }}
          defaultMonth={dates?.from ?? new Date()}
          endMonth={new Date()}
          disabled={{ after: new Date() }}
          autoFocus
        />
      </PopoverContent>
    </Popover>
  );
}

/** Items in the range and matching the search, by day (newest first, as stored), up to `limit`. */
function useDays<Item>(
  items: Item[],
  timeOf: (item: Item) => number,
  matches: (item: Item) => boolean,
  bounds: { since: number; until: number },
  limit = Infinity,
) {
  const locale = useLocale();
  return useMemo(() => {
    const dateFormat = new Intl.DateTimeFormat(locale, { dateStyle: "medium" });
    const days: { key: number; label: string; items: Item[] }[] = [];
    let shown = 0;
    for (const item of items) {
      const time = timeOf(item);
      if (time < bounds.since || time >= bounds.until || !matches(item)) continue;
      if (shown === limit) return { groups: days, more: true };
      shown += 1;
      const key = startOfDay(time);
      const last = days[days.length - 1];
      if (last?.key === key) last.items.push(item);
      else days.push({ key, label: dateFormat.format(key), items: [item] });
    }
    return { groups: days, more: false };
  }, [items, timeOf, matches, bounds, locale, limit]);
}

const visitTime = (item: HistoryItem) => item.visitedAt;

function HistoryPage({ tabId }: { tabId: string }) {
  const t = useT();
  const locale = useLocale();
  const history = useBrowserHistoryStore((state) => state.history);
  const [query, setQuery] = useState("");
  const filter = useRangeFilter();
  const { range, heading } = filter;
  const [selection, setSelection] = useState<ReadonlySet<string>>(() => new Set());
  const [clearOpen, setClearOpen] = useState(false);
  const needle = query.trim().toLowerCase();
  const [limit, setLimit] = useState(HISTORY_PAGE_ROWS);
  const matches = useCallback(
    (item: HistoryItem) =>
      !needle || item.title.toLowerCase().includes(needle) || item.url.toLowerCase().includes(needle),
    [needle],
  );
  const { groups, more } = useDays(history, visitTime, matches, filter.bounds, limit);
  const timeFormat = useMemo(() => new Intl.DateTimeFormat(locale, { timeStyle: "short" }), [locale]);
  const selected = useMemo(() => {
    const ids = new Set(history.map((item) => item.id));
    return new Set([...selection].filter((id) => ids.has(id)));
  }, [history, selection]);
  const select = (id: string, on: boolean) =>
    setSelection((current) => {
      const next = new Set(current);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });

  return (
    <div className="size-full overflow-auto bg-background">
      <PageCrumbs current={t("browser.historySetting")} />
      <div className="mx-auto flex w-full max-w-3xl flex-col px-6 pb-12 pt-10">
        <h1 className="text-ui-30 font-medium text-foreground">{t("browser.historySetting")}</h1>
        <SearchWithFilter
          query={query}
          onQueryChange={setQuery}
          placeholder={t("browser.pages.searchHistory")}
          filter={filter}
        />
        <div className="mt-8 flex h-9 items-center gap-2">
          {selected.size > 0 ? (
            <>
              <span className="flex-1 text-ui-15 font-medium text-foreground">
                {t("browser.pages.selected", { count: selected.size })}
              </span>
              <Button type="button" variant="ghost" size="sm" className="rounded-full" onClick={() => setSelection(new Set())}>
                {t("browser.pages.cancel")}
              </Button>
              <Button
                type="button"
                variant="destructive"
                size="sm"
                className="rounded-full"
                onClick={() => {
                  useBrowserHistoryStore.getState().removeVisits(selected);
                  setSelection(new Set());
                }}
              >
                {t("browser.pages.remove")}
              </Button>
            </>
          ) : (
            <>
              <h2 className="flex-1 text-ui-15 font-medium text-foreground">{heading}</h2>
              <button
                type="button"
                disabled={history.length === 0}
                onClick={() => setClearOpen(true)}
                className={cn(
                  "h-9 cursor-pointer rounded-full px-4 text-ui-14 text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-50",
                  BUTTON_WASH,
                )}
              >
                {t("browser.menu.clearData")}
              </button>
            </>
          )}
        </div>
        <div className="mt-4 flex flex-col gap-3">
          {groups.length === 0 ? (
            <p className="py-10 text-center text-ui-13 text-muted-foreground">
              {t(needle || range !== "all" ? "browser.pages.noMatches" : "browser.pages.noHistory")}
            </p>
          ) : (
            groups.map((group) => (
              <DayCard key={group.key} label={group.label}>
                {group.items.map((item) => (
                  <HistoryRow
                    key={item.id}
                    item={item}
                    tabId={tabId}
                    time={timeFormat.format(item.visitedAt)}
                    selected={selected.has(item.id)}
                    onSelect={(on) => select(item.id, on)}
                  />
                ))}
              </DayCard>
            ))
          )}
          {more ? (
            <button
              type="button"
              onClick={() => setLimit((current) => current + HISTORY_PAGE_ROWS)}
              className={cn(
                "h-9 cursor-pointer self-center rounded-full px-4 text-ui-14 text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                BUTTON_WASH,
              )}
            >
              {t("browser.pages.showMore")}
            </button>
          ) : null}
        </div>
      </div>
      <ClearBrowsingDataDialog open={clearOpen} onOpenChange={setClearOpen} />
    </div>
  );
}

/** Each platform's own name for showing a file in its folder. */
function revealLabelKey() {
  const platform = typeof navigator === "undefined" ? "" : navigator.userAgent;
  if (/Mac/i.test(platform)) return "browser.pages.showInFinder" as const;
  if (/Windows/i.test(platform)) return "browser.pages.showInExplorer" as const;
  return "browser.pages.showInFolder" as const;
}

/** Desktop downloads missing from disk, by native id; rechecked on window focus. */
function useMissingDownloads(downloads: DownloadItem[]) {
  const [missing, setMissing] = useState<ReadonlySet<string>>(() => new Set());
  const [checks, setChecks] = useState(0);
  useEffect(() => {
    if (!isTauri) return;
    const ids = downloads.flatMap((item) => (item.nativeId ? [item.nativeId] : []));
    let live = true;
    const check = () =>
      nativeDownloadsExist(ids).then(
        (found) => live && setMissing(new Set(ids.filter((_, index) => !found[index]))),
        () => undefined,
      );
    void check();
    window.addEventListener("focus", check);
    return () => {
      live = false;
      window.removeEventListener("focus", check);
    };
  }, [downloads, checks]);
  return { missing, recheck: () => setChecks((count) => count + 1) };
}

const ROW_ACTION = cn(
  "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-lg text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
  HOVER_WASH,
);

function DownloadRow({
  item,
  time,
  deleted,
  onRevealFailed,
}: {
  item: DownloadItem;
  time: string;
  deleted: boolean;
  onRevealFailed: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const kind = attachmentFileKind(item.name, item.contentType);
  const source = item.url ? hostOf(item.url) : t("browser.pages.fromChat");
  const revealLabel = t(revealLabelKey());
  const remove = () => useBrowserHistoryStore.getState().removeDownload(item.id);
  const nativeId = isTauri && !deleted ? item.nativeId : undefined;
  const reveal = nativeId
    ? () =>
        void revealNativeDownload(nativeId).catch(() => {
          toast.error(t("browser.pages.revealFailed", { name: item.name }));
          onRevealFailed();
        })
    : undefined;
  // Files aren't kept; the row reopens the source page.
  const open = item.url ? () => useBrowserStore.getState().openUrl(item.url ?? "", { newTab: true }) : undefined;
  return (
    <LinkContextMenu
      url={item.url}
      extra={
        <>
          {reveal ? (
            <MenuRow icon={Folder01Icon} onSelect={reveal}>
              {revealLabel}
            </MenuRow>
          ) : null}
          <MenuRow icon={Delete02Icon} onSelect={remove}>
            {t("browser.pages.removeFromHistory")}
          </MenuRow>
        </>
      }
    >
      <li
        className={cn(
          "relative flex h-16 items-center gap-3 px-4 before:absolute before:inset-x-4 before:top-0 before:h-px before:bg-border",
          HOVER_WASH,
          "data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]",
        )}
      >
        <button
          type="button"
          onClick={open}
          disabled={!open}
          title={`${formatSize(item.size, locale)} · ${source}`}
          className="flex h-full min-w-0 flex-1 cursor-pointer items-center gap-3 text-start focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring disabled:cursor-default"
        >
          <span
            className={cn(
              "flex size-10 shrink-0 items-center justify-center rounded-lg",
              "bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)]",
            )}
          >
            <HugeiconsIcon
              icon={ATTACHMENT_KIND_ICONS[kind]}
              strokeWidth={1.75}
              className={cn("size-4.5", deleted ? "text-muted-foreground" : ATTACHMENT_KIND_ICON_CLASS[kind])}
            />
          </span>
          <span
            className={cn(
              "min-w-0 truncate text-ui-14",
              deleted ? "text-muted-foreground line-through" : "text-foreground",
            )}
          >
            {item.name}
          </span>
          {deleted ? (
            <span className="shrink-0 text-ui-14 text-muted-foreground">{t("browser.pages.deleted")}</span>
          ) : null}
          <span className="ml-auto shrink-0 pl-3 text-ui-13 tabular-nums text-muted-foreground">{time}</span>
        </button>
        {reveal ? (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button type="button" aria-label={revealLabel} onClick={reveal} className={ROW_ACTION}>
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-4.5" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="tooltip-compact">{revealLabel}</TooltipContent>
          </Tooltip>
        ) : null}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={t("browser.pages.removeFromHistory")}
              onClick={remove}
              className={ROW_ACTION}
            >
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="tooltip-compact">{t("browser.pages.removeFromHistory")}</TooltipContent>
        </Tooltip>
      </li>
    </LinkContextMenu>
  );
}

const nameMatches = (needle: string) => (item: DownloadItem) =>
  !needle ||
  item.name.toLowerCase().includes(needle) ||
  (item.url?.toLowerCase().includes(needle) ?? false);

function DownloadsPage() {
  const t = useT();
  const locale = useLocale();
  const downloads = useBrowserHistoryStore((state) => state.downloads);
  const [query, setQuery] = useState("");
  const filter = useRangeFilter();
  const needle = query.trim().toLowerCase();
  const matches = useMemo(() => nameMatches(needle), [needle]);
  const { groups } = useDays(downloads, downloadTime, matches, filter.bounds);
  const timeFormat = useMemo(() => new Intl.DateTimeFormat(locale, { timeStyle: "short" }), [locale]);
  const { missing, recheck } = useMissingDownloads(downloads);
  return (
    <div className="size-full overflow-auto bg-background">
      <PageCrumbs current={t("browser.downloadsSetting")} />
      <div className="mx-auto flex w-full max-w-3xl flex-col px-6 pb-12 pt-10">
        <h1 className="text-ui-30 font-medium text-foreground">{t("browser.downloadsSetting")}</h1>
        <SearchWithFilter
          query={query}
          onQueryChange={setQuery}
          placeholder={t("browser.pages.searchDownloads")}
          filter={filter}
        />
        <div className="mt-8 flex h-9 items-center gap-2">
          <h2 className="flex-1 text-ui-15 font-medium text-foreground">{filter.heading}</h2>
          <button
            type="button"
            disabled={downloads.length === 0}
            onClick={() => useBrowserHistoryStore.getState().clearDownloads()}
            className={cn(
              "h-9 cursor-pointer rounded-full px-4 text-ui-14 text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-50",
              BUTTON_WASH,
            )}
          >
            {t("browser.pages.clearAll")}
          </button>
        </div>
        <div className="mt-4 flex flex-col gap-3">
          {groups.length === 0 ? (
            <p className="py-10 text-center text-ui-13 text-muted-foreground">
              {t(needle || filter.range !== "all" ? "browser.pages.noMatches" : "browser.pages.noDownloads")}
            </p>
          ) : (
            groups.map((group) => (
              <DayCard key={group.key} label={group.label}>
                {group.items.map((item) => (
                  <DownloadRow
                    key={item.id}
                    item={item}
                    time={timeFormat.format(item.downloadedAt)}
                    deleted={item.nativeId !== undefined && missing.has(item.nativeId)}
                    onRevealFailed={recheck}
                  />
                ))}
              </DayCard>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function BookmarkRow({ bookmark, tabId, time }: { bookmark: Bookmark; tabId: string; time: string }) {
  const t = useT();
  const [editing, setEditing] = useState(false);
  const editChosen = useRef(false);
  return (
    <BookmarkEditPopover bookmark={bookmark} open={editing} onOpenChange={setEditing} align="start">
      <Row
        icon={
          <SiteFavicon
            url={bookmark.url}
            icon={bookmark.icon}
            className="size-4 rounded-[3px]"
            fallbackClassName="size-4 text-muted-foreground"
          />
        }
        title={bookmarkTitle(bookmark)}
        detail={bookmark.url}
        time={time}
        url={bookmark.url}
        onOpen={() => useBrowserStore.getState().navigate(tabId, { url: bookmark.url })}
        onRemove={() => removeBookmarkWithUndo(bookmark, t)}
        menu={{
          rows: (
            <MenuRow
              icon={PencilEdit02Icon}
              onSelect={() => {
                editChosen.current = true;
                setEditing(true);
              }}
            >
              {t("browser.bookmarks.edit")}
            </MenuRow>
          ),
          onCloseAutoFocus: (event) => {
            // Focus stays in the editor the menu opened.
            if (!editChosen.current) return;
            editChosen.current = false;
            event.preventDefault();
          },
        }}
        anchor={<PopoverAnchor className="pointer-events-none absolute inset-x-3 bottom-0 h-0" />}
      />
    </BookmarkEditPopover>
  );
}

function BookmarksPage({ tabId }: { tabId: string }) {
  const t = useT();
  const locale = useLocale();
  const bookmarks = useBrowserBookmarksStore((state) => state.bookmarks);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const groups = useMemo(() => {
    const matches = needle
      ? bookmarks.filter(
          (bookmark) =>
            bookmark.title.toLowerCase().includes(needle) || bookmark.url.toLowerCase().includes(needle),
        )
      : bookmarks;
    return (["toolbar", "other"] as const)
      .map((folder) => ({
        label: t(folder === "toolbar" ? "browser.bookmarks.toolbar" : "browser.bookmarks.other"),
        items: matches.filter((bookmark) => bookmark.folder === folder),
      }))
      .filter((group) => group.items.length > 0);
  }, [bookmarks, needle, t]);
  const dateFormat = useMemo(() => new Intl.DateTimeFormat(locale, { dateStyle: "medium" }), [locale]);
  return (
    <PageShell title={t("browser.pages.bookmarks")} query={query} onQueryChange={setQuery} empty={bookmarks.length === 0}>
      <Groups
        groups={groups}
        emptyLabel={t(needle ? "browser.pages.noMatches" : "browser.pages.noBookmarks")}
        render={(bookmark) => (
          <BookmarkRow
            key={bookmark.id}
            bookmark={bookmark}
            tabId={tabId}
            time={dateFormat.format(bookmark.addedAt)}
          />
        )}
      />
    </PageShell>
  );
}

export function InternalPageView({ page, tabId }: { page: InternalPage; tabId: string }) {
  if (page === "history") return <HistoryPage tabId={tabId} />;
  if (page === "bookmarks") return <BookmarksPage tabId={tabId} />;
  return <DownloadsPage />;
}
