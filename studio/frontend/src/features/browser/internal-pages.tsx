// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ATTACHMENT_KIND_ICONS, ATTACHMENT_KIND_ICON_CLASS, attachmentFileKind } from "@/features/chat";
import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Cancel01Icon, InternetIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useMemo, useState } from "react";
import { hostOf } from "./address";
import { type DownloadItem, type HistoryItem, useBrowserHistoryStore } from "./history-store";
import { type InternalPage, useBrowserStore } from "./store";

const DAY_MS = 24 * 60 * 60 * 1000;

function startOfDay(time: number): number {
  const date = new Date(time);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

/** Items grouped under Today, Yesterday or their date, newest first. */
function useDayGroups<Item>(items: Item[], timeOf: (item: Item) => number) {
  const t = useT();
  const locale = useLocale();
  return useMemo(() => {
    const today = startOfDay(Date.now());
    const dateFormat = new Intl.DateTimeFormat(locale, { weekday: "long", month: "long", day: "numeric" });
    const groups: { label: string; items: Item[] }[] = [];
    for (const item of items) {
      const day = startOfDay(timeOf(item));
      const label =
        day === today
          ? t("browser.pages.today")
          : day === today - DAY_MS
            ? t("browser.pages.yesterday")
            : dateFormat.format(day);
      const last = groups[groups.length - 1];
      if (last?.label === label) last.items.push(item);
      else groups.push({ label, items: [item] });
    }
    return groups;
  }, [items, timeOf, locale, t]);
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
  onClear: () => void;
  clearLabel: string;
  empty: boolean;
  children: ReactNode;
}) {
  const t = useT();
  return (
    <div className="size-full overflow-auto bg-background">
      <div className="mx-auto flex w-full max-w-2xl flex-col gap-5 px-6 pb-10 pt-8">
        <div className="flex items-center gap-3">
          <h1 className="min-w-0 flex-1 truncate text-ui-18 font-medium text-foreground">{title}</h1>
          <Button type="button" variant="ghost" size="sm" disabled={empty} onClick={onClear}>
            {clearLabel}
          </Button>
        </div>
        <label className="flex h-9 items-center gap-2 rounded-full border border-border/80 bg-card px-3.5 focus-within:ring-2 focus-within:ring-ring/40 dark:border-transparent dark:bg-accent">
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
  onOpen,
  onRemove,
}: {
  icon: ReactNode;
  title: string;
  detail: string;
  time: string;
  onOpen?: () => void;
  onRemove: () => void;
}) {
  const t = useT();
  return (
    <li className="group/row flex items-center gap-1 rounded-xl pr-1 hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]">
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
    </li>
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

const historyTime = (item: HistoryItem) => item.visitedAt;
const downloadTime = (item: DownloadItem) => item.downloadedAt;

function HistoryPage({ tabId }: { tabId: string }) {
  const t = useT();
  const locale = useLocale();
  const history = useBrowserHistoryStore((state) => state.history);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const matches = useMemo(
    () =>
      needle
        ? history.filter((item) => item.title.toLowerCase().includes(needle) || item.url.toLowerCase().includes(needle))
        : history,
    [history, needle],
  );
  const groups = useDayGroups(matches, historyTime);
  const timeFormat = useMemo(() => new Intl.DateTimeFormat(locale, { timeStyle: "short" }), [locale]);
  const { removeVisit, clearHistory } = useBrowserHistoryStore.getState();
  return (
    <PageShell
      title={t("browser.pages.history")}
      query={query}
      onQueryChange={setQuery}
      onClear={clearHistory}
      clearLabel={t("browser.pages.clearHistory")}
      empty={history.length === 0}
    >
      <Groups
        groups={groups}
        emptyLabel={t(needle ? "browser.pages.noMatches" : "browser.pages.noHistory")}
        render={(item) => (
          <Row
            key={item.id}
            icon={<HugeiconsIcon icon={InternetIcon} strokeWidth={1.75} className="size-4 shrink-0 text-muted-foreground" />}
            title={item.title || hostOf(item.url)}
            detail={hostOf(item.url)}
            time={timeFormat.format(item.visitedAt)}
            onOpen={() => useBrowserStore.getState().navigate(tabId, { url: item.url })}
            onRemove={() => removeVisit(item.id)}
          />
        )}
      />
    </PageShell>
  );
}

function DownloadsPage() {
  const t = useT();
  const locale = useLocale();
  const downloads = useBrowserHistoryStore((state) => state.downloads);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const matches = useMemo(
    () => (needle ? downloads.filter((item) => item.name.toLowerCase().includes(needle)) : downloads),
    [downloads, needle],
  );
  const groups = useDayGroups(matches, downloadTime);
  const timeFormat = useMemo(() => new Intl.DateTimeFormat(locale, { timeStyle: "short" }), [locale]);
  const { removeDownload, clearDownloads } = useBrowserHistoryStore.getState();
  return (
    <PageShell
      title={t("browser.pages.downloads")}
      query={query}
      onQueryChange={setQuery}
      onClear={clearDownloads}
      clearLabel={t("browser.pages.clearDownloads")}
      empty={downloads.length === 0}
    >
      <Groups
        groups={groups}
        emptyLabel={t(needle ? "browser.pages.noMatches" : "browser.pages.noDownloads")}
        render={(item) => {
          const kind = attachmentFileKind(item.name, item.contentType);
          const source = item.url ? hostOf(item.url) : t("browser.pages.fromChat");
          return (
            <Row
              key={item.id}
              icon={
                <HugeiconsIcon
                  icon={ATTACHMENT_KIND_ICONS[kind]}
                  strokeWidth={1.75}
                  className={cn("size-4 shrink-0", ATTACHMENT_KIND_ICON_CLASS[kind])}
                />
              }
              title={item.name}
              detail={`${formatSize(item.size, locale)} · ${source}`}
              time={timeFormat.format(item.downloadedAt)}
              // Files are not kept; reopen the page they came from.
              onOpen={item.url ? () => useBrowserStore.getState().openUrl(item.url ?? "", { newTab: true }) : undefined}
              onRemove={() => removeDownload(item.id)}
            />
          );
        }}
      />
    </PageShell>
  );
}

export function InternalPageView({ page, tabId }: { page: InternalPage; tabId: string }) {
  return page === "history" ? <HistoryPage tabId={tabId} /> : <DownloadsPage />;
}
