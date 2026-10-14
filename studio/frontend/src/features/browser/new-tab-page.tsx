// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  Cancel01Icon,
  Clock01Icon,
  StarIcon,
  Delete02Icon,
  Download01Icon,
  ViewOffSlashIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactNode, useMemo, useState } from "react";
import { hostOf } from "./address";
import { type HistoryItem, useBrowserHistoryStore } from "./history-store";
import { LinkContextMenu, MenuRow } from "./link-context-menu";
import { useNativeBrowser } from "./native-support";
import { useBrowserPrefsStore } from "./prefs-store";
import { SiteFavicon } from "./site-favicon";
import { useBrowserStore } from "./store";

type Site = { title: string; url: string; icon?: string };

// Unsloth's sites have no /favicon.ico, so they use Studio's own sticker.
const UNSLOTH_SITES: Site[] = [
  { title: "Unsloth", url: "https://unsloth.ai", icon: "/sticker.png" },
  {
    title: "Unsloth Docs",
    url: "https://docs.unsloth.ai",
    icon: "/sticker.png",
  },
];

const OTHER_SITES: Site[] = [
  {
    title: "Unsloth on GitHub",
    url: "https://github.com/unslothai/unsloth",
    icon: "https://github.com/fluidicon.png",
  },
  { title: "Hugging Face", url: "https://huggingface.co/unsloth" },
];

const KNOWN_SITES = [...UNSLOTH_SITES, ...OTHER_SITES];

const SUGGESTED_COUNT = 4;
const RECENTS_PER_PAGE = 5;

/** Most visited sites first, topped up with the defaults, leaving out the ones taken off. */
function suggestedSites(
  history: HistoryItem[],
  hidden: readonly string[],
  defaults: Site[],
): Site[] {
  const byHost = new Map<string, { site: Site; visits: number }>();
  for (const item of history) {
    const host = hostOf(item.url);
    const seen = byHost.get(host);
    if (seen) seen.visits += 1;
    else {
      const icon = KNOWN_SITES.find(
        (site) => hostOf(site.url) === host,
      )?.icon;
      byHost.set(host, {
        site: { title: item.title || host, url: item.url, icon },
        visits: 1,
      });
    }
  }
  const sites = [...byHost.values()]
    .filter((site) => site.visits > 1)
    .sort((a, b) => b.visits - a.visits)
    .map((site) => site.site);
  const hosts = new Set(sites.map((site) => hostOf(site.url)));
  for (const site of defaults)
    if (!hosts.has(hostOf(site.url))) sites.push(site);
  const off = new Set(hidden);
  return sites
    .filter((site) => !off.has(hostOf(site.url)))
    .slice(0, SUGGESTED_COUNT);
}

function recentPages(history: HistoryItem[]): HistoryItem[] {
  const seen = new Set<string>();
  return history.filter((item) => !seen.has(item.url) && seen.add(item.url));
}

function SiteIcon({ site }: { site: Site }) {
  return (
    <SiteFavicon
      url={site.url}
      icon={site.icon}
      className="size-9 rounded-lg"
      fallbackClassName="size-9 text-foreground"
    />
  );
}

function SuggestedSite({ site, tabId }: { site: Site; tabId: string }) {
  const t = useT();
  const { navigate } = useBrowserStore.getState();
  const remove = () =>
    useBrowserPrefsStore.getState().hideSuggestion(hostOf(site.url));
  return (
    <LinkContextMenu
      url={site.url}
      tabId={tabId}
      extra={
        <MenuRow icon={ViewOffSlashIcon} onSelect={remove}>
          {t("browser.suggestedMenu.remove")}
        </MenuRow>
      }
    >
      <div className="group/site relative min-w-0">
        <button
          type="button"
          title={`${site.title}\n${site.url}`}
          onClick={() => navigate(tabId, { url: site.url })}
          className="flex w-full min-w-0 cursor-pointer flex-col items-center gap-4 rounded-2xl px-2 pb-4 pt-5 transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-data-[state=open]/site:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]"
        >
          <SiteIcon site={site} />
          <span className="w-full truncate text-center text-ui-15 text-foreground">
            {site.title}
          </span>
        </button>
        <button
          type="button"
          aria-label={t("browser.suggestedMenu.remove")}
          title={t("browser.suggestedMenu.remove")}
          onClick={remove}
          className="absolute right-1.5 top-1.5 flex size-6 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-opacity hover:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/site:opacity-100"
        >
          <HugeiconsIcon
            icon={Cancel01Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </button>
      </div>
    </LinkContextMenu>
  );
}

function SectionTitle({
  children,
  actions,
}: { children: string; actions?: ReactNode }) {
  return (
    <div className="flex h-8 items-center justify-between">
      <h2 className="text-ui-15 font-medium text-foreground">{children}</h2>
      {actions}
    </div>
  );
}

function ToolButton({
  icon,
  label,
  onClick,
}: { icon: IconSvgElement; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex h-11 min-w-0 cursor-pointer items-center gap-3 rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] px-3.5 text-start text-ui-15 text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-5 shrink-0 text-muted-foreground"
      />
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </button>
  );
}

function PageButton({
  label,
  icon,
  disabled,
  onClick,
}: {
  label: string;
  icon: IconSvgElement;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="flex size-8 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground disabled:cursor-default disabled:opacity-35 disabled:hover:bg-transparent"
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4.5" />
    </button>
  );
}

export function NewTabPage({ tabId }: { tabId: string }) {
  const t = useT();
  const locale = useLocale();
  const history = useBrowserHistoryStore((state) => state.history);
  const { navigate, openInternal } = useBrowserStore.getState();
  const [page, setPage] = useState(0);
  const hidden = useBrowserPrefsStore((state) => state.hiddenSuggestions);
  const showSuggested = useBrowserPrefsStore((state) => state.showSuggestedSites);
  const showRecents = useBrowserPrefsStore((state) => state.showRecentPages);
  // Cloudflare blocks Unsloth's sites in the proxied view, so only native views suggest them.
  const native = useNativeBrowser((state) => state.enabled);
  const sites = useMemo(
    () => suggestedSites(history, hidden, native ? KNOWN_SITES : OTHER_SITES),
    [history, hidden, native],
  );
  const recents = useMemo(() => recentPages(history), [history]);
  // A recent stands for its page, so taking it off takes every visit to that page.
  const removeFromHistory = (url: string) =>
    useBrowserHistoryStore
      .getState()
      .removeVisits(
        new Set(
          history.filter((item) => item.url === url).map((item) => item.id),
        ),
      );
  const pages = Math.max(1, Math.ceil(recents.length / RECENTS_PER_PAGE));
  const shownPage = Math.min(page, pages - 1);
  const shown = recents.slice(
    shownPage * RECENTS_PER_PAGE,
    (shownPage + 1) * RECENTS_PER_PAGE,
  );
  const timeFormat = useMemo(
    () => new Intl.DateTimeFormat(locale, { timeStyle: "short" }),
    [locale],
  );
  const dateFormat = useMemo(
    () => new Intl.DateTimeFormat(locale, { month: "short", day: "numeric" }),
    [locale],
  );
  const today = new Date().toDateString();
  const when = (time: number) =>
    new Date(time).toDateString() === today
      ? timeFormat.format(time)
      : dateFormat.format(time);

  return (
    <div className="size-full overflow-auto">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-9 px-6 pb-10 pt-8">
        <section className="flex flex-col gap-3">
          <SectionTitle>{t("browser.tools")}</SectionTitle>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <ToolButton
              icon={StarIcon}
              label={t("browser.pages.bookmarks")}
              onClick={() => openInternal("bookmarks")}
            />
            <ToolButton
              icon={Clock01Icon}
              label={t("browser.pages.history")}
              onClick={() => openInternal("history")}
            />
            <ToolButton
              icon={Download01Icon}
              label={t("browser.pages.downloads")}
              onClick={() => openInternal("downloads")}
            />
          </div>
        </section>
        {showSuggested && sites.length > 0 ? (
          <section className="flex flex-col gap-3">
            <SectionTitle>{t("browser.suggested")}</SectionTitle>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {sites.map((site) => (
                <SuggestedSite key={site.url} site={site} tabId={tabId} />
              ))}
            </div>
          </section>
        ) : null}
        {showRecents && recents.length > 0 ? (
          <section className="flex flex-col gap-3">
            <SectionTitle
              actions={
                <div className="flex items-center gap-1">
                  <PageButton
                    label={t("browser.recentNewer")}
                    icon={ArrowUp01Icon}
                    disabled={shownPage === 0}
                    onClick={() => setPage(shownPage - 1)}
                  />
                  <PageButton
                    label={t("browser.recentOlder")}
                    icon={ArrowDown01Icon}
                    disabled={shownPage >= pages - 1}
                    onClick={() => setPage(shownPage + 1)}
                  />
                </div>
              }
            >
              {t("browser.recents")}
            </SectionTitle>
            <div className="flex flex-col gap-1">
              {shown.map((item) => (
                <LinkContextMenu
                  key={item.id}
                  url={item.url}
                  tabId={tabId}
                  extra={
                    <MenuRow
                      icon={Delete02Icon}
                      onSelect={() => removeFromHistory(item.url)}
                    >
                      {t("browser.pages.removeFromHistory")}
                    </MenuRow>
                  }
                >
                  <button
                    type="button"
                    title={item.url}
                    onClick={() => navigate(tabId, { url: item.url })}
                    className={cn(
                      "flex min-w-0 cursor-pointer items-center gap-4 rounded-2xl px-3 py-3 text-start transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]",
                    )}
                  >
                    <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)]">
                      <SiteFavicon
                        url={item.url}
                        className="size-5 rounded-[4px]"
                        fallbackClassName="size-5 text-muted-foreground"
                      />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-ui-15 text-foreground">
                        {item.title || hostOf(item.url)}
                      </span>
                      <span className="block truncate text-ui-13 text-muted-foreground">
                        {t("browser.website")}
                      </span>
                    </span>
                    <span className="shrink-0 text-ui-13 tabular-nums text-muted-foreground">
                      {when(item.visitedAt)}
                    </span>
                  </button>
                </LinkContextMenu>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
