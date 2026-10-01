// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  Clock01Icon,
  Download01Icon,
  InternetIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { hostOf } from "./address";
import { proxiedFavicon } from "./favicon";
import { type HistoryItem, useBrowserHistoryStore } from "./history-store";
import { useBrowserStore } from "./store";

// Unsloth's sites have no /favicon.ico, so they use Studio's own sticker.
const DEFAULT_SITES: Site[] = [
  { title: "Unsloth", url: "https://unsloth.ai", icon: "/sticker.png" },
  {
    title: "Unsloth Docs",
    url: "https://docs.unsloth.ai",
    icon: "/sticker.png",
  },
  {
    title: "Unsloth on GitHub",
    url: "https://github.com/unslothai/unsloth",
    icon: "https://github.com/fluidicon.png",
  },
  { title: "Hugging Face", url: "https://huggingface.co/unsloth" },
];

type Site = { title: string; url: string; icon?: string };

const SUGGESTED_COUNT = 4;
const RECENTS_PER_PAGE = 5;

/** Most visited sites first, topped up with the defaults. */
function suggestedSites(history: HistoryItem[]): Site[] {
  const byHost = new Map<string, { site: Site; visits: number }>();
  for (const item of history) {
    const host = hostOf(item.url);
    const seen = byHost.get(host);
    if (seen) seen.visits += 1;
    else {
      const icon = DEFAULT_SITES.find(
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
  for (const site of DEFAULT_SITES)
    if (!hosts.has(hostOf(site.url))) sites.push(site);
  return sites.slice(0, SUGGESTED_COUNT);
}

/** The latest visit to each page, newest first. */
function recentPages(history: HistoryItem[]): HistoryItem[] {
  const seen = new Set<string>();
  return history.filter((item) => !seen.has(item.url) && seen.add(item.url));
}

function SiteIcon({ site }: { site: Site }) {
  const local = site.icon?.startsWith("/") ? site.icon : null;
  const [icon, setIcon] = useState<string | null>(local);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (local) return;
    let live = true;
    void proxiedFavicon(
      site.icon ?? `${new URL(site.url).origin}/favicon.ico`,
    ).then((blobUrl) => {
      if (!live) return;
      setIcon(blobUrl);
      setFailed(!blobUrl);
    });
    return () => {
      live = false;
    };
  }, [local, site.icon, site.url]);
  if (failed || !icon) {
    return (
      <HugeiconsIcon
        icon={InternetIcon}
        strokeWidth={1.5}
        className="size-9 text-foreground"
      />
    );
  }
  return (
    <img
      src={icon}
      alt=""
      onError={() => setFailed(true)}
      className="size-9 rounded-lg object-contain"
    />
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

/** A new tab: tools, suggested sites and recent pages. */
export function NewTabPage({ tabId }: { tabId: string }) {
  const t = useT();
  const locale = useLocale();
  const history = useBrowserHistoryStore((state) => state.history);
  const { navigate, openInternal } = useBrowserStore.getState();
  const [page, setPage] = useState(0);
  const sites = useMemo(() => suggestedSites(history), [history]);
  const recents = useMemo(() => recentPages(history), [history]);
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
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
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
        <section className="flex flex-col gap-3">
          <SectionTitle>{t("browser.suggested")}</SectionTitle>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {sites.map((site) => (
              <button
                key={site.url}
                type="button"
                title={site.url}
                onClick={() => navigate(tabId, { url: site.url })}
                className="flex min-w-0 cursor-pointer flex-col items-center gap-4 rounded-2xl px-2 pb-4 pt-5 transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <SiteIcon site={site} />
                <span className="w-full truncate text-center text-ui-15 text-foreground">
                  {site.title}
                </span>
              </button>
            ))}
          </div>
        </section>
        {recents.length > 0 ? (
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
                <button
                  key={item.id}
                  type="button"
                  title={item.url}
                  onClick={() => navigate(tabId, { url: item.url })}
                  className={cn(
                    "flex min-w-0 cursor-pointer items-center gap-4 rounded-2xl px-3 py-3 text-start transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  )}
                >
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)]">
                    <HugeiconsIcon
                      icon={InternetIcon}
                      strokeWidth={1.75}
                      className="size-5 text-muted-foreground"
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
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
