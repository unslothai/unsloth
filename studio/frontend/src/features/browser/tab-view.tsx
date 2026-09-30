// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { memo, useCallback, useEffect, useState } from "react";
import { fileNameFromUrl, hostOf } from "./address";
import { type BrowserPage, fetchBrowserPage } from "./api";
import { proxiedFavicon } from "./favicon";
import { FileView } from "./file-view";
import { useBrowserHistoryStore } from "./history-store";
import { InternalPageView } from "./internal-pages";
import { NewTabPage } from "./new-tab-page";
import type { FrameMessage } from "./page-frame";
import { PageFrame } from "./page-frame";
import {
  type BrowserEntry,
  type BrowserTab,
  browserFile,
  cachePage,
  cachedPage,
  currentEntry,
  entryKey,
  setPageDownload,
  useBrowserStore,
} from "./store";

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; page: BrowserPage };

function sameOrigin(url: string, origin: string): boolean {
  try {
    return new URL(url).origin === origin;
  } catch {
    return false;
  }
}

/** The page's favicon, unless it points at Studio. */
function safeFavicon(url: string | null): string | null {
  if (!url) return null;
  if (/^data:image\//i.test(url)) return url;
  try {
    const parsed = new URL(url);
    if (!/^https?:$/.test(parsed.protocol) || parsed.origin === window.location.origin) return null;
    return parsed.href;
  } catch {
    return null;
  }
}

/** Messages from a tab's page. `origin` is the site the page was loaded from, null until loaded. */
function useFrameMessages(tabId: string, origin: string | null) {
  return useCallback(
    (message: FrameMessage) => {
      const store = useBrowserStore.getState();
      switch (message.type) {
        case "navigate":
          if (message.newTab) store.openUrl(message.url, { newTab: true, background: message.background });
          else store.navigate(tabId, message, { replace: message.replace });
          break;
        case "external":
          openExternalLink(message.url);
          break;
        case "loaded": {
          store.updateTab(tabId, { title: message.title, favicon: null, loading: false });
          const tab = store.tabs.find((candidate) => candidate.id === tabId);
          const entry = tab ? currentEntry(tab) : null;
          const favicon = safeFavicon(message.favicon);
          if (favicon && entry) {
            void proxiedFavicon(favicon).then((icon) => {
              const now = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
              // Skip if the tab has moved on to another page.
              if (icon && now && currentEntry(now) === entry) useBrowserStore.getState().updateTab(tabId, { favicon: icon });
            });
          }
          // POST results can't be revisited, so they stay out of history.
          if (tab && entry?.kind === "web" && entry.method !== "POST") {
            useBrowserHistoryStore.getState().recordVisit(tab.displayUrl ?? entry.url, message.title);
          }
          break;
        }
        case "found":
          store.setFindMiss(!message.found);
          break;
        case "title":
          if (message.title) store.updateTab(tabId, { title: message.title });
          break;
        case "url":
          // Same origin only, or a page could spoof the address bar.
          if (origin && sameOrigin(message.url, origin)) store.updateTab(tabId, { displayUrl: message.url });
          break;
        case "reload":
          store.reload(tabId);
          break;
        case "shortcut":
          if (message.key === "l") store.focusAddress();
          else if (message.key === "f") store.setFindOpen(true);
          else if (message.key === "t") store.newTab();
          else if (message.key === "w") store.closeTab(tabId);
          else if (message.key === "r") store.reload(tabId);
          break;
      }
    },
    [tabId, origin],
  );
}

function PageError({ url, message, onRetry }: { url: string; message: string; onRetry: () => void }) {
  const t = useT();
  return (
    <div className="m-auto flex max-w-md flex-col items-center gap-3 px-6 text-center">
      <p className="text-base font-medium text-foreground">{t("browser.error.title")}</p>
      <p className="text-sm text-muted-foreground">
        {t("browser.error.description", { host: hostOf(url) })}
      </p>
      <p className="break-all rounded-lg bg-muted/60 px-3 py-2 font-mono text-xs text-muted-foreground">{message}</p>
      <div className="mt-1 flex gap-2">
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          {t("browser.error.retry")}
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={() => openExternalLink(url)}>
          {t("browser.openExternal")}
        </Button>
      </div>
    </div>
  );
}

function WebPage({
  tab,
  entry,
}: {
  tab: BrowserTab;
  entry: Extract<BrowserEntry, { kind: "web" }>;
}) {
  const { url, method, body } = entry;
  const { reloadKey } = tab;
  // Keyed per load; starts from the cache on back and forward.
  const [state, setState] = useState<LoadState>(() => {
    const page = cachedPage(entry, reloadKey);
    return page ? { status: "ready", page } : { status: "loading" };
  });
  const pageOrigin = state.status === "ready" ? new URL(state.page.url).origin : null;
  const onFrameMessage = useFrameMessages(tab.id, pageOrigin);
  const updateTab = useBrowserStore((store) => store.updateTab);
  const reload = useBrowserStore((store) => store.reload);

  useEffect(() => {
    const show = (page: BrowserPage) => {
      if (page.kind === "raw") {
        const name = fileNameFromUrl(page.url);
        setPageDownload(tab.id, { blob: page.blob, name, contentType: page.contentType });
        updateTab(tab.id, { loading: false, title: name, displayUrl: page.url, documentType: page.contentType });
        if (method !== "POST") useBrowserHistoryStore.getState().recordVisit(page.url, name);
      } else {
        // Host until the frame reports the title.
        updateTab(tab.id, { title: hostOf(page.url), displayUrl: page.url === url ? null : page.url });
      }
    };
    const cached = cachedPage(entry, reloadKey);
    if (cached) {
      show(cached);
      return () => setPageDownload(tab.id, null);
    }
    const controller = new AbortController();
    updateTab(tab.id, { loading: true });
    fetchBrowserPage({ url, method, body }, controller.signal)
      .then((page) => {
        cachePage(entry, reloadKey, page);
        setState({ status: "ready", page });
        show(page);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({ status: "error", message: error instanceof Error ? error.message : String(error) });
        updateTab(tab.id, { loading: false, title: hostOf(url) });
      });
    return () => {
      controller.abort();
      setPageDownload(tab.id, null);
    };
  }, [entry, url, method, body, reloadKey, tab.id, updateTab]);

  if (state.status === "loading") return <div className="size-full bg-background" />;
  if (state.status === "error") {
    return <PageError url={url} message={state.message} onRetry={() => reload(tab.id)} />;
  }
  const { page } = state;
  if (page.kind === "raw") {
    return (
      <FileView blob={page.blob} name={fileNameFromUrl(page.url)} contentType={page.contentType} scale={tab.zoom} />
    );
  }
  return (
    <PageFrame
      html={page.html}
      url={page.url}
      base={page.base}
      refresh={page.refresh}
      title={tab.title || hostOf(page.url)}
      tabId={tab.id}
      zoom={tab.zoom}
      onMessage={onFrameMessage}
    />
  );
}

function LocalFile({ tab }: { tab: BrowserTab }) {
  const t = useT();
  const entry = currentEntry(tab);
  if (entry.kind !== "file") return null;
  const blob = browserFile(entry.fileId);
  if (!blob) {
    return <p className="m-auto text-sm text-muted-foreground">{t("browser.fileGone")}</p>;
  }
  return (
    <FileView
      blob={blob}
      name={entry.name}
      contentType={entry.contentType}
      plainText={entry.plainText}
      scale={tab.zoom}
      tabId={tab.id}
      reloadNonce={tab.reloadKey}
    />
  );
}

/** One tab's content. Stays mounted while recently shown, keeping scroll and state. */
export const TabView = memo(function TabView({ tab, active }: { tab: BrowserTab; active: boolean }) {
  const entry = currentEntry(tab);
  return (
    <div className={cn("absolute inset-0 flex min-h-0 flex-col", !active && "hidden")} aria-hidden={!active}>
      {entry.kind === "newtab" ? (
        <NewTabPage tabId={tab.id} />
      ) : entry.kind === "web" ? (
        <WebPage key={`${entryKey(entry)}:${tab.reloadKey}`} tab={tab} entry={entry} />
      ) : entry.kind === "internal" ? (
        <InternalPageView page={entry.page} tabId={tab.id} />
      ) : (
        <LocalFile key={entry.fileId} tab={tab} />
      )}
    </div>
  );
});
