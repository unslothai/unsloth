// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { requestFind } from "@/features/find-in-page";
import { zoomScopeFromChord } from "@/features/interface-zoom";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { memo, useCallback, useEffect, useState } from "react";
import { fileNameFromUrl, hostOf } from "./address";
import { BrowserFetchError, type BrowserPage, fetchBrowserPage } from "./api";
import { proxiedFavicon } from "./favicon";
import { saveBrowserDownload } from "./downloads";
import { canShowFile } from "./file-kind";
import { FileView } from "./file-view";
import { BROWSER_FIND_TARGET, pageLoadedForFind, receiveFindResult } from "./find";
import { useBrowserHistoryStore } from "./history-store";
import { InternalPageView } from "./internal-pages";
import { useNativeBrowser } from "./native-view";
import { NewTabPage } from "./new-tab-page";
import { useBrowserPrefsStore } from "./prefs-store";
import { fitZoomToPage, zoomTab } from "./zoom";
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
  sentPosts,
  setPageDownload,
  useBrowserStore,
} from "./store";

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string; botCheck: boolean; resubmit?: boolean }
  | { status: "ready"; page: BrowserPage };

const resubmits = new WeakSet<BrowserEntry>();

function sameOrigin(url: string, origin: string): boolean {
  try {
    return new URL(url).origin === origin;
  } catch {
    return false;
  }
}

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

function pageAddress(tab: BrowserTab | undefined): string | undefined {
  const entry = tab ? currentEntry(tab) : null;
  return tab && entry?.kind === "web" ? (tab.displayUrl ?? entry.url) : undefined;
}

function useFrameMessages(tabId: string, origin: string | null) {
  const t = useT();
  return useCallback(
    (message: FrameMessage) => {
      const store = useBrowserStore.getState();
      switch (message.type) {
        case "navigate": {
          const from = pageAddress(store.tabs.find((candidate) => candidate.id === tabId));
          if (message.newTab) {
            store.openUrl(message.url, {
              newTab: true,
              background: message.background && !useBrowserPrefsStore.getState().switchToNewTabs,
              method: message.method,
              body: message.body,
              from,
            });
          } else {
            store.navigate(tabId, { url: message.url, method: message.method, body: message.body, from }, { replace: message.replace });
          }
          break;
        }
        case "external":
          openExternalLink(message.url);
          break;
        case "loaded": {
          store.updateTab(tabId, { title: message.title, favicon: null, loading: false });
          pageLoadedForFind(tabId);
          const tab = store.tabs.find((candidate) => candidate.id === tabId);
          const entry = tab ? currentEntry(tab) : null;
          const favicon = safeFavicon(message.favicon);
          // Kept by site, so Recents, History and Suggested show it once the tab is gone.
          if (favicon && tab && entry?.kind === "web") {
            useBrowserHistoryStore.getState().recordIcon(hostOf(tab.displayUrl ?? entry.url), favicon, entry.temporary);
          }
          if (favicon && entry) {
            void proxiedFavicon(favicon).then((icon) => {
              const now = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
              if (icon && now && currentEntry(now) === entry) useBrowserStore.getState().updateTab(tabId, { favicon: icon });
            });
          }
          // POST results can't be revisited, so they stay out of history.
          if (tab && entry?.kind === "web" && entry.method !== "POST") {
            useBrowserHistoryStore.getState().recordVisit(tab.displayUrl ?? entry.url, message.title, entry.temporary);
          }
          break;
        }
        case "findResult":
          receiveFindResult(tabId, message.count, message.active);
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
        case "upload":
        case "scriptNavigation": {
          const tab = store.tabs.find((candidate) => candidate.id === tabId);
          const entry = tab ? currentEntry(tab) : null;
          const url = tab?.displayUrl ?? (entry?.kind === "web" ? entry.url : null);
          toast(t(message.type === "upload" ? "browser.error.upload" : "browser.error.scriptNavigation"), {
            action: url ? { label: t("browser.openExternal"), onClick: () => openExternalLink(url) } : undefined,
          });
          break;
        }
        case "shortcut":
          if (message.key === "l") store.focusAddress();
          else if (message.key === "f") requestFind(BROWSER_FIND_TARGET);
          else if (message.key === "t") store.newTab();
          else if (message.key === "w") store.closeTab(tabId);
          else if (message.key === "r") store.reload(tabId);
          else if (message.key === "d") store.bookmarkPage();
          break;
        case "zoom":
          // A key goes through the scope, so the View menu repeating it doesn't zoom twice.
          if (message.wheel) zoomTab(tabId, message.direction);
          else zoomScopeFromChord({ contains: () => true, zoom: (direction) => zoomTab(tabId, direction) }, message.direction);
          break;
      }
    },
    [tabId, origin, t],
  );
}

// Stands in for the app name while translating, so it can be a link.
const APP_MARK = "\u0000";
const DESKTOP_APP_URL = "https://github.com/unslothai/unsloth";

function PageError({
  url,
  message,
  botCheck = false,
  resubmit = false,
  onRetry,
}: {
  url: string;
  message: string;
  botCheck?: boolean;
  resubmit?: boolean;
  onRetry: () => void;
}) {
  const t = useT();
  return (
    <div className="m-auto flex max-w-md flex-col items-center gap-3 px-6 text-center">
      <p className="text-base font-medium text-foreground">
        {t(botCheck ? "browser.error.botCheckTitle" : "browser.error.title")}
      </p>
      <p className="text-sm text-muted-foreground">
        {resubmit
          ? t("browser.error.resubmit")
          : botCheck && !isTauri
            ? // The desktop app's native views pass these checks; point web users there.
              t("browser.error.botCheckDescriptionWeb", { host: hostOf(url), app: APP_MARK })
                .split(APP_MARK)
                .flatMap((part, index) =>
                  index === 0
                    ? [part]
                    : [
                        <a
                          key={index}
                          href={DESKTOP_APP_URL}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-semibold text-foreground underline decoration-border underline-offset-2 transition-colors hover:decoration-foreground"
                        >
                          {t("browser.error.desktopApp")}
                        </a>,
                        part,
                      ],
                )
            : t(botCheck ? "browser.error.botCheckDescription" : "browser.error.description", {
                host: hostOf(url),
              })}
      </p>
      {message ? (
        <p className="break-all rounded-lg bg-muted/60 px-3 py-2 font-mono text-xs text-muted-foreground">{message}</p>
      ) : null}
      <div className="mt-1 flex gap-2">
        <Button type="button" variant={botCheck ? "ghost" : "outline"} size="sm" onClick={onRetry}>
          {t("browser.error.retry")}
        </Button>
        <Button type="button" variant={botCheck ? "default" : "ghost"} size="sm" onClick={() => openExternalLink(url)}>
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
  const [state, setState] = useState<LoadState>(() => {
    const page = cachedPage(entry);
    if (page) return { status: "ready", page };
    if (method === "POST" && sentPosts.has(entry) && !resubmits.has(entry)) {
      return { status: "error", message: "", botCheck: false, resubmit: true };
    }
    return { status: "loading" };
  });
  const blocked = state.status === "error" && state.resubmit === true;
  const pageOrigin = state.status === "ready" ? new URL(state.page.url).origin : null;
  const onFrameMessage = useFrameMessages(tab.id, pageOrigin);
  const updateTab = useBrowserStore((store) => store.updateTab);
  const reload = useBrowserStore((store) => store.reload);

  useEffect(() => {
    const show = (page: BrowserPage) => {
      if (page.kind === "raw") {
        const name = page.fileName ?? fileNameFromUrl(page.url);
        setPageDownload(tab.id, { blob: page.blob, name, contentType: page.contentType });
        fitZoomToPage(tab.id, true);
        updateTab(tab.id, {
          loading: false,
          title: name,
          displayUrl: page.url,
          documentType: page.contentType,
          pageError: false,
        });
        if (method !== "POST") useBrowserHistoryStore.getState().recordVisit(page.url, name, entry.temporary);
      } else {
        fitZoomToPage(tab.id, false);
        updateTab(tab.id, {
          title: hostOf(page.url),
          displayUrl: page.url === url ? null : page.url,
          pageError: false,
        });
      }
    };
    const cached = cachedPage(entry);
    if (cached) {
      show(cached);
      return () => setPageDownload(tab.id, null);
    }
    if (blocked) {
      updateTab(tab.id, { loading: false, title: hostOf(url), pageError: true });
      return;
    }
    if (method === "POST") {
      sentPosts.add(entry);
      resubmits.delete(entry);
    }
    const controller = new AbortController();
    updateTab(tab.id, { loading: true });
    fetchBrowserPage({ url, method, body }, controller.signal)
      .then((page) => {
        cachePage(entry, page);
        setState({ status: "ready", page });
        show(page);
        // A file the panel can't show downloads, as in a browser; fresh loads only, so returning to the tab doesn't ask again.
        if (page.kind === "raw") {
          const name = page.fileName ?? fileNameFromUrl(page.url);
          if (!canShowFile(name, page.contentType)) {
            // The sending page, else the address asked for (not the redirect target), so another site's "allow" can't cover it.
            void saveBrowserDownload({ blob: page.blob, name, contentType: page.contentType, url: page.url, site: entry.from ?? url });
            if (entry.kind === "web" && entry.from) useBrowserStore.getState().leaveDownload(tab.id, entry);
          }
        }
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({
          status: "error",
          message: error instanceof Error ? error.message : String(error),
          botCheck: error instanceof BrowserFetchError && error.botCheck,
        });
        updateTab(tab.id, { loading: false, title: hostOf(url), pageError: true });
      });
    return () => {
      controller.abort();
      setPageDownload(tab.id, null);
    };
  }, [entry, url, method, body, reloadKey, blocked, tab.id, updateTab]);

  if (state.status === "loading") return <div className="size-full bg-background" />;
  if (state.status === "error") {
    return (
      <PageError
        url={url}
        message={state.message}
        botCheck={state.botCheck}
        resubmit={state.resubmit}
        onRetry={() => {
          if (state.resubmit) resubmits.add(entry);
          reload(tab.id);
        }}
      />
    );
  }
  const { page } = state;
  if (page.kind === "raw") {
    return (
      <FileView blob={page.blob} name={page.fileName ?? fileNameFromUrl(page.url)} contentType={page.contentType} scale={tab.zoom} />
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
      muted={tab.muted}
      onMessage={onFrameMessage}
    />
  );
}

function NativePage({ tab, entry }: { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "web" }> }) {
  const reload = useBrowserStore((store) => store.reload);
  if (tab.nativeError) {
    return (
      <PageError
        url={tab.displayUrl ?? entry.url}
        message={tab.nativeError}
        onRetry={() => {
          useBrowserStore.getState().updateTab(tab.id, { nativeError: null });
          reload(tab.id);
        }}
      />
    );
  }
  return <div data-native-page={tab.id} className="size-full bg-background" />;
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

export const TabView = memo(function TabView({ tab, active }: { tab: BrowserTab; active: boolean }) {
  const entry = currentEntry(tab);
  const native = useNativeBrowser((state) => state.enabled);
  return (
    <div className={cn("absolute inset-0 flex min-h-0 flex-col", !active && "hidden")} aria-hidden={!active}>
      {entry.kind === "newtab" ? (
        <NewTabPage tabId={tab.id} />
      ) : entry.kind === "web" ? (
        native ? (
          <NativePage tab={tab} entry={entry} />
        ) : (
          <WebPage key={`${entryKey(entry)}:${tab.reloadKey}`} tab={tab} entry={entry} />
        )
      ) : entry.kind === "internal" ? (
        <InternalPageView page={entry.page} tabId={tab.id} />
      ) : (
        <LocalFile key={entry.fileId} tab={tab} />
      )}
    </div>
  );
});
