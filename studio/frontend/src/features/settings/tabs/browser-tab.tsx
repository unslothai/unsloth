// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  type AnnotationScreenshots,
  type BookmarksToolbarMode,
  type DownloadFolder,
  type DownloadSiteDecision,
  ClearBrowsingDataDialog,
  DEFAULT_ZOOM_STEPS,
  HISTORY_RETENTION_DAYS,
  MAX_BOOKMARKS,
  SEARCH_ENGINES,
  type SearchEngineId,
  browserPanelAvailable,
  canAskWhereToSave,
  canScreenshot,
  exportBookmarksFile,
  importBookmarksFile,
  nativeDownloadFolder,
  pickNativeDownloadFolder,
  resetNativeDownloadFolder,
  useBrowserPrefsStore,
  useBrowserStore,
  useDownloadSitesStore,
  useNativeBrowser,
} from "@/features/browser";
import { useChatRuntimeStore } from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

const RETENTION_LABELS: Record<number, TranslationKey> = {
  0: "browser.retention.forever",
  90: "browser.retention.days90",
  30: "browser.retention.days30",
  7: "browser.retention.days7",
  1: "browser.retention.days1",
};

function DownloadLocationRow() {
  const t = useT();
  const [folder, setFolder] = useState<DownloadFolder | null>(null);
  useEffect(() => {
    let live = true;
    nativeDownloadFolder().then(
      (current) => live && setFolder(current),
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, []);
  const failed = (error: unknown) => toast.error(error instanceof Error ? error.message : String(error));
  const change = () =>
    pickNativeDownloadFolder().then((picked) => picked && setFolder(picked), failed);
  const reset = () => resetNativeDownloadFolder().then(setFolder, failed);
  return (
    <SettingsRow
      label={t("browser.downloadLocationSetting")}
      description={
        <span className="break-all">
          {folder ? (folder.custom ? folder.path : t("browser.downloadLocationDefault")) : null}
        </span>
      }
    >
      <div className="flex gap-2">
        {folder?.custom ? (
          <Button variant="ghost" size="sm" onClick={reset}>
            {t("browser.downloadLocationReset")}
          </Button>
        ) : null}
        <Button variant="outline" size="sm" onClick={change}>
          {t("browser.downloadLocationChange")}
        </Button>
      </div>
    </SettingsRow>
  );
}

function DownloadSiteRows() {
  const t = useT();
  const sites = useDownloadSitesStore((state) => state.sites);
  const { setSite: setDownloadSite } = useDownloadSitesStore.getState();
  const origins = Object.keys(sites).sort();
  return (
    <SettingsRow
      label={t("browser.downloadSitesSetting")}
      description={t("browser.downloadSitesSettingDescription")}
      below={
        origins.length > 0 ? (
          <ul className="flex w-full flex-col divide-y divide-border/60 rounded-lg border border-border/80">
            {origins.map((origin) => (
              <li key={origin} className="flex items-center gap-3 py-1.5 pl-3 pr-1.5">
                <span className="min-w-0 flex-1 truncate text-sm text-foreground">{origin}</span>
                <Select
                  value={sites[origin]}
                  onValueChange={(value) => setDownloadSite(origin, value as DownloadSiteDecision)}
                >
                  <SelectTrigger className="w-40" aria-label={t("browser.downloadSitesFor", { host: origin })}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="allow">{t("browser.downloadSiteAllow")}</SelectItem>
                    <SelectItem value="block">{t("browser.downloadSiteBlock")}</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  aria-label={t("browser.downloadSitesRemove", { host: origin })}
                  onClick={() => setDownloadSite(origin, null)}
                >
                  <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4" />
                </Button>
              </li>
            ))}
          </ul>
        ) : (
          <span className="w-full text-xs text-muted-foreground">{t("browser.downloadSitesNone")}</span>
        )
      }
    />
  );
}

export function BrowserTab() {
  const t = useT();
  const locale = useLocale();
  const native = useNativeBrowser((state) => state.enabled);
  const openLinksInBrowser = useBrowserPrefsStore((state) => state.openLinksInBrowser);
  const openFilesInBrowser = useBrowserPrefsStore((state) => state.openFilesInBrowser);
  const searchEngine = useBrowserPrefsStore((state) => state.searchEngine);
  const showFullUrl = useBrowserPrefsStore((state) => state.showFullUrl);
  const bookmarksToolbar = useBrowserPrefsStore((state) => state.bookmarksToolbar);
  const showBookmarkEditor = useBrowserPrefsStore((state) => state.showBookmarkEditor);
  const switchToNewTabs = useBrowserPrefsStore((state) => state.switchToNewTabs);
  const defaultZoom = useBrowserPrefsStore((state) => state.defaultZoom);
  const showSuggestedSites = useBrowserPrefsStore((state) => state.showSuggestedSites);
  const showRecentPages = useBrowserPrefsStore((state) => state.showRecentPages);
  const hiddenSuggestions = useBrowserPrefsStore((state) => state.hiddenSuggestions.length);
  const saveHistory = useBrowserPrefsStore((state) => state.saveHistory);
  const historyRetentionDays = useBrowserPrefsStore((state) => state.historyRetentionDays);
  const saveDownloadHistory = useBrowserPrefsStore((state) => state.saveDownloadHistory);
  const askWhereToSave = useBrowserPrefsStore((state) => state.askWhereToSave);
  const askBeforeDownloading = useBrowserPrefsStore((state) => state.askBeforeDownloading);
  const annotationScreenshots = useBrowserPrefsStore((state) => state.annotationScreenshots);
  const {
    setOpenLinksInBrowser,
    setOpenFilesInBrowser,
    setSearchEngine,
    setShowFullUrl,
    setBookmarksToolbar,
    setShowBookmarkEditor,
    setSwitchToNewTabs,
    setDefaultZoom,
    setShowSuggestedSites,
    setShowRecentPages,
    restoreSuggestions,
    setSaveHistory,
    setHistoryRetentionDays,
    setSaveDownloadHistory,
    setAskWhereToSave,
    setAskBeforeDownloading,
    setAnnotationScreenshots,
  } = useBrowserPrefsStore.getState();
  const [clearOpen, setClearOpen] = useState(false);
  const importInput = useRef<HTMLInputElement | null>(null);
  const percent = useMemo(
    () => new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 0 }),
    [locale],
  );
  const canAsk = canAskWhereToSave();
  const screenshots = canScreenshot();

  const importBookmarks = async (file: File) => {
    try {
      const { added: count, leftOut } = await importBookmarksFile(file);
      if (leftOut > 0) toast.warning(t("browser.importBookmarksPartial", { count, leftOut, max: MAX_BOOKMARKS }));
      else if (count === 0) toast.info(t("browser.importBookmarksNone"));
      else toast.success(t(count === 1 ? "browser.importBookmarksDoneOne" : "browser.importBookmarksDoneMany", { count }));
    } catch {
      toast.error(t("browser.importBookmarksFailed"));
    }
  };
  const exportBookmarks = () =>
    exportBookmarksFile().catch((error: unknown) => {
      if (!isDownloadCancelled(error)) toast.error(error instanceof Error ? error.message : String(error));
    });
  const collapseHtmlArtifacts = useChatRuntimeStore((state) => state.collapseHtmlArtifacts);
  const allowArtifactNetworkAccess = useChatRuntimeStore((state) => state.allowArtifactNetworkAccess);
  const { setCollapseHtmlArtifacts, setAllowArtifactNetworkAccess } = useChatRuntimeStore.getState();
  // The network-blocked banner deep-links here.
  const networkAccessRowRef = useRef<HTMLDivElement | null>(null);
  const scrollTarget = useSettingsDialogStore((state) => state.scrollTarget);
  useEffect(() => {
    if (scrollTarget !== "browser-html-network") return;
    const frame = window.requestAnimationFrame(() => {
      networkAccessRowRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
      useSettingsDialogStore.getState().consumeScrollTarget("browser-html-network");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [scrollTarget]);
  // History and downloads open as browser tabs, so only beside a chat.
  const canOpenPages = browserPanelAvailable();
  const openPage = (page: "history" | "downloads" | "bookmarks") => {
    useSettingsDialogStore.getState().closeDialog();
    useBrowserStore.getState().openInternal(page);
  };

  return (
    <div className="settings-page">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">{t("browser.settingsTitle")}</h1>
      </header>

      <SettingsSection title={t("browser.linksTitle")}>
        <SettingsRow
          label={t("browser.openLinksSetting")}
          description={t("browser.openLinksSettingDescription")}
        >
          <Select
            value={openLinksInBrowser ? "panel" : "default"}
            onValueChange={(value) => setOpenLinksInBrowser(value === "panel")}
          >
            <SelectTrigger className="w-40" aria-label={t("browser.openLinksSetting")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">{t("browser.linkDestinationDefault")}</SelectItem>
              <SelectItem value="panel">{t("browser.linkDestinationPanel")}</SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("browser.openFilesSetting")}
          description={t("browser.openFilesSettingDescription")}
        >
          <Switch
            aria-label={t("browser.openFilesSetting")}
            checked={openFilesInBrowser}
            onCheckedChange={setOpenFilesInBrowser}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.chat.artifacts.title")}>
        <div ref={networkAccessRowRef}>
          <SettingsRow
            label={t("settings.chat.artifacts.allowNetworkAccess")}
            description={t("settings.chat.artifacts.allowNetworkAccessDescription")}
          >
            <Switch
              aria-label={t("settings.chat.artifacts.allowNetworkAccess")}
              checked={allowArtifactNetworkAccess}
              onCheckedChange={setAllowArtifactNetworkAccess}
            />
          </SettingsRow>
        </div>
        <SettingsRow
          label={t("settings.chat.artifacts.collapseHtmlBlocks")}
          description={t("settings.chat.artifacts.collapseHtmlBlocksDescription")}
        >
          <Switch
            aria-label={t("settings.chat.artifacts.collapseHtmlBlocks")}
            checked={collapseHtmlArtifacts}
            onCheckedChange={setCollapseHtmlArtifacts}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.tabsTitle")}>
        <SettingsRow
          label={t("browser.switchToNewTabsSetting")}
          description={t(
            native ? "browser.switchToNewTabsNative" : "browser.switchToNewTabsSettingDescription",
          )}
        >
          {/* Native views can't tell a background click, so their new tabs always open in front. */}
          <Switch
            aria-label={t("browser.switchToNewTabsSetting")}
            checked={native || switchToNewTabs}
            disabled={native}
            onCheckedChange={setSwitchToNewTabs}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.defaultZoomSetting")}
          description={t("browser.defaultZoomSettingDescription")}
        >
          <Select value={String(defaultZoom)} onValueChange={(value) => setDefaultZoom(Number(value))}>
            <SelectTrigger className="w-40" aria-label={t("browser.defaultZoomSetting")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DEFAULT_ZOOM_STEPS.map((step) => (
                <SelectItem key={step} value={String(step)}>
                  {percent.format(step)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.addressBarTitle")}>
        <SettingsRow
          label={t("browser.searchEngineSetting")}
          description={t("browser.searchEngineSettingDescription")}
        >
          <Select value={searchEngine} onValueChange={(value) => setSearchEngine(value as SearchEngineId)}>
            <SelectTrigger className="w-40" aria-label={t("browser.searchEngineSetting")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(SEARCH_ENGINES) as SearchEngineId[]).map((id) => (
                <SelectItem key={id} value={id}>
                  {SEARCH_ENGINES[id].label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("browser.showFullUrlSetting")}
          description={t("browser.showFullUrlSettingDescription")}
        >
          <Switch
            aria-label={t("browser.showFullUrlSetting")}
            checked={showFullUrl}
            onCheckedChange={setShowFullUrl}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.newTabPageTitle")}>
        <SettingsRow
          label={t("browser.showSuggestedSetting")}
          description={t("browser.showSuggestedSettingDescription")}
        >
          <Switch
            aria-label={t("browser.showSuggestedSetting")}
            checked={showSuggestedSites}
            onCheckedChange={setShowSuggestedSites}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.showRecentsSetting")}
          description={t("browser.showRecentsSettingDescription")}
        >
          <Switch
            aria-label={t("browser.showRecentsSetting")}
            checked={showRecentPages}
            onCheckedChange={setShowRecentPages}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.hiddenSuggestionsSetting")}
          description={
            hiddenSuggestions > 0
              ? t("browser.hiddenSuggestionsSettingDescription", { count: hiddenSuggestions })
              : t("browser.hiddenSuggestionsNone")
          }
        >
          <Button variant="outline" size="sm" disabled={hiddenSuggestions === 0} onClick={restoreSuggestions}>
            {t("browser.restore")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.bookmarksTitle")}>
        <SettingsRow
          label={t("browser.bookmarksToolbarSetting")}
          description={t("browser.bookmarksToolbarSettingDescription")}
        >
          <Select
            value={bookmarksToolbar}
            onValueChange={(value) => setBookmarksToolbar(value as BookmarksToolbarMode)}
          >
            <SelectTrigger className="w-48" aria-label={t("browser.bookmarksToolbarSetting")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="always">{t("browser.bookmarks.toolbarAlways")}</SelectItem>
              <SelectItem value="newtab">{t("browser.bookmarks.toolbarNewTab")}</SelectItem>
              <SelectItem value="never">{t("browser.bookmarks.toolbarNever")}</SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("browser.bookmarks.showEditor")}
          description={t("browser.bookmarkEditorSettingDescription")}
        >
          <Switch
            aria-label={t("browser.bookmarks.showEditor")}
            checked={showBookmarkEditor}
            onCheckedChange={setShowBookmarkEditor}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.pages.bookmarks")}
          description={t(canOpenPages ? "browser.bookmarksSettingDescription" : "browser.pagesFromChat")}
        >
          <Button variant="outline" size="sm" disabled={!canOpenPages} onClick={() => openPage("bookmarks")}>
            {t("browser.manage")}
          </Button>
        </SettingsRow>
        <SettingsRow
          label={t("browser.importBookmarksSetting")}
          description={t("browser.importBookmarksSettingDescription")}
        >
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => importInput.current?.click()}>
              {t("browser.importBookmarks")}
            </Button>
            <Button variant="outline" size="sm" onClick={exportBookmarks}>
              {t("browser.exportBookmarks")}
            </Button>
          </div>
          <input
            ref={importInput}
            type="file"
            accept=".html,.htm,text/html"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              // Cleared, so choosing the same file again still imports it.
              event.target.value = "";
              if (file) void importBookmarks(file);
            }}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.downloadsTitle")}>
        {isTauri ? <DownloadLocationRow /> : null}
        <SettingsRow
          label={t("browser.askWhereToSaveSetting")}
          description={t(
            canAsk ? "browser.askWhereToSaveSettingDescription" : "browser.askWhereToSaveUnsupported",
          )}
        >
          <Switch
            aria-label={t("browser.askWhereToSaveSetting")}
            checked={canAsk && askWhereToSave}
            disabled={!canAsk}
            onCheckedChange={setAskWhereToSave}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.askBeforeDownloadingSetting")}
          description={t("browser.askBeforeDownloadingSettingDescription")}
        >
          <Switch
            aria-label={t("browser.askBeforeDownloadingSetting")}
            checked={askBeforeDownloading}
            onCheckedChange={setAskBeforeDownloading}
          />
        </SettingsRow>
        <DownloadSiteRows />
        <SettingsRow
          label={t("browser.saveDownloadHistorySetting")}
          description={t("browser.saveDownloadHistorySettingDescription")}
        >
          <Switch
            aria-label={t("browser.saveDownloadHistorySetting")}
            checked={saveDownloadHistory}
            onCheckedChange={setSaveDownloadHistory}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.downloadsSetting")}
          description={t(canOpenPages ? "browser.downloadsSettingDescription" : "browser.pagesFromChat")}
        >
          <Button variant="outline" size="sm" disabled={!canOpenPages} onClick={() => openPage("downloads")}>
            {t("browser.manage")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.browsingDataTitle")}>
        <SettingsRow
          label={t("browser.saveHistorySetting")}
          description={t("browser.saveHistorySettingDescription")}
        >
          <Switch
            aria-label={t("browser.saveHistorySetting")}
            checked={saveHistory}
            onCheckedChange={setSaveHistory}
          />
        </SettingsRow>
        <SettingsRow
          label={t("browser.historyRetentionSetting")}
          description={t("browser.historyRetentionSettingDescription")}
        >
          <Select
            value={String(historyRetentionDays)}
            onValueChange={(value) => setHistoryRetentionDays(Number(value))}
          >
            <SelectTrigger className="w-40" aria-label={t("browser.historyRetentionSetting")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HISTORY_RETENTION_DAYS.map((days) => (
                <SelectItem key={days} value={String(days)}>
                  {t(RETENTION_LABELS[days] ?? "browser.retention.forever")}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("browser.historySetting")}
          description={t(canOpenPages ? "browser.historySettingDescription" : "browser.pagesFromChat")}
        >
          <Button variant="outline" size="sm" disabled={!canOpenPages} onClick={() => openPage("history")}>
            {t("browser.manage")}
          </Button>
        </SettingsRow>
        <SettingsRow
          label={t("browser.clearDataSetting")}
          description={t(
            native ? "browser.native.clearDataSettingDescription" : "browser.clearDataSettingDescription",
          )}
        >
          <Button variant="outline" size="sm" onClick={() => setClearOpen(true)}>
            {t("browser.menu.clearData")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("browser.annotationsTitle")}>
        <SettingsRow
          label={t("browser.annotationScreenshotsSetting")}
          description={t(
            screenshots
              ? "browser.annotationScreenshotsSettingDescription"
              : "browser.annotationScreenshotsUnsupported",
          )}
          hint={screenshots && !isTauri ? t("browser.annotationScreenshotsHint") : undefined}
        >
          <Select
            value={screenshots ? annotationScreenshots : "never"}
            disabled={!screenshots}
            onValueChange={(value) => setAnnotationScreenshots(value as AnnotationScreenshots)}
          >
            <SelectTrigger className="w-44" aria-label={t("browser.annotationScreenshotsSetting")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="always">{t("browser.annotationScreenshotsAlways")}</SelectItem>
              <SelectItem value="never">{t("browser.annotationScreenshotsNever")}</SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
      </SettingsSection>
      <ClearBrowsingDataDialog open={clearOpen} onOpenChange={setClearOpen} />
    </div>
  );
}
