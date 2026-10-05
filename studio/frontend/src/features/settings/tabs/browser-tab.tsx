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
  type BookmarksToolbarMode,
  ClearBrowsingDataDialog,
  SEARCH_ENGINES,
  type SearchEngineId,
  browserPanelAvailable,
  useBrowserPrefsStore,
  useBrowserStore,
  useNativeBrowser,
} from "@/features/browser";
import { useT } from "@/i18n";
import { useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function BrowserTab() {
  const t = useT();
  const native = useNativeBrowser((state) => state.enabled);
  const openLinksInBrowser = useBrowserPrefsStore((state) => state.openLinksInBrowser);
  const openFilesInBrowser = useBrowserPrefsStore((state) => state.openFilesInBrowser);
  const searchEngine = useBrowserPrefsStore((state) => state.searchEngine);
  const showFullUrl = useBrowserPrefsStore((state) => state.showFullUrl);
  const bookmarksToolbar = useBrowserPrefsStore((state) => state.bookmarksToolbar);
  const showBookmarkEditor = useBrowserPrefsStore((state) => state.showBookmarkEditor);
  const {
    setOpenLinksInBrowser,
    setOpenFilesInBrowser,
    setSearchEngine,
    setShowFullUrl,
    setBookmarksToolbar,
    setShowBookmarkEditor,
  } = useBrowserPrefsStore.getState();
  const [clearOpen, setClearOpen] = useState(false);
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
      </SettingsSection>

      <SettingsSection title={t("browser.browsingDataTitle")}>
        <SettingsRow
          label={t("browser.historySetting")}
          description={t(canOpenPages ? "browser.historySettingDescription" : "browser.pagesFromChat")}
        >
          <Button variant="outline" size="sm" disabled={!canOpenPages} onClick={() => openPage("history")}>
            {t("browser.manage")}
          </Button>
        </SettingsRow>
        <SettingsRow
          label={t("browser.downloadsSetting")}
          description={t(canOpenPages ? "browser.downloadsSettingDescription" : "browser.pagesFromChat")}
        >
          <Button variant="outline" size="sm" disabled={!canOpenPages} onClick={() => openPage("downloads")}>
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
      <ClearBrowsingDataDialog open={clearOpen} onOpenChange={setClearOpen} />
    </div>
  );
}
