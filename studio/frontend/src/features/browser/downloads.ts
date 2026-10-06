// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getLocale, translate } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { DownloadCancelledError, downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fileNameFromUrl, withBaseUrl } from "./address";
import { type BrowserPage, fetchBrowserPage } from "./api";
import { useBrowserHistoryStore } from "./history-store";
import { type SavedNativeDownload, saveNativeDownload } from "./native-downloads";
import { useBrowserPrefsStore } from "./prefs-store";

export type BrowserDownload = { blob: Blob; name: string; contentType: string; url: string | null };

type SaveHandle = {
  name: string;
  createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }>;
};
type SaveFilePicker = (options: { suggestedName: string }) => Promise<SaveHandle>;

function saveFilePicker(): SaveFilePicker | null {
  const picker = (globalThis as { showSaveFilePicker?: SaveFilePicker }).showSaveFilePicker;
  return typeof picker === "function" ? picker : null;
}

/** The desktop app always asks; the web build needs the browser's save dialog (Chromium). */
export function canAskWhereToSave(): boolean {
  return !isTauri && saveFilePicker() !== null;
}

function asksWhereToSave(): boolean {
  return !isTauri && saveFilePicker() !== null && useBrowserPrefsStore.getState().askWhereToSave;
}

/** True when Settings asks where to save but the click that started this has expired,
 *  so the dialog can't open without another one. */
export function saveNeedsClick(): boolean {
  return asksWhereToSave() && !navigator.userActivation?.isActive;
}

/** The browser's save dialog when Settings asks for it; null when off or it can't open
 *  (e.g. no recent click). Throws DownloadCancelledError if the reader cancels. */
async function pickSaveTarget(name: string): Promise<SaveHandle | null> {
  const picker = saveFilePicker();
  if (!picker || !asksWhereToSave()) return null;
  try {
    return await picker({ suggestedName: name });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new DownloadCancelledError();
    return null;
  }
}

/** Save a file from the panel and add it to the download history; a file that runs code asks first.
 *  `target` is a save location already picked, or null for none; left out, the dialog opens here
 *  when Settings asks. */
export async function saveBrowserDownload(download: BrowserDownload, target?: SaveHandle | null): Promise<void> {
  const { isDangerousDownload, safeDownloadName } = await import("./download-safety");
  const { blob, contentType, url } = download;
  const name = safeDownloadName(download.name);
  // Before any dialog: its Save anyway click is also the fresh click a save dialog needs.
  if (isDangerousDownload(name) && !(await confirmDangerous(name))) return;
  let saved: SavedNativeDownload | null = null;
  let picked: SaveHandle | null = null;
  try {
    if (isTauri) {
      // The app keeps the path so Download history can reveal it, and marks the file with its source.
      saved = await saveNativeDownload(blob, name, url);
      if (!saved) return;
    } else {
      picked = target === undefined ? await pickSaveTarget(name) : target;
      if (picked) {
        const writable = await picked.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        await downloadFile(blob, name, contentType || undefined);
      }
    }
  } catch (error) {
    if (!isDownloadCancelled(error)) toast.error(error instanceof Error ? error.message : String(error));
    return;
  }
  const savedName = saved?.name || picked?.name || name;
  useBrowserHistoryStore.getState().recordDownload({
    name: savedName,
    url,
    size: blob.size,
    contentType,
    nativeId: saved?.id,
  });
  if (saved?.marked === false) toast.warning(translate("browser.downloadSafety.notMarked", { name: savedName }, getLocale()));
}

let nextConfirm = 0;

/** Resolves true on Save anyway; Cancel, closing or swiping it away is false. */
function confirmDangerous(name: string): Promise<boolean> {
  const t = (key: Parameters<typeof translate>[0]) => translate(key, { name }, getLocale());
  return new Promise((resolve) => {
    let answered = false;
    const answer = (save: boolean) => {
      if (answered) return;
      answered = true;
      resolve(save);
    };
    toast(t("browser.downloadSafety.savePrompt"), {
      id: `browser-save-${nextConfirm++}`,
      duration: Number.POSITIVE_INFINITY,
      action: { label: t("browser.downloadSafety.saveAnyway"), onClick: () => answer(true) },
      cancel: { label: t("browser.downloadSafety.cancel"), onClick: () => answer(false) },
      onDismiss: () => answer(false),
    });
  });
}

// Well inside the ~5 s a click lets a page open the save dialog.
const RESOLVE_BEFORE_ASK_MS = 1000;

/** A fetched link as a file: the server's name, or the page as .html. */
function linkDownload(page: BrowserPage, url: string): BrowserDownload {
  if (page.kind === "raw") {
    return { blob: page.blob, name: page.fileName ?? fileNameFromUrl(page.url), contentType: page.contentType, url };
  }
  const name = fileNameFromUrl(page.url);
  return {
    blob: new Blob([withBaseUrl(page.html, page.base)], { type: "text/html" }),
    name: /\.html?$/i.test(name) ? name : `${name}.html`,
    contentType: "text/html",
    url,
  };
}

/** Save what a link points at, fetched through the panel's proxy so any site works. */
export async function saveLinkAs(url: string): Promise<void> {
  const controller = new AbortController();
  const pending = fetchBrowserPage({ url }, controller.signal).then((page) => linkDownload(page, url));
  // The dialog needs the menu click, which a slow fetch outlasts: ask with the resolved name
  // when the fetch is quick, else with the URL's.
  let target: SaveHandle | null | undefined;
  if (asksWhereToSave()) {
    const quick = await Promise.race([
      pending.then(
        (download) => ({ download }),
        (error: unknown) => ({ error }),
      ),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), RESOLVE_BEFORE_ASK_MS)),
    ]);
    // A link that already failed has nothing to save: report it without asking for a name.
    if (quick && "error" in quick) throw quick.error;
    try {
      target = await pickSaveTarget(quick?.download.name ?? fileNameFromUrl(url));
    } catch (error) {
      // No save after all: stop the fetch, which the backend drops on disconnect.
      controller.abort();
      pending.catch(() => undefined);
      if (isDownloadCancelled(error)) return;
      throw error;
    }
  }
  await saveBrowserDownload(await pending, target);
}
