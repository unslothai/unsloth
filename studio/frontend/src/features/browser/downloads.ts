// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getLocale, translate } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { DownloadCancelledError, downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fileNameFromUrl, isWebUrl, withBaseUrl } from "./address";
import { type BrowserPage, fetchBrowserPage } from "./api";
import { approveDownload } from "./download-approval-queue";
import { useBrowserHistoryStore } from "./history-store";
import { saveNativeDownload } from "./native-downloads";
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

/** The desktop app has its own dialog; the web build needs the browser's (Chromium). */
export function canAskWhereToSave(): boolean {
  return isTauri || saveFilePicker() !== null;
}

/** The web build's save dialog is on in Settings. */
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

/** Whether a file from `url` may be saved; files from websites wait for the user's approval. */
function approved(url: string | null, name: string): Promise<boolean> {
  return url && isWebUrl(url) ? approveDownload(url, name) : Promise.resolve(true);
}

/** Save a file from the panel and add it to the download history. A file from a website
 *  waits for approval first. `target` is a save location already picked (approval included),
 *  or null for none; left out, the dialog opens here when Settings asks. */
export async function saveBrowserDownload(download: BrowserDownload, target?: SaveHandle | null): Promise<void> {
  if (target === undefined) {
    if (!(await approved(download.url, download.name))) return;
    // The save dialog opens only right after a click, and one that finished later (a slow file,
    // or one nobody was asked about) would save without it: wait for a click on Save instead.
    if (saveNeedsClick()) {
      const locale = getLocale();
      toast(translate("browser.downloadPrompt.ready", { name: download.name }, locale), {
        action: {
          label: translate("browser.downloadPrompt.save", {}, locale),
          onClick: () => void writeDownload(download, undefined),
        },
      });
      return;
    }
  }
  await writeDownload(download, target);
}

/** Write an approved download where Settings says, and add it to the download history. */
async function writeDownload(
  { blob, name, contentType, url }: BrowserDownload,
  target: SaveHandle | null | undefined,
): Promise<void> {
  let saved: { id: string; name: string } | null = null;
  let picked: SaveHandle | null = null;
  try {
    if (isTauri) {
      // The app keeps the path so Download history can reveal it.
      saved = await saveNativeDownload(blob, name, useBrowserPrefsStore.getState().askWhereToSave, url);
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
  useBrowserHistoryStore.getState().recordDownload({
    name: saved?.name || picked?.name || name,
    url,
    size: blob.size,
    contentType,
    nativeId: saved?.id,
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
    const name = quick?.download.name ?? fileNameFromUrl(url);
    try {
      // Approved before the dialog, which creates the file. Answering is a fresh click for it.
      if (!(await approved(url, name))) throw new DownloadCancelledError();
      target = await pickSaveTarget(name);
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
