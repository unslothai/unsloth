// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { DownloadCancelledError, downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fileNameFromUrl, withBaseUrl } from "./address";
import { fetchBrowserPage } from "./api";
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

/** The desktop app always asks; the web build needs the browser's save dialog (Chromium). */
export function canAskWhereToSave(): boolean {
  return !isTauri && saveFilePicker() !== null;
}

/** The browser's save dialog when Settings asks for it; null when off or it can't open
 *  (e.g. no recent click). Throws DownloadCancelledError if the reader cancels. */
async function pickSaveTarget(name: string): Promise<SaveHandle | null> {
  const picker = saveFilePicker();
  if (isTauri || !picker || !useBrowserPrefsStore.getState().askWhereToSave) return null;
  try {
    return await picker({ suggestedName: name });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new DownloadCancelledError();
    return null;
  }
}

/** Save a file from the panel and add it to the download history. `target` is a save location
 *  already picked, or null for none; left out, the dialog opens here when Settings asks. */
export async function saveBrowserDownload(
  { blob, name, contentType, url }: BrowserDownload,
  target?: SaveHandle | null,
): Promise<void> {
  let saved: { id: string; name: string } | null = null;
  let picked: SaveHandle | null = null;
  try {
    if (isTauri) {
      // The app keeps the path so Download history can reveal it.
      saved = await saveNativeDownload(blob, name);
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

/** Save what a link points at, fetched through the panel's proxy so any site works. */
export async function saveLinkAs(url: string): Promise<void> {
  // Ask before fetching: the dialog needs the menu click, which a slow fetch outlasts.
  let target: SaveHandle | null;
  try {
    target = await pickSaveTarget(fileNameFromUrl(url));
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    throw error;
  }
  const page = await fetchBrowserPage({ url }, new AbortController().signal);
  if (page.kind === "raw") {
    await saveBrowserDownload(
      { blob: page.blob, name: page.fileName ?? fileNameFromUrl(page.url), contentType: page.contentType, url },
      target,
    );
    return;
  }
  const name = fileNameFromUrl(page.url);
  await saveBrowserDownload(
    {
      blob: new Blob([withBaseUrl(page.html, page.base)], { type: "text/html" }),
      name: /\.html?$/i.test(name) ? name : `${name}.html`,
      contentType: "text/html",
      url,
    },
    target,
  );
}
