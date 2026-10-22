// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat";
import { getLocale, translate } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { DownloadCancelledError, downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fileNameFromUrl, isWebUrl, safeDownloadName, withBaseUrl } from "./address";
import { type BrowserPage, fetchBrowserPage } from "./api";
import { abandonDownload, beginDownload, finishDownload } from "./download-activity";
import type { DownloadsRule } from "./agent/permissions";
import { approveDownload } from "./download-approval-queue";
import { isDangerousDownload, withTypeExtension } from "./download-safety";
import { useBrowserHistoryStore } from "./history-store";
import { type SavedNativeDownload, saveNativeDownload } from "./native-downloads";
import { useBrowserPrefsStore } from "./prefs-store";

export type BrowserDownload = {
  blob: Blob;
  name: string;
  contentType: string;
  url: string | null;
  site?: string;
  temporary?: boolean;
  /** Saved from the browser agent's tab: its Downloads permission (agent/permissions.ts). */
  agentRule?: DownloadsRule | null;
};

type SaveHandle = {
  name: string;
  createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }>;
};
type SaveFilePicker = (options: { suggestedName: string }) => Promise<SaveHandle>;

function saveFilePicker(): SaveFilePicker | null {
  const picker = (globalThis as { showSaveFilePicker?: SaveFilePicker }).showSaveFilePicker;
  return typeof picker === "function" ? picker : null;
}

export function canAskWhereToSave(): boolean {
  return isTauri || saveFilePicker() !== null;
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
    return await picker({ suggestedName: safeDownloadName(name) });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new DownloadCancelledError();
    return null;
  }
}

function approved(url: string | null, name: string, site?: string, agentRule?: DownloadsRule | null): Promise<boolean> {
  return url && isWebUrl(url) ? approveDownload(url, name, site ?? url, agentRule) : Promise.resolve(true);
}

function downloadName(name: string, contentType: string): string {
  const typed = withTypeExtension(safeDownloadName(name), contentType, !isTauri);
  return isTauri ? typed : typed.replace(/^\.+/, "").replace(/[. ]+$/, "") || "download";
}

const SAVE_PROMPTS = 3;
let savePrompts = 0;

function offerSave(name: string, save: () => Promise<void>, id?: string): void {
  const locale = getLocale();
  toast(translate("browser.downloadPrompt.ready", { name: safeDownloadName(name) }, locale), {
    id,
    action: { label: translate("browser.downloadPrompt.save", {}, locale), onClick: () => void save() },
  });
}

const nextPrompt = () => `page-download-${savePrompts++ % SAVE_PROMPTS}`;

export function saveBrowserDownloadOnClick(download: BrowserDownload): void {
  const temporary = download.temporary ?? useChatRuntimeStore.getState().incognito;
  offerSave(downloadName(download.name, download.contentType), () => saveBrowserDownload({ ...download, temporary }), nextPrompt());
}

type LinkedFile = Pick<BrowserDownload, "site" | "temporary"> & { agentRule: (fileUrl: string) => DownloadsRule | null };

export function saveLinkedFile(url: string, name: string, file: LinkedFile, askFirst = false): void {
  const temporary = file.temporary ?? useChatRuntimeStore.getState().incognito;
  const save = () =>
    fetchBrowserPage({ url }, new AbortController().signal).then(
      (page) => saveBrowserDownload({ ...linkDownload(page, url, name), site: file.site, temporary, agentRule: file.agentRule(page.url) }),
      (error: unknown) => void toast.error(error instanceof Error ? error.message : String(error)),
    );
  if (askFirst) offerSave(name || fileNameFromUrl(url), save, nextPrompt());
  else void save();
}

/** Website files wait for approval first. `target`: a location already picked, null for none; omitted, the dialog opens when Settings asks. */
export async function saveBrowserDownload(given: BrowserDownload, target?: SaveHandle | null): Promise<void> {
  const download = { ...given, name: downloadName(given.name, given.contentType) };
  // Taken now: the fetch, approval and the save dialog can outlast the temporary chat.
  const temporary = download.temporary ?? useChatRuntimeStore.getState().incognito;
  if (target === undefined) {
    if (!(await approved(download.url, download.name, download.site, download.agentRule))) return;
    // A save dialog needs a fresh click; a late approval waits for a click on Save instead.
    if (saveNeedsClick()) {
      offerSave(download.name, () => writeDownload(download, undefined, temporary));
      return;
    }
  }
  await writeDownload(download, target, temporary);
}

async function writeDownload(
  download: BrowserDownload,
  target: SaveHandle | null | undefined,
  temporary: boolean,
): Promise<void> {
  const { blob, contentType, url } = download;
  const name = safeDownloadName(download.name);
  let saved: SavedNativeDownload | null = null;
  let picked: SaveHandle | null = null;
  const key = `save:${Date.now().toString(36)}:${Math.random().toString(36).slice(2)}`;
  beginDownload(key, name);
  try {
    if (isTauri) {
      // The app keeps the path so Download history can reveal it.
      saved = await saveNativeDownload(blob, name, useBrowserPrefsStore.getState().askWhereToSave, url);
      if (!saved) {
        abandonDownload(key);
        return;
      }
    } else {
      picked = target === undefined ? await pickSaveTarget(name) : target;
      if (picked) {
        const writable = await picked.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        // The browser must not infer an unapproved extension from its MIME registry.
        await downloadFile(blob.slice(0, blob.size, "application/octet-stream"), name);
      }
    }
  } catch (error) {
    abandonDownload(key);
    if (!isDownloadCancelled(error)) toast.error(error instanceof Error ? error.message : String(error));
    return;
  }
  const savedName = saved?.name || picked?.name || name;
  const item = { name: savedName, url, size: blob.size, contentType, nativeId: saved?.id };
  const historyId = useBrowserHistoryStore.getState().recordDownload(item, temporary);
  // Open uses the saved file when the app kept it; only otherwise does this session hold the bytes.
  const shown = finishDownload(
    key,
    { name: savedName, size: blob.size, contentType, url, nativeId: saved?.id, historyId, failed: false },
    saved && historyId ? undefined : { blob, name: savedName, contentType },
  );
  if (saved?.marked === false) {
    toast.warning(translate("browser.native.notMarked", { name: savedName }, getLocale()));
  } else if (!shown) {
    toast.success(translate("browser.downloads.complete", {}, getLocale()), { description: savedName });
  }
}

// Well inside the ~5 s a click lets a page open the save dialog.
const RESOLVE_BEFORE_ASK_MS = 1000;

function linkDownload(page: BrowserPage, url: string, linkName = ""): BrowserDownload {
  if (page.kind === "raw") {
    const name = downloadName(page.fileName ?? (linkName || fileNameFromUrl(page.url)), page.contentType);
    return { blob: page.blob, name, contentType: page.contentType, url };
  }
  const name = linkName || fileNameFromUrl(page.url);
  return {
    blob: new Blob([withBaseUrl(page.html, page.base)], { type: "text/html" }),
    name: /\.html?$/i.test(name) ? name : `${name}.html`,
    contentType: "text/html",
    url,
  };
}

/** Save what a link points at, fetched through the panel's proxy so any site works. */
export async function saveLinkAs(url: string): Promise<void> {
  const temporary = useChatRuntimeStore.getState().incognito;
  const controller = new AbortController();
  const pending = fetchBrowserPage({ url }, controller.signal).then((page) => linkDownload(page, url));
  // The dialog needs the menu click, which a slow fetch outlasts: ask with the resolved name
  // when the fetch is quick, else with the URL's.
  let target: SaveHandle | null | undefined;
  let asked: string | undefined;
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
    asked = name;
    try {
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
  const download = await pending;
  // Approved under the address's name: a file that turns out to run code asks again, by its own name.
  if (asked !== undefined && isDangerousDownload(download.name) && !isDangerousDownload(asked)) {
    if (!(await approved(url, download.name))) return;
  }
  await saveBrowserDownload({ ...download, temporary }, target);
}
