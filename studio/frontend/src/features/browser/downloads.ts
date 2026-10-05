// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fileNameFromUrl, withBaseUrl } from "./address";
import { fetchBrowserPage } from "./api";
import { useBrowserHistoryStore } from "./history-store";

export type BrowserDownload = { blob: Blob; name: string; contentType: string; url: string | null };

/** Save a file from the panel and add it to the download history. */
export async function saveBrowserDownload({ blob, name, contentType, url }: BrowserDownload): Promise<void> {
  try {
    await downloadFile(blob, name, contentType || undefined);
  } catch (error) {
    if (!isDownloadCancelled(error)) toast.error(error instanceof Error ? error.message : String(error));
    return;
  }
  useBrowserHistoryStore.getState().recordDownload({ name, url, size: blob.size, contentType });
}

/** Save what a link points at, fetched through the panel's proxy so any site works. */
export async function saveLinkAs(url: string): Promise<void> {
  const page = await fetchBrowserPage({ url }, new AbortController().signal);
  if (page.kind === "raw") {
    await saveBrowserDownload({ blob: page.blob, name: page.fileName ?? fileNameFromUrl(page.url), contentType: page.contentType, url });
    return;
  }
  const name = fileNameFromUrl(page.url);
  await saveBrowserDownload({
    blob: new Blob([withBaseUrl(page.html, page.base)], { type: "text/html" }),
    name: /\.html?$/i.test(name) ? name : `${name}.html`,
    contentType: "text/html",
    url,
  });
}
