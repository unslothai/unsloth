// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
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
