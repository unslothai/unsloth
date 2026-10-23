// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Open and Show in folder for a download, shared by the Downloads button and the Downloads page.

import { isTauri } from "@/lib/api-base";
import { keptDownloadFile } from "./download-activity";
import { isDangerousDownload } from "./download-safety";
import type { DownloadItem } from "./history-store";
import { openNativeDownload, revealNativeDownload } from "./native-downloads";
import { useBrowserStore } from "./store";

export type Target = Pick<DownloadItem, "name" | "contentType" | "url" | "nativeId"> & { keptId?: string };

/** Open a download: the saved file on the desktop, this session's copy in a tab, else its page. */
export function openTarget(target: Target, onFailed: (name: string) => void): (() => void) | undefined {
  if (isTauri && target.nativeId) {
    // The app refuses to open programs and scripts (browser_download_open); Show in folder still works.
    if (isDangerousDownload(target.name)) return undefined;
    const id = target.nativeId;
    return () => void openNativeDownload(id).catch(() => onFailed(target.name));
  }
  const kept = keptDownloadFile(target.keptId);
  if (kept) {
    return () =>
      useBrowserStore.getState().openFile({ ...kept, key: `download:${target.keptId}` });
  }
  const url = target.url;
  return url ? () => useBrowserStore.getState().openUrl(url, { newTab: true }) : undefined;
}

export function revealTarget(target: Target, onFailed: (name: string) => void): (() => void) | undefined {
  if (!isTauri || !target.nativeId) return undefined;
  const id = target.nativeId;
  return () => void revealNativeDownload(id).catch(() => onFailed(target.name));
}
