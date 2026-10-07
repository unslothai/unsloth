// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Desktop only: the app keeps each download's path (browser_downloads.rs) and the page gets an
// opaque id to reveal the file or check it still exists.

import { isTauri } from "@/lib/api-base";
import { NATIVE_FILE_NAME_HEADER, encodeNativeFilename } from "@/lib/native-files";
import { isWebUrl } from "./address";

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const core = await import("@tauri-apps/api/core");
  return core.invoke<T>(command, args);
}

/** Null if cancelled. A website file (`source`) is quarantine-marked so Gatekeeper/SmartScreen check it. */
export async function saveNativeDownload(
  blob: Blob,
  name: string,
  ask: boolean,
  source: string | null,
): Promise<{ id: string; name: string } | null> {
  const core = await import("@tauri-apps/api/core");
  const headers: Record<string, string> = {
    [NATIVE_FILE_NAME_HEADER]: encodeNativeFilename(name),
    "x-unsloth-ask": ask ? "1" : "0",
  };
  // href is ASCII (punycode host, escaped path), as a header value must be.
  const href = source && isWebUrl(source) ? safeHref(source) : null;
  if (href) headers["x-unsloth-source"] = href;
  return core.invoke<{ id: string; name: string } | null>("browser_download_save", new Uint8Array(await blob.arrayBuffer()), {
    headers,
  });
}

function safeHref(url: string): string | null {
  try {
    return new URL(url).href;
  } catch {
    return null;
  }
}

export function revealNativeDownload(id: string): Promise<void> {
  return invoke<void>("browser_download_reveal", { id });
}

/** Whether each download is still where it was saved; all true outside the desktop app. */
export async function nativeDownloadsExist(ids: string[]): Promise<boolean[]> {
  if (!isTauri || ids.length === 0) return ids.map(() => true);
  return invoke<boolean[]>("browser_download_exists", { ids });
}

export type DownloadFolder = { path: string; custom: boolean };

export function nativeDownloadFolder(): Promise<DownloadFolder> {
  return invoke<DownloadFolder>("browser_download_folder");
}

export function pickNativeDownloadFolder(): Promise<DownloadFolder | null> {
  return invoke<DownloadFolder | null>("browser_download_folder_pick");
}

export function resetNativeDownloadFolder(): Promise<DownloadFolder> {
  return invoke<DownloadFolder>("browser_download_folder_reset");
}

export function decideNativeDownload(id: string, allow: boolean, ask: boolean): Promise<void> {
  return invoke<void>("browser_download_decide", { id, allow, ask });
}

/** Forget downloads taken off the history; the files stay. */
export function forgetNativeDownloads(ids: string[]): void {
  if (!isTauri || ids.length === 0) return;
  void invoke<void>("browser_download_forget", { ids }).catch(() => undefined);
}
