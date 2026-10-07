// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Desktop only: the app keeps each download's path (browser_downloads.rs) and the page gets an
// opaque id to reveal the file or check it still exists.

import { isTauri } from "@/lib/api-base";
import { NATIVE_FILE_NAME_HEADER, encodeNativeFilename } from "@/lib/native-files";

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const core = await import("@tauri-apps/api/core");
  return core.invoke<T>(command, args);
}

/** Save through the desktop app's dialog; the saved name and its id, or null if cancelled. */
export async function saveNativeDownload(blob: Blob, name: string): Promise<{ id: string; name: string } | null> {
  const core = await import("@tauri-apps/api/core");
  return core.invoke<{ id: string; name: string } | null>("browser_download_save", new Uint8Array(await blob.arrayBuffer()), {
    headers: { [NATIVE_FILE_NAME_HEADER]: encodeNativeFilename(name) },
  });
}

export function revealNativeDownload(id: string): Promise<void> {
  return invoke<void>("browser_download_reveal", { id });
}

/** Whether each download is still where it was saved; all true outside the desktop app. */
export async function nativeDownloadsExist(ids: string[]): Promise<boolean[]> {
  if (!isTauri || ids.length === 0) return ids.map(() => true);
  return invoke<boolean[]>("browser_download_exists", { ids });
}

/** Forget downloads taken off the history; the files stay. */
export function forgetNativeDownloads(ids: string[]): void {
  if (!isTauri || ids.length === 0) return;
  void invoke<void>("browser_download_forget", { ids }).catch(() => undefined);
}
